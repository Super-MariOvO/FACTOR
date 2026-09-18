"""Trajectory-Anchored Critic (TAC).

TAC restores sparse intermediate states from a realized trajectory, samples
short inference-only continuations under a frozen behavior policy, and uses
the resulting Monte Carlo returns as regression targets for intermediate
values V_bar_phi. One-step TD residuals on boundary-adjusted values then
decompose the trajectory advantage across actions (paper Eq. 3-5).

Design (per paper):
    - Value head: 2-layer MLP (hidden 1024, GELU, output 1) applied to the
      last hidden state at the pre-action boundary.
    - The value head shares the policy backbone with stop-gradient: gradients
      from the critic loss never propagate into the backbone.
    - Polyak (EMA) target head with decay 0.995.
    - Trained with MSE on pooled MC targets drawn from a 10-step replay
      buffer, Adam lr 1e-4, no weight decay.
    - Checkpoint selection (paper supplement): one checkpoint uniformly at
      random from each trajectory half when T >= 4; trajectories with
      1 < T < 4 contribute one checkpoint from the midpoint; single-action
      trajectories contribute none.
    - Continuation budget: B = 2 checkpoints x M = 4 continuations per
      checkpoint, each rolling out up to 15 additional turns under the
      frozen behavior policy.
    - Boundary-adjusted TD credit (paper Eq. 3-5): with V~(x_0) = b_i the
      per-environment baseline, V~(x_t) = V_bar_phi(x_t) for 0 < t < T, and
      V~(x_T) = 0, the per-action credits
      A*_t = r_t + V~(x_{t+1}) - V~(x_t) telescope exactly:
      sum_t A*_t = G_i - b_i = A^seq.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """A sparse intermediate state restored from a realized trajectory.

    Attributes:
        trajectory_id: Identifier of the source trajectory.
        token_position: Absolute token index of the pre-action boundary.
        action_index: Index of the action (turn) this checkpoint precedes.
        hidden_state: Detached last hidden state at the boundary, shape [H].
        prefix_token_ids: Token ids of the realized prefix up to the boundary,
            used to condition inference-only continuations.
        realized_return: Return realized by the source trajectory from this
            checkpoint onward (used for sanity logging / fallback targets).
    """

    trajectory_id: int
    token_position: int
    action_index: int
    hidden_state: torch.Tensor
    prefix_token_ids: torch.Tensor
    realized_return: float = 0.0


def select_checkpoints(
    num_actions: int,
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    """Select sparse anchor positions along a trajectory (paper supplement).

    Checkpoints are selected without reference to hidden-state drift:

    - ``T >= 4``: one checkpoint uniformly at random from the first half
      (actions ``0 .. floor(T/2) - 1``) and one uniformly at random from the
      second half (actions ``floor(T/2) .. T - 1``);
    - ``1 < T < 4``: one checkpoint from the midpoint (action ``T // 2``);
    - ``T <= 1``: no checkpoints.

    Args:
        num_actions: Number of executed actions ``T`` in the trajectory.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Sorted list of action indices (in ``[0, T)``) chosen as checkpoints.
    """
    T = int(num_actions)
    if T <= 1:
        return []
    if T < 4:
        return [T // 2]
    half = T // 2
    first = int(torch.randint(0, half, (1,), generator=generator).item())
    second = int(torch.randint(half, T, (1,), generator=generator).item())
    return sorted((first, second))


# ---------------------------------------------------------------------------
# One-step TD action credit (paper Eq. 3-5)
# ---------------------------------------------------------------------------


def boundary_adjusted_values(
    values: torch.Tensor,
    baseline: torch.Tensor | float,
) -> torch.Tensor:
    """Boundary-adjusted potentials V~ of paper Eq. 3.

    For a trajectory with T actions and per-action boundary values
    ``values[t] = V_bar_phi(x_t)`` (t = 0 .. T-1):

        V~(x_0) = b_i            (the baseline used to build A^seq)
        V~(x_t) = V_bar_phi(x_t) for 0 < t < T
        V~(x_T) = 0

    Args:
        values: 1-D tensor [T] of value-head estimates at each action's
            pre-action boundary.
        baseline: Scalar baseline b_i (leave-one-out group baseline on
            ALFWorld, 0 on WebShop and ScienceWorld).

    Returns:
        1-D tensor [T + 1] with V~(x_0) .. V~(x_T).
    """
    T = values.shape[0]
    v_tilde = torch.zeros(T + 1, dtype=values.dtype, device=values.device)
    if not torch.is_tensor(baseline):
        baseline = torch.as_tensor(
            baseline, dtype=values.dtype, device=values.device
        )
    v_tilde[0] = baseline.to(dtype=values.dtype, device=values.device)
    if T > 1:
        v_tilde[1:T] = values[1:T]
    # v_tilde[T] is already 0.
    return v_tilde


def compute_tac_advantages(
    action_rewards: torch.Tensor,
    values: torch.Tensor,
    baseline: torch.Tensor | float,
) -> torch.Tensor:
    """One-step TD action credits of paper Eq. 4 with boundary values Eq. 3.

        A*_t = r_t + V~(x_{t+1}) - V~(x_t),   t = 0 .. T-1.

    The credits telescope exactly (paper Eq. 5):

        sum_t A*_t = sum_t r_t + V~(x_T) - V~(x_0) = G_i - b_i = A^seq.

    The identity holds independently of value-head accuracy; it is exact in
    exact arithmetic and holds up to floating-point rounding in finite
    precision (bit-exact whenever the inputs are exactly representable,
    e.g. dyadic rationals in float64).

    Args:
        action_rewards: 1-D tensor [T] of per-action environment rewards in
            the environment's reward coordinate (terminal reward on the last
            action; typically zero elsewhere).
        values: 1-D tensor [T] with V_bar_phi(x_t) at each action's
            pre-action boundary (from the Polyak target head).
        baseline: Scalar b_i used for A^seq (see
            :func:`boundary_adjusted_values`).

    Returns:
        1-D tensor [T] of per-action credits A*_t.
    """
    if action_rewards.shape != values.shape:
        raise ValueError(
            f"action_rewards {tuple(action_rewards.shape)} and values "
            f"{tuple(values.shape)} must both be [T]"
        )
    v_tilde = boundary_adjusted_values(values, baseline)
    return action_rewards + v_tilde[1:] - v_tilde[:-1]


# ---------------------------------------------------------------------------
# Value head
# ---------------------------------------------------------------------------


class TACValueHead(nn.Module):
    """2-layer MLP value head: Linear(H -> 1024), GELU, Linear(1024 -> 1)."""

    def __init__(self, hidden_size: int, mlp_hidden: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        # Small output scale so initial values are near zero.
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Map hidden state [..., H] to scalar value [...]."""
        return self.net(hidden_state).squeeze(-1)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass
class MCTarget:
    """A pooled Monte Carlo regression target for the critic.

    Attributes:
        hidden_state: Detached boundary hidden state [H].
        target: Monte Carlo return estimate (float scalar).
    """

    hidden_state: torch.Tensor
    target: float


class TACReplayBuffer:
    """Fixed-window replay buffer pooling MC targets from the last K steps."""

    def __init__(self, window_steps: int = 10) -> None:
        self.window_steps = window_steps
        self._steps: Deque[List[MCTarget]] = deque(maxlen=window_steps)

    def add_step(self, targets: List[MCTarget]) -> None:
        """Register all MC targets produced at one training step."""
        self._steps.append(list(targets))

    def sample_all(self) -> List[MCTarget]:
        """Return the pooled targets from the last ``window_steps`` steps."""
        pooled: List[MCTarget] = []
        for step_targets in self._steps:
            pooled.extend(step_targets)
        return pooled

    def __len__(self) -> int:
        return sum(len(t) for t in self._steps)


# ---------------------------------------------------------------------------
# TAC module
# ---------------------------------------------------------------------------


class TACModule(nn.Module):
    """Full TAC critic: online head, Polyak target head, buffer, optimizer.

    The module owns its own optimizer (Adam, lr 1e-4, no weight decay) and is
    stepped independently of the PPO policy update.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden: int = 1024,
        ema_decay: float = 0.995,
        lr: float = 1e-4,
        buffer_window: int = 10,
        num_checkpoints: int = 2,            # B: restored checkpoints
        continuations_per_checkpoint: int = 4,  # M: continuations per checkpoint
        continuation_horizon: int = 15,      # max additional turns per continuation
        warmup_steps: int = 10,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.online_head = TACValueHead(hidden_size, mlp_hidden)
        self.target_head = copy.deepcopy(self.online_head)
        for p in self.target_head.parameters():
            p.requires_grad_(False)

        self.ema_decay = ema_decay
        self.buffer = TACReplayBuffer(window_steps=buffer_window)
        self.num_checkpoints = num_checkpoints
        self.continuations_per_checkpoint = continuations_per_checkpoint
        self.continuation_horizon = continuation_horizon
        self.warmup_steps = warmup_steps
        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = torch.Generator().manual_seed(seed)

        self.optimizer = torch.optim.Adam(
            self.online_head.parameters(), lr=lr, weight_decay=0.0
        )

        if device is not None:
            self.to(device)

    # -- target head maintenance -------------------------------------------

    @torch.no_grad()
    def polyak_update(self) -> None:
        """EMA update of the target head toward the online head."""
        for tgt, src in zip(
            self.target_head.parameters(), self.online_head.parameters()
        ):
            tgt.mul_(self.ema_decay).add_(src, alpha=1.0 - self.ema_decay)
        for tgt, src in zip(
            self.target_head.buffers(), self.online_head.buffers()
        ):
            tgt.copy_(src)

    # -- value computation ---------------------------------------------------

    def value(
        self, hidden_states: torch.Tensor, use_target: bool = False
    ) -> torch.Tensor:
        """Compute V(h) with stop-gradient on the backbone features.

        Args:
            hidden_states: Backbone hidden states [..., H]. They are detached
                here so the critic never backprops into the shared backbone.
            use_target: If True, use the Polyak target head.
        """
        head = self.target_head if use_target else self.online_head
        return head(hidden_states.detach())

    # -- MC target generation -------------------------------------------------

    @torch.no_grad()
    def build_targets_for_trajectory(
        self,
        trajectory_id: int,
        hidden_states: torch.Tensor,
        action_boundaries: Sequence[int],
        prefix_tokens_fn: Callable[[int], torch.Tensor],
        continuation_fn: Callable[[torch.Tensor, int, int], Sequence[float]],
        realized_return: float,
        discount: float = 1.0,
    ) -> List[MCTarget]:
        """Generate pooled MC targets for one realized trajectory.

        Args:
            trajectory_id: Source trajectory identifier.
            hidden_states: [T, H] detached hidden states for the trajectory.
            action_boundaries: Pre-action boundary indices.
            prefix_tokens_fn: Maps a boundary index to the token ids of the
                realized prefix, conditioning continuations.
            continuation_fn: ``continuation_fn(prefix_ids, M, horizon)``;
                samples M short inference-only continuations of up to
                ``horizon`` additional turns under the frozen behavior
                policy and returns their realized terminal outcomes in the
                per-environment reward coordinate.
            realized_return: Return of the source trajectory (fallback when a
                continuation cannot be scored, e.g. environment error at
                restore).
            discount: Discount factor for the continuation returns.

        Returns:
            List of MCTarget, one per selected checkpoint.
        """
        anchor_positions = select_checkpoints(
            len(action_boundaries), generator=self._generator
        )
        anchor_positions = anchor_positions[: self.num_checkpoints]
        targets: List[MCTarget] = []
        for pos in anchor_positions:
            boundary = action_boundaries[pos]
            prefix_ids = prefix_tokens_fn(boundary)
            cont_returns = continuation_fn(
                prefix_ids,
                self.continuations_per_checkpoint,
                self.continuation_horizon,
            )
            if len(cont_returns) == 0:
                mc = realized_return
            else:
                mc = sum(cont_returns) / len(cont_returns)
            targets.append(
                MCTarget(hidden_state=hidden_states[boundary].clone(), target=mc)
            )
        return targets

    # -- training --------------------------------------------------------------

    def train_step(self, targets: List[MCTarget], batch_size: int = 256) -> Dict[str, float]:
        """One MSE regression step over pooled replay targets.

        Args:
            targets: Newly generated targets for the current step; they are
                pushed into the replay buffer before sampling.
            batch_size: Max number of pooled targets per gradient step.

        Returns:
            Dict with the MSE loss and the number of pooled targets.
        """
        self.buffer.add_step(targets)
        pooled = self.buffer.sample_all()
        if len(pooled) == 0:
            return {"critic_loss": 0.0, "num_targets": 0}

        if len(pooled) > batch_size:
            idx = torch.randperm(len(pooled))[:batch_size].tolist()
            pooled = [pooled[i] for i in idx]

        device = next(self.online_head.parameters()).device
        h = torch.stack([t.hidden_state for t in pooled]).to(device)
        y = torch.tensor(
            [t.target for t in pooled], dtype=torch.float32, device=device
        )

        pred = self.value(h)
        loss = F.mse_loss(pred, y)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.polyak_update()

        return {"critic_loss": float(loss.item()), "num_targets": len(pooled)}
