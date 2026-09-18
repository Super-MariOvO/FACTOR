"""FACTOR trainer: integrates TAC, HTA and APM into a PPO loop.

The trainer is framework-anchored to verl/SERL (commit b338174,
serl_action_mask branch): the rollout worker supplies token ids, per-token
log-probs of the behavior policy, pre-action boundary hidden states, action
spans, per-action rewards and group ids; the trainer computes FACTOR
token-level advantages and runs one PPO epoch.

Per training step k (1-indexed):

1.  Roll out ``batch_size`` trajectories (group size 8 per task) with
    temperature 1.0, context 4096, generation 512.
2.  TAC: restore sparse checkpoints (one uniformly at random from each
    trajectory half), sample M=4 inference-only continuations of up to 15
    turns from each of B=2 checkpoints under the frozen behavior policy,
    pool MC returns into the 10-step replay buffer, and fit the value head
    with MSE (lr 1e-4, no weight decay, Polyak 0.995). During the first 10
    steps the critic only warms up and action credit falls back to the
    uniform (telescoping) split A^seq / T.
3.  Compute per-action TD credits A*_t = r_t + V~(x_{t+1}) - V~(x_t)
    (paper Eq. 3-5) with V~(x_0) = b_i the per-environment baseline and
    V~(x_T) = 0, so that sum_t A*_t = A^seq exactly.
4.  HTA: during steps 11..49, re-score every action token with the frozen
    teacher (feedback Phi_t visible), form the gap Delta, turn it into the
    outcome-aligned score s = sgn(A*_t) * clip(Delta, -d, d) (Eq. 6), and
    build the mean-one multipliers omega with eta_k = 0.7 * max(1 - k/50, 0).
5.  Token-level advantage: A_{t,j} = omega_{t,j} * A_t*.
6.  One PPO epoch with clip eps 0.2, Adam lr 5e-7, grad clip 1.0, and the
    full objective (Eq. 8): action-mean L_RL^act + token-mean L_RL^non-act
    (coefficient A^seq) + lambda_k * L_act^SERL with lambda_k = alpha_k.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from factor.apm import apm_token_advantages
from factor.hta import (
    HTAConfig,
    HindsightTeacher,
    build_action_token_weights,
    compute_hindsight_gap,
    compute_serl_action_kl,
)
from factor.tac import MCTarget, TACModule, compute_tac_advantages
from factor.training.config import FACTORConfig

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """One rolled-out trajectory with all FACTOR annotations."""

    input_ids: torch.Tensor            # [T] full token stream
    action_spans: List[slice]          # token slices of each agent action
    boundary_hidden: torch.Tensor      # [num_boundaries, H] detached h at
                                       # pre-action boundaries
    action_rewards: torch.Tensor       # [num_actions] env reward per action
    old_logprobs: torch.Tensor         # [T] behavior-policy token log-probs
    group_id: int                      # trajectories sharing a task prompt
    terminal_reward: float             # raw final outcome G_i (or score)
    env_feedback_ids: Optional[List[torch.Tensor]] = None  # Phi_t token ids
    trajectory_id: int = 0


@dataclass
class RewardSpec:
    """Per-environment reward construction for one trajectory.

    Attributes:
        terminal: Terminal reward r_{T-1} in the environment's reward
            coordinate (raw success for ALFWorld/ScienceWorld, the
            standardized score A^seq for WebShop).
        a_seq: Trajectory-level advantage A^seq = G_i - b_i.
        baseline: Baseline b_i used as V~(x_0) in the TAC boundary values
            (leave-one-out group baseline on ALFWorld, 0 elsewhere).
    """

    terminal: float
    a_seq: float
    baseline: float


@dataclass
class TrajectoryBatch:
    """Collated batch of trajectories for one training step."""

    trajectories: List[Trajectory] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.trajectories)

    def _groups(self) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        for i, tr in enumerate(self.trajectories):
            groups.setdefault(tr.group_id, []).append(i)
        return groups

    def construct_rewards(self, env_name: str) -> List[RewardSpec]:
        """Environment-specific reward construction (paper supplement).

        ALFWorld: binary success G_i in {0, 1}; leave-one-out group baseline
            b_i = (1 / (K - 1)) * sum_{j != i} G_j within group size K;
            A^seq = G_i - b_i; terminal reward r_{T-1} = G_i; V~(x_0) = b_i.
        WebShop: continuous score G~_i; group standardization
            A^seq = (G~_i - mu) / (sigma + 1e-8), with A^seq = 0 for groups
            with sigma < 1e-6; terminal reward r_{T-1} = A^seq; V~(x_0) = 0.
        ScienceWorld: binary success, no baseline; A^seq = G_i;
            terminal reward r_{T-1} = G_i; V~(x_0) = 0.

        Args:
            env_name: One of ``alfworld``, ``webshop``, ``scienceworld``
                (matched case-insensitively on the prefix).

        Returns:
            List of :class:`RewardSpec`, one per trajectory.
        """
        env = env_name.lower()
        specs = [RewardSpec(0.0, 0.0, 0.0) for _ in self.trajectories]
        for idxs in self._groups().values():
            g = torch.tensor(
                [self.trajectories[i].terminal_reward for i in idxs],
                dtype=torch.float64,
            )
            if env.startswith("alfworld"):
                # Leave-one-out baseline within the group.
                K = len(idxs)
                for pos, i in enumerate(idxs):
                    if K > 1:
                        b = float((g.sum() - g[pos]) / (K - 1))
                    else:
                        b = 0.0
                    a_seq = float(g[pos]) - b
                    specs[i] = RewardSpec(
                        terminal=float(g[pos]), a_seq=a_seq, baseline=b
                    )
            elif env.startswith("webshop"):
                mu = g.mean()
                sigma = g.std(unbiased=False)
                for pos, i in enumerate(idxs):
                    if float(sigma) < 1e-6:
                        a_seq = 0.0
                    else:
                        a_seq = float((g[pos] - mu) / (sigma + 1e-8))
                    specs[i] = RewardSpec(terminal=a_seq, a_seq=a_seq, baseline=0.0)
            elif env.startswith("scienceworld"):
                for pos, i in enumerate(idxs):
                    specs[i] = RewardSpec(
                        terminal=float(g[pos]), a_seq=float(g[pos]), baseline=0.0
                    )
            else:
                raise ValueError(f"unknown environment: {env_name!r}")
        return specs


class FACTORTrainer:
    """Owns the policy, the TAC critic and the FACTOR advantage pipeline."""

    def __init__(
        self,
        policy: nn.Module,
        config: FACTORConfig,
        hidden_size: int,
        continuation_sampler: Optional[
            Callable[[torch.Tensor, int, int], Sequence[float]]
        ] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.device = device or torch.device("cpu")

        self.tac = TACModule(
            hidden_size=hidden_size,
            mlp_hidden=config.tac_mlp_hidden,
            ema_decay=config.tac_ema_decay,
            lr=config.tac_lr,
            buffer_window=config.tac_buffer_window,
            num_checkpoints=config.tac_num_checkpoints,
            continuations_per_checkpoint=config.tac_continuations_per_checkpoint,
            continuation_horizon=config.tac_continuation_horizon,
            warmup_steps=config.tac_warmup_steps,
            device=self.device,
        )
        self.hta_config = HTAConfig(
            eta_0=config.hta_eta_0,
            decay_steps=config.hta_decay_steps,
            active_start=config.hta_active_start,
            active_end=config.hta_active_end,
            tau=config.hta_tau,
            clip_d=config.hta_clip_d,
        )
        self.teacher: Optional[HindsightTeacher] = None
        self.continuation_sampler = continuation_sampler

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.ppo.lr
        )
        self.step_idx = 0

    # -- teacher lifecycle -------------------------------------------------

    def refresh_teacher(self) -> None:
        """Snapshot the current policy as the frozen hindsight teacher."""
        self.teacher = HindsightTeacher(self.policy)
        self.teacher.teacher.to(self.device)

    # -- TAC -----------------------------------------------------------------

    def update_critic(
        self,
        batch: TrajectoryBatch,
        prefix_tokens_fn: Callable[[Trajectory, int], torch.Tensor],
    ) -> Dict[str, float]:
        """Generate MC targets for the batch and run one critic step."""
        all_targets: List[MCTarget] = []
        if self.continuation_sampler is not None:
            for tr in batch.trajectories:
                boundaries = [s.start for s in tr.action_spans]
                targets = self.tac.build_targets_for_trajectory(
                    trajectory_id=tr.trajectory_id,
                    hidden_states=tr.boundary_hidden,
                    action_boundaries=list(range(len(boundaries))),
                    prefix_tokens_fn=lambda b, tr=tr: prefix_tokens_fn(tr, b),
                    continuation_fn=self.continuation_sampler,
                    realized_return=tr.terminal_reward,
                )
                all_targets.extend(targets)
        stats = self.tac.train_step(all_targets)
        logger.info("step %d TAC: %s", self.step_idx + 1, stats)
        return stats

    # -- advantages -------------------------------------------------------------

    def compute_action_credits(
        self,
        batch: TrajectoryBatch,
        specs: Sequence[RewardSpec],
        step: int,
    ) -> List[torch.Tensor]:
        """Per-action TD credits A*_t for every trajectory (paper Eq. 3-5).

        After the value-head warm-up the credits are the one-step TD
        residuals ``r_t + V~(x_{t+1}) - V~(x_t)`` on boundary-adjusted
        values, which telescope exactly to A^seq. During warm-up (steps
        1..10) the value head is not yet trusted, so credit falls back to
        the uniform split A^seq / T, which also telescopes exactly.

        Args:
            batch: Rolled-out trajectories.
            specs: Per-trajectory reward construction from
                :meth:`TrajectoryBatch.construct_rewards`.
            step: Current 1-indexed training step.

        Returns:
            Per-trajectory 1-D tensors [num_actions] of A*_t.
        """
        credits: List[torch.Tensor] = []
        for i, tr in enumerate(batch.trajectories):
            T = len(tr.action_spans)
            if T == 0:
                credits.append(torch.zeros(0))
                continue
            if step <= self.config.tac_warmup_steps:
                # Uniform fallback: A*_t = A^seq / T (telescopes exactly).
                credits.append(
                    torch.full((T,), specs[i].a_seq / T, dtype=torch.float32)
                )
                continue
            rewards = tr.action_rewards.clone().float()
            rewards[-1] = specs[i].terminal
            with torch.no_grad():
                values = self.tac.value(tr.boundary_hidden, use_target=True)
            credits.append(
                compute_tac_advantages(
                    action_rewards=rewards,
                    values=values.float(),
                    baseline=specs[i].baseline,
                )
            )
        return credits

    def compute_token_advantages(
        self,
        batch: TrajectoryBatch,
        teacher_logprobs: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """Full FACTOR advantage pipeline for one batch.

        Args:
            batch: Rolled-out trajectories.
            teacher_logprobs: Per-trajectory teacher token log-probs with
                feedback visible (required during steps 11..49; may be None
                when eta_k = 0).

        Returns:
            Per-trajectory dense token advantage tensors aligned with
            ``Trajectory.input_ids`` (zeros on non-action tokens).
        """
        self.step_idx += 1
        step = self.step_idx
        specs = batch.construct_rewards(self.config.rollout.env_name)
        action_credits = self.compute_action_credits(batch, specs, step)

        dense_advantages: List[torch.Tensor] = []
        for i, tr in enumerate(batch.trajectories):
            action_adv = action_credits[i]

            # HTA token allocation weights omega_{t,j} (Eq. 6-7).
            token_weights = torch.zeros_like(tr.old_logprobs)
            eta_k = self.hta_config.eta(step)
            if eta_k > 0.0 and teacher_logprobs is not None:
                gap = compute_hindsight_gap(
                    teacher_logprobs[i], tr.old_logprobs
                ).unsqueeze(0)
                weights = build_action_token_weights(
                    gaps=gap,
                    action_spans=[tr.action_spans],
                    action_advantages=[action_adv],
                    step=step,
                    config=self.hta_config,
                ).squeeze(0)
            else:
                # Uniform allocation: omega = 1 on action tokens.
                weights = torch.zeros_like(tr.old_logprobs)
                for s in tr.action_spans:
                    weights[s] = 1.0

            # APM broadcast: A_{t,j} = omega_{t,j} * A_t*.
            per_action_omegas = [weights[s] for s in tr.action_spans]
            token_adv_list = apm_token_advantages(action_adv, per_action_omegas)
            for s, adv_t in zip(tr.action_spans, token_adv_list):
                token_weights[s] = adv_t
            dense_advantages.append(token_weights)

        return dense_advantages

    # -- PPO ------------------------------------------------------------------

    def ppo_update(
        self,
        batch: TrajectoryBatch,
        new_logprobs: List[torch.Tensor],
        advantages: List[torch.Tensor],
        action_masks: List[torch.Tensor],
        non_action_masks: Optional[List[torch.Tensor]] = None,
        teacher_logits: Optional[List[torch.Tensor]] = None,
        policy_logits: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """One PPO epoch with the full FACTOR objective (paper Eq. 7-8).

        The total loss is

            L = L_RL^act + L_RL^non-act + lambda_k * L_act^SERL,

        where

        - ``L_RL^act`` is the action-mean PPO surrogate (Eq. 7): tokens are
          averaged within each action (1/L_t) and actions are averaged
          across the batch (1 / sum_i T_i), with token coefficients
          C_{t,j} = omega_{t,j} * A*_t;
        - ``L_RL^non-act`` is a separate token-mean PPO branch over
          non-action (reasoning/formatting) tokens with the trajectory-level
          coefficient A^seq;
        - ``L_act^SERL`` is SERL's auxiliary action-only distillation KL
          (teacher with feedback visible vs. policy, forward KL on action
          tokens), weighted by lambda_k = alpha_k = max(1 - k/50, 0).

        Args:
            batch: Rolled-out trajectories.
            new_logprobs: Per-trajectory current-policy token log-probs.
            advantages: Per-trajectory dense token coefficients C_{t,j}
                (nonzero only on action tokens), from
                :meth:`compute_token_advantages`.
            action_masks: Per-trajectory mask that is 1 on action tokens.
            non_action_masks: Per-trajectory mask that is 1 on non-action
                (reasoning/formatting) response tokens. Defaults to the
                complement of ``action_masks``.
            teacher_logits: Per-trajectory feedback-conditioned teacher
                logits [T, V] for the SERL auxiliary KL. If None (or
                ``policy_logits`` is None), the auxiliary term is skipped.
            policy_logits: Per-trajectory current-policy logits [T, V].

        Returns:
            Dict with the total loss and each component.
        """
        eps = self.config.ppo.clip_eps
        specs = batch.construct_rewards(self.config.rollout.env_name)
        lambda_k = self.config.serl_lambda(max(self.step_idx, 1))

        # Accumulators for the two reward-weighted branches (exact global
        # reductions: action-mean for L_RL^act, token-mean for L_RL^non-act).
        act_surrogate_sum = torch.zeros((), device=self.device)
        total_actions = 0
        nonact_surrogate_sum = torch.zeros((), device=self.device)
        total_nonact_tokens = 0
        serl_kl_sum = torch.zeros((), device=self.device)
        total_act_tokens = 0

        use_serl = (
            lambda_k > 0.0
            and teacher_logits is not None
            and policy_logits is not None
        )

        for i, (tr, logp, adv, mask) in enumerate(
            zip(batch.trajectories, new_logprobs, advantages, action_masks)
        ):
            logp = logp.to(self.device)
            adv = adv.to(self.device)
            mask = mask.to(self.device).float()
            ratio = torch.exp(logp - tr.old_logprobs.to(self.device))

            # -- L_RL^act: action-mean surrogate (Eq. 7) -------------------
            clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
            min_surr = torch.min(ratio * adv, clipped * adv)
            for s in tr.action_spans:
                act_surrogate_sum = act_surrogate_sum + min_surr[s].mean()
                total_actions += 1

            # -- L_RL^non-act: token-mean branch with A^seq ----------------
            if non_action_masks is not None:
                non_mask = non_action_masks[i].to(self.device).float()
            else:
                non_mask = 1.0 - mask
            a_seq = torch.as_tensor(
                specs[i].a_seq, dtype=logp.dtype, device=self.device
            )
            surr1 = ratio * a_seq
            surr2 = clipped * a_seq
            min_surr_na = torch.min(surr1, surr2)
            nonact_surrogate_sum = (
                nonact_surrogate_sum + (min_surr_na * non_mask).sum()
            )
            total_nonact_tokens += int(non_mask.sum().item())

            # -- lambda_k * L_act^SERL: action-only distillation KL --------
            if use_serl:
                kl_sum = compute_serl_action_kl(
                    teacher_logits[i].to(self.device),
                    policy_logits[i].to(self.device),
                    mask,
                    reduction="sum",
                )
                serl_kl_sum = serl_kl_sum + kl_sum
                total_act_tokens += int(mask.sum().item())

        loss_act = -act_surrogate_sum / max(total_actions, 1)
        loss_nonact = -nonact_surrogate_sum / max(total_nonact_tokens, 1)
        serl_kl = serl_kl_sum / max(total_act_tokens, 1) if use_serl else (
            torch.zeros((), device=self.device)
        )
        total_loss = loss_act + loss_nonact + lambda_k * serl_kl

        self.policy_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.ppo.grad_clip
        )
        self.policy_optimizer.step()
        return {
            "policy_loss": float(total_loss.item()),
            "loss_rl_act": float(loss_act.item()),
            "loss_rl_nonact": float(loss_nonact.item()),
            "serl_lambda": float(lambda_k),
            "serl_kl": float(serl_kl.item()),
            "num_actions": total_actions,
            "num_nonact_tokens": total_nonact_tokens,
        }
