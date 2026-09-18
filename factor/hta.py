"""Hindsight Token Allocation (HTA).

The teacher pi_T is a frozen copy of the current policy pi_theta, re-scored
with the environment feedback Phi_t made visible. For each token j of action
t we form the hindsight likelihood gap

    Delta_{t,j} = log pi_T(a_j | x_t, Phi_t) - log pi_theta(a_j | x_t)

which measures how much the feedback would have raised the likelihood of the
token. The outcome-aligned token score (paper Eq. 6) is

    s_{t,j} = sgn(A*_t) * clip(Delta_{t,j}, -d, +d),   d = 3.0,

so that when A*_t > 0 tokens with higher gaps receive more positive credit,
and when A*_t < 0 tokens with lower gaps bear more negative credit. The
token allocation weight within an action of length L_t is (paper Eq. 7)

    rho_{t,j} = (1 - eta_k) / L_t + eta_k * softmax_j(s_{t,j} / tau)

with tau the temperature, and a linear decay schedule

    eta_k = eta_0 * max(1 - k / 50, 0),   eta_0 = 0.7,

active during training steps 11..49 (eta_k = 0 during the 10-step value-head
warm-up and from step 50 on). The mean-one multiplier is

    omega_{t,j} = L_t * rho_{t,j},  so that  (1/L_t) sum_j omega_{t,j} = 1.

This module also implements SERL's auxiliary action-only distillation term
(SERL Eq. 14), which FACTOR keeps unchanged with lambda_k = alpha_k:
the forward KL from the feedback-conditioned teacher to the policy,
restricted to executable-action tokens.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HTAConfig:
    """Hyper-parameters for hindsight token allocation."""

    eta_0: float = 0.7            # initial mixing coefficient
    decay_steps: int = 50         # eta_k hits zero at step 50
    active_start: int = 11        # first step with teacher allocation
    active_end: int = 49          # last step with teacher allocation
    tau: float = 1.0              # softmax temperature over scores
    clip_d: float = 3.0           # symmetric clip bound d on the gap (Eq. 6)

    def eta(self, step: int) -> float:
        """Mixing coefficient eta_k for a 1-indexed training step."""
        if step < self.active_start or step > self.active_end:
            return 0.0
        return self.eta_0 * max(1.0 - step / self.decay_steps, 0.0)


class HindsightTeacher:
    """Frozen copy of the policy re-scored with feedback visible."""

    def __init__(self, policy: nn.Module) -> None:
        self.teacher = copy.deepcopy(policy)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token teacher log-probs under the feedback-visible context.

        Args:
            input_ids: [B, T] token ids of the teacher input, where each
                action is re-scored with the realized feedback Phi_t spliced
                into the context.
            attention_mask: [B, T] attention mask.
            action_token_mask: [B, T] mask that is 1 on tokens belonging to
                agent actions (tokens to be scored).

        Returns:
            Tensor [B, T] of teacher log-probs aligned with ``input_ids``
            (positions outside ``action_token_mask`` are zero).
        """
        outputs = self.teacher(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = outputs.logits[:, :-1]
        labels = input_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        token_logp = torch.gather(logp, 2, labels.unsqueeze(-1)).squeeze(-1)
        mask = action_token_mask[:, 1:].float()
        token_logp = token_logp * mask
        return F.pad(token_logp, (1, 0), value=0.0)


def compute_hindsight_gap(
    teacher_logprobs: torch.Tensor,
    policy_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Delta_{t,j} = log pi_T(a_j | x_t, Phi_t) - log pi_theta(a_j | x_t).

    The gap is returned unclipped; clipping happens inside the
    outcome-aligned score of Eq. 6 (:func:`compute_outcome_score`).
    """
    return teacher_logprobs - policy_logprobs.detach()


def compute_outcome_score(
    gap: torch.Tensor,
    action_advantage: torch.Tensor | float,
    d: float = 3.0,
) -> torch.Tensor:
    """Outcome-aligned token score of paper Eq. 6.

        s_{t,j} = sgn(A*_t) * clip(Delta_{t,j}, -d, +d).

    Args:
        gap: 1-D tensor of hindsight gaps Delta_{t,j} for one action.
        action_advantage: Scalar action credit A*_t (only its sign is used).
            A*_t = 0 yields all-zero scores, hence uniform allocation.
        d: Symmetric clip bound (paper uses d = 3.0).

    Returns:
        1-D tensor of scores s_{t,j}, same shape as ``gap``.
    """
    if not torch.is_tensor(action_advantage):
        action_advantage = torch.as_tensor(
            action_advantage, dtype=gap.dtype, device=gap.device
        )
    sign = torch.sign(action_advantage.to(dtype=gap.dtype, device=gap.device))
    return sign * gap.clamp(min=-d, max=d)


def compute_token_allocation(
    scores: Sequence[torch.Tensor],
    eta_k: float,
    tau: float = 1.0,
) -> List[torch.Tensor]:
    """Compute mean-one token multipliers omega for each action (Eq. 7).

    Args:
        scores: Sequence of 1-D tensors; ``scores[t]`` holds the
            outcome-aligned score s_{t,j} for each token j of action t
            (length L_t), i.e. the output of :func:`compute_outcome_score`.
        eta_k: Mixing coefficient for the current step (0 disables the
            teacher term and yields uniform allocation).
        tau: Softmax temperature.

    Returns:
        List of 1-D tensors ``omega_t`` with mean one within each action,
        i.e. ``omega_t.mean() == 1`` and ``sum_j rho_{t,j} == 1``.
    """
    omegas: List[torch.Tensor] = []
    for s in scores:
        L = s.numel()
        if L == 0:
            omegas.append(s)
            continue
        if eta_k <= 0.0:
            rho = torch.full_like(s, 1.0 / L)
        else:
            soft = torch.softmax(s / tau, dim=0)
            rho = (1.0 - eta_k) / L + eta_k * soft
        omegas.append(L * rho)
    return omegas


def build_action_token_weights(
    gaps: torch.Tensor,
    action_spans: Sequence[Sequence[slice]],
    action_advantages: Sequence[torch.Tensor],
    step: int,
    config: HTAConfig,
) -> torch.Tensor:
    """Dense per-token weights for a batch, from per-action gaps and credits.

    For every action t the hindsight gaps are first turned into
    outcome-aligned scores via Eq. 6 (``sgn(A*_t) * clip(Delta, -d, d)``)
    and only then into the mean-one allocation of Eq. 7
    (``softmax(s / tau)`` mixed with the uniform floor).

    Args:
        gaps: [B, T] hindsight gaps (zeros outside action tokens).
        action_spans: ``action_spans[b]`` is the list of token slices that
            make up each action of trajectory b.
        action_advantages: ``action_advantages[b]`` is a 1-D tensor of
            per-action credits A*_t for trajectory b (signs drive Eq. 6).
        step: Current 1-indexed training step (drives the eta schedule).
        config: HTA hyper-parameters.

    Returns:
        Tensor [B, T] of mean-one (per action) multipliers omega_{t,j};
        positions outside any action are zero.
    """
    eta_k = config.eta(step)
    weights = torch.zeros_like(gaps)
    for b, spans in enumerate(action_spans):
        scores = [
            compute_outcome_score(gaps[b, s], action_advantages[b][t], config.clip_d)
            for t, s in enumerate(spans)
        ]
        omegas = compute_token_allocation(scores, eta_k=eta_k, tau=config.tau)
        for s, w in zip(spans, omegas):
            weights[b, s] = w
    return weights


def compute_serl_action_kl(
    teacher_logits: torch.Tensor,
    policy_logits: torch.Tensor,
    action_mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """SERL's auxiliary action-only distillation term (SERL Eq. 14).

        L_act = mean over action tokens of
                KL( pi_T(. | x_t, Phi_t, y_{t,<j}) || pi_theta(. | x_t, y_{t,<j}) )

    FACTOR keeps this term unchanged, weighted by lambda_k = alpha_k in the
    full objective (paper Eq. 8).

    Args:
        teacher_logits: [..., V] feedback-conditioned teacher logits
            (stop-gradient is applied internally).
        policy_logits: [..., V] current-policy logits, same shape.
        action_mask: [...] mask that is 1 on executable-action tokens.
        reduction: ``"mean"`` for the token-mean over masked positions,
            ``"sum"`` for the masked sum (used to build an exact batch-level
            token-mean), ``"none"`` for the per-token KL tensor.

    Returns:
        Scalar (``"mean"``/``"sum"``) or per-token tensor (``"none"``) of
        forward KL values on masked positions.
    """
    mask = action_mask.float()
    log_p_t = torch.log_softmax(teacher_logits.detach(), dim=-1)
    log_p_s = torch.log_softmax(policy_logits, dim=-1)
    p_t = log_p_t.exp()
    kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)
    if reduction == "none":
        return kl * mask
    if reduction == "sum":
        return (kl * mask).sum()
    if reduction == "mean":
        return (kl * mask).sum() / mask.sum().clamp_min(1.0)
    raise ValueError(f"unknown reduction: {reduction!r}")
