"""Per-Action Mean Preservation (APM).

Standard token-mean reduction dilutes credit across long actions. APM
instead reduces advantages at the *action* level and broadcasts them back to
tokens with a mean-preserving weighting:

    A_t*     = (1/L_t) * sum_j A_{t,j}        (action-mean reduction)
    sum_j rho_{t,j} = 1                        (per-action mean preservation)
    A_{t,j}  = omega_{t,j} * A_t*,  omega_{t,j} = L_t * rho_{t,j}

so that the token-level advantages of each action average exactly to the
action advantage A_t*, regardless of the allocation shape rho.

The per-action credits A_t* themselves are the one-step TD residuals of
paper Eq. 3-5; they are computed by
:func:`factor.tac.compute_tac_advantages` (re-exported here for backwards
compatibility).
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from factor.tac import compute_tac_advantages as _tac_advantages


def action_mean_reduce(
    token_advantages: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Reduce token-level advantages to one scalar per action (action mean).

    Args:
        token_advantages: ``token_advantages[t]`` is a 1-D tensor of
            token-level advantages for action t (length L_t).

    Returns:
        List of 0-D tensors A_t*, one per action.
    """
    action_advantages: List[torch.Tensor] = []
    for adv in token_advantages:
        if adv.numel() == 0:
            action_advantages.append(adv.new_zeros(()))
        else:
            action_advantages.append(adv.mean())
    return action_advantages


def apm_token_advantages(
    action_advantages: torch.Tensor,
    token_weights: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Broadcast per-action advantages to tokens with mean preservation.

    Args:
        action_advantages: 1-D tensor [num_actions] of action advantages A_t*.
        token_weights: ``token_weights[t]`` is the mean-one multiplier
            omega_{t,j} for action t (from HTA; uniform if teacher inactive).

    Returns:
        List of 1-D tensors with per-token advantages A_{t,j}. For each
        action t, ``result[t].mean() == action_advantages[t]`` exactly (up to
        floating point), because omega has mean one.
    """
    out: List[torch.Tensor] = []
    for t, omega in enumerate(token_weights):
        out.append(omega * action_advantages[t])
    return out


def dense_apm_advantages(
    action_advantages: Sequence[torch.Tensor],
    token_weights: Sequence[Sequence[torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Batch wrapper around :func:`apm_token_advantages`.

    Args:
        action_advantages: Per-trajectory action advantage tensors.
        token_weights: Per-trajectory, per-action mean-one multipliers.

    Returns:
        Nested list ``[traj][action] -> token advantage tensor``.
    """
    return [
        apm_token_advantages(adv, weights)
        for adv, weights in zip(action_advantages, token_weights)
    ]


def compute_action_advantages(
    action_rewards: torch.Tensor,
    values: torch.Tensor,
    baseline: torch.Tensor | float,
) -> torch.Tensor:
    """Per-action TD credits A*_t (paper Eq. 3-5).

    Thin wrapper around :func:`factor.tac.compute_tac_advantages`; see that
    function for the exact semantics. The credits telescope exactly:
    ``sum_t A*_t == sum_t r_t - baseline == A^seq``.

    Args:
        action_rewards: [T] per-action environment rewards in the
            environment's reward coordinate (terminal reward on the last
            action).
        values: [T] V_bar_phi(x_t) at each action's pre-action boundary.
        baseline: Scalar b_i used to construct A^seq (leave-one-out group
            baseline on ALFWorld, 0 on WebShop and ScienceWorld).

    Returns:
        Tensor [T] of per-action credits A*_t.
    """
    return _tac_advantages(action_rewards, values, baseline)
