"""FACTOR: Trajectory-Anchored Critics with Hindsight Token Allocation.

Implementation of the FACTOR algorithm for multi-turn agentic RL, built on
top of the verl/SERL framework (commit b338174, serl_action_mask branch).

Components:
    - TAC (Trajectory-Anchored Critic): dense intermediate value estimation
      via inference-only Monte Carlo continuations anchored at sparse
      trajectory checkpoints, with one-step TD action credits that
      telescope exactly to the trajectory advantage (paper Eq. 3-5).
    - HTA (Hindsight Token Allocation): teacher-guided per-token credit
      allocation using the outcome-aligned score
      s = sgn(A*) * clip(Delta, -d, d) (paper Eq. 6).
    - APM (Per-Action Mean Preservation): action-mean advantage reduction
      with mean-preserving token-level broadcasting.
"""

from factor.tac import (
    TACValueHead,
    TACModule,
    TACReplayBuffer,
    Checkpoint,
    boundary_adjusted_values,
    compute_tac_advantages,
    select_checkpoints,
)
from factor.hta import (
    HTAConfig,
    HindsightTeacher,
    compute_hindsight_gap,
    compute_outcome_score,
    compute_serl_action_kl,
    compute_token_allocation,
)
from factor.apm import apm_token_advantages, action_mean_reduce

__all__ = [
    "TACValueHead",
    "TACModule",
    "TACReplayBuffer",
    "Checkpoint",
    "boundary_adjusted_values",
    "compute_tac_advantages",
    "select_checkpoints",
    "HTAConfig",
    "HindsightTeacher",
    "compute_hindsight_gap",
    "compute_outcome_score",
    "compute_serl_action_kl",
    "compute_token_allocation",
    "apm_token_advantages",
    "action_mean_reduce",
]

__version__ = "1.0.0"
