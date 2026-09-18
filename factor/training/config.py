"""Configuration for FACTOR training runs.

Defaults match the paper setup: Qwen2.5-7B-Instruct backbone, 150 steps,
group size 8, batch size 128 trajectories, one PPO epoch, Adam lr 5e-7,
gradient clipping 1.0, PPO clip eps 0.2, context 4096, generation 512,
rollout temperature 1.0. FACTOR constants: B=2 checkpoints x M=4
continuations per checkpoint (each up to 15 additional turns), d=3.0,
tau=1.0, eta_0=0.7; the value head warms up for 10 steps and teacher
allocation is active during steps 11-49. The SERL auxiliary action-only
KL keeps SERL's coefficient lambda_k = alpha_k = max(1 - k/50, 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RolloutConfig:
    """Rollout / generation settings."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    context_length: int = 4096
    generation_length: int = 512
    temperature: float = 1.0
    eval_greedy: bool = True
    group_size: int = 8              # trajectories sampled per task prompt
    batch_size: int = 128            # trajectories per training step
    env_name: str = "alfworld"
    max_turns: int = 50


@dataclass
class PPOConfig:
    """PPO update settings."""

    total_steps: int = 150
    ppo_epochs: int = 1
    lr: float = 5e-7
    grad_clip: float = 1.0
    clip_eps: float = 0.2
    kl_coef: float = 0.0             # reference KL disabled (beta = 0) per SERL
    seeds: List[int] = field(default_factory=lambda: [42, 43, 1337])


@dataclass
class FACTORConfig:
    """FACTOR-specific hyper-parameters."""

    # TAC: B checkpoints x M continuations per checkpoint.
    tac_mlp_hidden: int = 1024
    tac_ema_decay: float = 0.995
    tac_lr: float = 1e-4
    tac_buffer_window: int = 10
    tac_num_checkpoints: int = 2             # B restored checkpoints
    tac_continuations_per_checkpoint: int = 4  # M continuations per checkpoint
    tac_continuation_horizon: int = 15       # max additional turns per continuation
    tac_warmup_steps: int = 10

    # HTA
    hta_eta_0: float = 0.7
    hta_decay_steps: int = 50
    hta_active_start: int = 11
    hta_active_end: int = 49
    hta_tau: float = 1.0
    hta_clip_d: float = 3.0        # gap clip bound d in Eq. 6

    # SERL auxiliary action-only KL: lambda_k = alpha_k.
    serl_lambda_decay_steps: int = 50  # alpha_k = max(1 - k/50, 0)

    # APM has no extra hyper-parameters beyond HTA weights.

    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def eta(self, step: int) -> float:
        """HTA mixing coefficient eta_k for a 1-indexed step."""
        if step < self.hta_active_start or step > self.hta_active_end:
            return 0.0
        return self.hta_eta_0 * max(1.0 - step / self.hta_decay_steps, 0.0)

    def serl_lambda(self, step: int) -> float:
        """SERL auxiliary coefficient lambda_k = alpha_k for a 1-indexed step."""
        return max(1.0 - step / self.serl_lambda_decay_steps, 0.0)
