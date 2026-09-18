#!/usr/bin/env python3
"""Main FACTOR training entry point.

Usage:
    python scripts/train_factor.py --env alfworld --seed 42 \
        --model Qwen/Qwen2.5-7B-Instruct --output runs/factor_alfworld_s42

The script assumes a verl/SERL checkout (commit b338174,
serl_action_mask branch) providing the rollout workers and the PPO actor
data-parallel engine. This entry point wires FACTOR's TAC critic, HTA
teacher and APM advantage computation into that loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import torch

from factor.environments import make_env
from factor.training.config import FACTORConfig
from factor.training.trainer import FACTORTrainer, Trajectory, TrajectoryBatch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_factor")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FACTOR training")
    p.add_argument("--env", default="alfworld",
                   choices=["alfworld", "webshop", "scienceworld"])
    p.add_argument("--seed", type=int, default=42, choices=[42, 43, 1337])
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output", default="runs/factor")
    p.add_argument("--total-steps", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint directory to resume from.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rollout_batch(
    policy: Any,
    env_name: str,
    batch_size: int,
    group_size: int,
    seed: int,
    temperature: float = 1.0,
) -> TrajectoryBatch:
    """Collect one batch of trajectories via the verl rollout worker.

    NOTE: This is the integration seam with verl/SERL. In production, the
    rollout worker returns token ids, old log-probs, pre-action boundary
    hidden states (detached), action spans, rewards and group ids. Here we
    construct the container and delegate the actual generation to the
    worker callback registered on the policy wrapper.
    """
    if not hasattr(policy, "generate_trajectories"):
        raise RuntimeError(
            "Policy wrapper must expose `generate_trajectories` from the "
            "verl/SERL rollout worker. See README for integration details."
        )
    raw = policy.generate_trajectories(
        env_name=env_name,
        n_trajectories=batch_size,
        group_size=group_size,
        temperature=temperature,
        max_tokens=512,
        seed=seed,
    )
    batch = TrajectoryBatch(
        trajectories=[Trajectory(**item) for item in raw]
    )
    return batch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = FACTORConfig()
    config.ppo.total_steps = args.total_steps
    config.rollout.batch_size = args.batch_size
    config.rollout.group_size = args.group_size
    config.rollout.env_name = args.env

    with open(out_dir / "config.json", "w") as f:
        json.dump({"env": args.env, "seed": args.seed,
                   "model": args.model}, f, indent=2)

    # ------------------------------------------------------------------
    # Policy loading is delegated to verl; here we require an already
    # constructed wrapper on `args.resume`/environment-specific factory.
    # ------------------------------------------------------------------
    policy = build_policy(args)  # noqa: F821 - provided by verl integration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = policy.config.hidden_size

    trainer = FACTORTrainer(
        policy=policy,
        config=config,
        hidden_size=hidden_size,
        continuation_sampler=getattr(policy, "sample_continuations", None),
        device=device,
    )

    for step in range(1, config.ppo.total_steps + 1):
        logger.info("=== FACTOR step %d/%d (eta_k=%.3f) ===",
                    step, config.ppo.total_steps, config.eta(step))

        batch = rollout_batch(
            policy, args.env, config.rollout.batch_size,
            config.rollout.group_size, seed=args.seed + step,
        )

        # Teacher is a frozen copy of the *current* policy each step. It is
        # needed by HTA token allocation (steps 11..49) and by SERL's
        # auxiliary action-only KL (lambda_k = alpha_k > 0 for steps 1..49).
        need_teacher = config.eta(step) > 0.0 or config.serl_lambda(step) > 0.0
        if need_teacher:
            trainer.refresh_teacher()
            teacher_logprobs = policy.score_with_teacher(
                trainer.teacher, batch
            )
        else:
            teacher_logprobs = None

        critic_stats = trainer.update_critic(
            batch, prefix_tokens_fn=policy.prefix_tokens
        )
        advantages = trainer.compute_token_advantages(batch, teacher_logprobs)
        new_logprobs, action_masks = policy.compute_logprobs(batch)

        # Full logits for the SERL auxiliary KL, when the wrapper exposes
        # them; the term is skipped gracefully otherwise.
        teacher_logits = policy_logits = None
        if need_teacher and hasattr(policy, "compute_logits"):
            policy_logits = policy.compute_logits(batch)
            teacher_logits = policy.compute_teacher_logits(
                trainer.teacher, batch
            )
        ppo_stats = trainer.ppo_update(
            batch, new_logprobs, advantages, action_masks,
            teacher_logits=teacher_logits, policy_logits=policy_logits,
        )

        logger.info("step %d stats: %s | %s", step, critic_stats, ppo_stats)
        if step % 10 == 0:
            ckpt = out_dir / f"checkpoint_step{step}"
            policy.save_pretrained(ckpt)
            torch.save(trainer.tac.state_dict(), ckpt / "tac.pt")

    policy.save_pretrained(out_dir / "final")
    torch.save(trainer.tac.state_dict(), out_dir / "final" / "tac.pt")
    logger.info("Training complete. Artifacts in %s", out_dir)


if __name__ == "__main__":
    main()
