#!/usr/bin/env python3
"""FACTOR evaluation script.

Runs greedy-decoding evaluation on the held-out splits:

    - ALFWorld:     134 unseen games (6 categories), max 50 turns.
    - WebShop:      1000 held-out instructions, max 15 turns.
    - ScienceWorld: 540 episodes (30 tasks x 3 levels x 6 eps), max 50 turns.

Usage:
    python scripts/evaluate.py --env alfworld --checkpoint runs/factor/final \
        --output results/alfworld_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch

from factor.environments import (
    ALFWORLD_CATEGORIES,
    make_env,
)
from factor.environments.scienceworld import NUM_EVAL_EPISODES
from factor.environments.webshop import NUM_EVAL_INSTRUCTIONS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

NUM_ALFWORLD_UNSEEN = 134


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FACTOR evaluation")
    p.add_argument("--env", required=True,
                   choices=["alfworld", "webshop", "scienceworld"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def num_episodes(env_name: str) -> int:
    return {
        "alfworld": NUM_ALFWORLD_UNSEEN,
        "webshop": NUM_EVAL_INSTRUCTIONS,
        "scienceworld": NUM_EVAL_EPISODES,
    }[env_name]


@torch.no_grad()
def run_episode(policy: Any, env: Any, episode_index: int) -> Dict[str, Any]:
    """Run one greedy episode and return per-episode metrics."""
    obs, info = env.reset(episode_index=episode_index)
    total_reward = 0.0
    done = False
    while not done:
        context = env.render_context(include_last_feedback=True)
        action = policy.generate_action(context, greedy=True, max_tokens=512)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    return {
        "episode_index": episode_index,
        "num_turns": env.num_turns,
        "total_reward": total_reward,
        "success": bool(env.success),
        "score": getattr(env, "score", total_reward),
    }


def summarize(env_name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    success = sum(r["success"] for r in results) / max(len(results), 1)
    summary: Dict[str, Any] = {
        "env": env_name,
        "episodes": len(results),
        "success_rate": success,
        "avg_turns": sum(r["num_turns"] for r in results) / max(len(results), 1),
    }
    if env_name == "webshop":
        summary["avg_score"] = (
            sum(r["score"] for r in results) / max(len(results), 1)
        )
    if env_name == "alfworld":
        per_cat: Dict[str, List[bool]] = {c: [] for c in ALFWORLD_CATEGORIES}
        for r in results:
            cat = ALFWORLD_CATEGORIES[r["episode_index"] % len(ALFWORLD_CATEGORIES)]
            per_cat[cat].append(r["success"])
        summary["per_category_success"] = {
            c: (sum(v) / len(v) if v else 0.0) for c, v in per_cat.items()
        }
    return summary


def main() -> None:
    args = parse_args()
    policy = load_policy(args.checkpoint)  # noqa: F821 - verl integration
    env = make_env(args.env, split="test", seed=args.seed)

    results: List[Dict[str, Any]] = []
    for idx in range(num_episodes(args.env)):
        r = run_episode(policy, env, idx)
        results.append(r)
        if (idx + 1) % 20 == 0:
            logger.info("%s: %d/%d episodes done", args.env, idx + 1,
                        num_episodes(args.env))

    summary = summarize(args.env, results)
    logger.info("Summary: %s", json.dumps(summary, indent=2))

    out = Path(args.output or f"results/{args.env}_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
