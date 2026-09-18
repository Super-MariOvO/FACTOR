"""ScienceWorld wrapper (v1.1) for FACTOR rollouts.

Setup:
    - 540 evaluation episodes: 30 task types x 3 difficulty levels x 6
      episodes (variations) each.
    - Maximum 50 turns per episode.
    - Text interface via the ScienceWorld ``ScienceWorldEnv`` backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

MAX_TURNS = 50
NUM_TASK_TYPES = 30
NUM_LEVELS = 3
EPISODES_PER_LEVEL = 6
NUM_EVAL_EPISODES = NUM_TASK_TYPES * NUM_LEVELS * EPISODES_PER_LEVEL  # 540


class ScienceWorldEnv:
    """Uniform text interface over a ScienceWorld episode."""

    def __init__(
        self,
        split: str = "test",
        seed: int = 42,
        max_turns: int = MAX_TURNS,
        env_step_limit: int = 100,
    ) -> None:
        self.split = split
        self.seed = seed
        self.max_turns = max_turns
        self.env_step_limit = env_step_limit

        self._backend = None
        self.task_name: str = ""
        self.variation_idx: int = 0
        self.task_description: str = ""
        self.history: List[Tuple[str, str]] = []
        self._done = True
        self._score: float = 0.0

    # -- backend lifecycle ------------------------------------------------

    def _lazy_init(self) -> None:
        if self._backend is not None:
            return
        try:
            from scienceworld import ScienceWorldEnv as _SWEnv
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "scienceworld v1.1 is required. Install with "
                "`pip install scienceworld==1.1` (requires Java 8+)."
            ) from e
        self._backend = _SWEnv(envStepLimit=self.env_step_limit)

    @staticmethod
    def decode_episode_index(idx: int) -> Tuple[int, int, int]:
        """Map a flat episode id in [0, 540) to (task, level, variation)."""
        task = idx // (NUM_LEVELS * EPISODES_PER_LEVEL)
        rem = idx % (NUM_LEVELS * EPISODES_PER_LEVEL)
        level = rem // EPISODES_PER_LEVEL
        variation = rem % EPISODES_PER_LEVEL
        return task, level, variation

    # -- uniform interface ---------------------------------------------------

    def reset(self, episode_index: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        self._lazy_init()
        if episode_index is not None:
            task_i, level_i, var_i = self.decode_episode_index(
                episode_index % NUM_EVAL_EPISODES
            )
            task_names = sorted(self._backend.get_task_names())
            self.task_name = task_names[task_i % len(task_names)]
            self.variation_idx = var_i
        obs, info = self._backend_reset()
        self.task_description = self._backend.taskdescription() if hasattr(
            self._backend, "taskdescription"
        ) else obs
        self.history = []
        self._done = False
        self._score = 0.0
        return obs, info

    def _backend_reset(self) -> Tuple[str, Dict[str, Any]]:
        self._backend.load(self.task_name, self.variation_idx)
        obs, _ = self._backend.reset()
        return obs, {}

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; reset first.")
        obs, reward, done, info = self._backend.step(action.strip())
        if len(self.history) + 1 >= self.max_turns:
            done = True
            info = dict(info, truncated=True)
        self.history.append((action, obs))
        self._score = max(self._score, float(reward))
        self._done = done
        return obs, float(reward), done, info

    # -- context rendering ------------------------------------------------

    def render_context(self, include_last_feedback: bool = False) -> str:
        parts = [f"Task: {self.task_description}"]
        for i, (action, obs) in enumerate(self.history):
            parts.append(f"> {action}")
            if i < len(self.history) - 1 or include_last_feedback:
                parts.append(obs)
        return "\n".join(parts)

    @property
    def num_turns(self) -> int:
        return len(self.history)

    @property
    def score(self) -> float:
        return self._score

    @property
    def success(self) -> bool:
        return self._score >= 100.0  # ScienceWorld scores on a 0-100 scale
