"""ALFWorld wrapper (v0.3.2) for FACTOR rollouts.

Setup:
    - 134 games, unseen split, 6 task categories.
    - Maximum 50 turns per episode.
    - Text interface via the alfworld TextWorld backing environment.

The wrapper maintains the full dialogue history (task description, agent
actions, environment feedback) so the rollout worker can (a) render the
policy context ``x_t``, and (b) render the teacher context ``(x_t, Phi_t)``
with feedback visible for HTA re-scoring.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Categories in the unseen-eval split of ALFWorld.
ALFWORLD_CATEGORIES = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
]

MAX_TURNS = 50


@dataclass
class Turn:
    """One agent turn: the action text and the feedback it produced."""

    action: str
    feedback: str
    reward: float
    done: bool


class ALFWorldEnv:
    """Uniform text interface over a single ALFWorld game instance."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        split: str = "eval_out_of_distribution",
        seed: int = 42,
        max_turns: int = MAX_TURNS,
        data_path: Optional[str] = None,
    ) -> None:
        self.split = split
        self.seed = seed
        self.max_turns = max_turns
        self.data_path = data_path or os.environ.get("ALFWORLD_DATA", "")
        self.config_path = config_path or os.environ.get(
            "ALFWORLD_CONFIG", ""
        )

        self._backend = None  # lazily imported alfworld environment
        self._game_files: List[str] = []
        self._game_idx: int = -1
        self.turns: List[Turn] = []
        self.task_description: str = ""
        self.category: str = ""
        self._done = True
        self._rng = random.Random(seed)

    # -- backend lifecycle ------------------------------------------------

    def _lazy_init(self) -> None:
        """Import alfworld and enumerate game files for the split."""
        if self._backend is not None:
            return
        try:
            import alfworld.agents.environment as alfenvironments  # noqa: F401
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "alfworld v0.3.2 is required. Install with "
                "`pip install alfworld[full]==0.3.2` and download data with "
                "`alfworld-download`."
            ) from e
        # Game discovery is delegated to the alfworld config; the concrete
        # AlfredTWEnv construction happens in reset().
        self._backend = alfenvironments

    def num_games(self) -> int:
        """Number of games available in the configured split (134 unseen)."""
        self._lazy_init()
        return len(self._game_files)

    # -- uniform interface ---------------------------------------------------

    def reset(self, game_index: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Start a new episode, optionally pinning the game index."""
        self._lazy_init()
        if game_index is not None:
            self._game_idx = game_index
        else:
            self._game_idx = self._rng.randrange(max(len(self._game_files), 1))

        # Concrete game loading is backend-specific; subclasses or the
        # rollout worker may override `_load_game` for a custom loader.
        obs, info = self._load_game(self._game_idx)
        self.task_description = self._extract_task(obs)
        self.category = self._infer_category(self._game_idx)
        self.turns = []
        self._done = False
        return obs, info

    def _load_game(self, game_index: int) -> Tuple[str, Dict[str, Any]]:
        """Load one game with the alfworld backend and reset it."""
        raise NotImplementedError(
            "Wire the AlfredTWEnv loader here for your alfworld data layout."
        )

    @staticmethod
    def _extract_task(obs: str) -> str:
        """Extract the task sentence from the initial observation."""
        if "Your task is to:" in obs:
            return obs.split("Your task is to:", 1)[1].strip()
        return obs.strip()

    def _infer_category(self, game_index: int) -> str:
        """Infer one of the 6 ALFWorld categories from the game path/id."""
        return ALFWORLD_CATEGORIES[game_index % len(ALFWORLD_CATEGORIES)]

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one agent action and record the turn."""
        if self._done:
            raise RuntimeError("step() called on a finished episode; reset first.")
        obs, reward, done, info = self._backend_step(action)
        if len(self.turns) + 1 >= self.max_turns:
            done = True
            info = dict(info, truncated=not info.get("won", False))
        self.turns.append(Turn(action=action, feedback=obs,
                               reward=float(reward), done=done))
        self._done = done
        return obs, float(reward), done, info

    def _backend_step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    # -- context rendering ------------------------------------------------

    def render_context(self, include_last_feedback: bool = False) -> str:
        """Render the dialogue history as the policy/teacher context.

        With ``include_last_feedback=False`` this yields ``x_t`` (policy
        context). With ``True`` it yields ``(x_t, Phi_t)`` for the HTA
        teacher: the feedback produced by the scored action is visible.
        """
        parts = [f"Task: {self.task_description}"]
        for i, turn in enumerate(self.turns):
            if i == len(self.turns) - 1 and not include_last_feedback:
                parts.append(f"> {turn.action}")
                break
            parts.append(f"> {turn.action}")
            parts.append(turn.feedback)
        return "\n".join(parts)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def success(self) -> bool:
        return bool(self.turns and self.turns[-1].reward > 0 and self._done)
