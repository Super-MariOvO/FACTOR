"""WebShop wrapper for FACTOR rollouts.

Setup:
    - 1000 held-out instructions for evaluation.
    - Maximum 15 turns per episode.
    - Greedy decoding at evaluation time.
    - Reward in [0, 1] from the WebShop goal-matching scorer.

The agent acts with structured commands, e.g. ``search[query]`` and
``click[element]``, following the WebShop action grammar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

MAX_TURNS = 15
NUM_EVAL_INSTRUCTIONS = 1000

_ACTION_RE = re.compile(r"^(search|click)\[(.+)\]$", re.IGNORECASE)


class WebShopEnv:
    """Uniform text interface over a WebShop session."""

    def __init__(
        self,
        split: str = "test",
        seed: int = 42,
        max_turns: int = MAX_TURNS,
        server_url: Optional[str] = None,
        num_instructions: int = NUM_EVAL_INSTRUCTIONS,
    ) -> None:
        self.split = split
        self.seed = seed
        self.max_turns = max_turns
        self.server_url = server_url
        self.num_instructions = num_instructions

        self._session = None  # lazily created WebShop session handle
        self._instruction_idx: int = -1
        self.instruction: str = ""
        self.history: List[Tuple[str, str]] = []  # (action, observation)
        self._done = True
        self._score: float = 0.0

    # -- backend lifecycle ------------------------------------------------

    def _lazy_init(self) -> None:
        if self._session is not None:
            return
        try:
            from webshop.web_agent_site.envs import web_agent_text_env  # noqa: F401
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "WebShop is required. Follow the setup at "
                "https://github.com/princeton-nlp/WebShop (Python 3.8+, "
                "web_agent_site data and products)."
            ) from e
        self._session = web_agent_text_env

    # -- uniform interface ---------------------------------------------------

    def reset(self, instruction_index: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        self._lazy_init()
        if instruction_index is not None:
            self._instruction_idx = instruction_index % self.num_instructions
        obs, info = self._backend_reset(self._instruction_idx)
        self.instruction = self._extract_instruction(obs)
        self.history = []
        self._done = False
        self._score = 0.0
        return obs, info

    def _backend_reset(self, idx: int) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError("Wire the WebShop session reset here.")

    @staticmethod
    def _extract_instruction(obs: str) -> str:
        if "Instruction:" in obs:
            return obs.split("Instruction:", 1)[1].split("\n")[0].strip()
        return obs.strip()

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; reset first.")
        action = action.strip()
        if not _ACTION_RE.match(action):
            obs = "Invalid action format. Use 'search[query]' or 'click[element]'."
            reward, done, info = 0.0, False, {}
        else:
            obs, reward, done, info = self._backend_step(action)
        if len(self.history) + 1 >= self.max_turns:
            done = True
            info = dict(info, truncated=True)
        self.history.append((action, obs))
        self._score = max(self._score, float(reward))
        self._done = done
        return obs, float(reward), done, info

    def _backend_step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    # -- context rendering ------------------------------------------------

    def render_context(self, include_last_feedback: bool = False) -> str:
        parts = [f"Instruction: {self.instruction}"]
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
        """Best goal-matching score achieved during the episode."""
        return self._score

    @property
    def success(self) -> bool:
        return self._score >= 1.0
