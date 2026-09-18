"""Environment wrappers for FACTOR training.

Each wrapper exposes a uniform text-based interface used by the rollout
worker:

    env = make_env(name, split="train", seed=42)
    obs, info = env.reset()
    obs, reward, done, info = env.step(action_text)

The wrappers track the pre-action boundary of every agent turn (the token
position at which the agent began generating) so TAC can restore sparse
intermediate states and APM can segment token streams into actions.
"""

from factor.environments.alfworld import ALFWorldEnv
from factor.environments.webshop import WebShopEnv
from factor.environments.scienceworld import ScienceWorldEnv

_ENV_REGISTRY = {
    "alfworld": ALFWorldEnv,
    "webshop": WebShopEnv,
    "scienceworld": ScienceWorldEnv,
}


def make_env(name: str, **kwargs):
    """Instantiate a wrapped environment by name."""
    key = name.lower()
    if key not in _ENV_REGISTRY:
        raise ValueError(f"Unknown environment '{name}'. "
                         f"Available: {sorted(_ENV_REGISTRY)}")
    return _ENV_REGISTRY[key](**kwargs)


__all__ = [
    "ALFWorldEnv",
    "WebShopEnv",
    "ScienceWorldEnv",
    "make_env",
]
