"""Environment provider registry."""

from typing import Any, Callable

import gymnasium

from programmatic_policy_learning.envs.providers.ggg_provider import create_ggg_env
from programmatic_policy_learning.envs.providers.kinder_provider import (
    create_kinder_env,
)
from programmatic_policy_learning.envs.providers.maze_provider import create_maze_env


class EnvRegistry:
    """Registry for environment providers."""

    def __init__(self) -> None:
        self._providers: dict[str, Callable[[Any], Any]] = {
            "ggg": create_ggg_env,
            "kinder": create_kinder_env,
            "maze": create_maze_env,
        }

    def load(self, env_config: Any, instance_num: int | None = None) -> Any:
        """Load environment from provider or fallback to gymnasium."""
        if "provider" in env_config:
            provider: str = env_config["provider"]
            if provider in ("ggg", "kinder") and instance_num is not None:
                return self._providers[provider](
                    env_config, instance_num
                )  # type: ignore
            return self._providers[provider](env_config)

        mk = dict(env_config.make_kwargs)
        env_id = mk.pop("id")
        mk.pop("base_name", None)
        mk.pop("description", None)

        # Fall back to gymnasium if nothing was found...
        return gymnasium.make(env_id, **mk)
