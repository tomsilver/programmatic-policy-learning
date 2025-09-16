"""Environment provider registry."""

from typing import Any, Callable, Dict

import gymnasium

from programmatic_policy_learning.envs.providers.ggg import ggg
from programmatic_policy_learning.envs.providers.pr_bench import pr_bench


class EnvRegistry:
    """Registry for environment providers."""

    def __init__(self) -> None:
        self._providers: Dict[str, Callable[[Any], Any]] = {
            "ggg": ggg,
            "prbench": pr_bench,
        }

    def load(self, env_config: Any) -> Any:
        """Load environment from provider or fallback to gymnasium."""
        if "provider" in env_config:
            provider: str = env_config["provider"]
            return self._providers[provider](env_config)
        # Fall back to gymnasium if nothing was found...
        return gymnasium.make(**env_config.make_kwargs)
