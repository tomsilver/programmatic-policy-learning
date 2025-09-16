"""Adapter for interoperating with legacy Gym envs in Gymnasium code."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from gymnasium import spaces


class GymToGymnasium:
    """Wrap a legacy Gym environment to present the Gymnasium API.

    - Maps `reset()` → `(obs, info)`; seeds via `env.seed(seed)` if available.
    - Maps `step()` → `(obs, reward, terminated, truncated, info)`; derives
      `truncated` from `info.get('TimeLimit.truncated', False)`.
    - Forwards `observation_space`, `action_space`, `spec`, `render()`, `close()`.
    """

    def __init__(self, base_env: Any) -> None:
        """Wrap a legacy gym environment."""
        self._env = base_env
        self.observation_space: Optional[spaces.Space] = getattr(
            base_env, "observation_space", None
        )
        self.action_space: Optional[spaces.Space] = getattr(
            base_env, "action_space", None
        )
        self.spec = getattr(base_env, "spec", None)

    @property
    def unwrapped(self) -> Any:
        """Get the underlying unwrapped environment."""
        return getattr(self._env, "unwrapped", self._env)

    def reset(self, *, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment and return (obs, info)."""
        if seed is not None:
            if hasattr(self._env, "seed") and callable(self._env.seed):
                try:
                    self._env.seed(seed)
                except (AttributeError, TypeError):
                    # Legacy env has broken seed implementation, continue without seeding
                    _ = None
        res = self._env.reset()
        if isinstance(res, tuple) and len(res) == 2:
            return res  # already (obs, info)
        return res, {}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Step environment and return (obs, reward, terminated, truncated,
        info)."""
        obs, reward, done, info = self._env.step(action)
        terminated = bool(done)
        truncated = bool(info.get("TimeLimit.truncated", False))
        return obs, float(reward), terminated, truncated, info

    def render(self) -> Any:
        """Render the environment."""
        return getattr(self._env, "render", lambda: None)()

    def close(self) -> None:
        """Close the environment."""
        getattr(self._env, "close", lambda: None)()


__all__ = ["GymToGymnasium"]
