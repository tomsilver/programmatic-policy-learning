"""Residual policy wrapper and SB3 backend for learning action residuals."""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    SupportsFloat,
    TypeVar,
    cast,
)

import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper
from gymnasium.spaces import Box, Space
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class ResidualActionWrapper(ActionWrapper):
    """Adds a residual to the base policy."""

    def __init__(self, env: gym.Env, base_policy: Callable[[np.ndarray], np.ndarray]):
        super().__init__(env)
        assert isinstance(
            env.action_space, Box
        ), "ResidualActionWrapper requires a Box action space."
        self._base: Callable[[np.ndarray], np.ndarray] = base_policy

        self.action_space = env.action_space
        self._act_box: Box = env.action_space

        self._last_obs: np.ndarray | None = None

    def action(self, action: np.ndarray) -> np.ndarray:
        """Action is determined in step()"""
        return action

    def reset(self, **kw: Any) -> tuple[np.ndarray, dict]:
        """Reset env and remember observation."""
        obs, info = self.env.reset(**kw)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Compose base action with residual and step the env."""
        if self._last_obs is None:
            obs, _ = self.reset()
            self._last_obs = np.asarray(obs, dtype=np.float32)

        residual = np.asarray(action, dtype=np.float32)
        base = self._base(self._last_obs)
        total = base + residual

        obs, r, term, trunc, info = self.env.step(total)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        info["base_action"] = base
        info["residual_action"] = residual
        info["total_action"] = total
        return obs, float(r), bool(term), bool(trunc), cast(dict[str, Any], info)


_BackendName = Literal["sb3-td3", "sb3-ddpg"]


class _SB3Backend:
    """Tiny SB3 wrapper that handles TD3/DDPG with Gaussian action noise."""

    def __init__(
        self,
        name: _BackendName,
        env: gym.Env,
        seed: int,
        lr: float = 1e-3,
        noise_std: float = 0.1,
        verbose: int = 1,
    ):
        self._vec = env if hasattr(env, "get_attr") else DummyVecEnv([lambda: env])
        if not isinstance(self._vec.action_space, Box):
            raise TypeError("SB3 backend requires a Box action space.")
        act_dim = int(np.prod(self._vec.action_space.shape))

        noise = NormalActionNoise(
            mean=np.zeros(act_dim, dtype=np.float32),
            sigma=np.ones(act_dim, dtype=np.float32) * noise_std,
        )

        algo_cls = TD3 if name == "sb3-td3" else DDPG
        self._model = algo_cls(
            "MlpPolicy",
            self._vec,
            learning_rate=lr,
            action_noise=noise,
            verbose=verbose,
            seed=seed,
        )

    def learn(self, total_timesteps: int) -> None:
        """Train the SB3 model."""
        self._model.learn(total_timesteps=total_timesteps, progress_bar=True)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic policy action given observation."""
        a, _ = self._model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float32)

    def save(self, path: str) -> None:
        """Save SB3 model."""
        self._model.save(path)

    def load(self, path: str) -> None:
        """Load SB3 model."""
        self._model = self._model.load(path, env=self._model.get_env())


class ResidualApproach(BaseApproach[_ObsType, _ActType], Generic[_ObsType, _ActType]):
    """Approach that learns a residual on top of a provided base policy."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
        env_builder: Callable[[], gym.Env],
        base_fn: Callable[[np.ndarray], np.ndarray],
        backend: _BackendName = "sb3-td3",
        total_timesteps: int = 100_000,
        lr: float = 1e-3,
        noise_std: float = 0.1,
        verbose: int = 1,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)

        self._base_fn = base_fn

        env = env_builder()
        env.reset(seed=seed)
        self._env: gym.Env = ResidualActionWrapper(env, self._base_fn)
        self._backend = _SB3Backend(
            backend,
            env=self._env,
            seed=seed,
            lr=lr,
            noise_std=noise_std,
            verbose=verbose,
        )
        self._total = total_timesteps

        # Keep a Box-typed handle to action space for mypy (low/high/dtype).
        if not isinstance(self._action_space, Box):
            raise TypeError("ResidualApproach requires a Box action space.")
        self._act_box: Box = cast(Box, self._action_space)

    def train(self) -> None:
        """Train the residual policy."""
        self._backend.learn(total_timesteps=self._total)

    def _get_action(self) -> _ActType:
        """Return clipped action = base + residual."""
        # pylint: disable=protected-access
        obs = np.asarray(self._last_observation, dtype=np.float32)
        base = self._base_fn(obs)  # fixed/base policy
        residual = self._backend.predict(obs)  # learned residual
        total = np.clip(base + residual, self._act_box.low, self._act_box.high)
        return total.astype(self._act_box.dtype)  # type: ignore[return-value]

    def save(self, path: str) -> None:
        """Save residual policy backend."""
        self._backend.save(path)

    def load(self, path: str) -> None:
        """Load residual policy backend."""
        self._backend.load(path)
