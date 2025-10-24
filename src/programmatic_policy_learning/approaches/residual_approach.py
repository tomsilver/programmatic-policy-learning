"""Residual policy wrapper and SB3 backend for learning action residuals."""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
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


# ---------------------------------------------------------------------
# Env wrapper: executes base(obs) + residual action from the agent
# ---------------------------------------------------------------------
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

    def action(self, action: np.ndarray) -> np.ndarray:  # noqa: D401
        """Action is determined in step()."""
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
        info = cast(dict[str, Any], info)
        info["base_action"] = base
        info["residual_action"] = residual
        info["total_action"] = total
        return obs, float(r), bool(term), bool(trunc), info


_BackendName = Literal["sb3-td3", "sb3-ddpg"]


# ---------------------------------------------------------------------
# Tiny SB3 wrapper (TD3/DDPG + Gaussian action noise)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Hydra adapter residual approach (accepts expert + env_factory)
# API matches experiments/run_experiment.py expectations
# ---------------------------------------------------------------------
class ResidualApproach(BaseApproach[np.ndarray, np.ndarray]):
    """Hydra-compatible residual approach: action = clip(expert(obs) + residual(obs))."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Space[np.ndarray],
        action_space: Space[np.ndarray],
        seed: int,
        expert: Any,
        env_factory: Callable[[int], gym.Env],
        *_: Any,
        env_specs: Optional[dict[str, Any]] = None,
        backend: str = "sb3-td3",
        total_timesteps: int = 100_000,
        lr: float = 1e-3,
        noise_std: float = 0.1,
        verbose: int = 1,
        train_before_eval: bool = True,
        train_env_instance: int = 0,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)

        if not isinstance(action_space, Box):
            raise TypeError("ResidualFromExpert requires a Box action space.")
        self._act_box: Box = cast(Box, action_space)

        self._env_factory = env_factory
        self._env_specs = env_specs or {}
        self._expert = expert

        def _base_fn(obs: np.ndarray) -> np.ndarray:
            o = np.asarray(obs, dtype=np.float32)
            if hasattr(expert, "get_action"):
                a = expert.get_action(o)
            elif hasattr(expert, "policy"):
                a = expert.policy(o)
            elif callable(expert):
                a = expert(o)
            else:
                a = np.zeros(self._act_box.shape, dtype=np.float32)
            return np.asarray(a, dtype=np.float32)

        self._base_fn: Callable[[np.ndarray], np.ndarray] = _base_fn

        train_env = self._env_factory(train_env_instance)
        train_env.reset(seed=seed)
        self._train_env = ResidualActionWrapper(train_env, self._base_fn)
        self._backend = _SB3Backend(
            name=cast(_BackendName, backend),
            env=self._train_env,
            seed=seed,
            lr=lr,
            noise_std=noise_std,
            verbose=verbose,
        )
        self._total = int(total_timesteps)

        self._is_trained: bool = False
        self._train_before_eval = bool(train_before_eval)
        self._last_obs: Optional[np.ndarray] = None

    # ------- pipeline hooks used by the Hydra runner -------
    def train(self) -> None:
        self._backend.learn(total_timesteps=self._total)
        self._is_trained = True

    def reset(self, obs: np.ndarray, info: dict) -> None:
        if self._train_before_eval and not self._is_trained:
            self.train()
        self._last_observation = np.asarray(obs, dtype=np.float32)
        self._last_obs = self._last_observation

    def _get_action(self) -> np.ndarray:
        """Return clipped action = base + residual."""
        assert self._last_observation is not None, "Call reset() before _get_action()."
        obs = np.asarray(self._last_observation, dtype=np.float32)
        base = self._base_fn(obs)
        residual = self._backend.predict(obs)
        total = np.clip(base + residual, self._act_box.low, self._act_box.high)
        return total.astype(self._act_box.dtype)

    def step(self) -> np.ndarray:
        assert self._last_obs is not None, "Call reset() before step()."
        obs = np.asarray(self._last_obs, dtype=np.float32)
        base = self._base_fn(obs)
        residual = self._backend.predict(obs)
        total = np.clip(base + residual, self._act_box.low, self._act_box.high)
        return total.astype(self._act_box.dtype)

    def update(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        self._last_observation = np.asarray(obs, dtype=np.float32)
        self._last_obs = self._last_observation

    def test_policy_on_envs(
        self,
        test_env_nums: Iterable[int] = range(11, 20),
        max_num_steps: int = 50,
        *,
        _base_class_name: Optional[str] = None,
        _record_videos: bool = False,
        _video_format: str = "mp4",
        **_extra_env_kwargs: Any,
    ) -> dict[int, float]:
        """Evaluate current (base + residual) policy on env instances; return
        {env_num: total_reward}."""
        results: dict[int, float] = {}
        for env_num in test_env_nums:
            env = self._env_factory(env_num)
            obs, _ = env.reset(seed=self._seed if hasattr(self, "_seed") else None)
            total_reward = 0.0
            for _ in range(int(max_num_steps)):
                obs = np.asarray(obs, dtype=np.float32)
                base = self._base_fn(obs)
                residual = self._backend.predict(obs)
                act = np.clip(
                    base + residual, self._act_box.low, self._act_box.high
                ).astype(self._act_box.dtype)
                obs, rew, terminated, truncated, _ = env.step(act)
                total_reward += float(rew)
                if terminated or truncated:
                    break
            results[int(env_num)] = total_reward
        return results

    # ------- persistence -------
    def save(self, path: str) -> None:
        """Save this object's state to the given filesystem path via the
        backend."""
        self._backend.save(path)

    def load(self, path: str) -> None:
        """Load this object's state from the given filesystem path via the
        backend."""
        self._backend.load(path)
