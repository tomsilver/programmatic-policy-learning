# src/programmatic_policy_learning/approaches/residual_from_expert.py
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, cast

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Space

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.approaches.residual_approach import (
    ResidualActionWrapper,
    _SB3Backend,
)

class ResidualFromExpert(BaseApproach[np.ndarray, np.ndarray]):
    """Hydra-compatible residual approach: action = clip(expert(obs) + residual(obs))."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Space,
        action_space: Space,
        seed: int,
        expert: Any,
        env_factory: Callable[[int], gym.Env],
        *,
        env_specs: Optional[dict[str, Any]] = None,
        backend: str = "sb3-td3",         # or "sb3-ddpg"
        total_timesteps: int = 100_000,
        lr: float = 1e-3,
        noise_std: float = 0.1,
        verbose: int = 1,
        train_before_eval: bool = True,   # train once at first reset()
        train_env_instance: int = 0,      # which instance_num to use for training
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)

        if not isinstance(action_space, Box):
            raise TypeError("ResidualFromExpert requires a continuous Box action space.")
        self._act_box: Box = cast(Box, action_space)

        # Keep references for later (testing on new envs, ablations, etc.)
        self._env_factory = env_factory
        self._expert = expert
        self._env_specs = env_specs or {}

        # --- Build zero-arg env builder from the runner's env_factory ---
        def _env_builder() -> gym.Env:
            return env_factory(train_env_instance)

        # --- Derive a stateless base policy from the passed expert ---
        def _base_fn(obs: np.ndarray) -> np.ndarray:
            if hasattr(expert, "get_action"):
                a = expert.get_action(obs)
            elif hasattr(expert, "policy"):
                a = expert.policy(obs)
            else:
                a = np.zeros(self._act_box.shape, dtype=np.float32)
            return np.asarray(a, dtype=np.float32)

        self._base_fn = _base_fn

        # --- Training env: residual wrapper + SB3 backend ---
        train_env = _env_builder()
        train_env.reset(seed=seed)
        self._train_env = ResidualActionWrapper(train_env, self._base_fn)

        self._backend = _SB3Backend(
            name=cast(Any, backend),
            env=self._train_env,
            seed=seed,
            lr=lr,
            noise_std=noise_std,
            verbose=verbose,
        )
        self._total = total_timesteps

        # Runtime state for the runner’s eval loop
        self._last_obs: Optional[np.ndarray] = None
        self._is_trained: bool = False
        self._train_before_eval = train_before_eval

    # ---------------- Pipeline hooks expected by run_experiment.py ----------------

    def train(self) -> None:
        """Train the residual policy on the training env."""
        self._backend.learn(total_timesteps=self._total)
        self._is_trained = True

    def reset(self, observation, info) -> None:
        """Called by the runner at the start of each episode."""
        if self._train_before_eval and not self._is_trained:
            self.train()
        self._last_observation = np.asarray(observation, dtype=np.float32)
        self._last_obs = self._last_observation

    def step(self) -> np.ndarray:
        """Return (base + residual) action, clipped to action space."""
        assert self._last_obs is not None, "Call reset() before step()."
        obs = np.asarray(self._last_obs, dtype=np.float32)
        base = self._base_fn(obs)
        residual = self._backend.predict(obs)
        total = np.clip(base + residual, self._act_box.low, self._act_box.high)
        return total.astype(self._act_box.dtype)

    def update(self, observation, reward: float, done: bool, info: dict) -> None:
        """Update internal observation state after env.step()."""
        self._last_observation = np.asarray(observation, dtype=np.float32)
        self._last_obs = self._last_observation

    # ---------------- Test sweep (to match runner expectations) ----------------
    def test_policy_on_envs(
        self,
        base_class_name: str,                  # kept for signature compatibility
        test_env_nums: Iterable[int] = range(11, 20),
        max_num_steps: int = 50,
        record_videos: bool = False,          # simple version: ignore video
        video_format: str = "mp4",
    ) -> dict[int, float]:
        """Evaluate current (base + residual) policy on env instances; return {env_num: total_reward}."""
        results: dict[int, float] = {}
        for env_num in test_env_nums:
            env = self._env_factory(env_num)
            obs, info = env.reset(seed=self._seed if hasattr(self, "_seed") else None)
            total_reward = 0.0
            for _ in range(int(max_num_steps)):
                obs = np.asarray(obs, dtype=np.float32)
                base = self._base_fn(obs)
                residual = self._backend.predict(obs)
                act = np.clip(base + residual, self._act_box.low, self._act_box.high).astype(self._act_box.dtype)
                obs, rew, terminated, truncated, info = env.step(act)
                total_reward += float(rew)
                if terminated or truncated:
                    break
            results[int(env_num)] = total_reward
            try:
                env.close()
            except Exception:
                pass
        return results

    # ---------------- Persistence passthroughs ----------------
    def save(self, path: str) -> None:
        self._backend.save(path)

    def load(self, path: str) -> None:
        self._backend.load(path)
