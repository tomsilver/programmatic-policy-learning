# src/programmatic_policy_learning/approaches/residual_from_expert.py
from __future__ import annotations

from typing import Any, Callable, Optional, cast

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Space

from programmatic_policy_learning.approaches.base_approach import BaseApproach

# If your ResidualActionWrapper/_SB3Backend live in a different module, adjust this import:
from programmatic_policy_learning.approaches.residual_approach import (
    ResidualActionWrapper,
    _SB3Backend,
)

class ResidualFromExpert(BaseApproach[np.ndarray, np.ndarray]):
    """
    Adapter so your existing Hydra runner (run_experiment.py) can instantiate a
    residual policy that learns a correction on top of a provided expert policy.

    Constructor signature matches what run_experiment.py passes:
        (environment_description, observation_space, action_space, seed,
         expert, env_factory, *, env_specs=..., backend=..., ...)

    Internally:
      - builds an env via env_factory(0)
      - derives base_fn(obs)->action from the `expert`
      - trains an SB3 residual (TD3/DDPG) that adds to the base_fn action
    """

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
        train_before_eval: bool = True,   # do a one-time train() on first reset()
        train_env_instance: int = 0,      # which instance_num to use for training
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)

        if not isinstance(action_space, Box):
            raise TypeError("ResidualFromExpert requires a continuous Box action space.")
        self._act_box: Box = cast(Box, action_space)

        # --- Build zero-arg env builder from the runner's env_factory ---
        def _env_builder() -> gym.Env:
            return env_factory(train_env_instance)

        # --- Derive a stateless base policy from the passed expert ---
        def _base_fn(obs: np.ndarray) -> np.ndarray:
            # Try common expert APIs; fall back to zeros if unknown.
            if hasattr(expert, "get_action"):
                a = expert.get_action(obs)
            elif hasattr(expert, "policy"):
                a = expert.policy(obs)
            else:
                a = np.zeros(self._act_box.shape, dtype=np.float32)
            return np.asarray(a, dtype=np.float32)

        self._base_fn = _base_fn

        # --- Wrap env with residual combiner and create SB3 backend ---
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

        # Runtime state for the runner's eval loop
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
            # Optional: perform training once before the first evaluation episode
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
        # Note: residual training happens in train(); no online updates here.

    # ---------------- Persistence passthroughs ----------------
    def save(self, path: str) -> None:
        self._backend.save(path)

    def load(self, path: str) -> None:
        self._backend.load(path)
