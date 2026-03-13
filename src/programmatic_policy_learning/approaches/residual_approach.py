"""Residual policy wrapper and SB3 backend for learning action residuals.

Works for BOTH:
- Pendulum (action_dim=1)
- LunarLanderContinuous (action_dim=2)

Key design goals:
- Pendulum behavior remains unchanged by default.
- LunarLanderContinuous can be made safer via optional residual scaling + clipping.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Literal, SupportsFloat, cast

import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper
from gymnasium.spaces import Box
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from programmatic_policy_learning.approaches.base_approach import BaseApproach


# ---------------------------------------------------------------------
# Env wrapper: executes base(obs) + residual action from the agent
# ---------------------------------------------------------------------
class ResidualActionWrapper(ActionWrapper):
    """Adds a residual to the base policy."""

    def __init__(
        self,
        env: gym.Env,
        base_policy: Callable[[np.ndarray], np.ndarray],
        *,
        residual_scale: float | None = None,
        clip_total_action: bool = True,
    ):
        """
        Args:
            env: Gymnasium environment with Box action space.
            base_policy: Callable mapping obs -> base action (np.ndarray).
            residual_scale:
                - If provided, multiplies residual action by this factor.
                - If None, we choose a default:
                    * action_dim==1 (Pendulum): 1.0 (UNCHANGED behavior)
                    * action_dim>=2: 1/sqrt(action_dim) for stability
            clip_total_action:
                Whether to clip (base + scaled_residual) into action bounds before stepping.
                Recommended True; Pendulum won't break, and LunarLander is happier.
        """
        super().__init__(env)
        assert isinstance(env.action_space, Box), (
            "ResidualActionWrapper requires a Box action space."
        )
        self._base: Callable[[np.ndarray], np.ndarray] = base_policy

        self.action_space = env.action_space
        self._act_box: Box = env.action_space
        self._action_dim = int(np.prod(self._act_box.shape))

        # Default: keep Pendulum EXACTLY the same (scale=1).
        # For higher-dim actions, modest downscaling is often more stable.
        if residual_scale is None:
            self._res_scale = 1.0 if self._action_dim == 1 else float(
                1.0 / np.sqrt(max(self._action_dim, 1))
            )
        else:
            self._res_scale = float(residual_scale)

        self._clip_total_action = bool(clip_total_action)
        self._last_obs: np.ndarray | None = None

    def action(self, action: np.ndarray) -> np.ndarray:
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
        base = np.asarray(self._base(self._last_obs), dtype=np.float32)

        total = base + self._res_scale * residual
        if self._clip_total_action:
            total = np.clip(total, self._act_box.low, self._act_box.high)

        obs, r, term, trunc, info = self.env.step(total)
        self._last_obs = np.asarray(obs, dtype=np.float32)

        info["base_action"] = base
        info["residual_action"] = residual
        info["residual_scale"] = self._res_scale
        info["total_action"] = total
        return obs, float(r), bool(term), bool(trunc), info


_BackendName = Literal[
    "sb3-td3",
    "sb3-ddpg",
    "sb3-ppo",
    "sb3-sac",
    "sb3-a2c",
]


class _SB3Backend:
    """Tiny SB3 wrapper that handles several continuous-control algorithms."""

    def __init__(
        self,
        name: _BackendName,
        env: gym.Env,
        seed: int,
        lr: float = 1e-3,
        noise_std: float = 0.1,
        verbose: int = 1,
    ):
        # Wrap in VecEnv if needed
        self._vec = env if hasattr(env, "get_attr") else DummyVecEnv([lambda: env])
        if not isinstance(self._vec.action_space, Box):
            raise TypeError("SB3 backend requires a Box action space.")
        act_dim = int(np.prod(self._vec.action_space.shape))

        # Gaussian exploration noise (only used for some off-policy algos)
        noise = NormalActionNoise(
            mean=np.zeros(act_dim, dtype=np.float32),
            sigma=np.ones(act_dim, dtype=np.float32) * float(noise_std),
        )

        if name in ("sb3-td3", "sb3-ddpg"):
            # Off-policy, deterministic actors with action noise
            algo_cls = TD3 if name == "sb3-td3" else DDPG
            self._model = algo_cls(
                "MlpPolicy",
                self._vec,
                learning_rate=float(lr),
                action_noise=noise,
                verbose=int(verbose),
                seed=int(seed),
            )

        elif name == "sb3-sac":
            # Off-policy, stochastic actor (no external action_noise)
            self._model = SAC(
                "MlpPolicy",
                self._vec,
                learning_rate=float(lr),
                verbose=int(verbose),
                seed=int(seed),
            )

        elif name == "sb3-ppo":
            # On-policy PPO for continuous control
            self._model = PPO(
                "MlpPolicy",
                self._vec,
                learning_rate=float(lr),
                verbose=int(verbose),
                seed=int(seed),
            )

        elif name == "sb3-a2c":
            # On-policy actor-critic
            self._model = A2C(
                "MlpPolicy",
                self._vec,
                learning_rate=float(lr),
                verbose=int(verbose),
                seed=int(seed),
            )

        else:
            raise ValueError(f"Unknown SB3 backend name: {name!r}")

    def learn(self, total_timesteps: int) -> None:
        """Train the SB3 model."""
        self._model.learn(total_timesteps=int(total_timesteps), progress_bar=True)

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
        observation_space: Box,
        action_space: Box,
        seed: int,
        expert: Any,
        env_factory: Callable[[int], gym.Env],
        backend: str = "sb3-td3",
        total_timesteps: int = 100_000,
        lr: float = 1e-3,
        noise_std: float = 0.1,
        verbose: int = 1,
        train_before_eval: bool = True,
        train_env_instance: int = 0,
        *,
        residual_scale: float | None = None,
        clip_total_action: bool = True,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)

        if not isinstance(action_space, Box):
            raise TypeError("ResidualApproach requires a Box action space.")
        self._act_box: Box = cast(Box, action_space)

        self._env_factory = env_factory
        self._expert = expert

        def _base_fn(obs: np.ndarray) -> np.ndarray:
            o = np.asarray(obs, dtype=np.float32)
            if hasattr(expert, "get_action"):
                a = expert.get_action(o)
            elif hasattr(expert, "policy"):
                a = expert.policy(o)
            elif hasattr(expert, "act"):
                a = expert.act(o)
            elif callable(expert):
                a = expert(o)
            else:
                a = np.zeros(self._act_box.shape, dtype=np.float32)
            return np.asarray(a, dtype=np.float32)

        self._base_fn: Callable[[np.ndarray], np.ndarray] = _base_fn

        train_env = self._env_factory(train_env_instance)
        train_env.reset(seed=seed)

        # Wrapper executes base(obs) + scaled residual (learned) and (optionally) clips.
        self._train_env = ResidualActionWrapper(
            train_env,
            self._base_fn,
            residual_scale=residual_scale,
            clip_total_action=clip_total_action,
        )

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
        self._last_obs: np.ndarray | None = None

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
        _base_class_name: str | None = None,
        _record_videos: bool = False,
        _video_format: str = "mp4",
        **_extra_env_kwargs: Any,
    ) -> dict[int, float]:
        """Evaluate current (base + residual) policy on env instances; return {env_num: total_reward}."""
        results: dict[int, float] = {}
        for env_num in test_env_nums:
            env = self._env_factory(env_num)
            obs, _ = env.reset(seed=self._seed if hasattr(self, "_seed") else None)

            total_reward = 0.0
            for _ in range(int(max_num_steps)):
                obs = np.asarray(obs, dtype=np.float32)
                base = self._base_fn(obs)
                residual = self._backend.predict(obs)
                act = np.clip(base + residual, self._act_box.low, self._act_box.high).astype(
                    self._act_box.dtype
                )
                obs, rew, terminated, truncated, _ = env.step(act)
                total_reward += float(rew)
                if terminated or truncated:
                    break

            results[int(env_num)] = float(total_reward)

        return results

    # ------- persistence -------
    def save(self, path: str) -> None:
        """Save this object's state to the given filesystem path via the backend."""
        self._backend.save(path)

    def load(self, path: str) -> None:
        """Load this object's state from the given filesystem path via the backend."""
        self._backend.load(path)
