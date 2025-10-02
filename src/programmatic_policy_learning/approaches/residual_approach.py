# python -m ensurepip --upgrade
# python -m pip install "stable-baselines3[extra]>=2.0"

import contextlib
import io
from typing import Callable, Generic, Literal, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Space

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


def _adapt_approach(approach_instance) -> Callable[[np.ndarray], np.ndarray]:
    act_shape = approach_instance._action_space.shape

    def f(obs: np.ndarray) -> np.ndarray:
        approach_instance._last_observation = obs
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            a = approach_instance._get_action()
        return np.asarray(a, dtype=np.float32).reshape(act_shape)

    return f


class ResidualActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, base_policy: Callable[[np.ndarray], np.ndarray]):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        self._base = base_policy
        self.action_space = env.action_space

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._last_obs = obs
        return obs, info

    def step(self, residual: np.ndarray):
        base = self._base(np.asarray(self._last_obs, dtype=np.float32))
        total = (
            base + residual
        )  # np.clip(base + residual, self.action_space.low, self.action_space.high)
        obs, r, term, trunc, info = self.env.step(total)
        self._last_obs = obs
        info["base_action"], info["residual_action"], info["total_action"] = (
            base,
            residual,
            total,
        )
        return obs, r, term, trunc, info


_BackendName = Literal["sb3-td3", "sb3-ddpg", "cleanrl-td3"]


class _SB3Backend:
    def __init__(
        self, name: _BackendName, env, seed: int, lr=1e-3, noise_std=0.1, verbose=1
    ):
        from stable_baselines3 import DDPG, TD3
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.vec_env import DummyVecEnv

        self._vec = env if hasattr(env, "get_attr") else DummyVecEnv([lambda: env])
        act_dim = int(np.prod(self._vec.action_space.shape))
        noise = NormalActionNoise(
            mean=np.zeros(act_dim), sigma=np.ones(act_dim) * noise_std
        )
        Algo = TD3 if name == "sb3-td3" else DDPG
        self._model = Algo(
            "MlpPolicy",
            self._vec,
            learning_rate=lr,
            action_noise=noise,
            verbose=verbose,
            seed=seed,
        )

    def learn(self, total_timesteps: int):
        self._model.learn(total_timesteps=total_timesteps, progress_bar=True)

    def predict(self, obs: np.ndarray):
        a, _ = self._model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float32)

    def save(self, path: str):
        self._model.save(path)

    def load(self, path: str):
        self._model = self._model.load(path, env=self._model.get_env())  


class ResidualApproach(BaseApproach[_ObsType, _ActType], Generic[_ObsType, _ActType]):
    def __init__(
        self,
        environment_description: str,
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
        env_builder: Callable[[], gym.Env],
        base_approach_instance,
        backend: _BackendName = "sb3-td3",
        total_timesteps: int = 100_000,
        lr: float = 1e-3,
        noise_std: float = 0.1,
        verbose: int = 1,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self._base_fn: Callable[[np.ndarray], np.ndarray] = _adapt_approach(
            base_approach_instance
        )

        def make_env():
            env = env_builder()
            env.reset(seed=seed)
            return ResidualActionWrapper(env, self._base_fn)

        self._env = make_env()
        self._backend = _SB3Backend(
            backend,
            env=self._env,
            seed=seed,
            lr=lr,
            noise_std=noise_std,
            verbose=verbose,
        )
        self._total = total_timesteps

    def train(self) -> None:
        self._backend.learn(total_timesteps=self._total)

    def _get_action(self) -> _ActType:
        obs = np.asarray(self._last_observation, dtype=np.float32)
        base = self._base_fn(obs)  # fixed
        residual = self._backend.predict(obs)  # learned
        total = np.clip(
            base + residual, self._action_space.low, self._action_space.high
        )
        return total.astype(self._action_space.dtype)

    def save(self, path: str) -> None:
        self._backend.save(path)

    def load(self, path: str) -> None:
        self._backend.load(path)
