"""Oracle/expert policy wrapper for any environment."""

from typing import Any, TypeVar

import numpy as np

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class ExpertApproach(BaseApproach[_ObsType, _ActType]):
    """Oracle/expert policy wrapper for any environment."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Any,
        action_space: Any,
        seed: int,
        expert_fn: Any,
        env_factory: Any | None = None,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)

        if hasattr(expert_fn, "get_action"):
            self._expert_fn = expert_fn.get_action
        elif hasattr(expert_fn, "act"):
            self._expert_fn = expert_fn.act
        elif callable(expert_fn):
            self._expert_fn = expert_fn
        else:
            raise TypeError("Expert must be callable or have get_action/act.")
        self._last_observation: _ObsType | None = None
        self._last_info: dict[str, Any] | None = None
        self._env_factory = env_factory

    def reset(self, obs: _ObsType, info: dict[str, Any]) -> None:
        self._last_observation = obs
        self._last_info = info
        self._timestep = 0

    def _get_action(self) -> Any:
        assert self._last_observation is not None
        return self._expert_fn(self._last_observation)

    def test_policy_on_envs(
        self,
        test_env_nums: range = range(11, 20),
        max_num_steps: int = 50,
        *,
        _base_class_name: str | None = None,
        _record_videos: bool = False,
        _video_format: str = "mp4",
        **_extra_env_kwargs: Any,
    ) -> dict[int, float]:
        """Currently necessary to conform with run_experiments.py."""
        if self._env_factory is None:
            raise NotImplementedError("ExpertApproach needs env_factory for testing.")
        results: dict[int, float] = {}
        for env_num in test_env_nums:
            env = self._env_factory(env_num)
            obs, _ = env.reset(seed=getattr(self, "_seed", None))
            terminated = truncated = False
            total = 0.0
            steps = 0
            while not (terminated or truncated) and steps < int(max_num_steps):
                act = self._expert_fn(np.asarray(obs, dtype=np.float32))
                obs, rew, terminated, truncated, _ = env.step(act)
                total += float(rew)
                steps += 1
            results[int(env_num)] = total
        return results
