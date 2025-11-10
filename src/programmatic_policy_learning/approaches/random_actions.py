"""A very naive baseline approach that samples random actions."""

from typing import TypeVar, Any, Callable

import gymnasium as gym
import numpy as np

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class RandomActionsApproach(BaseApproach[_ObsType, _ActType]):
    """A very naive baseline approach that samples random actions."""

    # init to swallow extra kwargs from hydra passes 
    def __init__(
        self,
        environment_description: str,
        observation_space: Any,
        action_space: Any,
        seed: int,
        expert: Any | None = None,                            
        env_factory: Callable[[int], gym.Env] | None = None,  
        base_name: str | None = None,                         
        *_,                                                   
        env_specs: dict | None = None,                        
        **__,                                                 
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self._env_factory = env_factory                       
        self._rng = np.random.default_rng(seed)
    def _get_action(self) -> _ActType:
        return self._action_space.sample()

    # Match the runner's expectations
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
        if self._env_factory is None:
            raise NotImplementedError("RandomActionsApproach needs env_factory for testing.")
        results: dict[int, float] = {}
        for env_num in test_env_nums:
            env = self._env_factory(env_num)
            obs, _ = env.reset(seed=getattr(self, "_seed", None))
            total_reward = 0.0
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated) and steps < int(max_num_steps):
                action = self._action_space.sample()
                obs, rew, terminated, truncated, _ = env.step(action)
                total_reward += float(rew)
                steps += 1
            results[int(env_num)] = total_reward
        return results