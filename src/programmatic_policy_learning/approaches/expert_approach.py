"""Oracle/expert policy wrapper for any environment."""

from typing import Any

from programmatic_policy_learning.approaches.base_approach import BaseApproach


class ExpertApproach(BaseApproach[Any, Any]):
    """Oracle/expert policy wrapper for any environment."""

    def __init__(
        self, environment_description, observation_space, action_space, seed, expert_fn
    ):
        super().__init__(environment_description, observation_space, action_space, seed)
        self._expert_fn = expert_fn
        self._current_obs = None
        self._current_info = None

    def reset(self, obs, info):
        self._current_obs = obs
        self._current_info = info

    def _get_action(self):  # type: ignore
        return self._expert_fn(self._current_obs)
