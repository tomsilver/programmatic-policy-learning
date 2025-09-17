"""A very naive baseline approach that samples random actions."""

from typing import TypeVar

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class SameActionApproach(BaseApproach[_ObsType, _ActType]):
    """Always take the same action (chosen once at init)."""

    def __init__(self, name, observation_space, action_space, seed=None):
        super().__init__(name, observation_space, action_space, seed=seed)
        self._fixed_action = action_space.sample()  # pick one random action once

    def _get_action(self) -> _ActType:
        return self._fixed_action
