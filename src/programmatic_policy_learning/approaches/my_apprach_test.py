"""A very naive baseline approach that samples random actions."""

from typing import TypeVar

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class MyApproachTest(BaseApproach[_ObsType, _ActType]):
    """Always take the same action (chosen once at init)."""

    def __init__(self, name, observation_space, action_space, seed=None):
        super().__init__(name, observation_space, action_space, seed=seed)
        self._action1 = action_space.sample()
        self._action2 = action_space.sample()

    def _get_action(self) -> _ActType:
        act = self._action1 if self._i == 0 else self._action2
        self._i ^= 1  # flip between 0 and 1
        return act