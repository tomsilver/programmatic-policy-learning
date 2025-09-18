"""A very naive baseline approach that samples random actions."""

from typing import TypeVar

from gymnasium.spaces import Space

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class ConstantApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that always takes the same, fixed action."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self.constant_action = self._action_space.sample()

    def _get_action(self) -> _ActType:
        return self.constant_action
