"""An approach that learns a logical programmatic policy from data."""

from typing import Any, TypeVar

from gymnasium.spaces import Space

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class LogicProgrammaticPolicyApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that learns a logical programmatic policy from data."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
    ) -> None:
        """LPP APProach."""
        super().__init__(environment_description, observation_space, action_space, seed)
        self._policy: StateActionProgram | None = None

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        # sketch
        # self._policy = self._train_policy() -> need to implement policyObject first
        self._timestep = 0

    def _get_action(self) -> _ActType:
        assert self._policy is not None, "Call reset() first."
        assert self._last_observation is not None
        # Use the logical policy to select an action
        return self._policy(self._last_observation)
