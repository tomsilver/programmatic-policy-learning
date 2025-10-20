"""Oracle/expert policy wrapper for any environment."""

from typing import Any, TypeVar

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
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self._expert_fn = expert_fn
        self._last_observation: _ObsType | None = None
        self._last_info: dict[str, Any] | None = None

    def reset(self, obs: _ObsType, info: dict[str, Any]) -> None:
        self._last_observation = obs
        self._last_info = info
        self._timestep = 0

    def _get_action(self) -> Any:  # type: ignore
        return self._expert_fn(self._last_observation)
