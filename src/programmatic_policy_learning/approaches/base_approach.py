"""Base class for approaches."""

import abc
from typing import Any, Generic, TypeVar

from gymnasium.spaces import Space
from prpl_utils.gym_agent import Agent

from programmatic_policy_learning.data.demo_types import Trajectory

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class BaseApproach(Agent[_ObsType, _ActType], Generic[_ObsType, _ActType], abc.ABC):
    """Base class for approaches."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
    ) -> None:
        super().__init__(seed)
        self._environment_description = environment_description
        self._action_space = action_space
        self._action_space.seed(seed)
        self._observation_space = observation_space
        self._observation_space.seed(seed)
        # Each Trajectory represents one demonstration, which is a sequence of steps
        # we are keeping a lost of trajectories, called demonstrations
        self._demonstrations: list[Trajectory[Any, Any]] | None = (
            None  # Optional offline data
        )

    def set_demonstrations(self, demonstrations: list[Trajectory[Any, Any]]) -> None:
        """Set offline demonstration data for training or evaluation."""
        self._demonstrations = demonstrations

    # Subclasses should override this method if offline training is supported.
    def train_offline(self) -> None:
        """Train the approach using offline demonstration data, if
        available."""
        if self._demonstrations is None:
            raise ValueError("No demonstrations set for offline training.")

