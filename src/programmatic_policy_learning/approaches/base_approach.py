"""Base class for approaches."""

import abc
from typing import Generic, TypeVar

from gymnasium.spaces import Space
from prpl_utils.gym_agent import Agent

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
        self._is_trained = False

    def fit(  # pylint: disable=unused-argument
        self, demonstrations: list[list[tuple[_ObsType, _ActType]]]
    ) -> None:
        """Train the approach on demonstration data.

        Args:
            demonstrations: List of episodes,
            each episode is a list of (state, action) pairs

        Note: Default no-op implementation. Override if training is needed.
        """
        self._is_trained = True

    def save(self, filepath: str) -> None:
        """Save the trained model to disk."""
        raise NotImplementedError("Save method not implemented")

    def load(self, filepath: str) -> None:
        """Load a trained model from disk."""
        raise NotImplementedError("Load method not implemented")

    @property
    def is_trained(self) -> bool:
        """Whether the approach has been trained."""
        return self._is_trained
