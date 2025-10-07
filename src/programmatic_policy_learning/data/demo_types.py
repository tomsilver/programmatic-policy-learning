"""Types for demonstrations and trajectories."""

from dataclasses import dataclass
from typing import Generic, TypeVar

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


@dataclass(frozen=True)
class Trajectory(Generic[ObsT, ActT]):
    """A sequence of (observation, action) steps."""

    steps: list[tuple[ObsT, ActT]]

    def __post_init__(self) -> None:
        for step in self.steps:
            assert (
                isinstance(step, tuple) and len(step) == 2
            ), "Each step must be a (obs, act) tuple"
