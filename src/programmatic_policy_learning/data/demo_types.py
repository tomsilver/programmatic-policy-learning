"""Types for demonstrations and trajectories."""

from dataclasses import dataclass
from typing import Generic, TypeVar

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


@dataclass(frozen=True)
class Trajectory(Generic[ObsT, ActT]):
    """A sequence of steps, containing observation and action."""

    obs: list[ObsT]
    act: list[ActT]

    def __post_init__(self) -> None:
        assert len(self.obs) == len(self.act)
