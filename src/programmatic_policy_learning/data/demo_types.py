"""Types for demonstrations and trajectories."""

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


@dataclass(frozen=True)
class Demo(Generic[ObsT, ActT]):
    """A single demonstration step, containing observation and action."""

    obs: ObsT
    act: ActT


@dataclass(frozen=True)
class Trajectory(Generic[ObsT, ActT]):
    """A sequence of demonstration steps (Demo), called a trajectory."""

    steps: Sequence[Demo[ObsT, ActT]]
