"""Base class for program generators."""

import abc
from typing import Any, Generic, Iterator, TypeVar

from programmatic_policy_learning.dsl.core import DSL

ProgT = TypeVar("ProgT")
InT = TypeVar("InT")
OutT = TypeVar("OutT")


class ProgramGenerator(Generic[ProgT, InT, OutT], abc.ABC):
    """Base class for program generators."""

    def __init__(self, dsl: DSL[ProgT, InT, OutT], env_spec: dict[str, Any]) -> None:
        self._dsl = dsl
        # NOTE: env specs can be used, for example, to extract possible object types
        # in GGG environments which are included as part of primitive programs.
        self._env_spec = env_spec

    @abc.abstractmethod
    def generate_programs(
        self,
    ) -> Iterator[tuple[str, float]]:  # changed to str to match stringify output
        """Generate a potentially infinite number of programs from the DSL.

        Typically, program generators should go from "simplest" to "most
        complex".
        """
