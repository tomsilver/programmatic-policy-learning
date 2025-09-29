"""A program generator that uses a given grammar to generate programs."""

from dataclasses import dataclass
from typing import Any, Generic, Iterator, TypeAlias, TypeVar

from programmatic_policy_learning.dsl.core import DSL
from programmatic_policy_learning.dsl.generators.base import ProgramGenerator

ProgT = TypeVar("ProgT")
InT = TypeVar("InT")
OutT = TypeVar("OutT")

GrammarSymbol: TypeAlias = int


@dataclass(frozen=True)
class Grammar(Generic[ProgT, InT, OutT]):
    """A probabilistic context-free grammar over programs."""

    # Each rule maps a grammar symbol to possible "substitutions", where each sub is a
    # list of programs or other grammar symbols, with a corresponding probability.
    rules: dict[GrammarSymbol, tuple[list[list[ProgT | GrammarSymbol]], list[float]]]


class GrammarBasedProgramGenerator(ProgramGenerator[ProgT, InT, OutT]):
    """A program generator that uses a given grammar to generate programs."""

    def __init__(
        self,
        grammar: Grammar[ProgT, InT, OutT],
        dsl: DSL[ProgT, InT, OutT],
        env_spec: dict[str, Any],
    ) -> None:
        self._grammar = grammar
        super().__init__(dsl, env_spec)

    def generate_programs(self) -> Iterator[ProgT]:
        import ipdb

        ipdb.set_trace()
