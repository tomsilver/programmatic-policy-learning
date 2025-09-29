"""A program generator that uses a given grammar to generate programs."""

import heapq as hq
import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Iterator, TypeAlias, TypeVar

import numpy as np

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
        start_symbol: int = 0,
    ) -> None:
        self._grammar = grammar
        self._start_symbol = start_symbol
        super().__init__(dsl, env_spec)

    def generate_programs(self) -> Iterator[ProgT]:
        queue = []
        counter = itertools.count()
        hq.heappush(queue, (0, 0, next(counter), [self._start_symbol]))
        while True:
            priority, production_neg_log_prob, _, program = hq.heappop(queue)
            for (
                child_program,
                child_production_prob,
                child_priority,
            ) in _get_child_programs(program, self._grammar):
                if _program_is_complete(child_program):
                    yield _stringify(child_program)
                else:
                    hq.heappush(
                        queue,
                        (
                            priority + child_priority,
                            production_neg_log_prob - np.log(child_production_prob),
                            next(counter),
                            child_program,
                        ),
                    )


def _find_symbol(deconstructed_program):
    for idx, elm in enumerate(deconstructed_program):
        if isinstance(elm, int):
            return elm, idx
        if isinstance(elm, list):
            rec_result = _find_symbol(elm)
            if rec_result is not None:
                return rec_result[0], [idx, rec_result[1]]
    return None


def _copy_program(deconstructed_program):
    return deepcopy(deconstructed_program)


def _program_is_complete(deconstructed_program):
    return _find_symbol(deconstructed_program) == None


def _update_program(deconstructed_program, idx, new_symbol):
    if isinstance(idx, int):
        deconstructed_program[idx] = new_symbol
        return
    if len(idx) == 2:
        next_idx = idx[1]
    else:
        next_idx = idx[1:]
    _update_program(deconstructed_program[idx[0]], next_idx, new_symbol)


def _get_child_programs(
    deconstructed_program, grammar: Grammar[ProgT, InT, OutT]
) -> Iterator[tuple[ProgT, float, float]]:
    symbol, idx = _find_symbol(deconstructed_program)
    substitutions, production_probs = grammar.rules[symbol]
    priorities = -np.log(production_probs)

    for substitution, prob, priority in zip(
        substitutions, production_probs, priorities
    ):
        child_program = _copy_program(deconstructed_program)
        _update_program(child_program, idx, substitution)
        yield child_program, prob, priority

def _stringify(program):
    if isinstance(program, str):
        return program
    if isinstance(program, int):
        raise Exception("Should not stringify incomplete programs")
    s = ''
    for x in program:
        s = s + ' ' + _stringify(x)
    return s.strip().lstrip()
