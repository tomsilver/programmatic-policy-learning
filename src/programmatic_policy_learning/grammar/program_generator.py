"""Program generator for creating logical programs using grammar-based
synthesis."""

import heapq as hq
import itertools
from copy import deepcopy
from typing import Any, Iterator

import numpy as np

from programmatic_policy_learning.approaches.policies.lpp_policy import (
    StateActionProgram,
)


class ProgramGenerator:
    """Generates programs from grammar."""

    @staticmethod
    def find_symbol(program: list) -> tuple[int, Any] | None:
        """Find the first non-terminal symbol in the program."""
        for idx, elm in enumerate(program):
            if isinstance(elm, int):
                return elm, idx
            if isinstance(elm, list):
                rec_result = ProgramGenerator.find_symbol(elm)
                if rec_result is not None:
                    return rec_result[0], [idx, rec_result[1]]
        return None

    @staticmethod
    def copy_program(program: list) -> list:
        """Create a deep copy of the program."""
        return deepcopy(program)

    @staticmethod
    def update_program(program: list, idx: Any, new_symbol: Any) -> None:
        """Update program at given index with new symbol."""
        if isinstance(idx, int):
            program[idx] = new_symbol
            return
        if len(idx) == 2:
            next_idx = idx[1]
        else:
            next_idx = idx[1:]
        ProgramGenerator.update_program(program[idx[0]], next_idx, new_symbol)

    @staticmethod
    def stringify(program: Any) -> str:
        """Convert program to string representation."""
        if isinstance(program, str):
            return program
        if isinstance(program, int):
            raise ValueError("Should not stringify incomplete programs")
        s = ""
        for x in program:
            s = s + " " + ProgramGenerator.stringify(x)
        return s.strip().lstrip()

    @staticmethod
    def get_child_programs(
        program: list, grammar: dict
    ) -> Iterator[tuple[list, float, float]]:
        """Get all possible child programs from current program."""
        symbol_result = ProgramGenerator.find_symbol(program)
        if symbol_result is None:
            return  # No more symbols to expand
        symbol, idx = symbol_result
        substitutions, production_probs = grammar[symbol]
        priorities = -np.log(production_probs)

        for substitution, prob, priority in zip(
            substitutions, production_probs, priorities
        ):
            child_program = ProgramGenerator.copy_program(program)
            ProgramGenerator.update_program(child_program, idx, substitution)
            yield child_program, prob, priority

    @staticmethod
    def program_is_complete(program: list) -> bool:
        """Check if program has no more non-terminal symbols."""
        return ProgramGenerator.find_symbol(program) is None

    @staticmethod
    def generate_programs(
        grammar: dict[int, tuple[list[list[str]], list[float]]],
        start_symbol: int = 0,
        num_iterations: int = 100000000,
    ) -> Iterator[tuple[StateActionProgram, float]]:
        """Generate programs with their prior log probabilities."""
        queue: list[tuple[float, float, int, list]] = []
        counter = itertools.count()

        hq.heappush(queue, (0, 0, next(counter), [start_symbol]))

        for _ in range(num_iterations):
            if not queue:
                break

            priority, production_neg_log_prob, _, program = hq.heappop(queue)

            for (
                child_program,
                child_production_prob,
                child_priority,
            ) in ProgramGenerator.get_child_programs(program, grammar):
                if ProgramGenerator.program_is_complete(child_program):
                    program_str = ProgramGenerator.stringify(child_program)
                    prior_log_prob = -production_neg_log_prob + np.log(
                        child_production_prob
                    )
                    yield StateActionProgram(program_str), prior_log_prob
                else:
                    new_priority = priority + child_priority
                    new_neg_log_prob = production_neg_log_prob - np.log(
                        child_production_prob
                    )
                    hq.heappush(
                        queue,
                        (new_priority, new_neg_log_prob, next(counter), child_program),
                    )
