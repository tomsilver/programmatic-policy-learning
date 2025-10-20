"""StateActionProgram Class."""

from typing import Any

from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    get_dsl_functions_dict,
)

DSL_FUNCTIONS = get_dsl_functions_dict()


class StateActionProgram:
    """A callable object with input (state, action) and Boolean output.

    Made a class to have nice strs and pickling and to avoid redundant
    evals.
    """

    def __init__(self, program: str) -> None:

        self.program: str = program
        self.wrapped: Any = None
        # self.dsl_functions = dsl_functions

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.wrapped is None:
            self.wrapped = eval("lambda s, a: " + self.program, DSL_FUNCTIONS)
        return self.wrapped(*args, **kwargs)

    def __repr__(self) -> str:
        return str(self.program)

    def __str__(self) -> str:
        return str(self.program)

    def __getstate__(self) -> str:
        return self.program

    def __setstate__(self, program: str) -> None:
        self.program = program
        self.wrapped = None

    def __add__(self, s: Any) -> "StateActionProgram":
        if isinstance(s, str):
            return StateActionProgram(self.program + s)
        if isinstance(s, StateActionProgram):
            return StateActionProgram(self.program + s.program)
        raise TypeError(
            f"Unsupported operand type(s) for +: '\
            {type(self).__name__}' and '{type(s).__name__}'"
        )

    def __radd__(self, s: Any) -> "StateActionProgram":
        if isinstance(s, str):
            return StateActionProgram(s + self.program)
        if isinstance(s, StateActionProgram):
            return StateActionProgram(s.program + self.program)
        raise TypeError(
            f"Unsupported operand type(s) for +: '\
                {type(s).__name__}' and '{type(self).__name__}'"
        )
