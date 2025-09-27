"""Core LPP program representation that wraps logical program strings."""

from typing import Any, Callable, Generic, TypeVar

# from programmatic_policy_learning.lpp.dsl import primitives

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class StateActionProgram(Generic[ObsType, ActType]):
    """A logical program that operates on state-action pairs.

    Takes a program string (like "cell_is_value(1, a, s)") and makes it
    executable by converting it to a Python function when called.

    This class is generic and can work with any observation and action
    types.
    """

    def __init__(self, program_string: str, primitives: dict[str, Any]) -> None:
        """Initialize with a program string and primitives dictionary.

        Args:
            program_string: The program string to execute
            primitives: Dictionary of primitive functions (required)

        Raises:
            ValueError: If primitives dictionary is empty or None
        """
        if not primitives:
            raise ValueError("Primitives dictionary cannot be empty or None")

        self.program = program_string
        self.primitives = primitives
        self.compiled_func: Callable[[ObsType, ActType], Any] | None = None

    def __call__(self, s: ObsType, a: ActType) -> Any:
        """Execute the program on a state-action pair.

        Args:
            s: Observation/state of any type
            a: Action of any type

        Returns:
            Result of program evaluation (any type)
        """
        if self.compiled_func is None:
            # Convert string to executable function
            # using primitives as evaluation context
            self.compiled_func = eval(f"lambda s, a: {self.program}", self.primitives)

        return self.compiled_func(s, a)

    def __str__(self) -> str:
        """String representation of the program."""
        return self.program

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"StateActionProgram('{self.program}')"
