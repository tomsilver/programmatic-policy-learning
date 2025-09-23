"""Core LPP program representation that wraps logical program strings."""

from programmatic_policy_learning.lpp.dsl.primitives import cell_is_value, out_of_bounds


class StateActionProgram:
    """A logical program that operates on state-action pairs.

    Takes a program string (like "cell_is_value(1, a, s)") and makes it
    executable by converting it to a Python function when called.
    """

    def __init__(self, program_string: str) -> None:
        """Initialize with a program string.

        Args:
            program_string: String representation of the logical program
        """
        self.program = program_string
        self.compiled_func = None

    def __call__(self, s, a):
        """Execute the program on a state-action pair.

        Args:
            state: Game state (numpy array)
            action: Action tuple (e.g., (row, col))

        Returns:
            Boolean result of program evaluation
        """
        if self.compiled_func is None:

            eval_context = {
                "cell_is_value": cell_is_value,
                "out_of_bounds": out_of_bounds,
            }
            # Convert string to executable function
            self.compiled_func = eval(f"lambda s, a: {self.program}", eval_context)

        return self.compiled_func(s, a)

    def __str__(self) -> str:
        """String representation of the program."""
        return self.program

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"StateActionProgram('{self.program}')"
