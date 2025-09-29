"""Tests for grammar_based_generator.py."""

from programmatic_policy_learning.dsl.core import DSL
from programmatic_policy_learning.dsl.generators.grammar_based_generator import (
    Grammar,
    GrammarBasedProgramGenerator,
)


def test_grammar_based_program_generator():
    """Tests for GrammarBasedProgramGenerator()."""

    # Create a simple test DSL.
    def add_one(x: int) -> int:
        """Add one to the input."""
        return x + 1

    prims = {
        "add_one": add_one,
    }

    def _eval(program: str, inputs: None):
        assert inputs is None
        return eval(program, {}, prims)

    dsl = DSL(id="dummy", primitives=prims, evaluate_fn=_eval)

    # Create a simple test grammar.
    INT, INCREMENT = 0, 1
    grammar = Grammar(
        rules={
            INT: ([[0], [1.0]]),
            INCREMENT: ([["add_one(", INCREMENT, ")"], [INT]], [0.5, 0.5]),
        }
    )

    # Create the program generator.
    program_generator = GrammarBasedProgramGenerator(grammar, dsl, env_spec={})

    # Generate programs from the generator in order from simplest to most complex.
    expected_programs = [
        "add_one(0)",
        "add_one(add_one(0))",
        "add_one(add_one(add_one(0)))",
    ]

    generated_programs = []
    gen = program_generator.generate_programs()
    for _ in range(3):
        generated_programs.append(next(gen))

    assert expected_programs == generated_programs
