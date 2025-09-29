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

    def _create_grammar(env_spec):
        del env_spec  # not used
        grammar: Grammar[str, None, None] = Grammar(
            rules={
                INT: ([["0"]], [1.0]),
                INCREMENT: ([["add_one(", INCREMENT, ")"], [INT]], [0.5, 0.5]),
            }
        )
        return grammar

    # Create the program generator.
    program_generator = GrammarBasedProgramGenerator(
        _create_grammar, dsl, env_spec={}, start_symbol=INCREMENT
    )

    # Generate programs from the generator in order from simplest to most complex.
    expected_programs = [
        "0",
        "add_one( 0 )",
        "add_one( add_one( 0 ) )",
    ]

    generated_programs = []
    gen = program_generator.generate_programs()
    for _ in range(3):
        generated_programs.append(next(gen))

    assert expected_programs == generated_programs


def test_grammar_based_program_generator_with_input():
    """Test GrammarBasedProgramGenerator with external input variable."""

    def add_one(x: int) -> int:
        return x + 1

    prims = {
        "add_one": add_one,
    }

    # Eval function expects 'program' as a string with 'x' as a variable,
    # and 'inputs' as the value for 'x'.
    def _eval(program: str, inputs: int):
        return eval(program, {}, {"x": inputs, **prims})

    dsl = DSL(id="dummy", primitives=prims, evaluate_fn=_eval)

    INT, INCREMENT = 0, 1

    def _create_grammar(env_spec):
        del env_spec
        grammar: Grammar[str, int, int] = Grammar(
            rules={
                INT: ([["x"]], [1.0]),
                INCREMENT: ([["add_one(", INT, ")"]], [1.0]),
            }
        )
        return grammar

    program_generator = GrammarBasedProgramGenerator(
        _create_grammar, dsl, env_spec={}, start_symbol=INCREMENT
    )

    gen = program_generator.generate_programs()
    program = next(gen)  # 'add_one( x )'

    assert dsl.evaluate(program, 0) == 1
    assert dsl.evaluate(program, 5) == 6
