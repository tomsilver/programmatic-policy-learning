"""Tests for grammar_based_generator.py."""

from typing import Any

from omegaconf import OmegaConf

from programmatic_policy_learning.dsl.core import DSL
from programmatic_policy_learning.dsl.generators.grammar_based_generator import (
    Grammar,
    GrammarBasedProgramGenerator,
)
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import create_grammar
from programmatic_policy_learning.envs.providers.ggg_provider import create_ggg_env


def test_grammar_based_program_generator() -> None:
    """Tests for GrammarBasedProgramGenerator()."""

    # Create a simple test DSL.
    def add_one(x: int) -> int:
        """Add one to the input."""
        return x + 1

    prims = {
        "add_one": add_one,
    }

    def _eval(program: str, inputs: None) -> int:
        assert inputs is None
        return eval(program, {}, prims)

    dsl = DSL(id="dummy", primitives=prims, evaluate_fn=_eval)

    # Create a simple test grammar.
    INT, INCREMENT = 0, 1

    def _create_grammar(env_spec: dict[str, Any]) -> Grammar[str, None, int]:
        del env_spec  # not used
        grammar: Grammar[str, None, int] = Grammar(
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
        generated_programs.append(next(gen)[0])

    assert expected_programs == generated_programs


def test_grammar_based_program_generator_with_input() -> None:
    """Test GrammarBasedProgramGenerator with external input variable."""

    def add_one(x: int) -> int:
        return x + 1

    prims = {"add_one": add_one}

    # Eval function expects 'program' as a string with 'x' as a variable,
    # and 'inputs' as the value for 'x'.
    def _eval(program: str, inputs: int) -> int:
        return eval(program, {}, {"x": inputs, **prims})

    dsl: DSL[str, int, int] = DSL(id="dummy", primitives=prims, evaluate_fn=_eval)

    INT, INCREMENT = 0, 1

    def _create_grammar(
        env_spec: dict[str, Any],  # pylint: disable=unused-argument
    ) -> Grammar[str, int, int]:
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
    program = next(gen)[0]  # 'add_one( x )'

    assert dsl.evaluate(program, 0) == 1
    assert dsl.evaluate(program, 5) == 6


def test_generated_programs_are_callable() -> None:
    """Test that env_spec is inferred from GGGEnvWithTypes and used in
    grammar."""
    cfg = OmegaConf.create({"make_kwargs": {"id": "TwoPileNim0-v0"}})
    env = create_ggg_env(cfg)
    object_types = env.get_object_types()
    env_spec = {"object_types": object_types}
    dsl: DSL[str, None, str] = DSL(
        id="grid_v1", primitives={}, evaluate_fn=lambda p, i: p
    )
    generator = GrammarBasedProgramGenerator(  # type:ignore
        create_grammar, dsl, env_spec=env_spec, start_symbol=6  # type:ignore
    )
    gen = generator.generate_programs()

    assert next(gen)[0] == "tpn.EMPTY"
    assert next(gen)[0] == "tpn.TOKEN"
    assert next(gen)[0] == "None"


def test_generate_program_with_custom_generator() -> None:
    """Test program generation with a custom generator function."""

    def custom_generator() -> str:
        return "42"

    dsl: DSL[str, Any, Any] = DSL(id="dummy", primitives={}, evaluate_fn=lambda p, _: p)

    # Create a dummy grammar with a single rule that always returns the custom program
    def _create_grammar(_: dict[str, Any]) -> Grammar[str, Any, Any]:
        return Grammar(rules={0: ([[custom_generator()]], [1.0])})

    program_generator = GrammarBasedProgramGenerator(
        _create_grammar, dsl, env_spec={}, start_symbol=0
    )

    gen = program_generator.generate_programs()
    assert next(gen)[0] == "42"


def test_generate_program_with_env_specific_grammar() -> None:
    """Test program generation with environment-specific grammar."""

    cfg = OmegaConf.create({"make_kwargs": {"id": "TwoPileNim0-v0"}})
    env = create_ggg_env(cfg)
    object_types = env.get_object_types()
    env_spec = {"object_types": object_types}

    # Create a dummy DSL that just returns the program as the result.
    dsl = DSL[str, None, str](id="dummy", primitives={}, evaluate_fn=lambda p, _: p)

    # Create the program generator with the environment-specific grammar.
    program_generator = GrammarBasedProgramGenerator(  # type: ignore
        create_grammar, dsl, env_spec=env_spec, start_symbol=6  # type: ignore
    )

    gen = program_generator.generate_programs()

    assert next(gen)[0] == "tpn.EMPTY"
    assert next(gen)[0] == "tpn.TOKEN"
    assert next(gen)[0] == "None"
