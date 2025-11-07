"""Tests for LLM primitive generator."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pytest
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

from programmatic_policy_learning.dsl.generators.grammar_based_generator import Grammar
from programmatic_policy_learning.dsl.llm_primitives.llm_generator import (
    LLMPrimitivesGenerator,
)
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    get_dsl_functions_dict,
)

runllms = pytest.mark.skipif("not config.getoption('runllms')")


@runllms
def test_create_grammar() -> None:
    """Test the create_grammar_from_response function to ensure it correctly
    constructs a Grammar object from the LLM's JSON output.

    Validates the structure of the grammar, including nonterminals,
    rules, and probabilities.
    """

    llm_output = """{
        "proposal": {
            "rationale_short": "One-step relative move improves compositional locality and search branching versus full scans.",
            "name": "step",
            "type_signature": "(direction: DIRECTION, next_prog: LOCAL_PROGRAM, cell: CELL, obs: OBS) -> Bool",
            "args": [
                {"name": "direction", "type": "DIRECTION"},
                {"name": "next_prog", "type": "LOCAL_PROGRAM"}
            ],
            "semantics_py_stub": "def step(direction, next_prog, cell, obs):\\n    if cell is None:\\n        return False\\n    next_cell = (cell[0] + direction[0], cell[1] + direction[1])\\n    if out_of_bounds(next_cell[0], next_cell[1], obs.shape):\\n        return False\\n    return next_prog(next_cell, obs)",
            "pcfg_insertion": {
                "nonterminal": "CONDITION",
                "production": "step(DIRECTION, LOCAL_PROGRAM, cell, obs)"
            }
        },
        "updated_grammar": {
            "nonterminals": ["START","LOCAL_PROGRAM","CONDITION","DIRECTION","VALUE"],
            "terminals": ["at_cell_with_value","at_action_cell","cell_is_value","scanning","step"],
            "productions": {
                "START": "at_cell_with_value(VALUE, LOCAL_PROGRAM, STATE) | at_action_cell(LOCAL_PROGRAM, ACTION, STATE)",
                "LOCAL_PROGRAM": "CONDITION",
                "CONDITION": "cell_is_value(VALUE, cell, obs) | scanning(DIRECTION, LOCAL_PROGRAM, LOCAL_PROGRAM, cell, obs) | step(DIRECTION, LOCAL_PROGRAM, cell, obs)",
                "DIRECTION": "(1,0) | (0,1) | (-1,0) | (0,-1) | (1,1) | (-1,1) | (1,-1) | (-1,-1)",
                "VALUE": "{OBJECT_TYPES}"
            }
        }
    }
    """
    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_client = OpenAIModel("gpt-4o-mini", cache)
    llm_output_dict = json.loads(llm_output)
    object_types = ["tpn.EMPTY", "tpn.TOKEN", "None"]
    generator = LLMPrimitivesGenerator(llm_client, None)
    new_grammar = generator.create_grammar_from_response(llm_output_dict, object_types)

    # Assertions to validate the grammar
    assert isinstance(new_grammar, Grammar)
    assert 0 in new_grammar.rules  # START nonterminal
    assert 1 in new_grammar.rules  # LOCAL_PROGRAM nonterminal
    assert 2 in new_grammar.rules  # CONDITION nonterminal
    assert 3 in new_grammar.rules  # DIRECTION nonterminal
    assert 4 in new_grammar.rules  # VALUE nonterminal

    # Check CONDITION rules
    condition_rules, condition_probs = new_grammar.rules[2]
    assert ["step(", 3, ", ", 1, ", cell, obs)"] in condition_rules
    assert len(condition_rules) == len(condition_probs)

    # Check VALUE rules
    value_rules, value_probs = new_grammar.rules[4]
    assert ["tpn.EMPTY"] in value_rules
    assert len(value_rules) == len(value_probs)
    assert all(prob == 1.0 / len(object_types) for prob in value_probs)


@runllms
def test_generate_grammar_with_real_llm() -> None:
    """Test the generate_grammar method with the real LLM."""

    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_client = OpenAIModel("gpt-4o-mini", cache)

    with open(
        "src/programmatic_policy_learning/dsl/llm_primitives/prompts/"
        + "one_missing_prompt_shifted.txt",
        "r",
        encoding="utf-8",
    ) as file:
        prompt = file.read()

    object_types = ["tpn.EMPTY", "tpn.TOKEN", "None"]

    generator = LLMPrimitivesGenerator(llm_client, "shifted")

    grammar, _, _ = generator.generate_and_process_grammar(prompt, object_types)
    logging.info(grammar)

    assert isinstance(grammar, Grammar)
    assert 0 in grammar.rules  # START nonterminal
    assert 1 in grammar.rules  # LOCAL_PROGRAM nonterminal
    assert 2 in grammar.rules  # CONDITION nonterminal
    assert 3 in grammar.rules  # DIRECTION nonterminal
    assert 4 in grammar.rules  # VALUE nonterminal

    # Check CONDITION rules
    condition_rules, condition_probs = grammar.rules[2]
    assert len(condition_rules) == len(condition_probs)

    # Check VALUE rules
    value_rules, value_probs = grammar.rules[4]
    assert len(value_rules) == len(value_probs)
    assert all(prob == 1.0 / len(object_types) for prob in value_probs)


def test_add_primitive_to_dsl() -> None:
    """Test that a new primitive is successfully added to the DSL."""

    def new_primitive(cell: tuple[int, int], obs: Any) -> bool:
        """Example of a new primitive."""
        return cell is not None and obs[cell[0]][cell[1]] == 42

    generator = LLMPrimitivesGenerator(None, None)

    # Add the new primitive to the DSL
    updated_get_dsl_functions_dict = generator.add_primitive_to_dsl(
        "new_primitive", new_primitive
    )

    # Retrieve the updated DSL functions
    updated_dsl_functions = updated_get_dsl_functions_dict()

    # Assertions
    assert "new_primitive" in updated_dsl_functions
    assert updated_dsl_functions["new_primitive"] is new_primitive

    # Ensure the original DSL functions are still present
    base_dsl_functions = get_dsl_functions_dict()
    for key, value in base_dsl_functions.items():
        assert key in updated_dsl_functions
        assert updated_dsl_functions[key] == value

    # Test the new primitive
    assert updated_dsl_functions["new_primitive"]((1, 1), [[0, 0], [0, 42]]) is True
    assert updated_dsl_functions["new_primitive"]((0, 0), [[0, 0], [0, 42]]) is False
