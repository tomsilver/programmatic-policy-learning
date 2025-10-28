"""This module defines utilities for transforming the structured output of a
LLM into a formal Grammar object.

It processes nonterminals, terminals, and production rules to
dynamically construct grammars for DSLs.
"""

import json
import logging
from typing import Any, Callable

from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.reprompting import query_with_reprompts
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.dsl.generators.grammar_based_generator import Grammar
from programmatic_policy_learning.dsl.llm_primitives.utils import (
    JSONStructureRepromptCheck,
    create_function_from_stub,
)
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    get_dsl_functions_dict,
)


class LLMPrimitivesGenerator:
    """A class to interact with an LLM, process its response, and generate a
    Grammar object."""

    def __init__(self, llm_client: PretrainedLargeModel | None):
        """Initialize the generator with an LLM client.

        Args:
            llm_client (Any): An object or function to interact
            with the LLM (e.g., OpenAI API client).
        """
        self.llm_client = llm_client

    def query_llm(self, prompt: str) -> dict:
        """Send a query to the LLM and return the parsed JSON response.

        Args:
            prompt (str): The query or prompt to send to the LLM.

        Returns:
            dict: The parsed JSON response from the LLM.
        """
        if self.llm_client is None:
            raise ValueError("LLM client is not initialized.")

        query = Query(prompt)
        reprompt_checks = [JSONStructureRepromptCheck()]
        response = query_with_reprompts(
            self.llm_client,
            query,
            reprompt_checks,  # type: ignore[arg-type]
            max_attempts=5,
        )
        logging.info("Response from LLM:")
        logging.info(response)
        return json.loads(response.text)

    def create_grammar_from_response(
        self, llm_output: dict[str, Any], object_types: list[Any]
    ) -> Grammar[str, int, int]:
        """Create a Grammar object from the LLM's JSON output.

        Args:
            llm_output (dict[str, Any]): Parsed JSON output from the LLM.
            object_types (list[Any]): list of object types for the VALUE nonterminal.

        Returns:
            Grammar[str, int, int]: A reconstructed Grammar object.
        """
        # Extract the proposal and updated grammar
        proposal = llm_output["proposal"]
        updated_grammar = llm_output["updated_grammar"]

        # Initialize the rules dictionary
        rules: dict[int, tuple[list[list[Any]], list[float]]] = {}

        # Map nonterminal names to integer constants
        nonterminal_map = {
            name: idx for idx, name in enumerate(updated_grammar["nonterminals"])
        }

        # Build the rules
        for nonterminal, production_rules in updated_grammar["productions"].items():
            # Parse the production rules into lists of tokens
            parsed_rules = [rule.split() for rule in production_rules.split(" | ")]
            probabilities = [1.0 / len(parsed_rules)] * len(
                parsed_rules
            )  # Uniform probabilities
            rules[nonterminal_map[nonterminal]] = (parsed_rules, probabilities)

        # Handle the VALUE nonterminal separately
        if "VALUE" in nonterminal_map:
            value_rules = [[str(v)] for v in object_types]
            value_probabilities = [1.0 / len(object_types)] * len(object_types)
            rules[nonterminal_map["VALUE"]] = (value_rules, value_probabilities)

        # Integrate the proposal into the grammar
        pcfg_insertion = proposal["pcfg_insertion"]
        nonterminal_to_update = pcfg_insertion["nonterminal"]
        new_production = pcfg_insertion["production"].split()

        if nonterminal_to_update in nonterminal_map:
            nonterminal_idx = nonterminal_map[nonterminal_to_update]
            if new_production not in rules[nonterminal_idx][0]:
                rules[nonterminal_idx][0].append(new_production)
                rules[nonterminal_idx][1].append(
                    1.0
                )  # Assign equal probability to the new rule

                # Normalize probabilities
                total = sum(rules[nonterminal_idx][1])
                rules[nonterminal_idx] = (
                    rules[nonterminal_idx][0],
                    [p / total for p in rules[nonterminal_idx][1]],
                )

        # Create and return the Grammar object
        return Grammar(rules=rules)

    def generate_grammar(
        self, prompt_text: str, object_types: list[Any]
    ) -> Grammar[str, int, int]:
        """Generate a Grammar object by querying the LLM and processing its
        response."""
        llm_response = self.query_llm(prompt_text)
        new_primitive_name = llm_response["proposal"]["name"]
        # pylint: disable=unused-variable
        new_get_dsl_functions_dict = self.add_primitive_to_dsl(
            new_primitive_name,
            create_function_from_stub(
                llm_response["proposal"]["semantics_py_stub"], new_primitive_name
            ),
        )

        return self.create_grammar_from_response(llm_response, object_types)

    def add_primitive_to_dsl(
        self, name: str, implementation: Callable[..., Any]
    ) -> Callable[[], dict[str, Any]]:
        """Add a new primitive to the DSL functions dictionary.

        Args:
            name: The name of the new primitive.
            implementation: The Python implementation of the primitive.

        Returns:
            A modified `get_dsl_functions_dict` function.
        """
        # Get the base DSL functions
        base_dsl_functions = get_dsl_functions_dict()

        if name in base_dsl_functions:
            raise ValueError(f"Primitive '{name}' already exists in the DSL.")

        # Add the new primitive
        base_dsl_functions[name] = implementation

        # Return a new function that includes the updated DSL
        def updated_get_dsl_functions_dict() -> dict[str, Any]:
            return base_dsl_functions

        return updated_get_dsl_functions_dict
