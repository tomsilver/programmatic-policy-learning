"""This module defines utilities for transforming the structured output of a
LLM into a formal Grammar object.

It processes nonterminals, terminals, and production rules to
dynamically construct grammars for DSLs.
"""

from typing import Any

from programmatic_policy_learning.dsl.generators.grammar_based_generator import Grammar


def create_grammar(
    llm_output: dict[str, Any], object_types: list[Any]
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
