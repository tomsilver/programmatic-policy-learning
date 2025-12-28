"""Serialize expert trajectories into textual hint blocks."""

from typing import Sequence

import numpy as np

# pylint: disable=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.hint_generator.llm_based_hint_extractor.grid_encoder import (
    GridStateEncoder,
)
from programmatic_policy_learning.dsl.llm_primitives.hint_generator.llm_based_hint_extractor.transition_analyzer import (
    GenericTransitionAnalyzer,
    extract_relational_facts,
)


def trajectory_to_text(
    trajectory: Sequence[tuple[np.ndarray, tuple[int, int], np.ndarray]],
    *,
    encoder: GridStateEncoder,
    analyzer: GenericTransitionAnalyzer,
    salient_tokens: list[str],
    encoding_method: str,
    max_steps: int | None = None,
) -> str:
    """Convert (obs, action, next_obs) tuples to structured text."""
    blocks: list[str] = []
    steps = trajectory[:max_steps] if max_steps else trajectory

    for i, (obs_t, action, obs_t1) in enumerate(steps):
        # ascii_t = encoder.to_ascii(obs_t, action, i)
        ascii_t = encoder.to_ascii_list_literal(obs_t, action, i)
        # ascii_t1 = encoder.to_ascii(obs_t1)
        tokens = list(dict.fromkeys([*salient_tokens, encoder.cfg.empty_token]))
        objs = encoder.extract_objects(obs_t, tokens)
        relational_facts = extract_relational_facts(
            obs_t,
            objs,
            symbol_map=encoder.cfg.symbol_map,
            reference_tokens=salient_tokens,
            empty_tokens=(encoder.cfg.empty_token,),
        )

        # objs_next = encoder.extract_objects(obs_t1, tokens)
        listing = encoder.format_cell_value_listing(
            objs,
            i,
            action=action,
        )
        events = analyzer.analyze(
            obs_t,
            action,
            obs_t1,
            encoder=encoder,
        )
        change_summary = "\n".join(f"- {event}" for event in events)
        # block = f"=== EXAMPLE ===\n{ascii_t}"

        if encoding_method == "1":
            block = ascii_t
            blocks.append(block)

        elif encoding_method == "2":
            block = listing
            blocks.append(block)
        elif encoding_method == "3":
            block = listing
            blocks.append(block)
            blocks.append(f"\nCHANGES SUMMARY:\n{change_summary}")
        elif encoding_method == "4":
            block = listing
            blocks.append(block)
            blocks.append(f"\nCHANGES SUMMARY:\n{change_summary}\n\n")
            blocks.append(
                f"Relational Facts:\n{', '.join(str(each) for each in relational_facts)}"
            )
    return "\n\n".join(blocks)
