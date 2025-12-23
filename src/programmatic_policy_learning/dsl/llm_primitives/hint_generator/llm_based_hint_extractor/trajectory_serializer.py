"""Serialize expert trajectories into textual hint blocks."""

from typing import Sequence

import numpy as np

# pylint: disable=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.hint_generator.llm_based_hint_extractor.grid_encoder import (
    GridStateEncoder,
)
from programmatic_policy_learning.dsl.llm_primitives.hint_generator.llm_based_hint_extractor.transition_analyzer import (
    GenericTransitionAnalyzer,
)


def trajectory_to_text(
    trajectory: Sequence[tuple[np.ndarray, tuple[int, int], np.ndarray]],
    *,
    encoder: GridStateEncoder,
    analyzer: GenericTransitionAnalyzer,
    salient_tokens: list[str],
    max_steps: int | None = None,
) -> str:
    """Convert (obs, action, next_obs) tuples to structured text."""
    blocks: list[str] = []
    steps = trajectory[:max_steps] if max_steps else trajectory

    for i, (obs_t, action, obs_t1) in enumerate(steps):

        ascii_t = encoder.to_ascii(obs_t, action)
        # ascii_t1 = encoder.to_ascii(obs_t1)
        # action_array = np.full(obs_t.shape, "_", dtype=object)
        # action_array[action] = "C"
        # rows: list[str] = []
        # for r in range(action_array.shape[0]):
        #     row_chars = []
        #     for c in range(action_array.shape[1]):
        #         token = action_array[r, c]
        #         row_chars.append(token)
        #     rows.append("".join(row_chars))
        # ascii_action = "\n".join(rows)
        objs = encoder.extract_objects(obs_t, salient_tokens)
        listing = encoder.format_coordinate_listing(objs)

        events = analyzer.analyze(
            obs_t,
            action,
            obs_t1,
            encoder=encoder,
        )

        change_summary = "\n".join(f"- {event}" for event in events)
        block = f"""
=== EXAMPLE ===
{ascii_t}
""".strip()
        blocks.append(block)

    return "\n\n".join(blocks)
