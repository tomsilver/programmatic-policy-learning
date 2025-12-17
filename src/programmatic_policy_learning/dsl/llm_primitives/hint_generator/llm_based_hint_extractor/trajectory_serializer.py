"""Serialize expert trajectories into textual hint blocks."""

from typing import List, Sequence, Tuple

import numpy as np

from .grid_encoder import GridStateEncoder
from .transition_analyzer import GenericTransitionAnalyzer


def trajectory_to_text(
    trajectory: Sequence[Tuple[np.ndarray, Tuple[int, int], np.ndarray]],
    *,
    encoder: GridStateEncoder,
    analyzer: GenericTransitionAnalyzer,
    salient_tokens: List[str],
    max_steps: int | None = None,
) -> str:
    """Convert (obs, action, next_obs) tuples to structured text."""
    blocks: list[str] = []
    steps = trajectory[:max_steps] if max_steps else trajectory

    for i, (obs_t, action, obs_t1) in enumerate(steps):
        ascii_t = encoder.to_ascii(obs_t)
        ascii_t1 = encoder.to_ascii(obs_t1)
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
=== TRANSITION {i} ===
ASCII(t):
{ascii_t}

Action: {action}

ASCII(t+1):
{ascii_t1}

Objects(t):
{listing}

Change summary:
{change_summary}
        """.strip()
        blocks.append(block)

    return "\n\n".join(blocks)
