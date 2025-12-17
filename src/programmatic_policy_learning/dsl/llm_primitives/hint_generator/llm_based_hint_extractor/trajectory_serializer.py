# trajectory_serializer.py

from typing import Any, List, Tuple
import numpy as np
from grid_encoder import GridStateEncoder
from transition_analyzer import GenericTransitionAnalyzer


def trajectory_to_text(
    trajectory: List[Tuple[np.ndarray, Tuple[int, int], np.ndarray]],
    *,
    encoder: GridStateEncoder,
    analyzer: GenericTransitionAnalyzer,
    salient_tokens: List[str],
    max_steps: int | None = None,
) -> str:
    blocks = []
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
""" + "\n".join(f"- {e}" for e in events)

        blocks.append(block.strip())

    return "\n\n".join(blocks)
