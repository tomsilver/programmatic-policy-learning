"""Utility functions for selecting top particles by log prob in LPP."""

from typing import Any

import numpy as np


def select_particles(
    particles: list[Any], particle_log_probs: list[float], max_num_particles: int
) -> tuple[list[Any], list[float]]:
    """Select top particles by log probability.

    Parameters
    ----------
    particles : [ Any ]
    particle_log_probs : [ float ]
    max_num_particles : int

    Returns
    -------
    selected_particles : [ Any ]
    selected_particle_log_probs : [ float ]
    """
    sorted_log_probs, _, sorted_particles = (
        list(t)
        for t in zip(
            *sorted(
                zip(
                    particle_log_probs, np.random.random(size=len(particles)), particles
                ),
                reverse=True,
            )
        )
    )
    end = min(max_num_particles, len(sorted_particles))
    try:
        idx = sorted_log_probs.index(-np.inf)
        end = min(idx, end)
    except ValueError:
        pass
    return sorted_particles[:end], sorted_log_probs[:end]
