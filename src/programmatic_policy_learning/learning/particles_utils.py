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
    if len(particles) != len(particle_log_probs):
        raise ValueError("particles and particle_log_probs must have the same length.")
    if not particles:
        return [], []

    ranked = sorted(
        enumerate(zip(particle_log_probs, particles)),
        key=lambda item: (-float(item[1][0]), int(item[0])),
    )
    sorted_log_probs = [float(item[1][0]) for item in ranked]
    sorted_particles = [item[1][1] for item in ranked]
    end = min(max_num_particles, len(sorted_particles))
    try:
        idx = sorted_log_probs.index(-np.inf)
        end = min(idx, end)
    except ValueError:
        pass
    return sorted_particles[:end], sorted_log_probs[:end]
