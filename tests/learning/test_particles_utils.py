"""Tests for particle selection utilities."""

import numpy as np

from programmatic_policy_learning.learning.particles_utils import select_particles


def test_select_particles_basic() -> None:
    """Test selecting top particles by log probability."""
    particles = ["a", "b", "c", "d"]
    log_probs = [0.1, -0.2, 0.5, -np.inf]
    max_num_particles = 2

    selected_particles, selected_log_probs = select_particles(
        particles, log_probs, max_num_particles
    )
    assert isinstance(selected_particles, list)
    assert isinstance(selected_log_probs, list)
    assert len(selected_particles) <= max_num_particles
    assert all(lp != -np.inf for lp in selected_log_probs)
    assert selected_particles[0] == "c"  # Highest log prob


def test_select_particles_all_inf() -> None:
    """Test when all log probabilities are -inf."""
    particles = ["a", "b"]
    log_probs = [-np.inf, -np.inf]
    max_num_particles = 2

    selected_particles, selected_log_probs = select_particles(
        particles, log_probs, max_num_particles
    )
    assert selected_particles == []
    assert selected_log_probs == []
