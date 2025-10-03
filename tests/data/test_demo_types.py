"""Tests for demo_types.py."""

import numpy as np

from programmatic_policy_learning.data.demo_types import Trajectory


def test_trajectory_dataclass() -> None:
    """Test Trajectory dataclass with steps as (obs, act) tuples."""
    obs = [np.ones((3, 3)), np.zeros((3, 3))]
    act = [42, 7]
    steps = list(zip(obs, act))
    traj = Trajectory(steps=steps)
    assert isinstance(traj, Trajectory)
    assert isinstance(traj.steps, list)
    assert isinstance(traj.steps[0], tuple)
    assert traj.steps[0][0].shape == (3, 3)
    assert traj.steps[0][1] == 42
    assert len(traj.steps) == 2
