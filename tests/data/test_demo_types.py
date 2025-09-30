"""Tests for demo_types.py."""

import numpy as np

from programmatic_policy_learning.data.demo_types import Demo, Trajectory


def test_demo_and_trajectory():
    """Test Demo and Trajectory dataclasses."""
    demo = Demo(obs=np.ones((3, 3)), act=42)
    traj = Trajectory(steps=[demo])
    assert isinstance(demo, Demo)
    assert isinstance(traj, Trajectory)
    assert traj.steps[0].obs.shape == (3, 3)
    assert traj.steps[0].act == 42
    assert len(traj.steps) == 1
