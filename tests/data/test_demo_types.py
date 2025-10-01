"""Tests for demo_types.py."""

import numpy as np

from programmatic_policy_learning.data.demo_types import Trajectory


def test_trajectory_dataclass():
    """Test Trajectory dataclass with obs and act lists."""
    obs = [np.ones((3, 3)), np.zeros((3, 3))]
    act = [42, 7]
    traj = Trajectory(obs=obs, act=act)
    assert isinstance(traj, Trajectory)
    assert isinstance(traj.obs, list)
    assert isinstance(traj.act, list)
    assert traj.obs[0].shape == (3, 3)
    assert traj.act[0] == 42
    assert len(traj.obs) == len(traj.act) == 2
