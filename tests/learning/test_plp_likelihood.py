"""Tests for PLP likelihood computation functions."""

import numpy as np

from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram
from programmatic_policy_learning.learning.plp_likelihood import compute_likelihood_plps


def test_compute_likelihood_plps() -> None:
    """Test likelihood computation for multiple PLPs."""
    obs = np.ones((2, 2))
    action = (0, 1)
    steps = [(obs, action)]
    traj = Trajectory(steps)
    plps: list[StateActionProgram] = [
        StateActionProgram("True"),
        StateActionProgram("False or True"),
    ]
    likelihoods = compute_likelihood_plps(plps, traj, {})
    assert isinstance(likelihoods, list)
    assert all(isinstance(ll, float) for ll in likelihoods)


def test_compute_likelihood_plps_real() -> None:
    """Test compute_likelihood_plps where one PLP accepts all actions."""
    obs1 = np.array([[1, 0], [0, 1]])
    obs2 = np.array([[0, 1], [1, 0]])
    action1 = (0, 0)
    action2 = (0, 1)
    steps = [(obs1, action1), (obs2, action2)]
    traj = Trajectory(steps)
    plps = [
        StateActionProgram("a[0] == 0"),  # top row
        StateActionProgram("a[0] == a[1]"),  # diagonal
    ]
    likelihoods = compute_likelihood_plps(plps, traj, {})
    assert likelihoods[0] > -np.inf  # SelectTopRow accepts both actions
    assert isinstance(likelihoods, list)
    assert all(isinstance(ll, float) for ll in likelihoods)
