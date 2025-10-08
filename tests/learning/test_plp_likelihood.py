"""Tests for PLP likelihood computation functions."""

import numpy as np

from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram
from programmatic_policy_learning.learning.plp_likelihood import (
    compute_likelihood_plps,
    compute_likelihood_single_plp,
)


class DummyPLP(StateActionProgram):
    """Dummy PLP."""

    def __call__(
        self, obs: np.ndarray, action: tuple[int, int]
    ) -> bool:  # pylint: disable=unused-argument
        """Call."""
        # Accept all actions for testing
        return True


def test_compute_likelihood_single_plp() -> None:
    """Test single PLP likelihood computation."""
    obs = np.ones((2, 2))
    action = (0, 1)
    steps = [(obs, action)]
    traj = Trajectory(steps)
    plp: StateActionProgram = DummyPLP("dummy")
    ll = compute_likelihood_single_plp(traj, plp)
    assert isinstance(ll, float)
    assert ll <= 0  # log likelihood should be <= 0


def test_compute_likelihood_plps() -> None:
    """Test likelihood computation for multiple PLPs."""
    obs = np.ones((2, 2))
    action = (0, 1)
    steps = [(obs, action)]
    traj = Trajectory(steps)
    plps: list[StateActionProgram] = [DummyPLP("dummy1"), DummyPLP("dummy2")]
    likelihoods = compute_likelihood_plps(plps, traj)
    assert isinstance(likelihoods, list)
    assert all(isinstance(ll, float) for ll in likelihoods)


# Python’s multiprocessing cannot pickle local classes or functions
# that's why these two classes are outside the corresponding test function
class SelectTopRow(StateActionProgram):
    """SelectTopRow."""

    def __call__(
        self, obs: np.ndarray, action: tuple[int, int]
    ) -> bool:  # pylint: disable=unused-argument
        """Call."""
        # Only allow actions in the top row
        return action[0] == 0


class SelectDiagonal(StateActionProgram):
    """SelectDiagonal."""

    def __call__(
        self, obs: np.ndarray, action: tuple[int, int]
    ) -> bool:  # pylint: disable=unused-argument
        """Call."""
        # Only allow actions on the diagonal
        return action[0] == action[1]


def test_compute_likelihood_plps_real() -> None:
    """Test compute_likelihood_plps where one PLP accepts all actions."""
    obs1 = np.array([[1, 0], [0, 1]])
    obs2 = np.array([[0, 1], [1, 0]])
    action1 = (0, 0)
    action2 = (0, 1)
    steps = [(obs1, action1), (obs2, action2)]
    traj = Trajectory(steps)
    plps: list[StateActionProgram] = [
        SelectTopRow("top_row"),
        SelectDiagonal("diagonal"),
    ]
    likelihoods = compute_likelihood_plps(plps, traj)
    assert likelihoods[0] > -np.inf  # SelectTopRow accepts both actions
    assert isinstance(likelihoods, list)
    assert all(isinstance(ll, float) for ll in likelihoods)
