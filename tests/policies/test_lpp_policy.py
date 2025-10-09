"""Tests for the LPPPolicy class."""

import numpy as np

from programmatic_policy_learning.policies.lpp_policy import LPPPolicy


class DummyPLP:
    """Dummy PLP that accepts all actions."""

    def __call__(self, obs: np.ndarray, action: tuple[int, int]) -> bool:
        return True


class TopRowPLP:
    """PLP that accepts only actions in the top row."""

    def __call__(self, obs: np.ndarray, action: tuple[int, int]) -> bool:
        return action[0] == 0


def test_lpp_policy_call() -> None:
    """Test __call__ method for selecting an action."""
    obs = np.ones((2, 2))
    plps = [DummyPLP()]
    probs = [1.0]
    policy: LPPPolicy[np.ndarray, tuple[int, int]] = LPPPolicy(plps, probs)
    action = policy(obs)
    assert isinstance(action, tuple)
    assert len(action) == 2
    assert all(isinstance(x, int) for x in action)


def test_lpp_policy_top_row() -> None:
    """Test policy with a PLP that only accepts top row actions."""
    obs = np.ones((2, 2))
    plps = [TopRowPLP()]
    probs = [1.0]
    policy: LPPPolicy[np.ndarray, tuple[int, int]] = LPPPolicy(plps, probs)
    action = policy(obs)
    assert action[0] == 0  # Should always select top row


def test_lpp_policy_action_probs() -> None:
    """Test get_action_probs returns a valid probability array."""
    obs = np.ones((2, 2))
    plps = [DummyPLP()]
    probs = [1.0]
    policy: LPPPolicy[np.ndarray, tuple[int, int]] = LPPPolicy(plps, probs)
    action_probs = policy.get_action_probs(obs)
    assert isinstance(action_probs, np.ndarray)
    assert action_probs.shape == obs.shape
    assert np.isclose(np.sum(action_probs), 1.0)


def test_lpp_policy_cache() -> None:
    """Test that action probabilities are cached."""
    obs = np.ones((2, 2))
    plps = [DummyPLP()]
    probs = [1.0]
    policy: LPPPolicy[np.ndarray, tuple[int, int]] = LPPPolicy(plps, probs)
    _ = policy.get_action_probs(obs)
    hashed_obs = policy.hash_obs(obs)
    assert hashed_obs in policy._action_prob_cache  # pylint: disable=protected-access


def test_lpp_policy_multiple_plps() -> None:
    """Test policy with multiple PLPs and probabilities."""
    obs = np.ones((2, 2))
    plps = [DummyPLP(), TopRowPLP()]
    probs = [0.6, 0.4]
    policy: LPPPolicy[np.ndarray, tuple[int, int]] = LPPPolicy(plps, probs)
    action_probs = policy.get_action_probs(obs)
    assert np.isclose(np.sum(action_probs), 1.0)
    assert action_probs.shape == obs.shape
