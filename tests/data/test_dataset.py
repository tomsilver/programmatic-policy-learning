"""Test for dataset creation workflow."""

import numpy as np

from programmatic_policy_learning.data.dataset import (
    extract_examples_from_demonstration,
    extract_examples_from_demonstration_item,
    run_all_programs_on_single_demonstration,
    sample_manual_negative_actions_continuous,
)
from programmatic_policy_learning.data.demo_types import Trajectory


def test_run_all_programs_on_single_demonstration() -> None:
    """Test running programs on a single demonstration."""
    # Create a dummy demonstration
    state = np.array([[1, 2], [3, 4]])
    action = (0, 1)
    traj: Trajectory[np.ndarray, tuple[int, int]] = Trajectory(steps=[(state, action)])

    programs = ["np.sum(s) > 10"]
    X, y, _examples = run_all_programs_on_single_demonstration(
        "DummyEnv",  # base_class_name
        0,  # demo_number
        programs,  # programs
        traj,  # demo_traj
        {},
    )

    assert X.shape[0] == len(y)
    assert X.shape[1] == len(programs)
    assert set(y) <= {0, 1}  # binary labels


def test_extract_examples_from_demonstration() -> None:
    """Test extracting positive and negative examples from demonstration."""

    state = np.array([[1, 2], [3, 4]])
    action = (0, 1)
    traj: Trajectory[np.ndarray, tuple[int, int]] = Trajectory(steps=[(state, action)])
    pos, neg = extract_examples_from_demonstration(traj)
    assert len(pos) == 1
    assert all(isinstance(x, tuple) for x in pos)
    assert all(isinstance(x, tuple) for x in neg)


def test_discrete_negative_sampling_fallback_uses_all_other_cells() -> None:
    """Discrete fallback should include all non-expert cells as negatives."""
    state = np.array([[1, 2, 3], [4, 5, 6]])
    action = (0, 1)
    traj: Trajectory[np.ndarray, tuple[int, int]] = Trajectory(steps=[(state, action)])

    pos, neg = extract_examples_from_demonstration(
        traj,
        action_mode="discrete",
    )
    assert len(pos) == 1
    assert len(neg) == (state.shape[0] * state.shape[1] - 1)
    assert all(a != action for _, a in neg)


def test_discrete_negative_sampling_enabled_respects_k_and_excludes_expert() -> None:
    """Discrete mixture sampling should return K negatives and exclude
    expert."""
    state = np.array([[1, 2, 2], [2, 2, 3], [4, 2, 5]])
    action = (1, 1)
    k = 5
    neg_cfg = {
        "enabled": True,
        "discrete": {
            "K": k,
            "local_radius": 1,
            "w_local": 0.5,
            "w_struct": 0.3,
            "w_random": 0.2,
        },
    }
    _pos, neg = extract_examples_from_demonstration_item(
        (state, action),
        negative_sampling=neg_cfg,
        action_mode="discrete",
    )
    assert len(neg) == k
    neg_actions = [a for _, a in neg]
    assert len(set(neg_actions)) == len(neg_actions)
    assert all(a != action for a in neg_actions)


def test_continuous_negative_sampling_fallback_uses_uniform_with_bounds() -> None:
    """Continuous fallback samples bounded negatives when bounds are
    provided."""
    obs = np.array([0.0, 1.0], dtype=np.float32)
    action = np.array([0.1, -0.2], dtype=np.float32)
    neg_cfg = {
        "enabled": False,
        "action_low": [-1.0, -0.5],
        "action_high": [1.0, 0.5],
    }
    _pos, neg = extract_examples_from_demonstration_item(
        (obs, action),
        negative_sampling=neg_cfg,
        action_mode="continuous",
    )
    assert len(neg) == 10
    for _s, a in neg:
        arr = np.asarray(a, dtype=float)
        assert arr.shape == (2,)
        assert np.all(arr >= np.array([-1.0, -0.5]))
        assert np.all(arr <= np.array([1.0, 0.5]))


def test_continuous_negative_sampling_enabled_respects_k_and_bounds() -> None:
    """Continuous mixture sampling should respect K and action bounds."""
    obs = np.array([0.0, 1.0], dtype=np.float32)
    action = np.array([0.1, -0.2], dtype=np.float32)
    k = 7
    neg_cfg = {
        "enabled": True,
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
        "continuous": {
            "K": k,
            "local_noise_scale": 0.2,
            "lambda_local": 0.6,
            "lambda_uniform": 0.4,
            "lambda_traj": 0.0,
        },
    }
    _pos, neg = extract_examples_from_demonstration_item(
        (obs, action),
        negative_sampling=neg_cfg,
        action_mode="continuous",
    )
    assert len(neg) == k
    for _s, a in neg:
        arr = np.asarray(a, dtype=float)
        assert arr.shape == (2,)
        assert np.all(arr >= -1.0)
        assert np.all(arr <= 1.0)


def test_manual_continuous_negative_sampling_returns_expected_variants() -> None:
    """Manual continuous negatives should match the fixed 10 variants."""
    action = np.array([0.0493, 0.0265, 0.0, 0.0, 0.0], dtype=np.float32)

    neg_actions = sample_manual_negative_actions_continuous(action)

    assert len(neg_actions) == 10
    got = [tuple(np.asarray(a, dtype=float).round(4).tolist()) for a in neg_actions]
    assert (-0.0493, 0.0265, 0.0, 0.0, 0.0) in got
    assert (0.0493, -0.0265, 0.0, 0.0, 0.0) in got
    assert (-0.0493, -0.0265, 0.0, 0.0, 0.0) in got
    assert (0.0493, -0.0132, 0.0, 0.0, 0.0) in got
    assert (-0.0247, 0.0265, 0.0, 0.0, 0.0) in got
    assert (0.0247, 0.0, 0.0, 0.0, 0.0) in got
    assert (0.0, 0.0132, 0.0, 0.0, 0.0) in got
    assert (0.0247, 0.0132, 0.0, 0.0, 0.0) in got
    assert any(
        np.allclose(
            np.asarray(a, dtype=float),
            np.array([0.07395, 0.03975, 0.0, 0.0, 0.0]),
            atol=1e-4,
        )
        for a in neg_actions
    )
    assert (0.0493, 0.0132, 0.0, 0.0, 0.0) in got


def test_continuous_negative_sampling_manual_mode() -> None:
    """Continuous negative sampling should support manual mode via config."""
    obs = np.array([0.0, 1.0], dtype=np.float32)
    action = np.array([0.0493, 0.0265, 0.0, 0.0, 0.0], dtype=np.float32)
    neg_cfg = {
        "enabled": True,
        "action_low": [-1.0, -1.0, -1.0, -1.0, -1.0],
        "action_high": [1.0, 1.0, 1.0, 1.0, 1.0],
        "continuous": {
            "mode": "manual",
        },
    }
    _pos, neg = extract_examples_from_demonstration_item(
        (obs, action),
        negative_sampling=neg_cfg,
        action_mode="continuous",
    )
    assert len(neg) == 10
