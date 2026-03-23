"""Test for dataset creation workflow."""

import numpy as np

from programmatic_policy_learning.data.dataset import (
    compute_cost_sensitive_bucket_weights,
    extract_examples_from_demonstration,
    extract_examples_from_demonstration_item,
    run_all_programs_on_single_demonstration,
)
from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.utils.action_quantization import Motion2DActionQuantizer


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


def test_continuous_quantized_expansion_uses_full_grid() -> None:
    """Continuous expansion always uses quantized full-grid negatives."""
    obs = np.array([0.0, 1.0], dtype=np.float32)
    action = np.array([0.1, -0.2], dtype=np.float32)
    neg_cfg = {
        "action_low": [-1.0, -0.5],
        "action_high": [1.0, 0.5],
    }
    _pos, neg = extract_examples_from_demonstration_item(
        (obs, action),
        negative_sampling=neg_cfg,
        action_mode="continuous",
    )
    print("\n[continuous-quantized] pos_count=", len(_pos), "neg_count=", len(neg))
    # default bucket_counts=5 for each of dx,dy => 25 total, 24 negatives
    assert len(_pos) == 1
    assert len(neg) == 24
    pos_action = np.asarray(_pos[0][1], dtype=float)
    q = Motion2DActionQuantizer.from_bounds([-1.0, -0.5], [1.0, 0.5], bucket_counts=5)
    expected_bucket = q.quantize(action)
    expected_center = q.dequantize(expected_bucket)
    print(
        "[continuous-quantized] expert_action=",
        action.tolist(),
        "bucket=",
        expected_bucket,
        "center=",
        expected_center.tolist(),
    )
    preview = [np.asarray(a, dtype=float).tolist() for _, a in neg[:5]]
    print("[continuous-quantized] first_5_negatives=", preview)
    assert np.allclose(pos_action, expected_center)
    for _s, a in neg:
        arr = np.asarray(a, dtype=float)
        assert arr.shape == (2,)
        assert np.all(arr >= np.array([-1.0, -1.0]))
        assert np.all(arr <= np.array([1.0, 1.0]))


def test_continuous_quantized_expansion_supports_per_dim_bucket_counts() -> None:
    """Continuous quantized expansion supports per-dimension bucket counts."""
    obs = np.array([0.0, 1.0], dtype=np.float32)
    action = np.array([0.1, -0.2], dtype=np.float32)
    neg_cfg = {
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
        "continuous": {
            "bucket_counts": [3, 7],
        },
    }
    _pos, neg = extract_examples_from_demonstration_item(
        (obs, action),
        negative_sampling=neg_cfg,
        action_mode="continuous",
    )
    assert len(neg) == 20
    for _s, a in neg:
        arr = np.asarray(a, dtype=float)
        assert arr.shape == (2,)
        assert np.all(arr >= -1.0)
        assert np.all(arr <= 1.0)


def test_continuous_quantized_expansion_ignores_enabled_flag() -> None:
    """Continuous quantized expansion is independent of enabled flag."""
    obs = np.array([0.0, 1.0], dtype=np.float32)
    action = np.array([0.0493, 0.0265, 0.0, 0.0, 0.0], dtype=np.float32)
    neg_cfg = {
        "enabled": False,
        "action_low": [-1.0, -1.0, -1.0, -1.0, -1.0],
        "action_high": [1.0, 1.0, 1.0, 1.0, 1.0],
        "continuous": {
            "bucket_counts": 3,
        },
    }
    _pos, neg = extract_examples_from_demonstration_item(
        (obs, action),
        negative_sampling=neg_cfg,
        action_mode="continuous",
    )
    assert len(neg) == 8
    pos_arr = np.asarray(_pos[0][1], dtype=float)
    assert np.isclose(pos_arr[2], 0.0)
    assert np.isclose(pos_arr[3], 0.0)
    assert np.isclose(pos_arr[4], 0.0)


def test_cost_sensitive_bucket_weights_positive_and_negative_mass() -> None:
    expert = (1, 1)
    candidates = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
    beta_pos = 2.0
    beta_neg = 3.0
    w = compute_cost_sensitive_bucket_weights(
        expert,
        candidates,
        beta_pos=beta_pos,
        beta_neg=beta_neg,
        alpha=1.0,
        lambda_per_dim=(1.0, 1.0),
    )
    assert np.isclose(float(w[2]), beta_pos)
    neg_total = float(np.sum(w)) - float(w[2])
    assert np.isclose(neg_total, beta_neg)


def test_cost_sensitive_bucket_weights_farther_negative_gets_more_weight() -> None:
    expert = (1, 1)
    # idx 0: near negative (dist=1), idx 1: far negative (dist=3), idx 2: expert
    candidates = [(1, 0), (3, 0), (1, 1)]
    w = compute_cost_sensitive_bucket_weights(
        expert,
        candidates,
        beta_pos=1.0,
        beta_neg=1.0,
        alpha=1.0,
        lambda_per_dim=(1.0, 1.0),
    )
    assert float(w[1]) > float(w[0])


def test_cost_sensitive_bucket_weights_supports_dimension_lambdas() -> None:
    expert = (1, 1)
    # Both negatives have |delta|=1 in one dimension, but lambda_x is larger.
    candidates = [(2, 1), (1, 2), (1, 1)]
    w = compute_cost_sensitive_bucket_weights(
        expert,
        candidates,
        beta_pos=1.0,
        beta_neg=1.0,
        alpha=1.0,
        lambda_per_dim=(2.0, 0.5),
    )
    assert float(w[0]) > float(w[1])
