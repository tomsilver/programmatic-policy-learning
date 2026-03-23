"""Step 2 tests: Verify row-aligned sample weights in dataset pipeline."""

import numpy as np

from programmatic_policy_learning.data.dataset import (
    extract_examples_from_demonstration,
    extract_examples_from_demonstration_item,
)
from programmatic_policy_learning.data.demo_types import Trajectory


def test_weights_length_matches_examples_discrete() -> None:
    """Test that sample weights length == total examples in discrete mode."""
    state = np.array([[1, 2, 3], [4, 5, 6]])
    action = (0, 1)
    traj: Trajectory[np.ndarray, tuple[int, int]] = Trajectory(steps=[(state, action)])

    pos, neg, weights = extract_examples_from_demonstration(
        traj,
        action_mode="discrete",
    )
    total_examples = len(pos) + len(neg)
    assert len(weights) == total_examples
    print(
        "\n[discrete-weight-length] "
        f"total_examples={total_examples}, "
        f"weight_len={len(weights)}"
    )


def test_weights_length_matches_examples_continuous() -> None:
    """Test that sample weights length == total examples in continuous mode."""
    obs = np.array([0.0, 1.0], dtype=np.float32)
    action = np.array([0.1, -0.2], dtype=np.float32)
    neg_cfg = {
        "action_low": [-1.0, -0.5],
        "action_high": [1.0, 0.5],
        "continuous": {
            "bucket_counts": 5,
        },
    }
    pos, neg, weights = extract_examples_from_demonstration(
        Trajectory(steps=[(obs, action)]),
        negative_sampling=neg_cfg,
        action_mode="continuous",
    )
    total_examples = len(pos) + len(neg)
    assert len(weights) == total_examples
    # 5x5 buckets = 25 total, so 1 positive + 24 negatives
    assert total_examples == 25


def test_weights_are_row_aligned_positives_first() -> None:
    """Test that weights are aligned with examples: positives first, then negatives."""
    obs = np.array([0.0, 1.0], dtype=np.float32)
    action = np.array([0.1, -0.2], dtype=np.float32)
    neg_cfg = {
        "action_low": [-1.0, -0.5],
        "action_high": [1.0, 0.5],
        "continuous": {
            "bucket_counts": 3,  # 3x3 = 9 total buckets
        },
    }
    pos, neg, weights = extract_examples_from_demonstration_item(
        (obs, action),
        negative_sampling=neg_cfg,
        action_mode="continuous",
    )

    total_examples = len(pos) + len(neg)
    assert len(weights) == total_examples

    # In the bucket-aligned return: first weight is for positive, rest are for negatives
    # The first weight should be the positive weight (default=1.0)
    # All weights should be positive floats
    assert weights[0] > 0.0  # positive weight
    assert all(w > 0.0 for w in weights)  # all weights positive
    assert len(weights) == 1 + 8  # 1 positive + 8 negatives


def test_weights_default_uniform_continuous() -> None:
    """Test that default continuous weights (compute_sample_weights=False) are
    uniform."""
    obs = np.array([0.0, 0.0], dtype=np.float32)
    action = np.array([0.0, 0.0], dtype=np.float32)
    neg_cfg = {
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
        "continuous": {
            "bucket_counts": 3,
        },
    }
    _pos, _neg, weights = extract_examples_from_demonstration_item(
        (obs, action),
        negative_sampling=neg_cfg,
        action_mode="continuous",
        compute_sample_weights=False,  # explicitly request default behavior
    )

    # All weights should be 1.0 when not computing cost-sensitive
    assert np.allclose(weights, 1.0)
    print(f"\n[uniform-weights] all_weights_equal_1={np.allclose(weights, 1.0)}")


def test_weights_discrete_are_uniform() -> None:
    """Test that discrete mode returns uniform weights."""
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
    _pos, _neg, weights = extract_examples_from_demonstration_item(
        (state, action),
        negative_sampling=neg_cfg,
        action_mode="discrete",
    )

    # Discrete mode returns uniform weights
    assert np.allclose(weights, 1.0)
    assert len(weights) == 1 + k


def test_weights_support_cost_sensitive_continuous() -> None:
    """Test that weights can be computed cost-sensitively in continuous
    mode."""
    obs = np.array([0.0, 0.0], dtype=np.float32)
    action = np.array([0.0, 0.0], dtype=np.float32)  # center of grid
    neg_cfg = {
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
        "continuous": {
            "bucket_counts": 3,
            "weight_config": {
                "beta_pos": 1.0,
                "beta_neg": 1.0,
                "alpha": 1.0,
                "lambda_per_dim": [1.0, 1.0],
            },
        },
    }
    _pos, _neg, weights = extract_examples_from_demonstration_item(
        (obs, action),
        negative_sampling=neg_cfg,
        action_mode="continuous",
        compute_sample_weights=True,  # enable cost-sensitive weighting
    )
    # With cost-sensitive weighting, weights should vary based on distance
    # Center bucket (expert) should have weight=1.0 (beta_pos)
    # Negatives should have varying weights based on distance
    pos_weight = weights[0]
    neg_weights = weights[1:]

    assert np.isclose(
        pos_weight, 1.0
    ), f"Expected positive weight ~1.0, got {pos_weight}"
    assert all(w > 0.0 for w in neg_weights), "All negative weights should be positive"
    assert not np.allclose(
        neg_weights, neg_weights[0]
    ), "Negative weights should vary with distance"


def test_weights_accumulate_across_trajectory() -> None:
    """Test that weights correctly accumulate when processing multiple
    trajectory steps."""
    state1 = np.array([0.0, 0.0], dtype=np.float32)
    action1 = np.array([0.0, 0.0], dtype=np.float32)
    state2 = np.array([1.0, 1.0], dtype=np.float32)
    action2 = np.array([0.5, 0.5], dtype=np.float32)

    neg_cfg = {
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
        "continuous": {
            "bucket_counts": 3,
        },
    }

    traj: Trajectory[np.ndarray, np.ndarray] = Trajectory(
        steps=[(state1, action1), (state2, action2)]
    )

    pos, neg, weights = extract_examples_from_demonstration(
        traj,
        negative_sampling=neg_cfg,
        action_mode="continuous",
    )

    # Should have 2 positives (one per step) + negatives from both steps
    assert len(pos) == 2
    assert len(neg) == 16  # 2 steps * 8 negatives per step (9 buckets - 1 positive)
    assert len(weights) == 18  # Total examples
