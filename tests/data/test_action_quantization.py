"""Tests for action quantization."""

import numpy as np
import pytest

from programmatic_policy_learning.utils.action_quantization import (
    Motion2DActionQuantizer,
)


def test_quantizer_requires_odd_bucket_count() -> None:
    """Check odd buckets."""
    print("\n[odd-count-check] trying bucket_counts=4 (should fail)")
    with pytest.raises(ValueError, match="must be odd"):
        Motion2DActionQuantizer.from_bounds(
            [-1.0, -1.0],
            [1.0, 1.0],
            bucket_counts=4,
        )


def test_quantize_zero_gets_dedicated_bucket() -> None:
    """Check zero bucket."""
    quantizer = Motion2DActionQuantizer.from_bounds(
        [-1.0, -2.0],
        [1.0, 2.0],
        bucket_counts=5,
    )

    bucket = quantizer.quantize([0.0, 0.0])
    print(f"\n[quantize-zero] action=[0,0] -> bucket={bucket}")
    assert bucket == (2, 2)


def test_dequantize_zero_bucket_returns_exact_zero() -> None:
    """Check zero center."""
    quantizer = Motion2DActionQuantizer.from_bounds(
        [-1.0, -1.0],
        [1.0, 1.0],
        bucket_counts=3,
    )

    center = quantizer.dequantize((1, 1))
    print(f"\n[dequantize-zero] bucket=(1,1) -> center={center.tolist()}")
    np.testing.assert_allclose(center, np.array([0.0, 0.0]))


def test_dynamic_bounds_split_changes_centers() -> None:
    """Check dynamic centers."""
    quantizer = Motion2DActionQuantizer.from_bounds(
        [-2.0, -1.0],
        [1.0, 3.0],
        bucket_counts=(5, 5),
    )

    # For dim 0, positive side [0, 1] with two bins -> upper bin center is 0.75.
    center = quantizer.dequantize((4, 2))
    print(f"\n[dynamic-bounds] bucket=(4,2) -> center={center.tolist()}")
    assert np.isclose(center[0], 0.75)

    # For dim 1, positive side [0, 3] with two bins -> upper bin center is 2.25.
    center = quantizer.dequantize((2, 4))
    print(f"[dynamic-bounds] bucket=(2,4) -> center={center.tolist()}")
    assert np.isclose(center[1], 2.25)


def test_quantize_clips_out_of_range_values() -> None:
    """Check clipping."""
    quantizer = Motion2DActionQuantizer.from_bounds(
        [-1.0, -1.0],
        [1.0, 1.0],
        bucket_counts=5,
    )

    bucket = quantizer.quantize([99.0, -99.0])
    print("\n[clip-out-of-range] " f"action=[99,-99] -> bucket={bucket}")
    assert bucket == (4, 0)


def test_all_bucket_indices_count() -> None:
    """Check bucket count."""
    quantizer = Motion2DActionQuantizer.from_bounds(
        [-1.0, -1.0],
        [1.0, 1.0],
        bucket_counts=(3, 5),
    )

    all_indices = quantizer.all_bucket_indices()
    print(
        "\n[all-buckets] "
        f"count={len(all_indices)} "
        f"first5={all_indices[:5]} "
        f"last5={all_indices[-5:]}"
    )
    assert len(all_indices) == 15
    assert len(set(all_indices)) == 15


def test_ragged_bucket_edges_are_supported_per_dimension() -> None:
    """Ragged per-dimension bucket edges should normalize correctly."""
    quantizer = Motion2DActionQuantizer.from_bounds(
        [-0.05, -0.05, -0.1, -0.1, 0.0],
        [0.05, 0.05, 0.1, 0.1, 1.0],
        bucket_edges=[
            [-0.05, -0.025, 0.0, 0.025, 0.05],
            [-0.05, -0.025, 0.0, 0.025, 0.05],
            [-0.1, -0.05, 0.0, 0.05, 0.1],
            [-0.1, -0.05, 0.0, 0.05, 0.1],
            [0.0, 0.5, 1.0],
        ],
    )

    bucket = quantizer.quantize([0.01, -0.03, 0.0, 0.08, 1.0])
    print(f"\n[ragged-edges] action -> bucket={bucket}")
    assert bucket == (2, 0, 2, 3, 1)

    center = quantizer.dequantize(bucket)
    print(f"[ragged-edges] bucket -> center={center.tolist()}")
    np.testing.assert_allclose(
        center,
        np.array([0.0125, -0.0375, 0.025, 0.075, 0.75]),
    )
