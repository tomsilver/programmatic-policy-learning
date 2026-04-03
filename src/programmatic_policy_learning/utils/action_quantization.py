"""Utilities for quantizing continuous Motion2D actions into buckets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np


def _as_1d_float_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    return arr


def _normalize_bucket_counts(
    bucket_counts: int | Sequence[int] | None, *, dims: int
) -> np.ndarray:
    if bucket_counts is None:
        bucket_counts = 5
    if isinstance(bucket_counts, int):
        counts = np.full((dims,), int(bucket_counts), dtype=int)
    else:
        counts = np.asarray(bucket_counts, dtype=int).reshape(-1)
        if counts.size != dims:
            raise ValueError(
                "bucket_counts length must match number of action dimensions: "
                f"got {counts.size}, expected {dims}."
            )

    if np.any(counts < 3):
        raise ValueError("Each bucket count must be >= 3 to include a zero bucket.")
    if np.any(counts % 2 == 0):
        raise ValueError("Each bucket count must be odd so zero can be its own bucket.")
    return counts


def _normalize_bucket_edges(
    bucket_edges: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    dims: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Normalize explicit bucket edges per dimension."""
    arr = np.asarray(bucket_edges, dtype=float)
    if arr.ndim == 1:
        edges_per_dim = tuple(arr.copy() for _ in range(dims))
    elif arr.ndim == 2:
        if arr.shape[0] != dims:
            raise ValueError(
                "bucket_edges row count must match number of action dimensions: "
                f"got {arr.shape[0]}, expected {dims}."
            )
        edges_per_dim = tuple(arr[d].copy() for d in range(dims))
    else:
        raise ValueError("bucket_edges must be a 1-D or 2-D array-like value.")

    normalized: list[np.ndarray] = []
    for d, edges in enumerate(edges_per_dim):
        if edges.size < 2:
            raise ValueError("Each bucket_edges entry must contain at least 2 values.")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("bucket_edges must be strictly increasing per dimension.")
        if not np.isclose(edges[0], action_low[d]):
            raise ValueError(
                f"bucket_edges for dim {d} must start at "
                f"action_low={action_low[d]:.6f}, "
                f"got {edges[0]:.6f}."
            )
        if not np.isclose(edges[-1], action_high[d]):
            raise ValueError(
                f"bucket_edges for dim {d} must end at "
                f"action_high={action_high[d]:.6f}, "
                f"got {edges[-1]:.6f}."
            )
        normalized.append(edges.astype(float, copy=False))
    return tuple(normalized)


@dataclass(frozen=True)
class Motion2DActionQuantizer:
    """Quantize/dequantize continuous actions into per-dimension bins."""

    action_low: np.ndarray
    action_high: np.ndarray
    bucket_counts: np.ndarray
    bucket_edges: tuple[np.ndarray, ...] | None = None

    def _log_creation(self) -> None:
        """Print/log quantizer configuration for debugging."""
        if self.bucket_edges is not None:
            edges_repr = [edges.tolist() for edges in self.bucket_edges]
            msg = (
                "Created Motion2DActionQuantizer with explicit bucket_edges="
                f"{edges_repr} action_low={self.action_low.tolist()} "
                f"action_high={self.action_high.tolist()}"
            )
        else:
            msg = (
                "Created Motion2DActionQuantizer with bucket_counts="
                f"{self.bucket_counts.tolist()} action_low={self.action_low.tolist()} "
                f"action_high={self.action_high.tolist()}"
            )
        logging.info(msg)

    @classmethod
    def from_bounds(
        cls,
        action_low: Sequence[float] | np.ndarray,
        action_high: Sequence[float] | np.ndarray,
        *,
        bucket_counts: int | Sequence[int] | None = 5,
        bucket_edges: (
            Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None
        ) = None,
    ) -> "Motion2DActionQuantizer":
        """Create a quantizer from per-dimension action bounds."""
        low = _as_1d_float_array(action_low, "action_low")
        high = _as_1d_float_array(action_high, "action_high")
        if low.shape != high.shape:
            raise ValueError(
                "action_low and action_high must have the same shape: "
                f"got {low.shape} vs {high.shape}."
            )
        if np.any(low >= high):
            raise ValueError("Each action_low must be strictly less than action_high.")
        # if np.any(low >= 0.0) or np.any(high <= 0.0):
        #     raise ValueError(
        #         "Each dimension must straddle zero to reserve a dedicated zero bucket."
        #     )

        if bucket_edges is not None:
            edges = _normalize_bucket_edges(
                bucket_edges,
                dims=low.size,
                action_low=low,
                action_high=high,
            )
            counts = np.asarray([len(e) - 1 for e in edges], dtype=int)
            quantizer = cls(
                action_low=low,
                action_high=high,
                bucket_counts=counts,
                bucket_edges=edges,
            )
            # quantizer._log_creation()
            return quantizer

        counts = _normalize_bucket_counts(bucket_counts, dims=low.size)
        quantizer = cls(action_low=low, action_high=high, bucket_counts=counts)
        # quantizer._log_creation()
        return quantizer

    @property
    def dims(self) -> int:
        """Return action dimensionality."""
        return int(self.action_low.size)

    @property
    def zero_bucket_index_per_dim(self) -> np.ndarray:
        """Return zero-bucket indices."""
        if self.bucket_edges is not None:
            zero_bins: list[int] = []
            for edges in self.bucket_edges:
                zero_idx = int(np.searchsorted(edges, 0.0, side="right") - 1)
                zero_idx = max(0, min(zero_idx, len(edges) - 2))
                zero_bins.append(zero_idx)
            return np.asarray(zero_bins, dtype=int)
        return self.bucket_counts // 2

    def _clipped_action(self, action: Sequence[float] | np.ndarray) -> np.ndarray:
        """Clip an action to bounds."""
        arr = _as_1d_float_array(action, "action")
        if arr.size != self.dims:
            raise ValueError(
                f"Action has wrong dimension: got {arr.size}, expected {self.dims}."
            )
        return np.clip(arr, self.action_low, self.action_high)

    def quantize(self, action: Sequence[float] | np.ndarray) -> tuple[int, ...]:
        """Map an action to bucket indices."""
        arr = self._clipped_action(action)
        indices: list[int] = []

        for d in range(self.dims):
            value = float(arr[d])
            if self.bucket_edges is not None:
                edges = self.bucket_edges[d]
                bucket = int(np.searchsorted(edges, value, side="right") - 1)
                bucket = max(0, min(bucket, len(edges) - 2))
                indices.append(bucket)
                continue
            low = float(self.action_low[d])
            high = float(self.action_high[d])
            count = int(self.bucket_counts[d])
            n_side = count // 2
            zero_idx = n_side

            if np.isclose(value, 0.0):
                indices.append(zero_idx)
                continue

            if value < 0.0:
                neg_edges = np.linspace(low, 0.0, n_side + 1)
                bucket = int(np.searchsorted(neg_edges, value, side="right") - 1)
                bucket = max(0, min(bucket, n_side - 1))
                indices.append(bucket)
                continue

            pos_edges = np.linspace(0.0, high, n_side + 1)
            bucket_pos = int(np.searchsorted(pos_edges, value, side="right") - 1)
            bucket_pos = max(0, min(bucket_pos, n_side - 1))
            indices.append(zero_idx + 1 + bucket_pos)

        return tuple(indices)

    def dequantize(self, bucket_index: Sequence[int]) -> np.ndarray:
        """Map bucket indices to centers."""
        bucket_arr = np.asarray(bucket_index, dtype=int).reshape(-1)
        if bucket_arr.size != self.dims:
            raise ValueError(
                "bucket_index length must match number of action dimensions: "
                f"got {bucket_arr.size}, expected {self.dims}."
            )

        centers = np.zeros(self.dims, dtype=float)
        for d in range(self.dims):
            idx = int(bucket_arr[d])
            count = int(self.bucket_counts[d])
            if self.bucket_edges is not None:
                if idx < 0 or idx >= count:
                    raise ValueError(
                        f"Bucket index out of range for dim {d}: "
                        f"{idx} not in [0, {count - 1}]"
                    )
                edges = self.bucket_edges[d]
                lo = float(edges[idx])
                hi = float(edges[idx + 1])
                centers[d] = 0.5 * (lo + hi)
                continue
            n_side = count // 2
            if idx < 0 or idx >= count:
                raise ValueError(
                    f"Bucket index out of range for dim {d}: "
                    f"{idx} not in [0, {count - 1}]"
                )

            if idx == n_side:
                centers[d] = 0.0
                continue

            if idx < n_side:
                edges = np.linspace(float(self.action_low[d]), 0.0, n_side + 1)
                lo = edges[idx]
                hi = edges[idx + 1]
                centers[d] = 0.5 * (lo + hi)
                continue

            pos_idx = idx - (n_side + 1)
            edges = np.linspace(0.0, float(self.action_high[d]), n_side + 1)
            lo = edges[pos_idx]
            hi = edges[pos_idx + 1]
            centers[d] = 0.5 * (lo + hi)

        return centers

    def all_bucket_indices(self) -> list[tuple[int, ...]]:
        """List all bucket index tuples."""
        return [
            tuple(idx)
            for idx in product(*(range(int(c)) for c in self.bucket_counts.tolist()))
        ]

    def all_bucket_centers(self) -> list[np.ndarray]:
        """List all bucket centers."""
        return [self.dequantize(idx) for idx in self.all_bucket_indices()]
