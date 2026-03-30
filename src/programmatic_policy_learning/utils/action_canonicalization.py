"""Helpers for canonicalizing continuous actions to task-salient dimensions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def get_active_action_dims(
    sampling_cfg: dict[str, Any] | None,
    *,
    total_dims: int,
    default_active_dims: Sequence[int] | None = None,
) -> np.ndarray:
    """Resolve task-salient action dimensions from config."""
    cont_cfg = dict((sampling_cfg or {}).get("continuous", {}))
    raw_dims = cont_cfg.get("active_action_dims", default_active_dims)
    if raw_dims is None:
        return np.arange(total_dims, dtype=int)

    dims = np.asarray(raw_dims, dtype=int).reshape(-1)
    if dims.size == 0:
        raise ValueError("active_action_dims cannot be empty.")
    if np.any(dims < 0) or np.any(dims >= total_dims):
        raise ValueError(
            f"active_action_dims must lie in [0, {total_dims - 1}], got {dims.tolist()}."
        )
    unique_dims = np.unique(dims)
    if unique_dims.size != dims.size:
        raise ValueError(f"active_action_dims must be unique, got {dims.tolist()}.")
    return unique_dims.astype(int, copy=False)


def get_inactive_action_fill_value(sampling_cfg: dict[str, Any] | None) -> float:
    """Return the fill value used for inactive continuous action dims."""
    cont_cfg = dict((sampling_cfg or {}).get("continuous", {}))
    return float(cont_cfg.get("inactive_action_fill_value", 0.0))


def canonicalize_continuous_action(
    action: Sequence[float] | np.ndarray,
    *,
    active_dims: Sequence[int] | np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Zero/fill non-salient dimensions while preserving salient ones."""
    arr = np.asarray(action, dtype=float).reshape(-1).copy()
    active = np.asarray(active_dims, dtype=int).reshape(-1)
    inactive_mask = np.ones(arr.shape[0], dtype=bool)
    inactive_mask[active] = False
    arr[inactive_mask] = float(fill_value)
    return arr


def active_action_bounds(
    action_low: Sequence[float] | np.ndarray,
    action_high: Sequence[float] | np.ndarray,
    *,
    active_dims: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project full action bounds to salient dimensions."""
    low = np.asarray(action_low, dtype=float).reshape(-1)
    high = np.asarray(action_high, dtype=float).reshape(-1)
    active = np.asarray(active_dims, dtype=int).reshape(-1)
    return low[active], high[active]


def embed_active_action(
    active_values: Sequence[float] | np.ndarray,
    *,
    template: Sequence[float] | np.ndarray,
    active_dims: Sequence[int] | np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Embed salient-dimension values back into a full action vector."""
    full = canonicalize_continuous_action(
        template, active_dims=active_dims, fill_value=fill_value
    )
    active = np.asarray(active_dims, dtype=int).reshape(-1)
    values = np.asarray(active_values, dtype=float).reshape(-1)
    if values.size != active.size:
        raise ValueError(
            f"active_values must have length {active.size}, got {values.size}."
        )
    full[active] = values
    return full
