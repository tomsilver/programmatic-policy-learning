"""Shared grid validation helpers."""

from typing import Any

import numpy as np


def require_grid_state_action(
    s: Any, a: Any, *, context: str
) -> tuple[np.ndarray, tuple[int, int]]:
    """Validate and normalize a grid-style (state, action) example."""
    if not isinstance(s, np.ndarray) or s.ndim < 2:
        raise TypeError(f"{context}: expected state as 2D+ np.ndarray.")
    if not isinstance(a, tuple) or len(a) != 2:
        raise TypeError(f"{context}: expected action as (row, col) tuple.")
    try:
        r = int(a[0])
        c = int(a[1])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise TypeError(f"{context}: action coordinates must be int-castable.") from exc
    return s, (r, c)
