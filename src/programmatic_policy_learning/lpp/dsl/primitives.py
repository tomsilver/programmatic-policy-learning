"""Core DSL primitives for grid-based environment interaction."""

from typing import Any

import numpy as np


def out_of_bounds(r: int, c: int, shape: tuple[int, ...]) -> bool:
    """Check if coordinates are outside the grid bounds."""
    return r < 0 or c < 0 or r >= shape[0] or c >= shape[1]


def cell_is_value(value: Any, cell: tuple[int, int] | None, obs: np.ndarray) -> bool:
    """Check if a cell contains a specific value."""
    if cell is None or value is None or out_of_bounds(cell[0], cell[1], obs.shape):
        return False
    return obs[cell[0], cell[1]] == value
