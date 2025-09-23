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


def shifted(direction, local_program, cell, obs):
    """Apply program to cell shifted by direction vector."""
    if cell is None:
        new_cell = None
    else:
        new_cell = (cell[0] + direction[0], cell[1] + direction[1])
    return local_program(new_cell, obs)


def at_cell_with_value(value, local_program, obs):
    """Apply program to first cell containing the specified value."""
    matches = np.argwhere(obs == value)
    if len(matches) == 0:
        cell = None
    else:
        cell = matches[0]
    return local_program(cell, obs)


def at_action_cell(local_program, cell, obs):
    """Apply program to the action cell (pass-through wrapper)."""
    return local_program(cell, obs)


def scanning(direction, true_condition, false_condition, cell, obs, max_timeout=50):
    """Scan in direction until true/false condition met or bounds reached."""
    if cell is None:
        return False

    for _ in range(max_timeout):
        cell = (cell[0] + direction[0], cell[1] + direction[1])

        if true_condition(cell, obs):
            return True

        if false_condition(cell, obs):
            return False

        # prevent infinite loops
        if out_of_bounds(cell[0], cell[1], obs.shape):
            return False

    return False
