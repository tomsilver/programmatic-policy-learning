"""Core DSL primitives for grid-based environment interaction."""


def out_of_bounds(r, c, shape):
    """Check if coordinates are outside the grid bounds."""
    return r < 0 or c < 0 or r >= shape[0] or c >= shape[1]


def cell_is_value(value, cell, obs):
    """Check if a cell contains a specific value."""
    if cell is None or value is None or out_of_bounds(cell[0], cell[1], obs.shape):
        return False
    return obs[cell[0], cell[1]] == value
