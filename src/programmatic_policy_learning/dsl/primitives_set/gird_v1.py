"""Grid DSL primitives and evaluation."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from programmatic_policy_learning.dsl.core import DSL  # generic core

Cell = tuple[int, int] | None
Local = Callable[[Cell, np.ndarray], Any]  # LocalProgram - (cell, obs) -> *


@dataclass(frozen=True)
class GridInput:
    """Input structure for grid-based DSL programs.

    Attributes:
        obs: The grid observation (numpy array).
        cell: The cell position (tuple of ints or None).
    """

    obs: np.ndarray
    cell: Cell = None


def out_of_bounds(r: int, c: int, shape: tuple[int, ...]) -> bool:
    """Check if coordinates are outside the grid bounds."""
    if len(shape) < 2:
        return True  # Invalid shape
    return r < 0 or c < 0 or r >= shape[0] or c >= shape[1]


def shifted(
    direction: tuple[int, int],
    local_program: Callable,
    cell: tuple[int, int] | None,
    obs: np.ndarray,
) -> bool:
    """Execute a local program at a shifted cell position."""
    if cell is None:
        new_cell = None
    else:
        new_cell = (cell[0] + direction[0], cell[1] + direction[1])
    return local_program(new_cell, obs)


def cell_is_value(value: Any, cell: tuple[int, int] | None, obs: np.ndarray) -> bool:
    """Check if a cell contains a specific value."""
    if cell is None or out_of_bounds(cell[0], cell[1], obs.shape):
        focus = None
    else:
        focus = obs[cell[0], cell[1]]

    return focus == value


def at_cell_with_value(value: Any, local_program: Callable, obs: np.ndarray) -> bool:
    """Execute a local program at the first cell containing a specific
    value."""
    matches = np.argwhere(obs == value)
    if len(matches) == 0:
        cell = None
    else:
        cell = tuple(matches[0])  # Convert to tuple for consistency
    return local_program(cell, obs)


def at_action_cell(
    local_program: Callable, cell: tuple[int, int] | None, obs: np.ndarray
) -> bool:
    """Execute a local program at the action cell position."""
    return local_program(cell, obs)


def scanning(
    direction: tuple[int, int],
    true_condition: Callable,
    false_condition: Callable,
    cell: tuple[int, int] | None,
    obs: np.ndarray,
    max_timeout: int = 50,
) -> bool:
    """Scan in a direction until a condition is met."""
    if cell is None:
        return False

    current_cell = cell
    for _ in range(max_timeout):
        current_cell = (current_cell[0] + direction[0], current_cell[1] + direction[1])

        if true_condition(current_cell, obs):
            return True

        if false_condition(current_cell, obs):
            return False

        # Prevent infinite loops
        if out_of_bounds(current_cell[0], current_cell[1], obs.shape):
            return False

    return False


# Domain-specific evaluator: program is a Local; inputs are GridInput.
def _eval(program: Local, inputs: GridInput) -> Any:
    """Evaluate a grid program on given inputs."""
    return program(inputs.cell, inputs.obs)


def make_dsl() -> DSL[Local, GridInput, Any]:
    """Construct the grid DSL object."""
    prims: Mapping[str, Callable[..., Any]] = {
        "cell_is_value": cell_is_value,
        "shifted": shifted,
        "at_cell_with_value": at_cell_with_value,
        "at_action_cell": at_action_cell,
        "scanning": scanning,
    }
    return DSL(id="grid_v1", primitives=prims, evaluate_fn=_eval)
