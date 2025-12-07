"""Grid DSL primitives and evaluation."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping

import numpy as np
from generalization_grid_games.envs import chase as ec
from generalization_grid_games.envs import checkmate_tactic as ct
from generalization_grid_games.envs import reach_for_the_star as rfts
from generalization_grid_games.envs import stop_the_fall as stf
from generalization_grid_games.envs import two_pile_nim as tpn

from programmatic_policy_learning.dsl.core import DSL
from programmatic_policy_learning.dsl.generators.grammar_based_generator import Grammar

Cell = tuple[int, int] | None
LocalProgram = Callable[[Cell, np.ndarray], Any]  # LocalProgram - (cell, obs) -> *


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


def at_cell_with_value(
    value: Any, local_program: Callable, cell: tuple[int, int] | None, obs: np.ndarray
) -> bool:
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


# Domain-specific evaluator: program is a LocalProgram; inputs are GridInput.
def _eval(program: LocalProgram, inputs: GridInput) -> Any:
    """Evaluate a grid program on given inputs."""
    return program(inputs.cell, inputs.obs)


def make_dsl() -> DSL[LocalProgram, GridInput, Any]:
    """Construct the grid DSL object."""
    prims: Mapping[str, Callable[..., Any]] = {
        "cell_is_value": cell_is_value,
        "shifted": shifted,
        "at_cell_with_value": at_cell_with_value,
        "at_action_cell": at_action_cell,
        "scanning": scanning,
    }
    return DSL(id="grid_v1", primitives=prims, evaluate_fn=_eval)


def make_ablated_dsl(removed_primitive: str) -> DSL[LocalProgram, GridInput, Any]:
    """Construct the grid DSL object."""
    prims: MutableMapping[str, Callable[..., Any]] = {
        "cell_is_value": cell_is_value,
        "shifted": shifted,
        "at_cell_with_value": at_cell_with_value,
        "at_action_cell": at_action_cell,
        "scanning": scanning,
    }
    if removed_primitive is not None:
        del prims[removed_primitive]
    return DSL(
        id=f"grid_v1_no_{removed_primitive}", primitives=prims, evaluate_fn=_eval
    )


# Grammar symbol constants
START, CONDITION, LOCAL_PROGRAM, DIRECTION, POSITIVE_NUM, NEGATIVE_NUM, VALUE = range(7)


def create_grammar(
    env_spec: dict[str, Any], removed_primitive: str | None = None
) -> Grammar[str, int, int]:
    """Create a grammar for grid programs, using env_spec for VALUE types."""
    object_types = env_spec["object_types"]
    grammar: Grammar[str, int, int] = Grammar(
        rules={
            START: (
                [
                    ["at_cell_with_value(", VALUE, ",", LOCAL_PROGRAM, ", a, s)"],
                    ["at_action_cell(", LOCAL_PROGRAM, ", a, s)"],
                ],
                [0.5, 0.5],
            ),
            LOCAL_PROGRAM: (
                [
                    [CONDITION],
                    [
                        "lambda cell,o : shifted(",
                        DIRECTION,
                        ",",
                        CONDITION,
                        ", cell, o)",
                    ],
                ],
                [0.5, 0.5],
            ),
            CONDITION: (
                [
                    ["lambda cell,o : cell_is_value(", VALUE, ", cell, o)"],
                    [
                        "lambda cell,o : scanning(",
                        DIRECTION,
                        ",",
                        LOCAL_PROGRAM,
                        ",",
                        LOCAL_PROGRAM,
                        ", cell, o)",
                    ],
                ],
                [0.5, 0.5],
            ),
            DIRECTION: (
                [
                    ["(", POSITIVE_NUM, ", 0)"],
                    ["(0,", POSITIVE_NUM, ")"],
                    ["(", NEGATIVE_NUM, ", 0)"],
                    ["(0,", NEGATIVE_NUM, ")"],
                    ["(", POSITIVE_NUM, ",", POSITIVE_NUM, ")"],
                    ["(", NEGATIVE_NUM, ",", POSITIVE_NUM, ")"],
                    ["(", POSITIVE_NUM, ",", NEGATIVE_NUM, ")"],
                    ["(", NEGATIVE_NUM, ",", NEGATIVE_NUM, ")"],
                ],
                [1.0 / 8] * 8,
            ),
            POSITIVE_NUM: (
                [["1"], [POSITIVE_NUM, "+1"]],
                [0.99, 0.01],
            ),
            NEGATIVE_NUM: (
                [["-1"], [NEGATIVE_NUM, "-1"]],
                [0.99, 0.01],
            ),
            VALUE: (
                [[str(v)] for v in object_types],
                [1.0 / len(object_types) for _ in object_types],
            ),
        }
    )

    # === Remove specified primitive if requested ===
    if removed_primitive is not None:
        for nonterminal, (productions, probs) in list(grammar.rules.items()):
            new_productions, new_probs = [], []

            for prod, p in zip(productions, probs):
                # Convert production (which can be a list or str)
                # to string for substring search
                prod_str = (
                    " ".join(map(str, prod)) if isinstance(prod, list) else str(prod)
                )
                if removed_primitive not in prod_str:
                    new_productions.append(prod)
                    new_probs.append(p)

            # Renormalize probabilities if something was removed
            if len(new_probs) > 0 and len(new_probs) < len(probs):
                total = sum(new_probs)
                new_probs = [p / total for p in new_probs]
                grammar.rules[nonterminal] = (new_productions, new_probs)

        logging.info(
            f"Removed primitive '{removed_primitive}' and renormalized probabilities."
        )

    return grammar


def get_dsl_functions_dict(removed_primitive: str | None = None) -> dict[str, Any]:
    """Return all grid_v1 DSL primitives as a dictionary.

    If there is a removed_primitive, remove it and then return it.
    """

    DSL_FUNCTIONS = {
        "cell_is_value": cell_is_value,
        "shifted": shifted,
        "at_cell_with_value": at_cell_with_value,
        "at_action_cell": at_action_cell,
        "scanning": scanning,
        "out_of_bounds": out_of_bounds,
        "START": START,
        "CONDITION": CONDITION,
        "LOCAL_PROGRAM": LOCAL_PROGRAM,
        "DIRECTION": DIRECTION,
        "POSITIVE_NUM": POSITIVE_NUM,
        "NEGATIVE_NUM": NEGATIVE_NUM,
        "VALUE": VALUE,
        "ec": ec,
        "ct": ct,
        "rfts": rfts,
        "stf": stf,
        "tpn": tpn,
    }
    if removed_primitive:
        del DSL_FUNCTIONS[removed_primitive]
    return DSL_FUNCTIONS


def get_core_boolean_primitives(
    removed_primitive: str | None = None,
) -> dict[str, Callable]:
    """Return a dictionary of core boolean primitives with their names as
    keys."""
    list_of_fns = {
        "cell_is_value": cell_is_value,
        "shifted": shifted,
        "at_cell_with_value": at_cell_with_value,
        "at_action_cell": at_action_cell,
        "scanning": scanning,
    }

    if removed_primitive:
        list_of_fns = {
            name: fn for name, fn in list_of_fns.items() if name != removed_primitive
        }

    return list_of_fns  # type: ignore


__all__ = [
    "cell_is_value",
    "shifted",
    "at_cell_with_value",
    "at_action_cell",
    "scanning",
    "out_of_bounds",
    "GridInput",
    "make_dsl",
    "create_grammar",
    "START",
    "CONDITION",
    "LOCAL_PROGRAM",
    "DIRECTION",
    "POSITIVE_NUM",
    "NEGATIVE_NUM",
    "VALUE",
]
