"""Partial Evaluator for DSLs."""
import random
from typing import Any, Callable

import numpy as np

from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    get_core_boolean_primitives,
)

# ---------------------------------------------------------
# RANDOM SAMPLERS FOR ARGUMENT TYPES
# ---------------------------------------------------------

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]


def sample_direction() -> tuple[int, int]:
    """Randomly sample a direction from predefined directions."""
    return random.choice(DIRECTIONS)


def sample_value() -> Any:
    """Randomly sample a value from predefined values."""
    return random.choice(VALUES)


def sample_local_program(existing_programs: list[Callable]) -> Callable:
    """Choose any primitive that has signature (cell, obs) → Bool."""
    return random.choice(existing_programs)


def sample_condition(existing_programs: list[Callable]) -> Callable:
    """Sample a condition from existing programs."""
    return sample_local_program(existing_programs)  # simplest version


def sample_value_from_env(object_types: list[Any]) -> Any:
    """Sample a value from the environment's object_types list.

    This ensures correctness across Chase, Nim, RFTS, STF, PRBench...
    """
    return random.choice(object_types)  # TODO: make it more general


def sample_obs_and_cell(env: Any) -> tuple[np.ndarray, tuple[int, int]]:
    """Sample a real observation and a valid action-cell from the environment.

    Returns:
        obs: np.ndarray     (H×W grid or possibly multi-channel)
        cell: (row, col)    random valid coordinate
    """
    obs, _ = env.reset()
    obs = np.asarray(obs)

    H, W = obs.shape[0], obs.shape[1]

    cell = (random.randint(0, H - 1), random.randint(0, W - 1))

    return obs, cell


def sample_argument(
    name: str,
    obs: np.ndarray,
    cell: tuple[int, int],
    existing_programs: list[Callable],
    object_types: list[Any],
) -> Any:
    """Sample an argument based on its name."""
    if name == "obs":
        return obs
    if name == "cell":
        return cell
    if name == "direction":
        return sample_direction()
    if name == "value":
        return sample_value_from_env(object_types)
    if name in (
        "local_program",
        "condition",
        "true_condition",
        "false_condition",
        "program",
    ):
        return random.choice(existing_programs)

    return random.choice([None, 0, 1])


# ---------------------------------------------------------
# EVALUATE SINGLE FUNCTION ON SAMPLE
# ---------------------------------------------------------
# CHECK IF OBS IS VALID AND THE CELL IS VALID
# check if the coverage is enough - at least 80% valid and also invalid becuase they are meaningful
def eval_on_random_inputs(
    fn: Callable,
    object_types: list[Any],
    env_factory: Callable[[int], Any],
    num_samples: int = 200,
) -> list[Any]:
    """Evaluate a function on random inputs."""
    outputs = []

    for i in range(num_samples):
        # inspect fn signature
        params = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
        env = env_factory(int(i / 20))
        obs, cell = sample_obs_and_cell(env)  # TODO: sample not just initial state
        args = {
            p: sample_argument(
                p,
                obs=obs,
                cell=cell,
                existing_programs=get_core_boolean_primitives(),
                object_types=object_types,
            )
            for p in params
        }
        # print(params)
        # print(args)
        # input()
        try:
            out = fn(**args)
        except Exception:
            out = None  # treat crashes as non-usable

        outputs.append(out)

    return outputs


# ---------------------------------------------------------
# LEVEL 1: DEGENERACY TEST
# ---------------------------------------------------------


def degeneracy_score(outputs: list[Any]) -> float:
    """Compute how constant the function is.

    1.0 = constant.
    """
    unique_vals = set(outputs)
    return 1.0 if len(unique_vals) <= 1 else 0.0


# ---------------------------------------------------------
# LEVEL 2: EQUIVALENCE CHECK
# ---------------------------------------------------------


def equivalence_score(outputs_new: list[Any], outputs_old: list[Any]) -> float:
    """Compute the agreement fraction between two output lists."""
    agree = sum(a == b for a, b in zip(outputs_new, outputs_old))
    return agree / len(outputs_new)


# ---------------------------------------------------------
# MAIN API CALLED BY LLM GENERATOR
# ---------------------------------------------------------


def evaluate_primitive(
    new_primitive_fn: Callable,
    existing_primitives: dict[str, Callable],
    object_types: list[Any],
    env_factory: Callable[[int], Any],
    num_samples: int = 200,
    degeneracy_threshold: float = 0.95,
    equivalence_threshold: float = 0.95,
) -> dict[str, Any]:
    """Evaluate a new primitive for degeneracy and redundancy."""
    # -----------------------------------------------------
    # Step 1: Compute outputs of new primitive
    # -----------------------------------------------------
    outputs_new = eval_on_random_inputs(
        new_primitive_fn, object_types, env_factory, num_samples
    )
    print("OUT", outputs_new)
    input()
    # -----------------------------------------------------
    # Level 1: Degenerate check
    # -----------------------------------------------------
    deg = degeneracy_score(outputs_new)
    if deg >= degeneracy_threshold:
        return {"keep": False, "reason": "degenerate", "degeneracy_score": deg}

    # -----------------------------------------------------
    # Level 2: Redundancy check
    # -----------------------------------------------------
    for name, old_fn in existing_primitives.items():
        outputs_old = eval_on_random_inputs(
            old_fn, object_types, env_factory, num_samples
        )
        eq = equivalence_score(outputs_new, outputs_old)
        if eq >= equivalence_threshold:
            return {
                "keep": False,
                "reason": f"redundant_wrt_{name}",
                "equivalence_score": eq,
            }

    return {"keep": True, "reason": "useful", "degeneracy_score": deg}
