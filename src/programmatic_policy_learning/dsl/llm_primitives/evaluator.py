"""Partial Evaluator for DSLs."""

import random
from typing import Any, Callable

import numpy as np

from programmatic_policy_learning.approaches.random_actions import RandomActionsApproach
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    get_core_boolean_primitives,
)

# ---------------------------------------------------------
# RANDOM SAMPLERS FOR ARGUMENT TYPES
# ---------------------------------------------------------

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]


def normalize_object_types(object_types: tuple[str, ...]) -> tuple[str, ...]:
    """
    Convert object types like "ec.WALL" or "tpn.EMPTY" or "None"
    into normalized lowercase names: "wall", "empty", "none".
    """
    normalized = []

    for obj in object_types:
        if obj == "None":
            normalized.append("None")
            continue

        # Split prefix (e.g., "ec.") and take the last component
        # e.g., "ec.AGENT"  →  "AGENT"
        if "." in obj:
            name = obj.split(".")[-1]
        else:
            name = obj

        normalized.append(name.lower())

    return tuple(normalized)


def sample_random_state_action(
    env: Any, max_steps: int = 20, seed: int = 123
) -> tuple[np.ndarray, Any]:
    """Roll out the environment using RandomActionsApproach to generate a
    diverse state."""
    # 1) Initialize approach
    approach: RandomActionsApproach[np.ndarray, np.ndarray] = RandomActionsApproach(
        "N/A",
        env.observation_space,
        env.action_space,
        seed=seed,
    )

    # 2) Reset env + approach
    obs, info = env.reset()
    obs = np.asarray(obs)
    approach.reset(obs, info)

    # 3) Pick random rollout depth
    k = random.randint(0, max_steps)

    for _ in range(k):
        # structured random action
        action = approach.step()

        # env step
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception:  # pylint: disable=broad-exception-caught
            break

        obs = np.asarray(obs)

        # update approach (not strictly needed but cleaner)
        approach.update(obs, float(reward), terminated, info)

        if terminated or truncated:
            break
    if k == 0:
        action = approach.step()
    return obs, action


def sample_direction() -> tuple[int, int]:
    """Randomly sample a direction from predefined directions."""
    return random.choice(DIRECTIONS)


def sample_local_program(existing_programs: list[Callable]) -> Callable:
    """Choose any primitive that has signature (cell, obs) → Bool."""
    return random.choice(existing_programs)


def sample_condition(existing_programs: list[Callable]) -> Callable:
    """Sample a condition from existing programs."""
    return sample_local_program(existing_programs)  # simplest version


def sample_value_from_env(object_types: tuple[str, ...]) -> Any:
    """Sample a value from the environment's object_types list.

    This ensures correctness across Chase, Nim, RFTS, STF, PRBench...
    """
    return random.choice(object_types)


def sample_argument(
    name: str,
    obs: np.ndarray,
    cell: tuple[int, int],
    existing_programs: list[Callable],
    object_types: tuple[str, ...],
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
# check if the coverage is enough -
# at least 80% valid and also invalid becuase they are meaningful
def eval_on_random_inputs(
    fn: Callable,
    object_types: tuple[str, ...],
    env_factory: Callable[[], Any],
    num_samples: int = 200,
) -> list[Any]:
    """Evaluate a function on random inputs."""
    outputs = []

    for i in range(num_samples):
        # inspect fn signature
        params = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
        env = env_factory(int(i % 20))  # type: ignore[call-arg]
        obs, cell = sample_random_state_action(env)
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
        except Exception:  # pylint: disable=broad-exception-caught
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
    object_types: tuple[Any],
    env_factory: Callable[[], Any],
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
    print(len(outputs_new))
    input("DONE")
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
