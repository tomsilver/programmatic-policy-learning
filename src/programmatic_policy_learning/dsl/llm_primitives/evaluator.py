"""Partial Evaluator for DSLs.

This module provides utilities for evaluating DSL primitives, including
random sampling, semantic similarity filtering, and degeneracy testing.
It also includes helper functions for working with DSLs and
environments.
"""

import inspect
import logging
import math
import random
from collections import Counter
from functools import partial
from typing import Any, Callable

import numpy as np

from programmatic_policy_learning.approaches.random_actions import RandomActionsApproach


# pylint: disable=cell-var-from-loop
# ---------------------------------------------------------
# RANDOM SAMPLERS FOR ARGUMENT TYPES
# ---------------------------------------------------------
def seed_all(seed: int) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)


DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]


def normalize_object_types(object_types: tuple[str, ...]) -> tuple[str, ...]:
    """
    Convert object types like "ec.WALL" or "tpn.EMPTY" or "None"
    into normalized lowercase names: "wall", "empty", "none".
    """
    normalized: list[str] = []

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


ARG_TYPE_MAP = {
    "cell": "CELL",
    "obs": "OBS",
    "direction": "DIRECTION",
    "value": "VALUE",
    "value_int": "Int",
    "int": "Int",
    "local_program": "LOCAL_PROGRAM",
    "condition": "CONDITION",
    "true_condition": "CONDITION",
    "false_condition": "CONDITION",
    "max_timeout": "Int",
}

IMPLICIT_ARGS = {"cell", "obs"}


def extract_signature_from_python_fn(fn: Callable) -> tuple[tuple[str, str], ...]:
    """Extract the DSL signature from a Python function."""
    sig = inspect.signature(fn)
    s: list[tuple[str, str]] = []
    for name, _ in sig.parameters.items():
        if name in IMPLICIT_ARGS:
            continue  # skip implicit runtime args
        inferred_type = ARG_TYPE_MAP.get(name, "UNKNOWN")
        s.append((name, inferred_type))
    return tuple(s)


def signature_matches(
    sig_new: tuple[tuple[str, str], ...], sig_existing: tuple[tuple[str, str], ...]
) -> bool:
    """Check if two DSL signatures are equivalent."""
    if len(sig_new) != len(sig_existing):
        return False
    return all(t1 == t2 for (_, t1), (_, t2) in zip(sig_new, sig_existing))


def compare_semantics(
    fn_new: Callable,
    fn_existing: Callable,
    signature: tuple[tuple[str, str], ...],
    existing_primitives: dict[str, Callable],
    normalized_object_types: tuple[str, ...],
    env_factory: Callable[[int], Any],
    seed: int,
    max_steps: int,
    num_samples: int = 200,
) -> float:
    """Compare the semantics of two functions based on their outputs."""
    matches = 0
    total = 0

    for i in range(num_samples):
        env = env_factory(i % 20)
        obs, cell = sample_random_state_action(env, seed, max_steps)

        # Build shared arguments
        args: dict[str, Any] = {}

        for arg_name, type_name in signature:

            # implicit runtime args
            if arg_name == "cell":
                args[arg_name] = cell
                continue
            if arg_name == "obs":
                args[arg_name] = obs
                continue

            # explicit typed args
            args[arg_name] = sample_argument(
                type_name=type_name,
                normalized_object_types=normalized_object_types,
                existing_primitives=existing_primitives,
            )

        try:
            out_new = fn_new(**args)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.info(f"Exception occurred: {e}")
            out_new = None

        try:
            out_existing = fn_existing(**args)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.info(f"Exception occurred: {e}")
            out_existing = None

        total += 1
        if out_new == out_existing:
            matches += 1

    return matches / max(total, 1)


def semantic_similarity_filter(
    new_primitive_fn: Callable,
    proposal_signature: tuple[tuple[str, str], ...],
    existing_primitives: dict[str, Callable],
    normalized_object_types: tuple[str, ...],
    env_factory: Callable[[int], Any],
    seed: int,
    max_steps: int,
    num_samples: int,
    threshold: float,
) -> dict[str, Any]:
    """Filter primitives based on semantic similarity."""
    similar: list[tuple[str, float]] = []

    for name, fn_existing in existing_primitives.items():
        sig_existing = extract_signature_from_python_fn(fn_existing)
        logging.info(f"CHECKING FOR: {name}")
        if signature_matches(proposal_signature, sig_existing):
            logging.info("Equal signature detected")
            score = compare_semantics(
                new_primitive_fn,
                fn_existing,
                proposal_signature,
                existing_primitives,
                normalized_object_types,
                env_factory,
                seed,
                max_steps,
                num_samples,
            )
            logging.info(f"SCORE: {score}")
            similar.append((name, score))

    # If ANY existing primitive is too similar → reject
    for name, score in similar:
        if score >= threshold:
            return {
                "keep": False,
                "reason": f"Semantic overlap with {name}, score={score:.3f}",
                "score": similar,
            }

    return {
        "keep": True,
        "score": similar,
    }


def sample_random_state_action(
    env: Any,
    seed: int,
    max_steps: int,
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


def sample_value_from_env(normalized_object_types: tuple[str, ...]) -> Any:
    """Sample a value from the environment's object_types list.

    This ensures correctness across Chase, Nim, RFTS, STF, PRBench...
    """
    return random.choice(normalized_object_types)


def sample_argument(
    type_name: str,
    normalized_object_types: tuple[str, ...],
    existing_primitives: dict[str, Callable],
) -> Any:
    """Sample an argument based on its type."""

    if type_name == "DIRECTION":
        return sample_direction()

    if type_name == "VALUE":
        return sample_value_from_env(normalized_object_types)

    if type_name in {"VALUE_INT", "Int"}:
        return random.randint(0, 3)

    if type_name in ("LOCAL_PROGRAM", "CONDITION"):
        return random.choice(
            make_closed_program_pool(normalized_object_types, existing_primitives)
        )

    # fallback
    raise ValueError(f"Unknown DSL type: {type_name}")


# ---------------------------------------------------------
# EVALUATE SINGLE FUNCTION ON SAMPLE
# ---------------------------------------------------------
def eval_on_random_inputs(
    fn: Callable,
    normalized_object_types: tuple[str, ...],
    env_factory: Callable[[int], Any],
    proposal_signature: tuple[tuple[str, str], ...],
    existing_primitives: dict[str, Callable],
    seed: int,
    max_steps: int,
    num_samples: int = 200,
) -> list[Any]:
    """Evaluate a function on random inputs."""

    sig_dict = dict(proposal_signature)
    print(sig_dict)
    params = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])

    outputs = []

    for i in range(num_samples):

        env = env_factory(int(i % 20))  # type: ignore[call-arg]
        obs, cell = sample_random_state_action(env, seed, max_steps)

        args = {}
        for p in params:

            # implicit runtime arguments
            if p == "cell":
                args[p] = cell
                continue
            if p == "obs":
                args[p] = obs
                continue

            # remaining args must be typed by DSL
            if p not in sig_dict:
                raise ValueError(f"Argument '{p}' missing from DSL signature")

            type_name = sig_dict[p]
            args[p] = sample_argument(
                type_name=type_name,
                normalized_object_types=normalized_object_types,
                existing_primitives=existing_primitives,
            )

        try:
            out = fn(**args)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.info(f"Exception occurred: {e}")
            out = None  # treat crashes as non-usable

        outputs.append(out)

    return outputs


# ---------------------------------------------------------
# LEVEL 1: DEGENERACY TEST
# ---------------------------------------------------------


def degeneracy_score(outputs: list[Any]) -> float:
    """Return a diversity score in [0,1].

    0 = completely constant (degenerate) 1 = maximally diverse across
    {True, False, None}
    """
    counter = Counter(outputs)
    probs = [c / len(outputs) for c in counter.values()]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(counter))
    return entropy / max_entropy


# ---------------------------------------------------------
# LEVEL 2: EQUIVALENCE CHECK
# ---------------------------------------------------------


def equivalence_score(outputs_new: list[Any], outputs_old: list[Any]) -> float:
    """Compute the agreement fraction between two output lists."""
    agree = sum(a == b for a, b in zip(outputs_new, outputs_old))
    return agree / len(outputs_new)


def make_closed_program_pool(
    normalized_object_types: tuple[str, ...],
    existing_primitives: dict[str, Callable],
) -> list[Callable]:
    """Generate a pool of closed programs using existing primitives."""

    # --------------------------------------------------------
    # First create 0-depth closed programs (simple base)
    # --------------------------------------------------------
    base_programs = [
        lambda cell, obs: True,
        lambda cell, obs: False,
    ]

    pool = base_programs.copy()

    # Need VALUE sampling helper
    norm_values = normalized_object_types

    def sample_for_type(type_name: str, pool_snapshot: list[Callable]) -> Any:
        """Sample a closed argument based on DSL TYPE."""
        if type_name == "DIRECTION":
            return random.choice(DIRECTIONS)

        if type_name == "VALUE":
            return random.choice(norm_values)

        if type_name in {"VALUE_INT", "Int"}:
            return random.randint(1, 3)

        if type_name in ("LOCAL_PROGRAM", "CONDITION"):
            #  pick from pool_snapshot (not existing_primitives!)
            return random.choice(pool_snapshot)

        raise ValueError(f"Unknown argument type: {type_name}")

    # --------------------------------------------------------
    # Build deeper programs by INSTANTIATING each primitive
    # --------------------------------------------------------
    for _, fn in existing_primitives.items():
        sig = inspect.signature(fn)

        # Extract the DSL-type arguments (skip implicit cell,obs)
        arg_order = []
        arg_types = []
        for arg_name, _ in sig.parameters.items():
            if arg_name in ("cell", "obs"):
                continue
            arg_type = ARG_TYPE_MAP.get(arg_name, None)
            if arg_type is None:
                # unknown/unsupported → skip this primitive safely
                break
            arg_order.append(arg_name)
            arg_types.append(arg_type)

        else:
            # Only executed if loop didn't break → we have a supported primitive
            def make_wrapper(fn: Callable, arg_types: list[str]) -> Callable:
                """Capture fully-instantiated primitive with closed args."""

                # Capture args NOW, not at call-time
                pool_snapshot = pool.copy()
                sampled_args = [
                    sample_for_type(typ, pool_snapshot) for typ in arg_types
                ]
                arg_order_copy = arg_order.copy()  # Avoid cell variable issue
                arg_dict = dict(zip(arg_order_copy, sampled_args))

                partial_fn = partial(fn, **arg_dict)

                def wrapped(
                    cell: tuple[int, int] | None,
                    obs: np.ndarray,
                    fn: Callable[..., Any] = partial_fn,
                ) -> Any:
                    """Wrap around (cell, obs)."""
                    return fn(cell=cell, obs=obs)

                return wrapped

            # Add many instantiations (loop a few times for diversity)
            for _ in range(20):  # hyperparameter: #instantiations per primitive
                pool.append(make_wrapper(fn, arg_types))

    return pool


# ---------------------------------------------------------
# MAIN API CALLED BY LLM GENERATOR
# ---------------------------------------------------------


def evaluate_primitive(
    new_primitive_fn: Callable,
    existing_primitives: dict[str, Callable],
    object_types: tuple[Any, ...],
    env_factory: Callable[[int], Any],
    proposal_signature: tuple[tuple[str, str], ...],
    seed: int = 123,
    max_steps: int = 20,
    num_samples: int = 200,
    degeneracy_threshold: float = 0.1,
    equivalence_threshold: float = 0.9,
) -> dict[str, Any]:
    """Evaluate a new primitive for degeneracy and redundancy."""
    seed_all(seed)
    normalized_object_types = normalize_object_types(object_types)
    # -----------------------------------------------------
    # Step 1: Compute outputs of new primitive
    # -----------------------------------------------------
    outputs_new = eval_on_random_inputs(
        new_primitive_fn,
        normalized_object_types,
        env_factory,
        proposal_signature,
        existing_primitives,
        seed,
        max_steps,
        num_samples,
    )
    print("OUT", outputs_new)
    # -----------------------------------------------------
    # Level 1: Degenerate check
    # -----------------------------------------------------
    deg = degeneracy_score(outputs_new)
    if deg <= degeneracy_threshold:
        return {"keep": False, "reason": "degenerate", "score": deg}
    # -----------------------------------------------------
    # Level 2: Semantic similarity
    # -----------------------------------------------------

    result = semantic_similarity_filter(
        new_primitive_fn,
        proposal_signature,
        existing_primitives,
        normalized_object_types,
        env_factory,
        seed,
        max_steps,
        num_samples,
        equivalence_threshold,
    )
    logging.info(result)
    return result
