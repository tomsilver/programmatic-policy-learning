"""Dataset creation and processing utilities for programmatic policy
learning."""

import inspect
import logging
import multiprocessing
import os
from importlib import import_module
from typing import Any

import cloudpickle
import numpy as np
from scipy.sparse import lil_matrix, vstack

from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.dsl.state_action_program import (
    StateActionProgram,
    set_dsl_functions,
)


def allowed_cpus() -> int:
    """Determine the number of CPUs available for use."""
    # Use SLURM allocation if available
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return int(slurm_cpus)
    # Otherwise fall back to system count
    return multiprocessing.cpu_count()


import random
from collections import defaultdict
from typing import Any, List, Tuple

Coord = Tuple[int, int]


def sample_negative_actions_stratified(
    state: np.ndarray,
    expert_action: Coord,
    K: int = 30,
    rng: random.Random | None = None,
    include_nearby: int = 8,
) -> List[Coord]:
    """Stratified negative sampling over cell *tokens* + some nearby negatives.

    - Tries to include negatives from as many token-types as possible.
    - Adds a few "nearby" negatives around the expert action (harder negatives).
    - Falls back to uniform sampling if buckets are small.

    Returns: list of (r,c) coords (excluding expert_action).
    """
    if rng is None:
        rng = random.Random(0)

    h = state.shape[0]
    w = state.shape[1]

    # 1) bucket all coords (except expert) by token value at that cell
    buckets = defaultdict(list)  # token -> list[Coord]
    all_coords: List[Coord] = []

    er, ec = expert_action
    for r in range(h):
        for c in range(w):
            if (r, c) == (er, ec):
                continue
            tok = state[r, c]  # works for np arrays
            buckets[tok].append((r, c))
            all_coords.append((r, c))

    if not all_coords:
        return []

    # 2) Build a "nearby" pool (hard negatives)
    nearby: List[Coord] = []
    for dr in (-2, -1, 0, 1, 2):
        for dc in (-2, -1, 0, 1, 2):
            if dr == 0 and dc == 0:
                continue
            rr, cc = er + dr, ec + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) != (er, ec):
                nearby.append((rr, cc))
    rng.shuffle(nearby)

    picked: List[Coord] = []
    picked_set = set()

    def add_coord(rc: Coord) -> None:
        if rc not in picked_set and rc != (er, ec):
            picked.append(rc)
            picked_set.add(rc)

    # 3) Take some nearby first (if requested)
    for rc in nearby[:include_nearby]:
        add_coord(rc)
        if len(picked) >= K:
            return picked[:K]

    # 4) Round-robin across token buckets to cover "all available ones"
    #    (as many distinct tokens as possible)
    token_keys = list(buckets.keys())
    rng.shuffle(token_keys)

    # Prepare shuffled lists so we sample without replacement per bucket
    for t in token_keys:
        rng.shuffle(buckets[t])

    idx = 0
    while len(picked) < K and token_keys:
        t = token_keys[idx % len(token_keys)]
        if buckets[t]:
            add_coord(buckets[t].pop())
        else:
            # remove empty bucket
            token_keys.remove(t)
            continue
        idx += 1
        if idx > 10_000:  # safety
            break

    # 5) If still short, fill uniformly from anywhere
    if len(picked) < K:
        remaining = [rc for rc in all_coords if rc not in picked_set]
        rng.shuffle(remaining)
        for rc in remaining:
            add_coord(rc)
            if len(picked) >= K:
                break

    return picked[:K]


def extract_examples_from_demonstration_item(
    demonstration_item: tuple[np.ndarray, tuple[int, int]],
    *,
    data_imbalance: dict[str, Any] | None = None,
) -> tuple[
    list[tuple[np.ndarray, tuple[int, int]]],
    list[tuple[np.ndarray, tuple[int, int]]],
]:
    """Convert a demonstrated (state, action) into positive and negative
    classification data.

    Parameters
    ----------
    demonstrations : (np.ndarray, (int, int))
        A state, action pair.

    Returns
    -------
    positive_examples : [(np.ndarray, (int, int))]
        A list with just the input state, action pair (for convenience).
    negative_examples : [(np.ndarray, (int, int))]
        A list with negative examples of state, actions.
    """
    state, action = demonstration_item

    positive_examples: list[tuple[np.ndarray, tuple[int, int]]] = [(state, action)]
    negative_examples: list[tuple[np.ndarray, tuple[int, int]]] = []

    if data_imbalance and data_imbalance.get("enabled", False):
        method = data_imbalance.get("method", "")
        if method == "downsample_majority":
            k = int(data_imbalance.get("K", 10))
            if k < 0:
                raise ValueError("data_imbalance.K must be >= 0")
            rng = random.Random(0)  # or pass seed from config
            neg_coords = sample_negative_actions_stratified(
                state=state,
                expert_action=action,
                K=k,
                rng=rng,
                include_nearby=8,
            )
            for rc in neg_coords:
                negative_examples.append((state, rc))
        else:
            raise ValueError(f"Unknown data_imbalance.method: {method}")
    else:
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                if (r, c) == action:
                    continue
                negative_examples.append((state, (r, c)))

    return positive_examples, negative_examples


def extract_examples_from_demonstration(
    demonstration: Trajectory[np.ndarray, tuple[int, int]],
    *,
    data_imbalance: dict[str, Any] | None = None,
) -> tuple[
    list[tuple[np.ndarray, tuple[int, int]]], list[tuple[np.ndarray, tuple[int, int]]]
]:
    """Convert demonstrated (state, action)s into positive and negative
    classification data.

    Parameters
    ----------
    demonstrations : [(np.ndarray, (int, int))]
        State, action pairs

    Returns
    -------
    positive_examples : [(np.ndarray, (int, int))]
        A list with just the input state, action pairs (for convenience).
    negative_examples : [(np.ndarray, (int, int))]
        A list with negative examples of state, actions.
    """
    positive_examples: list[tuple[np.ndarray, tuple[int, int]]] = []
    negative_examples: list[tuple[np.ndarray, tuple[int, int]]] = []

    for demonstration_item in demonstration.steps:
        demo_positive_examples, demo_negative_examples = (
            extract_examples_from_demonstration_item(
                demonstration_item, data_imbalance=data_imbalance
            )
        )
        positive_examples.extend(demo_positive_examples)
        negative_examples.extend(demo_negative_examples)

    return positive_examples, negative_examples


def _split_dsl(dsl: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Return (base, module_map).

    base is pickleable; module_map is name->import_path.
    """
    base, module_map = {}, {}
    for k, v in dsl.items():
        if inspect.ismodule(v):
            module_map[k] = v.__name__
        else:
            base[k] = v
    # Remove __builtins__ if present
    if "__builtins__" in base:
        del base["__builtins__"]
    return base, module_map


def eval_program_fn(s: np.ndarray, a: tuple[int, int], prog: str) -> bool | None:
    """Evaluate a program on a state-action pair."""
    try:
        result = eval("lambda s, a: " + prog, _WORKER_DSL)(s, a)
        logging.info(f"Program: {prog}, Input: (s={s}, a={a}), Result: {result}")
        return result
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.info(f"Program: {prog}, Input: (s={s}, a={a}), Exception: {e}")
        return None


# Global worker states
_WORKER_DSL = None
_WORKER_PROGRAMS = None


def worker_init(
    dsl_blob: bytes, module_map: dict[str, str], program_batch: list[str]
) -> None:
    """Set up the worker once.

    Loads the DSL, reimports modules, and compiles the given program
    batch. Runs only once per process before handling any examples.
    """
    base = cloudpickle.loads(dsl_blob)
    for name, modpath in module_map.items():
        base[name] = import_module(modpath)
    set_dsl_functions(base)

    from programmatic_policy_learning.dsl.state_action_program import (  # pylint: disable=import-outside-toplevel
        DSL_FUNCTIONS,
    )

    global _WORKER_DSL, _WORKER_PROGRAMS  # pylint: disable=global-statement
    _WORKER_DSL = DSL_FUNCTIONS
    _WORKER_PROGRAMS = [
        eval("lambda s, a: " + prog, DSL_FUNCTIONS) for prog in program_batch
    ]


def worker_eval_example(fn_input: tuple[np.ndarray, tuple[int, int]]) -> list[bool]:
    """Run all precompiled programs on one (state, action) example.

    Uses the DSL and program_batch already set up by worker_init.
    """
    s, a = fn_input

    if _WORKER_PROGRAMS is None:
        raise RuntimeError("_WORKER_PROGRAMS is not initialized.\
            Ensure worker_init is called before using worker_eval_example.")

    results = []
    for f in _WORKER_PROGRAMS:
        try:
            results.append(f(s, a))
        except Exception as e:  # pylint: disable=broad-exception-caught
            results.append(None)
            logging.info(f"Error type: {type(e).__name__}\n" f"Error message: {e}")
    return results


def run_all_programs_on_single_demonstration(
    base_class_name: str,
    demo_number: int,
    programs: list[StateActionProgram] | list[str],
    demo_traj: Trajectory[np.ndarray, tuple[int, int]],
    dsl_functions: dict,
    *,
    data_imbalance: dict[str, Any] | None = None,
    return_examples: bool = False,
    program_interval: int = 1000,  # unused in this fast path; keep for compat  # pylint: disable=unused-argument
) -> tuple[Any, np.ndarray, list[tuple[np.ndarray, tuple[int, int]]] | None]:
    """Run all programs on a single demonstration and return feature matrix and
    labels."""
    logging.info(f"Running all programs on {base_class_name}, {demo_number}")
    positive_examples, negative_examples = extract_examples_from_demonstration(
        demo_traj, data_imbalance=data_imbalance
    )

    fn_inputs = positive_examples + negative_examples
    y: list[int] = [1] * len(positive_examples) + [0] * len(negative_examples)
    base_dsl, module_map = _split_dsl(dsl_functions)

    try:
        dsl_blob = cloudpickle.dumps(base_dsl)
        cloudpickle.loads(dsl_blob)  # Test deserialization
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to serialize/deserialize DSL: {e}") from e

    # Extract program strings (don’t pickle heavy objects repeatedly)
    program_strs = [
        (p.program if isinstance(p, StateActionProgram) else str(p)) for p in programs
    ]

    num_data = len(fn_inputs)
    num_programs = len(program_strs)

    X = lil_matrix((num_data, num_programs), dtype=bool)

    # Combine the context initialization into a single block to avoid redefinition
    try:
        ctx = multiprocessing.get_context("spawn")  # type: ignore[assignment]
        # linux
    except (ValueError, RuntimeError):
        ctx = multiprocessing.get_context("fork")  # type: ignore[assignment]
        # macOS/Windows fallback (spawn)
    num_workers = allowed_cpus()
    num_workers = max(1, min(num_workers, len(fn_inputs)))
    for p_start in range(0, num_programs, program_interval):
        p_end = min(p_start + program_interval, num_programs)
        program_batch = program_strs[p_start:p_end]
        with ctx.Pool(
            processes=num_workers,
            initializer=worker_init,
            initargs=(dsl_blob, module_map, program_batch),
            maxtasksperchild=100,
        ) as pool:
            results_iter = pool.imap(worker_eval_example, fn_inputs, chunksize=64)
            batch_rows_list = list(results_iter)
        batch_matrix = np.array(batch_rows_list, dtype=bool)
        X[:, p_start:p_end] = batch_matrix
    examples = fn_inputs if return_examples else None
    return X.tocsr(), np.array(y, dtype=np.uint8), examples


def run_all_programs_on_demonstrations(
    base_class_name: str,
    demo_numbers: tuple[int, ...],
    programs: list,
    demo_dict: dict[int, Trajectory],
    dsl_functions: dict,
    *,
    data_imbalance: dict[str, Any] | None = None,
    return_examples: bool = False,
) -> tuple[
    Any | None, np.ndarray | None, list[tuple[np.ndarray, tuple[int, int]]] | None
]:
    """Run all programs on a set of demonstrations and aggregate results."""
    X, y = None, None
    examples_all: list[tuple[np.ndarray, tuple[int, int]]] = []
    for demo_number in demo_numbers:
        demo_X, demo_y, demo_examples = run_all_programs_on_single_demonstration(
            base_class_name,
            demo_number,
            programs,
            demo_dict[demo_number],
            dsl_functions,
            data_imbalance=data_imbalance,
            return_examples=return_examples,
        )

        if X is None:
            X = demo_X
            y = demo_y
        else:
            X = vstack([X, demo_X])
            y = np.concatenate([y, demo_y])
        if return_examples and demo_examples:
            examples_all.extend(demo_examples)
    return X, y, (examples_all if return_examples else None)
