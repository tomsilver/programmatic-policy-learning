"""Dataset creation and processing utilities for programmatic policy
learning."""

import inspect
import logging
from importlib import import_module
from typing import Any

import cloudpickle
import numpy as np
from pathos.helpers import cpu_count
from pathos.multiprocessing import Pool
from scipy.sparse import lil_matrix, vstack

from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.dsl.state_action_program import (
    StateActionProgram,
    set_dsl_functions,
)

# from programmatic_policy_learning.utils.cache_utils import manage_cache


def extract_examples_from_demonstration_item(
    demonstration_item: tuple[np.ndarray, tuple[int, int]],
) -> tuple[
    list[tuple[np.ndarray, tuple[int, int]]], list[tuple[np.ndarray, tuple[int, int]]]
]:
    """Convert a demonstrated (state, action) into positive and negative
    classification data.

    All actions not taken in the demonstration_item are considered negative.

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

    for r in range(state.shape[0]):
        for c in range(state.shape[1]):
            if (r, c) == action:
                continue
            negative_examples.append((state, (r, c)))

    return positive_examples, negative_examples


def extract_examples_from_demonstration(
    demonstration: Trajectory[np.ndarray, tuple[int, int]],
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
            extract_examples_from_demonstration_item(demonstration_item)
        )
        positive_examples.extend(demo_positive_examples)
        negative_examples.extend(demo_negative_examples)

    return positive_examples, negative_examples


def key_fn_for_all_p_one_demo(args: tuple, kwargs: dict) -> str:
    """Short id for caching: keep values but skip `programs` and `demo_traj`."""
    # args: base_class_name, demo_number, programs, demo_traj, program_interval
    parts = [str(a) for i, a in enumerate(args) if i not in (2, 3)]

    # include programs length
    try:
        parts.append(str(len(args[2])))
    except Exception as exc:
        raise TypeError("programs argument must be a sized list") from exc
    parts += [str(v) for k, v in kwargs.items() if k not in ("programs", "demo_traj")]
    return "-".join(parts)


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


def _eval_all_on_example_with_dsl(
    fn_input: tuple[np.ndarray, tuple[int, int]],
    dsl_blob: bytes,
    module_map: dict[str, str],
    program_strs: list[str],
) -> list[bool | None]:
    """Worker task: apply ALL programs to ONE example (s, a), with DSL setup."""

    base = cloudpickle.loads(dsl_blob)  # functions/constants
    for name, modpath in module_map.items():  # re-bind modules
        base[name] = import_module(modpath)

    set_dsl_functions(base)

    from programmatic_policy_learning.dsl.state_action_program import (  # pylint: disable=import-outside-toplevel
        DSL_FUNCTIONS,
    )

    worker_funs = [eval("lambda s, a: " + prog, DSL_FUNCTIONS) for prog in program_strs]

    results = []
    for f in worker_funs:
        try:
            result = f(*fn_input)
            results.append(result)
        except Exception as e:
            results.append(None)
            raise RuntimeError(f"Error executing program: {f}, Error: {e}")

    return results


# @manage_cache("cache", [".npz", ".pkl"], key_fn=key_fn_for_all_p_one_demo)
def run_all_programs_on_single_demonstration(
    base_class_name: str,
    demo_number: int,
    programs: list[StateActionProgram] | list[str],
    demo_traj: Trajectory[np.ndarray, tuple[int, int]],
    dsl_functions: dict,
    program_interval: int = 1000,  # unused in this fast path; keep for compat  # pylint: disable=unused-argument
) -> tuple[Any, np.ndarray]:
    """Run all programs on a single demonstration and return feature matrix and
    labels."""
    logging.info(f"Running all programs on {base_class_name}, {demo_number}")
    positive_examples, negative_examples = extract_examples_from_demonstration(
        demo_traj
    )
    fn_inputs = positive_examples + negative_examples
    y: list[int] = [1] * len(positive_examples) + [0] * len(negative_examples)

    base_dsl, module_map = _split_dsl(dsl_functions)

    try:
        dsl_blob = cloudpickle.dumps(base_dsl)
        cloudpickle.loads(dsl_blob)  # Test deserialization
    except Exception as e:
        raise RuntimeError(f"Failed to serialize/deserialize DSL: {e}")

    # Extract program strings (don’t pickle heavy objects repeatedly)
    program_strs = [
        (p.program if isinstance(p, StateActionProgram) else str(p)) for p in programs
    ]

    num_data = len(fn_inputs)
    num_programs = len(program_strs)
    X = lil_matrix((num_data, num_programs), dtype=bool)

    # Add program-level chunking for multiprocessing
    num_workers = cpu_count()
    # NOTE: one pool reused for the entire demo
    with Pool(processes=num_workers) as pool:
        # Chunk examples to cut scheduling overhead
        # Chunking number of (s,a) pairs assigned to the worker
        CHUNK = 64
        for program_start in range(0, len(program_strs), program_interval):
            program_end = min(program_start + program_interval, len(program_strs))
            program_batch = program_strs[program_start:program_end]

            results_iter = pool.imap(
                lambda fn_input, batch=program_batch: _eval_all_on_example_with_dsl(
                    fn_input, dsl_blob, module_map, batch
                ),
                fn_inputs,
                chunksize=CHUNK,
            )
            for row_idx, row in enumerate(results_iter):
                X[row_idx, program_start:program_end] = row

    return X.tocsr(), np.array(y, dtype=np.uint8)


def run_all_programs_on_demonstrations(
    base_class_name: str,
    demo_numbers: tuple[int, ...],
    programs: list,
    demo_dict: dict[int, Trajectory],
    dsl_functions: dict,
) -> tuple[Any | None, np.ndarray | None]:
    """Run all programs on a set of demonstrations and aggregate results."""
    X, y = None, None
    for demo_number in demo_numbers:
        demo_X, demo_y = run_all_programs_on_single_demonstration(
            base_class_name,
            demo_number,
            programs,
            demo_dict[demo_number],
            dsl_functions,
        )

        if X is None:
            X = demo_X
            y = demo_y
        else:
            X = vstack([X, demo_X])
            y = np.concatenate([y, demo_y])
    return X, y
