"""Dataset creation and processing utilities for programmatic policy
learning."""

import hashlib
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


def extract_examples_from_demonstration_item(
    demonstration_item: tuple[np.ndarray, tuple[int, int]],
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


# def key_fn_for_all_p_one_demo(args: tuple, kwargs: dict) -> str:
#     """Short id for caching: keep values but skip `programs` and `demo_traj`."""
#     # args: base_class_name, demo_number, programs, demo_traj, program_interval
#     parts = [str(a) for i, a in enumerate(args) if i not in (2, 3)]

#     # include programs length
#     try:
#         parts.append(str(len(args[2])))
#     except Exception as exc:
#         raise TypeError("programs argument must be a sized list") from exc
#     parts += [str(v) for k, v in kwargs.items() if k not in ("programs", "demo_traj")]
#     return "-".join(parts)


def key_fn_for_all_p_one_demo(args: tuple, kwargs: dict) -> str:
    """Short id for caching: skip large/unpicklable args like programs,
    demo_traj, dsl_functions."""
    # args: base_class_name, demo_number, programs,
    # demo_traj, dsl_functions, program_interval
    parts = []
    for i, a in enumerate(args):
        # skip heavy objects (programs, trajectories, dsl)
        if i in (2, 3, 4):
            continue
        parts.append(str(a))

    # include programs length for uniqueness
    try:
        parts.append(str(len(args[2])))
    except Exception as exc:
        raise TypeError("programs argument must be a sized list") from exc

    # include lightweight kwargs only
    for k, v in kwargs.items():
        if k not in ("programs", "demo_traj", "dsl_functions"):
            parts.append(str(v))

    # truncate long ids if necessary
    key = "-".join(parts)
    if len(key) > 200:
        key = hashlib.sha1(key.encode()).hexdigest()[:16]
    return key


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
        logging.debug(f"Program: {prog}, Input: (s={s}, a={a}), Result: {result}")
        return result
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.debug(f"Program: {prog}, Input: (s={s}, a={a}), Exception: {e}")
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
    #for i, prog in enumerate(program_batch):
     #   try:
      #      compiled = eval("lambda s, a: " + prog, DSL_FUNCTIONS)
       #     _WORKER_PROGRAMS[i] = compiled
        #except Exception as e:
         #   logging.error(f"[compile_error] Program #{i} failed to compile: {e}")
          #  _WORKER_PROGRAMS[i] = None

def worker_eval_example(fn_input: tuple[np.ndarray, tuple[int, int]]) -> list[bool]:
    """Run all precompiled programs on one (state, action) example.

    Uses the DSL and program_batch already set up by worker_init.
    """
    s, a = fn_input

    if _WORKER_PROGRAMS is None:
        raise RuntimeError(
            "_WORKER_PROGRAMS is not initialized.\
            Ensure worker_init is called before using worker_eval_example."
        )

    results = []
    for f in _WORKER_PROGRAMS:
        try:
        #    out = safe_eval_with_timeout(f, s, a)
        #    results.append(out)
            results.append(f(s, a))
        except Exception:  # pylint: disable=broad-exception-caught
            results.append(None)
            # logging.warning(
            #     f"[worker_eval_example] Error executing program #{i}:\n"
            #     f"Program source: {f}\n"
            #     f"Error type: {type(e).__name__}\n"
            #     f"Error message: {e}"
            # )

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
        ctx = multiprocessing.get_context(
        "spawn"
        )  # linux
    except (ValueError, RuntimeError):
        ctx = (  # type: ignore[no-redef]
            multiprocessing.get_context()
        )  # macOS/Windows fallback (spawn)

    num_workers = allowed_cpus()
    num_workers = max(1, min(num_workers, len(fn_inputs)))

    for p_start in range(0, num_programs, program_interval):
        p_end = min(p_start + program_interval, num_programs)
        program_batch = program_strs[p_start:p_end]
        # Convert to numpy and assign into big matrix
        with ctx.Pool(
            processes=num_workers,
            initializer=worker_init,
            initargs=(dsl_blob, module_map, program_batch),
            maxtasksperchild=100,
        ) as pool:
            results_iter = pool.imap(worker_eval_example, fn_inputs, chunksize=64)
            #batch_rows = np.array(list(results_iter), dtype=bool)
            #X[:, p_start:p_end] = batch_rows
            batch_rows_list = list(results_iter)
        batch_matrix = np.array(batch_rows_list, dtype=bool)
        X[:, p_start:p_end] = batch_matrix
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
