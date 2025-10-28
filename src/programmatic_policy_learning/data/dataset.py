"""Dataset creation and processing utilities for programmatic policy
learning."""

import logging
import multiprocessing
from functools import partial
from typing import Any

import numpy as np
from scipy.sparse import lil_matrix, vstack

from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.utils.cache_utils import manage_cache


def apply_programs(programs: list, fn_input: Any) -> list[bool]:
    """Worker function that applies a list of programs to a single given input.

    Parameters
    ----------
    programs : [ callable ]
    fn_input : Any

    Returns
    -------
    results : [ bool ]
        Program outputs in order.
    """
    x: list[bool] = []
    for program in programs:
        x_i = program(*fn_input)
        x.append(x_i)
    return x


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


@manage_cache("cache", [".npz", ".pkl"], key_fn=key_fn_for_all_p_one_demo)
def run_all_programs_on_single_demonstration(
    base_class_name: str,
    demo_number: int,
    programs: list,
    demo_traj: Trajectory[np.ndarray, tuple[int, int]],
    program_interval: int = 1000,
) -> tuple[Any, np.ndarray]:
    """Run all programs on a single demonstration and return feature matrix and
    labels."""

    logging.info(f"Running all programs on {base_class_name}, {demo_number}")
    positive_examples, negative_examples = extract_examples_from_demonstration(
        demo_traj
    )
    y: list[int] = [1] * len(positive_examples) + [0] * len(negative_examples)
    num_data = len(y)
    num_programs = len(programs)
    X = lil_matrix((num_data, num_programs), dtype=bool)
    for i in range(0, num_programs, program_interval):
        end = min(i + program_interval, num_programs)
        logging.info(f"Iteration {i} of {num_programs}")
        num_workers = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_workers)
        fn = partial(apply_programs, programs[i:end])
        fn_inputs = positive_examples + negative_examples
        results = pool.map(fn, fn_inputs)
        pool.close()
        for X_idx, x in enumerate(results):
            X[X_idx, i:end] = x
    X = X.tocsr()
    return X, np.array(y, dtype=np.uint8)  # y


def run_all_programs_on_demonstrations(
    base_class_name: str,
    demo_numbers: tuple[int, ...],
    programs: list,
    demo_dict: dict[int, Trajectory],
) -> tuple[Any | None, np.ndarray | None]:
    """Run all programs on a set of demonstrations and aggregate results."""
    X, y = None, None
    for demo_number in demo_numbers:
        demo_X, demo_y = run_all_programs_on_single_demonstration(
            base_class_name,
            demo_number,
            programs,
            demo_dict[demo_number],
        )
        if X is None:
            X = demo_X
            y = demo_y
        else:
            X = vstack([X, demo_X])
            y = np.concatenate([y, demo_y])
    return X, y
