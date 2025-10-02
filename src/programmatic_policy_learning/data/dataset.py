"""Dataset creation and processing utilities for programmatic policy learning."""

from typing import Any, Tuple, List
import numpy as np
from scipy.sparse import lil_matrix, vstack
import multiprocessing
from functools import partial
from programmatic_policy_learning.data.demo_types import Trajectory


def apply_programs(programs, fn_input):
    """
    Worker function that applies a list of programs to a single given input.

    Parameters
    ----------
    programs : [ callable ]
    fn_input : Any

    Returns
    -------
    results : [ bool ]
        Program outputs in order.
    """
    x = []
    for program in programs:
        x_i = program(*fn_input)
        x.append(x_i)
    return x

def extract_examples_from_demonstration_item(demonstration_item):
    """
    Convert a demonstrated (state, action) into positive and negative classification data.

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

    positive_examples = [(state, action)]
    negative_examples = []

    for r in range(state.shape[0]):
        for c in range(state.shape[1]):
            if (r, c) == action:
                continue
            else:
                negative_examples.append((state, (r, c)))

    return positive_examples, negative_examples

def extract_examples_from_demonstration(demonstration):
    """
    Convert demonstrated (state, action)s into positive and negative classification data.

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
    positive_examples = []
    negative_examples = []

    for demonstration_item in demonstration:
        demo_positive_examples, demo_negative_examples = extract_examples_from_demonstration_item(demonstration_item)
        positive_examples.extend(demo_positive_examples)
        negative_examples.extend(demo_negative_examples)

    return positive_examples, negative_examples


# cache management decorator
def run_all_programs_on_single_demonstration(base_class_name: str, demo_number: int, programs: list, demonstrations: list[Trajectory[Any, Any]], program_interval: int = 1000):
    """
    Run all programs on a single demonstration and return feature matrix and labels.
    """
    print(f"Running all programs on {base_class_name}, {demo_number}")
    positive_examples, negative_examples = extract_examples_from_demonstration(demonstrations)
    y = [1] * len(positive_examples) + [0] * len(negative_examples)
    num_data = len(y)
    num_programs = len(programs)
    X = lil_matrix((num_data, num_programs), dtype=bool)
    for i in range(0, num_programs, program_interval):
        end = min(i + program_interval, num_programs)
        print(f'Iteration {i} of {num_programs}', end='\r')
        num_workers = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_workers)
        fn = partial(apply_programs, programs[i:end])
        fn_inputs = positive_examples + negative_examples
        results = pool.map(fn, fn_inputs)
        pool.close()
        for X_idx, x in enumerate(results):
            X[X_idx, i:end] = x
    X = X.tocsr()
    print()
    return X, y



# cache management decorator
def run_all_programs_on_demonstrations(base_class_name: str, demo_numbers: List[int], programs: list, demonstrations: list[Trajectory[Any, Any]]) -> Tuple[Any, Any]:
    """
    Run all programs on a set of demonstrations and aggregate results.
    """
    X, y = None, None
    for demo_number in demo_numbers:
        demo_X, demo_y = run_all_programs_on_single_demonstration(base_class_name, demo_number, programs, demonstrations)
        if X is None:
            X = demo_X
            y = demo_y
        else:
            X = vstack([X, demo_X])
            y.extend(demo_y)
    y = np.array(y, dtype=np.uint8)
    return X, y
