"""Test for dataset creation workflow."""

import numpy as np

from programmatic_policy_learning.data.dataset import (
    extract_examples_from_demonstration,
    run_all_programs_on_single_demonstration,
)
from programmatic_policy_learning.data.demo_types import Trajectory


def test_run_all_programs_on_single_demonstration() -> None:
    """Test running programs on a single demonstration."""
    # Create a dummy demonstration
    state = np.array([[1, 2], [3, 4]])
    action = (0, 1)
    traj: Trajectory[np.ndarray, tuple[int, int]] = Trajectory(steps=[(state, action)])

    # Define a simpler dummy program for evaluation
    def dummy_program(state: np.ndarray, _: tuple[int, int]) -> bool:
        """Return True if the sum of the state is greater than 10."""
        return np.sum(state) > 10

    # programs = [dummy_program]
    programs = ["lambda s, a: np.sum(s) > 10"]
    X, y = run_all_programs_on_single_demonstration(
        "DummyEnv",  # base_class_name
        0,  # demo_number
        programs,  # programs
        traj,  # demo_traj
        dsl_functions={},
    )
    assert X.shape[0] == len(y)
    assert X.shape[1] == len(programs)
    assert set(y) <= {0, 1}  # binary labels


def test_extract_examples_from_demonstration() -> None:
    """Test extracting positive and negative examples from demonstration."""

    state = np.array([[1, 2], [3, 4]])
    action = (0, 1)
    traj: Trajectory[np.ndarray, tuple[int, int]] = Trajectory(steps=[(state, action)])
    pos, neg = extract_examples_from_demonstration(traj)
    assert len(pos) == 1
    assert all(isinstance(x, tuple) for x in pos)
    assert all(isinstance(x, tuple) for x in neg)
