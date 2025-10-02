"""Test for dataset creation workflow."""

import numpy as np

from programmatic_policy_learning.data.dataset import (
    extract_examples_from_demonstration,
    extract_examples_from_demonstration_item,
    run_all_programs_on_demonstrations,
    run_all_programs_on_single_demonstration,
)
from programmatic_policy_learning.data.demo_types import Trajectory


def dummy_program(state: np.ndarray, _: tuple[int, int]) -> bool:
    """Return True if sum of state is even, else False."""
    return np.sum(state) % 2 == 0


def test_run_all_programs_on_single_demonstration() -> None:
    """Test running programs on a single demonstration."""
    # Create a dummy demonstration
    state = np.array([[1, 2], [3, 4]])
    action = (0, 1)
    # traj = Trajectory(obs=[state], act=[action])
    programs = [dummy_program]
    X, y = run_all_programs_on_single_demonstration(
        base_class_name="DummyEnv",
        demo_number=0,
        programs=programs,
        demonstrations=[(state, action)],
    )
    assert X.shape[0] == len(y)
    assert X.shape[1] == len(programs)
    assert set(y) <= {0, 1}  # binary labels


def test_extract_examples_from_demonstration() -> None:
    """Test extracting positive and negative examples from demonstration."""

    state = np.array([[1, 2], [3, 4]])
    action = (0, 1)
    demonstration = [(state, action)]
    pos, neg = extract_examples_from_demonstration(demonstration)
    assert len(pos) == 1
    assert all(isinstance(x, tuple) for x in pos)
    assert all(isinstance(x, tuple) for x in neg)
