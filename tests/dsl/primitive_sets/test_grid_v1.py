"""Tests for DSL primitives - grid_v1."""

import numpy as np

from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    at_action_cell,
    at_cell_with_value,
    cell_is_value,
    out_of_bounds,
    scanning,
    shifted,
)


def test_out_of_bounds() -> None:
    """Basic and out of bounds case."""
    shape = (2, 2)
    assert not out_of_bounds(0, 0, shape)
    assert out_of_bounds(2, 2, shape)


def test_cell_is_value_basic_and_none() -> None:
    """Basic and None case for cell_is_value."""
    obs = np.array([[1, 2], [3, 4]])
    assert cell_is_value(4, (1, 1), obs)
    assert not cell_is_value(99, (0, 0), obs)


def test_cell_is_value_out_of_bounds() -> None:
    """Out of bounds returns False."""
    obs = np.array([[1, 2], [3, 4]])
    assert not cell_is_value(1, (-1, 0), obs)


def test_cell_is_value_none_position() -> None:
    """None position returns False."""
    obs = np.array([[1, 2], [3, 4]])
    assert not cell_is_value(1, None, obs)


def test_shifted_basic() -> None:
    """Basic shifted test."""
    obs = np.array([[1, 2], [3, 4]])

    def local_program(cell: tuple[int, int] | None, obs: np.ndarray) -> bool:
        return cell is not None and obs[cell[0], cell[1]] == 4

    assert shifted((1, 1), local_program, (0, 0), obs)


def test_shifted_none_cell() -> None:
    """None cell returns False."""
    obs = np.array([[1, 2], [3, 4]])

    def local_program(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> bool:
        return cell is not None

    assert not shifted((1, 0), local_program, None, obs)


def test_shifted_directions() -> None:
    """Shifted returns correct value for direction."""
    obs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def get_value_program(cell: tuple[int, int] | None, obs: np.ndarray) -> int:
        if cell is None:
            return -1
        return obs[cell[0], cell[1]]

    start_cell = (1, 1)
    assert shifted((0, 1), get_value_program, start_cell, obs) == 6


def test_at_cell_with_value_basic() -> None:
    """Basic at_cell_with_value test."""
    obs = np.array([[1, 2], [3, 4]])
    cell = (0, 1)

    def get_coordinates_program(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> tuple[int, int] | None:
        return cell

    assert np.array_equal(
        at_cell_with_value(4, get_coordinates_program, cell, obs), [1, 1]
    )


def test_at_cell_with_value_not_found() -> None:
    """Not found returns False."""
    obs = np.array([[1, 2], [3, 4]])
    cell = (0, 1)

    def check_cell_program(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> bool:
        return cell is not None

    assert not at_cell_with_value(99, check_cell_program, cell, obs)


def test_at_cell_with_value_multiple_occurrences() -> None:
    """Finds first occurrence."""
    obs = np.array([[1, 2, 1], [3, 1, 4]])
    cell = (0, 1)

    def get_coordinates_program(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> tuple[int, int] | None:
        return cell

    assert np.array_equal(
        at_cell_with_value(1, get_coordinates_program, cell, obs), [0, 0]
    )


def test_at_cell_with_value_with_logic() -> None:
    """Logic on found cell."""
    obs = np.array([[5, 2], [3, 8]])
    cell = (0, 1)

    def check_neighbors_program(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> bool:
        return cell is not None and obs[cell[0], cell[1]] < 5

    assert at_cell_with_value(3, check_neighbors_program, cell, obs)


def test_at_action_cell_basic() -> None:
    """Basic at_action_cell test."""
    obs = np.array([[1, 2], [3, 4]])
    cell = (1, 1)

    def get_value_program(cell: tuple[int, int] | None, obs: np.ndarray) -> int:
        return obs[cell[0], cell[1]] if cell is not None else -1

    assert at_action_cell(get_value_program, cell, obs) == 4


def test_at_action_cell_none() -> None:
    """None cell returns True."""
    obs = np.array([[1, 2], [3, 4]])

    def check_none_program(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> bool:
        return cell is None

    assert at_action_cell(check_none_program, None, obs)


def test_scanning_basic() -> None:
    """Basic scanning test."""
    obs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def true_condition(cell: tuple[int, int] | None, obs: np.ndarray) -> bool:
        return cell is not None and obs[cell[0], cell[1]] == 6

    def false_condition(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> bool:
        return False

    assert scanning((0, 1), true_condition, false_condition, (1, 0), obs)


def test_scanning_hits_false_condition() -> None:
    """Stops at false condition."""
    obs = np.array([[1, 2, 3], [4, 0, 6], [7, 8, 9]])

    def true_condition(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> bool:
        return False

    def false_condition(cell: tuple[int, int] | None, obs: np.ndarray) -> bool:
        return cell is not None and obs[cell[0], cell[1]] == 0

    assert not scanning((0, 1), true_condition, false_condition, (1, 0), obs)


def test_scanning_hits_boundary() -> None:
    """Stops at boundary."""
    obs = np.array([[1, 2], [3, 4]])

    def true_condition(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> bool:
        return False

    def false_condition(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> bool:
        return False

    assert not scanning((0, -1), true_condition, false_condition, (0, 0), obs)


def test_scanning_none_cell() -> None:
    """None cell returns False."""
    obs = np.array([[1, 2], [3, 4]])

    def true_condition(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> bool:
        return True

    def false_condition(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> bool:
        return False

    assert not scanning((1, 0), true_condition, false_condition, None, obs)


def test_scanning_timeout() -> None:
    """Respects max_timeout."""
    obs = np.array([[1] * 100])

    def true_condition(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> bool:
        return False

    def false_condition(
        cell: tuple[int, int] | None, obs: np.ndarray  # pylint: disable=unused-argument
    ) -> bool:
        return False

    assert not scanning(
        (0, 1), true_condition, false_condition, (0, 0), obs, max_timeout=5
    )
