"""Tests for LPP DSL primitives."""

import numpy as np

from programmatic_policy_learning.lpp.dsl import cell_is_value, out_of_bounds


class TestPrimitives:
    """Test cases for DSL primitive functions."""

    def test_out_of_bounds(self):
        """Test out_of_bounds function."""
        shape = (3, 3)

        # Valid positions
        assert out_of_bounds(0, 0, shape) is False
        assert out_of_bounds(2, 2, shape) is False
        assert out_of_bounds(1, 1, shape) is False

        # Invalid positions
        assert out_of_bounds(-1, 0, shape) is True
        assert out_of_bounds(0, -1, shape) is True
        assert out_of_bounds(3, 0, shape) is True
        assert out_of_bounds(0, 3, shape) is True

    def test_cell_is_value_valid_position(self):
        """Test cell_is_value with valid positions."""
        obs = np.array([[1, 2], [3, 4]])

        assert cell_is_value(1, (0, 0), obs)
        assert cell_is_value(2, (0, 1), obs)
        assert cell_is_value(3, (1, 0), obs)
        assert cell_is_value(4, (1, 1), obs)

        # Wrong values
        assert not cell_is_value(5, (0, 0), obs)
        assert not cell_is_value(1, (0, 1), obs)

    def test_cell_is_value_out_of_bounds(self):
        """Test cell_is_value with out of bounds positions."""
        obs = np.array([[1, 2], [3, 4]])

        # Out of bounds positions should return False
        assert cell_is_value(1, (-1, 0), obs) is False
        assert cell_is_value(1, (0, -1), obs) is False
        assert cell_is_value(1, (2, 0), obs) is False
        assert cell_is_value(1, (0, 2), obs) is False

    def test_cell_is_value_none_position(self):
        """Test cell_is_value with None position."""
        obs = np.array([[1, 2], [3, 4]])

        # Test with invalid position
        assert not cell_is_value(1, None, obs)
        assert not cell_is_value(None, None, obs)
