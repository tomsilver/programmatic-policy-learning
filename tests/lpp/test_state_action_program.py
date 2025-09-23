"""Tests for LPP StateActionProgram."""

import numpy as np
import pytest

from programmatic_policy_learning.lpp.state_action_program import StateActionProgram


def test_state_action_program_primitive_function():
    """Test program that uses a primitive function."""
    state = np.array([[1, 0], [0, 1]])
    action = (0, 0)

    # Program that uses cell_is_value primitive
    program = StateActionProgram("cell_is_value(1, a, s)")

    # Test basic execution
    result = program(state, action)
    assert result


def test_state_action_program_primitive_function_false():
    """Test program returns False when condition not met."""
    state = np.array([[1, 0], [0, 1]])
    action = (0, 1)  # This position has value 0, not 1

    program = StateActionProgram("cell_is_value(1, a, s)")

    # Test complex program
    result = program(state, action)
    assert not result


def test_state_action_program_direct_state_access():
    """Test program with direct state array access."""
    state = np.array([[1, 2], [3, 4]])
    action = (0, 0)

    # Program that directly accesses state
    program = StateActionProgram("s[0, 0] == 1")

    # Test with different parameters
    assert program(state, action)


def test_state_action_program_action_coordinate_access():
    """Test program that uses action coordinates."""
    state = np.array([[1, 2], [3, 4]])

    # Program that checks if action is in top row
    program = StateActionProgram("a[0] == 0")

    # Test first call
    assert program(state, (0, 0))
    assert not program(state, (1, 0))


def test_state_action_program_caching():
    """Test that programs are compiled and cached."""
    program = StateActionProgram("True")
    state = np.array([[1]])
    action = (0, 0)

    # First call should compile
    assert program.compiled_func is None
    result1 = program(state, action)
    assert program.compiled_func is not None

    # Second call should use cached version
    result2 = program(state, action)
    assert result1 == result2


def test_state_action_program_string_representation():
    """Test string representations of programs."""
    program = StateActionProgram("cell_is_value(1, a, s)")

    assert str(program) == "cell_is_value(1, a, s)"
    assert repr(program) == "StateActionProgram('cell_is_value(1, a, s)')"


def test_state_action_program_invalid_program():
    """Test that invalid programs raise appropriate errors."""
    program = StateActionProgram("invalid_syntax[")
    state = np.array([[1]])
    action = (0, 0)

    with pytest.raises(SyntaxError):
        program(state, action)
