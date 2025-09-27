"""Tests for LPP StateActionProgram."""

import numpy as np
import pytest

from programmatic_policy_learning.lpp.dsl.providers.ggg_primitives import (
    at_cell_with_value,
    cell_is_value,
    scanning,
    shifted,
)
from programmatic_policy_learning.lpp.state_action_program import StateActionProgram

# Create primitives dictionary for tests
TEST_PRIMITIVES = {
    "cell_is_value": cell_is_value,
    "at_cell_with_value": at_cell_with_value,
    "shifted": shifted,
    "scanning": scanning,
}


def test_state_action_program_primitive_function() -> None:
    """Test program that uses a primitive function."""
    state = np.array([[1, 0], [0, 1]])
    action = (0, 0)

    # Program that uses cell_is_value primitive
    program: StateActionProgram = StateActionProgram(
        "cell_is_value(1, a, s)", TEST_PRIMITIVES
    )

    # Test basic execution
    result = program(state, action)
    assert result


def test_state_action_program_primitive_function_false() -> None:
    """Test program returns False when condition not met."""
    state = np.array([[1, 0], [0, 1]])
    action = (0, 1)  # This position has value 0, not 1

    program: StateActionProgram = StateActionProgram(
        "cell_is_value(1, a, s)", TEST_PRIMITIVES
    )

    # Test complex program
    result = program(state, action)
    assert not result


def test_state_action_program_direct_state_access() -> None:
    """Test program with direct state array access."""
    state = np.array([[1, 2], [3, 4]])
    action = (0, 0)

    # Program that directly accesses state
    program: StateActionProgram = StateActionProgram("s[0, 0] == 1", TEST_PRIMITIVES)

    # Test with different parameters
    assert program(state, action)


def test_state_action_program_action_coordinate_access() -> None:
    """Test program that uses action coordinates."""
    state = np.array([[1, 2], [3, 4]])

    # Program that checks if action is in top row
    program: StateActionProgram = StateActionProgram("a[0] == 0", TEST_PRIMITIVES)

    # Test first call
    assert program(state, (0, 0))
    assert not program(state, (1, 0))


def test_state_action_program_caching() -> None:
    """Test that programs are compiled and cached."""
    program: StateActionProgram = StateActionProgram("True", TEST_PRIMITIVES)
    state = np.array([[1]])
    action = (0, 0)

    # First call should compile
    assert program.compiled_func is None
    result1 = program(state, action)
    assert program.compiled_func is not None

    # Second call should use cached version
    result2 = program(state, action)  # type: ignore[unreachable]
    # Verify results are the same
    assert result1 == result2


def test_state_action_program_string_representation() -> None:
    """Test string representations of programs."""
    program: StateActionProgram = StateActionProgram(
        "cell_is_value(1, a, s)", TEST_PRIMITIVES
    )

    assert str(program) == "cell_is_value(1, a, s)"
    assert repr(program) == "StateActionProgram('cell_is_value(1, a, s)')"


def test_state_action_program_invalid_program() -> None:
    """Test that invalid programs raise appropriate errors."""
    program: StateActionProgram = StateActionProgram("invalid_syntax[", TEST_PRIMITIVES)
    state = np.array([[1]])
    action = (0, 0)

    with pytest.raises(SyntaxError):
        program(state, action)


def test_state_action_program_custom_eval_context() -> None:
    """Test StateActionProgram with custom evaluation context."""
    custom_primitives = {**TEST_PRIMITIVES, "custom_func": lambda s, a: "custom"}
    program: StateActionProgram = StateActionProgram(
        "custom_func(s, a)", custom_primitives
    )
    state = np.array([[1]])
    action = (0, 0)

    result = program(state, action)
    assert result == "custom"


def test_state_action_program_default_vs_custom_context() -> None:
    """Test that custom primitives override default primitives."""
    # First, test with default primitives
    default_program: StateActionProgram = StateActionProgram(
        "cell_is_value(1, (0, 0), s)", TEST_PRIMITIVES
    )

    # Then test with custom primitives that overrides cell_is_value
    custom_primitives = {
        "cell_is_value": lambda value, position, obs: False  # Always return False
    }
    custom_program: StateActionProgram = StateActionProgram(
        "cell_is_value(1, (0, 0), s)", custom_primitives
    )

    state = np.array([[1]])
    action = (0, 0)

    # Default should return True (1 == 1)
    assert default_program(state, action)

    # Custom should return False (overridden function)
    assert not custom_program(state, action)
