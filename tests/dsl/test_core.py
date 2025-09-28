"""Core DSL tests."""

from dataclasses import FrozenInstanceError

import pytest

from programmatic_policy_learning.dsl.core import DSL


def test_dsl_evaluate_simple() -> None:
    """Test DSL evaluate returns correct output."""

    def eval_fn(program: int, inputs: int) -> int:
        return program + inputs

    dsl: DSL[int, int, int] = DSL(id="dummy", primitives={}, evaluate_fn=eval_fn)
    result = dsl.evaluate(2, 3)
    assert result == 5


def test_dsl_is_frozen_dataclass() -> None:
    """Test DSL is frozen dataclass."""
    dsl: DSL[str, None, None] = DSL(
        id="dummy", primitives={}, evaluate_fn=lambda p, i: None
    )
    with pytest.raises(FrozenInstanceError):
        dsl.id = "new"  # type: ignore
