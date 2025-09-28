"""Tests for OracleDSLLearner."""

from typing import Any

import pytest

from programmatic_policy_learning.dsl.learners.oracle import OracleDSLLearner


def test_oracle_loads_known_dsl() -> None:
    """Test that OracleDSLLearner loads a known DSL id."""
    learner: OracleDSLLearner[Any, Any, Any] = OracleDSLLearner("grid_v1")
    dsl = learner.generate_dsl()
    assert dsl.id == "grid_v1"
    assert "cell_is_value" in dsl.primitives


def test_oracle_unknown_id_raises() -> None:
    """Test that loading an unknown DSL id raises ValueError."""
    with pytest.raises(ValueError):
        OracleDSLLearner("does_not_exist")
