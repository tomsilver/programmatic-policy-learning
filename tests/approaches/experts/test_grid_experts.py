"""Tests for experts/grid_experts.py."""

import numpy as np
import pytest
from generalization_grid_games.envs import two_pile_nim as tpn

from programmatic_policy_learning.approaches.experts.grid_experts import get_grid_expert


def test_get_grid_expert_returns_callable() -> None:
    """Test that get_grid_expert returns a callable."""
    expert_fn = get_grid_expert("TwoPileNim0-v0")
    assert callable(expert_fn)


def test_get_grid_expert_valid_names() -> None:
    """Test get_grid_expert returns callable for valid names."""
    for name in [
        "TwoPileNim0-v0",
        "CheckmateTactic",
        "StopTheFall",
        "Chase",
        "ReachForTheStar",
    ]:
        expert = get_grid_expert(name)
        assert callable(expert)


def test_get_grid_expert_invalid_name() -> None:
    """Test get_grid_expert raises ValueError for invalid name."""
    with pytest.raises(ValueError):
        get_grid_expert("UnknownEnv")


def test_expert_nim_policy_returns_action() -> None:
    """Test expert function output type (action tuple)."""
    layout = np.zeros((2, 2), dtype=object)
    layout[0, 0] = tpn.EMPTY
    layout[0, 1] = tpn.TOKEN
    expert_fn = get_grid_expert("TwoPileNim0-v0")
    action = expert_fn(layout)
    assert isinstance(action, tuple)
