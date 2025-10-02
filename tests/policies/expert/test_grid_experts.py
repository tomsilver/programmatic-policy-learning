"""Tests for grid_experts.py."""

import numpy as np
import pytest
from generalization_grid_games.envs import two_pile_nim as tpn

from programmatic_policy_learning.policies.expert import grid_experts


def test_get_grid_expert_valid_names():
    """Test get_grid_expert returns callable for valid names."""
    for name in [
        "TwoPileNim",
        "CheckmateTactic",
        "StopTheFall",
        "Chase",
        "ReachForTheStar",
    ]:
        expert = grid_experts.get_grid_expert(name)
        assert callable(expert)


def test_get_grid_expert_invalid_name():
    """Test get_grid_expert raises ValueError for invalid name."""
    with pytest.raises(ValueError):
        grid_experts.get_grid_expert("UnknownEnv")


def test_expert_nim_policy_returns_action():
    """Test expert_nim_policy returns a tuple action."""
    layout = np.zeros((2, 2), dtype=object)
    layout[0, 0] = tpn.EMPTY
    layout[0, 1] = tpn.TOKEN
    action = grid_experts.expert_nim_policy(layout)
    assert isinstance(action, tuple)
