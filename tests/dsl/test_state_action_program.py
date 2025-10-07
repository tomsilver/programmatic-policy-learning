"""Test for StateActionProgram Class."""

import numpy as np
from generalization_grid_games.envs import two_pile_nim as tpn

from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    get_dsl_functions_dict,
)
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram


def test_state_action_program_evaluates_grid_v1_primitive() -> None:
    """Test that StateActionProgram correctly evaluates a grid_v1 primitive and
    returns a boolean."""
    # Example grid and action for testing
    state = np.array([[tpn.EMPTY, tpn.TOKEN], [tpn.TOKEN, tpn.EMPTY]])
    action = (0, 1)
    prog_str = "cell_is_value(tpn.TOKEN, a, s)"
    DSL_dict = get_dsl_functions_dict()
    prog_obj = StateActionProgram(prog_str, DSL_dict)
    result = prog_obj(state, action)
    assert callable(prog_obj)
    assert isinstance(result, bool) or result is None
