"Test for grid_v1 evaluation"

import numpy as np

from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    GridInput,
    get_dsl_functions_dict,
    make_ablated_dsl,
    make_dsl,
)


def test_evaluate_shift_right_equals_three() -> None:
    """Test that shifting right from a cell finds value 3."""
    dsl = make_dsl()
    P = dsl.primitives
    obs = np.array([[0, 3, 0], [0, 0, 0]])

    def program(cell: tuple[int, int] | None, obs: np.ndarray) -> bool:
        """Returns True if shifting right from cell lands on value 3."""
        return P["shifted"](
            (0, 1),
            lambda c, o: P["cell_is_value"](3, c, o),
            cell,
            obs,
        )

    assert dsl.evaluate(program, GridInput(obs=obs, cell=(0, 0)))
    assert not dsl.evaluate(program, GridInput(obs=obs, cell=(1, 0)))


def test_make_ablated_dsl_and_get_dsl_functions_dict() -> None:
    """Test make_ablated_dsl and get_dsl_functions_dict with a config-like
    structure."""

    # Simulate a config structure
    config = {
        "program_generation": {
            "strategy": "grid_v1_ablated",
            "removed_primitive": "shifted",
            "dsl_generator_prompt": "some_prompt_path.txt",
        }
    }

    removed_primitive = config["program_generation"]["removed_primitive"]

    # Test make_ablated_dsl
    dsl = make_ablated_dsl(removed_primitive)
    assert removed_primitive not in dsl.primitives
    assert "cell_is_value" in dsl.primitives
    assert "at_cell_with_value" in dsl.primitives

    # Test get_dsl_functions_dict
    dsl_functions = get_dsl_functions_dict(removed_primitive)
    assert removed_primitive not in dsl_functions
    assert "cell_is_value" in dsl_functions
    assert "at_cell_with_value" in dsl_functions
