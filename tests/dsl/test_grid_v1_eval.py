"Test for grid_v1 evaluation"

import numpy as np

from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import GridInput, make_dsl


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
