"""Tests for structs.py."""

from programmatic_policy_learning.approaches.structs import ParametricPolicyBase
from typing import Any


def test_parameteric_policy():
    """Tests for ParametericPolicyBase()."""

    class SimpleParametricPolicyBase(ParametricPolicyBase):
        """A simple parameteric policy that just returns the "act" parameter."""

        def act(self, obs: Any) -> Any:
            return self._params["act"]


    parameteric_policy = SimpleParametricPolicyBase(
        init_params={"act": 0.0},
        param_bounds={"act": (-1.0, 1.0)},
    )

    assert parameteric_policy.get_params() == {"act": 0.0}
    assert parameteric_policy.get_bounds() == {"act": (-1.0, 1.0)}
    parameteric_policy.set_params({"act": 0.5})
    assert parameteric_policy.get_params() == {"act": 0.5}
    assert parameteric_policy.act(None) == 0.5
