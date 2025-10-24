"""Single-prompt LLM synthesis that returns a ParametricPolicyBase instance.

- One prompt only (no iterations).
- LLM generates a function: policy(obs, params) -> action (np.ndarray).
- We adapt that function into a ParametricPolicyBase subclass so callers get a
  proper object with .act(obs) and self._params.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, TypeVar, cast

import numpy as np
from gymnasium.spaces import Box, Space
from prpl_llm_utils.code import (
    synthesize_python_function_with_llm,
)
from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.approaches.structs import ParametricPolicyBase

_ObsT = TypeVar("_ObsT")
_ActT = TypeVar("_ActT")


class LLMGeneratedParametricPolicy(ParametricPolicyBase):
    """Adapter that wraps an LLM-generated function into
    ParametricPolicyBase."""

    def __init__(
        self,
        fn: Callable[[Any, Dict[str, float]], Any],
        action_space: Box,
        init_params: Dict[str, float],
        param_bounds: Dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if param_bounds is None:
            param_bounds = {k: (-100.0, 100.0) for k in init_params.keys()}
        super().__init__(init_params, param_bounds)
        self._fn = fn
        self._action_space = action_space

    def act(self, obs: Any) -> np.ndarray:
        out = self._fn(obs, self._params)
        if not isinstance(out, np.ndarray):
            out = np.asarray(out, dtype=self._action_space.dtype)
        out = out.reshape(self._action_space.shape)
        low = np.asarray(self._action_space.low, dtype=self._action_space.dtype)
        high = np.asarray(self._action_space.high, dtype=self._action_space.dtype)
        out = np.clip(out, low, high)
        return out


def default_single_shot_prompt(
    environment_description: str,
    action_space: Space[_ActT],
    example_observation: _ObsT,
) -> str:
    """Prompt that asks for a short, policy function."""
    return f"""
You will write a concise PARAMETRIC control policy for a gymnasium environment task.

Return ONLY a Python function named `policy` with EXACT format:

```python
def policy(obs, params):
    \"\"\"Return an action compatible with the action_space.
    - obs: one environment observation (np array or list).
    - params: dict[str, float] with controller gains/constants (e.g., 'kp', 'kd').
    \"\"\"
    # The function MUST be self-contained: add `import numpy as np` inside the function
    # before using NumPy. Do not rely on external imports.
    # your code here
```

**Requirements**
- Use values in `params` (e.g., 'kp', 'kd') to compute the action.
- The action MUST be valid for the provided action_space (shape & bounds).
- Be numerically safe: avoid NaNs/inf; clip to bounds.
- Keep it short; include `import numpy as np` *inside* the function.
- Do NOT include any text besides the function; no examples or prints.

**Environment (natural language)**
{environment_description}

**Action space (repr)**
{action_space}

**Example observation**
{example_observation}
""".strip()


def synthesize_llm_parametric_policy(
    environment_description: str,
    action_space: Space[_ActT],
    llm: PretrainedLargeModel,
    example_observation: _ObsT,
    init_params: Dict[str, float] | None = None,
    param_bounds: Dict[str, tuple[float, float]] | None = None,
) -> ParametricPolicyBase:
    """One-prompt LLM synthesis and ParametricPolicyBase adapter."""
    if init_params is None:
        init_params = {"kp": 10.0, "kd": 1.0, "ki": 0.0}

    prompt = default_single_shot_prompt(
        environment_description, action_space, example_observation
    )
    function_name = "policy"
    reprompt_checks: list = []
    generated_fn = synthesize_python_function_with_llm(
        function_name,
        llm,
        Query(prompt),
        reprompt_checks=reprompt_checks,
    )
    return LLMGeneratedParametricPolicy(
        fn=generated_fn,
        action_space=cast(Box, action_space),
        init_params=init_params,
        param_bounds=param_bounds,
    )
