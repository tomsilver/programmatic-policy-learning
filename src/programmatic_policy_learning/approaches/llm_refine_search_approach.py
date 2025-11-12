"""Iterative-refinement LLM synthesis that returns a ParametricPolicyBase
instance.

- Multiple refinement attempts.
- On each attempt, if the generated policy fails a sanity check, we add on the
  feedback to the next prompt and re-synthesize.
- Produces a standard ParametricPolicyBase so we can use .act(obs).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, TypeVar, cast

import numpy as np
from gymnasium.spaces import Box, Space
from prpl_llm_utils.code import synthesize_python_function_with_llm
from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.approaches.structs import ParametricPolicyBase

_ObsT = TypeVar("_ObsT")
_ActT = TypeVar("_ActT")


class LLMGeneratedParametricPolicy(ParametricPolicyBase):
    """Adapter around an LLM-generated function."""

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

    @property
    def function(self) -> Callable[[Any, Dict[str, float]], Any]:
        """Callable self function."""
        return self._fn

    @property
    def action_space(self) -> Box:
        """Action space function."""
        return self._action_space

    def act(self, obs: Any) -> np.ndarray:
        out = self._fn(obs, self._params)
        if not isinstance(out, np.ndarray):
            out = np.asarray(out, dtype=self._action_space.dtype)
        out = out.reshape(self._action_space.shape)
        low = np.asarray(self._action_space.low, dtype=self._action_space.dtype)
        high = np.asarray(self._action_space.high, dtype=self._action_space.dtype)
        return np.clip(out, low, high)


def default_single_shot_prompt(
    environment_description: str,
    action_space: Space[_ActT],
    example_observation: _ObsT,
) -> str:
    """Default single shot prompt."""
    return f"""
You will write a concise parametric control policy for a gymnasium environment task.

Return ONLY a Python function named `policy` with EXACT format:

```python
def policy(obs, params):
    Return an action compatible with the action_space.
    - obs: one environment observation (np array or list).
    - params: dict[str, float] with controller gains/constants (e.g., 'kp', 'kd').
    
    # The function MUST be self-contained: add `import numpy as np` inside the function
    # before using NumPy. Do not rely on external imports.
```

Requirements:
- Use values in `params`.
- Action must match action_space shape & bounds.
- Avoid NaN/Inf, clip to bounds.
- No explanations, just the function.

Environment:
{environment_description}

Action space:
{action_space}

Example observation:
{example_observation}
""".strip()


def addfeedback(base_prompt: str, feedback: List[str]) -> str:
    if not feedback:
        return base_prompt
    fixes = "\n".join(f"- {msg}" for msg in feedback)
    return (
        base_prompt
        + "\n\n### Fix the following issues from the previous attempt:\n"
        + fixes
        + "\n\nReturn ONLY the corrected `policy` function."
    )


def checkingstep(
    candidate: LLMGeneratedParametricPolicy,
    example_obs: Any,
) -> Tuple[bool, str | None]:
    """Run a cheap, safe check that never raises.

    Returns (ok, err_msg).
    """
    try:
        act = candidate.act(example_obs)
    except BaseException as e:  # pylint: disable=broad-exception-caught
        return False, f"Policy call failed: {type(e).__name__}: {e}"

    space = candidate.action_space

    if act.shape != space.shape:
        return False, f"Wrong action shape {act.shape}, expected {space.shape}"

    if act.dtype != space.dtype:
        return False, f"Wrong dtype {act.dtype}, expected {space.dtype}"

    if not np.all(np.isfinite(act)):
        return False, "Action has NaN or Inf."

    return True, None


def synthesize_llm_parametric_policy_refine(
    environment_description: str,
    action_space: Space[_ActT],
    llm: PretrainedLargeModel,
    example_observation: _ObsT,
    init_params: Dict[str, float] | None = None,
    param_bounds: Dict[str, tuple[float, float]] | None = None,
    max_iters: int = 4,
) -> ParametricPolicyBase:
    """Synthesizing llm parametric policy after refinement."""
    if init_params is None:
        init_params = {"kp": 10.0, "kd": 1.0}

    base_prompt = default_single_shot_prompt(
        environment_description, action_space, example_observation
    )
    feedback: List[str] = []
    best_candidate: LLMGeneratedParametricPolicy | None = None
    last_candidate: LLMGeneratedParametricPolicy | None = None

    for _ in range(max_iters):
        prompt = addfeedback(base_prompt, feedback)

        generated_fn = synthesize_python_function_with_llm(
            "policy",
            llm,
            Query(prompt),
            reprompt_checks=[],
        )

        candidate = LLMGeneratedParametricPolicy(
            fn=generated_fn,
            action_space=cast(Box, action_space),
            init_params=init_params,
            param_bounds=param_bounds,
        )

        ok, err = checkingstep(candidate, example_observation)
        if ok:
            return candidate

        feedback.append(err or "Unknown failure")
        best_candidate = candidate

    return best_candidate or last_candidate  # type: ignore[return-value]
