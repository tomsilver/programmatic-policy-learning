"""Iterative-refinement tests for Pendulum-v1.

What we are testing:
  1) Refinement loop reprompts after a failing and succeeds on a later attempt.
  2) The resulting policy produces valid actions.

We include a "fake LLM" test and a real-LLM test 
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel, OrderedResponseModel
from prpl_llm_utils.structs import Response

from programmatic_policy_learning.approaches.llm_refine_search_approach import (
    synthesize_llm_parametric_policy_refine,
)

runllms = pytest.mark.skipif("not config.getoption('runllms')")


def test_iterative_refine_on_pendulum_fake() -> None:
    """Fake LLM test to verify the refinement cycle repairs a bad first
    policy."""
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=0)

    env_desc = (
        "Pendulum-v1: keep the pendulum upright. Observation is "
        "[cos(theta), sin(theta), theta_dot]. Action is 1-D torque in [-2, 2]."
    )

    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)

    bad = (
        "```python\n"
        "def policy(obs, params):\n"
        "    # Missing import; will raise NameError\n"
        "    return np.array([0.0], dtype=np.float32)\n"
        "```"
    )
    good = (
        "```python\n"
        "def policy(obs, params):\n"
        "    import numpy as np\n"
        "    # Simple bounded output; adapter will clip as needed\n"
        "    u = np.array([0.0], dtype=np.float32)\n"
        "    return u\n"
        "```"
    )

    llm = OrderedResponseModel(
        [Response(bad, {}), Response(good, {})],
        cache,
    )

    policy = synthesize_llm_parametric_policy_refine(
        environment_description=env_desc,
        action_space=env.action_space,
        llm=llm,
        example_observation=obs,
        init_params={"kp": 10.0, "kd": 1.0},
        max_iters=3,
    )

    total_return = 0.0
    for _ in range(10):
        act = policy.act(obs)
        assert env.action_space.contains(act)
        obs, r, term, trunc, _ = env.step(act)
        total_return += float(r)
        if term or trunc:
            break
    assert np.isfinite(total_return)


@runllms
def test_iterative_refine_on_pendulum_real() -> None:
    """Real LLM test (gated with --runllms) that should pass basic sanity +
    mild perf."""
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=0)

    env_desc = (
        "Pendulum-v1: keep the pendulum upright. Observation is "
        "[cos(theta), sin(theta), theta_dot]. Action is 1-D torque in [-2, 2]. "
        "A PD controller around upright typically works; clip to bounds."
    )

    cache = SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
    llm = OpenAIModel("gpt-4o-mini", cache)

    policy = synthesize_llm_parametric_policy_refine(
        environment_description=env_desc,
        action_space=env.action_space,
        llm=llm,
        example_observation=obs,
        init_params={"kp": 10.0, "kd": 1.0},
        max_iters=4,
    )

    total_return = 0.0
    for _ in range(150):
        act = policy.act(obs)
        assert env.action_space.contains(act)
        obs, r, term, trunc, _ = env.step(act)
        total_return += float(r)
        if term or trunc:
            break

    assert np.isfinite(total_return)
    assert total_return > -1900.0
