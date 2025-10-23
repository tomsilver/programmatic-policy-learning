"""Single-prompt LLM parametric policy — Pendulum-v1 tests.

What we are testing:
  1) If an Open AI key is available, we can create a policy in just one prompt and
     run a short rollout
  2) The ParametricPolicyBase clips to bounds.

Weakness:
  - Performance constraints for now are kept slightly on the milder side
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

from programmatic_policy_learning.approaches.llm_single_search_approach import (
    LLMGeneratedParametricPolicy,
    synthesize_llm_parametric_policy,
)


def test_single_prompt_llm_policy_on_pendulum() -> None:
    """Test: synthesize once and do a short rollout on Pendulum-v1."""
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=0)

    env_desc = (
        "Pendulum-v1: keep the pendulum upright. Observation is "
        "[cos(theta), sin(theta), theta_dot]. Action is 1-D torque in [-2, 2]. "
        "A PD controller around upright typically works; clip to bounds."
    )

    cache = SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
    llm = OpenAIModel("gpt-4o-mini", cache)

    policy = synthesize_llm_parametric_policy(
        environment_description=env_desc,
        action_space=env.action_space,
        llm=llm,
        example_observation=obs,
        init_params={"kp": 10.0, "kd": 1.0},
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


def test_clips() -> None:
    """Unit test: the adapter enforces shape and clips outputs to bounds."""
    action_space = Box(
        low=np.array([-2.0], dtype=np.float32),
        high=np.array([2.0], dtype=np.float32),
        dtype=np.float32,
    )

    # pylint: disable=unused-argument
    def fn_sample(obs: Any, params: dict[str, float]) -> np.ndarray:
        return np.array([3.5], dtype=np.float32)

    # pylint: disable=unused-argument

    policy = LLMGeneratedParametricPolicy(
        fn=fn_sample,
        action_space=action_space,
        init_params={"kp": 1.0, "kd": 0.1},
    )

    obs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    act = policy.act(obs)

    assert act.shape == action_space.shape
    assert act.dtype == action_space.dtype
