"""Tests for the LLM + grid-search approach.

Includes:
  - Fake LLM test (CI-safe)
  - Optional real LLM test (guarded by @runllms)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel, OrderedResponseModel
from prpl_llm_utils.structs import Response

from programmatic_policy_learning.approaches.llm_grid_search_approach import (
    synthesize_and_grid_search,
)

runllms = pytest.mark.skipif("not config.getoption('runllms')")


def test_llm_grid_search_fake() -> None:
    """
    The policy follows the interface and uses params['kp'] 
    """
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=0)
    env.close()

    env_desc = (
        "Pendulum-v1: keep upright. obs=[cos(theta), sin(theta), theta_dot]. "
        "Action is 1-D torque in [-2, 2]. PD around upright is fine; clip."
    )

    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)

    fake_llm = OrderedResponseModel(
        [
            Response(
                "```python\n"
                "def policy(obs, params):\n"
                "    import numpy as np\n"
                "    c, s, d = float(obs[0]), float(obs[1]), float(obs[2])\n"
                "    theta = np.arctan2(s, c)\n"
                "    kp = float(params.get('kp', 10.0))\n"
                "    kd = float(params.get('kd', 1.0))\n"
                "    u = -kp*theta - kd*d\n"
                "    u = np.clip(u, -2.0, 2.0)\n"
                "    return np.array([u], dtype=np.float32)\n"
                "```",
                {},
            )
        ],
        cache,
    )

    tuned_policy, best_kp, best_avg = synthesize_and_grid_search(
        env_factory=lambda: gym.make("Pendulum-v1"),
        environment_description=env_desc,
        action_space=Box(low=np.array([-2.0], dtype=np.float32),
                        high=np.array([2.0], dtype=np.float32),
                        dtype=np.float32),
        llm=fake_llm,
        example_observation=obs,
        param_name="kp",
        param_bounds=(2.0, 12.0),
        num_points=5,
        steps=150,
        episodes=3,
        fixed_params={"kd": 1.5},
        init_params={"kp": 6.0, "kd": 1.5},
        param_bounds_all={"kp": (0.0, 50.0), "kd": (0.0, 10.0)},
    )
    assert isinstance(best_kp, float)
    assert np.isfinite(best_avg)
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=1)
    act = tuned_policy.act(obs)
    assert env.action_space.contains(act)
    env.close()


@runllms
def test_llm_grid_search_real_llm() -> None:
    """Optional: use a real LLM if --runllms is enabled."""
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=0)
    env.close()

    env_desc = (
        "Pendulum-v1: keep the pendulum upright. Observation is "
        "[cos(theta), sin(theta), theta_dot]. Action is 1-D torque in [-2, 2]."
    )

    cache = SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
    llm = OpenAIModel("gpt-4o-mini", cache)

    tuned_policy, best_kp, best_avg = synthesize_and_grid_search(
        env_factory=lambda: gym.make("Pendulum-v1"),
        environment_description=env_desc,
        action_space=Box(low=np.array([-2.0], dtype=np.float32),
                        high=np.array([2.0], dtype=np.float32),
                        dtype=np.float32),
        llm=llm,
        example_observation=obs,
        param_name="kp",
        param_bounds=(2.0, 12.0),
        num_points=5,
        steps=150,
        episodes=3,
        fixed_params={"kd": 1.5},
        init_params={"kp": 6.0, "kd": 1.5},
        param_bounds_all={"kp": (0.0, 50.0), "kd": (0.0, 10.0)},
    )


    assert np.isfinite(best_avg)
    assert best_avg > -1900.0

    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=2)
    act = tuned_policy.act(obs)
    assert env.action_space.contains(act)
    env.close()
