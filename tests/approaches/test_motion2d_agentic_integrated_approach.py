"""Tests for agentic_integrated_approach.py with Motion2D environment."""

import os
from pathlib import Path

import gymnasium as gym
import kinder
import numpy as np
import pytest
from gymnasium.envs.registration import register, registry
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

from programmatic_policy_learning.approaches.agentic_integrated_approach import (
    BIRRT_PLANNER_DOC,
    AgenticIntegratedApproach,
)

runllms = pytest.mark.skipif("not config.getoption('runllms')")

_MOTION2D_P1 = "kinder/Motion2D-p1-v0"
_MOTION2D_P3 = "kinder/Motion2D-p3-v0"


@pytest.fixture(scope="module", autouse=True)
def _register_envs() -> None:
    if _MOTION2D_P1 not in registry:
        register(
            id=_MOTION2D_P1,
            entry_point="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
            kwargs={"num_passages": 1},
        )
    if _MOTION2D_P3 not in registry:
        register(
            id=_MOTION2D_P3,
            entry_point="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
            kwargs={"num_passages": 3},
        )


def _make_motion2d_env(
    env_id: str,
) -> gym.Env:
    """Create a Motion2D env with state access enabled."""
    return kinder.make(env_id, render_mode="rgb_array", allow_state_access=True)


def _check_terminated_motion2d(state: np.ndarray) -> bool:
    """Check if the robot center is inside the target region.

    Target region is an axis-aligned rectangle with bottom-left at
    (state[9], state[10]) and size (state[17], state[18]).
    """
    robot_x, robot_y = float(state[0]), float(state[1])
    target_x, target_y = float(state[9]), float(state[10])
    target_w, target_h = float(state[17]), float(state[18])
    return (
        target_x <= robot_x <= target_x + target_w
        and target_y <= robot_y <= target_y + target_h
    )


MOTION2D_ENVIRONMENT_DESCRIPTION = """
    A 2D continuous motion planning environment (Motion2D from the KinDER
    benchmark).

    WORLD:
    - A 2.5 x 2.5 continuous world with a circular robot on the left side
      and a rectangular target region on the right side.
    - Between the robot and target there are vertical wall obstacles with
      narrow passages. The number of walls varies across instances.

    OBSERVATION:
    The observation is a flat numpy array (float32). Key indices:
    - obs[0], obs[1]: robot x, y position
    - obs[2]: robot theta (orientation in radians)
    - obs[3]: robot base_radius (circular robot radius, ~0.1)
    - obs[9], obs[10]: target region x, y (bottom-left corner)
    - obs[17], obs[18]: target region width, height
    - obs[19+]: obstacle features, 10 per obstacle
      (x, y, theta, static, color_r, color_g, color_b, z_order, width, height)
    - Each wall passage consists of 2 obstacles (bottom wall segment and
      top wall segment). For passage i:
        bottom obstacle at index 19 + 20*i
        top obstacle at index 19 + 20*i + 10
      The gap (passage) is between bottom.y + bottom.height and top.y.

    ACTIONS:
    A 5-dimensional continuous numpy array:
    - action[0]: dx (change in x), range roughly [-0.05, 0.05]
    - action[1]: dy (change in y), range roughly [-0.05, 0.05]
    - action[2]: dtheta (change in angle), range roughly [-pi/16, pi/16]
    - action[3]: darm (change in arm length) — not needed for navigation
    - action[4]: vacuum on/off — not needed for navigation
    Only action[0] (dx) and action[1] (dy) are needed for navigation.
    Set action[2], action[3], action[4] to 0.

    TASK:
    Navigate the robot from its starting position to the target region.
    The episode terminates (success) when the robot center (obs[0], obs[1])
    is inside the target rectangle:
        obs[9] <= obs[0] <= obs[9] + obs[17]
        obs[10] <= obs[1] <= obs[10] + obs[18]

    The reward is -1.0 per step until success. The info dict is empty.
"""


@runllms
def test_agentic_integrated_approach_motion2d_with_real_llm() -> None:
    """Tests AgenticIntegratedApproach on Motion2D-p1 (1 passage)."""
    # --- TRAINING PHASE ---
    train_env = _make_motion2d_env(_MOTION2D_P1)
    train_inner = train_env.unwrapped
    obs, info = train_env.reset(seed=42)

    os.environ["astar_metrics_path"] = str(
        Path(__file__).parents[2]
        / "src"
        / "programmatic_policy_learning"
        / "metrics"
        / "astar_metrics.json"
    )

    cache_path = Path("llm_cache.db")
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm = OpenAIModel("gpt-5.2", cache)

    approach = AgenticIntegratedApproach(
        MOTION2D_ENVIRONMENT_DESCRIPTION,
        train_env.observation_space,
        train_env.action_space,
        seed=123,
        llm=llm,
        num_candidates=5,
        scoring_max_timesteps=500,
        planner_doc=BIRRT_PLANNER_DOC,
        env_callables={
            "get_next_state": train_inner.get_next_state,
            "action_space": train_env.action_space,
            "check_terminated": _check_terminated_motion2d,
        },
        check_terminated=_check_terminated_motion2d,
    )

    approach.reset(obs, info)
    train_env.close()

    # --- EVALUATION PHASE (different instance: 3 passages) ---
    eval_env = _make_motion2d_env(_MOTION2D_P3)
    eval_inner = eval_env.unwrapped
    obs, info = eval_env.reset(seed=42)

    approach.update_env_callables(
        get_next_state=eval_inner.get_next_state,
        action_space=eval_env.action_space,
        check_terminated=_check_terminated_motion2d,
    )
    approach.reset(obs, info)

    goal_reached = False
    for _ in range(500):
        action = approach.step()
        assert eval_env.action_space.contains(action)
        obs, reward, terminated, _, info = eval_env.step(action)
        approach.update(obs, float(reward), terminated, info)
        if terminated:
            goal_reached = True
            break

    print("Goal reached:", goal_reached)
    eval_env.close()
