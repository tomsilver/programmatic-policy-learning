"""Tests for agentic_integrated_approach.py with Motion2D environment."""

# pylint: disable=redefined-outer-name,protected-access

import json
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import gymnasium as gym
import kinder
import numpy as np
import pytest
from gymnasium.envs.registration import register, registry
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

from programmatic_policy_learning.approaches.agentic_integrated_approach import (
    BIRRT_INIT_DOC,
    BIRRT_PLANNER_DOC,
    AgenticIntegratedApproach,
    score_policy_motion2d,
)

runllms = pytest.mark.skipif("not config.getoption('runllms')")

_MOTION2D_P1 = "kinder/Motion2D-p1-v0"
_MOTION2D_P2 = "kinder/Motion2D-p2-v0"

MOTION2D_ENVIRONMENT_DESCRIPTION = """
    A 2D continuous motion planning environment (Motion2D from the KinDER
    benchmark).

    WORLD:
        - A 2.5 x 2.5 continuous world.
        - A circular robot starts on the left side, and a rectangular
          target region is on the right side.
        - Between the robot and target are vertical obstacle columns
          that span the environment from left to right.

    IMPORTANT:
        Each obstacle column is composed of TWO axis-aligned rectangles:
            (1) a bottom rectangle extending upward from the bottom
            (2) a top rectangle extending downward from the top
        These two rectangles share the same x-position and width, and
        together form a vertical wall with a single open gap.

        The only traversable region through each wall column is this
        vertical gap.

    OBSERVATION:
        - The observation is a flat numpy array (float32).

        - Robot:
            - obs[0], obs[1]: robot x, y position
            - obs[2]: robot theta (orientation in radians)
            - obs[3]: robot base radius (typically ~0.1)
            - The robot is a circle. The full circular body must fit
              through any passage.

        - Target region:
            - obs[9], obs[10]: bottom-left corner (x, y)
            - obs[17], obs[18]: width and height

        - Obstacles:
            - Starting from obs[19], obstacles are listed sequentially
            - Each obstacle has 10 values
            - For passage i:
                - bottom obstacle starts at index 19 + 20*i
                - top obstacle starts at index 19 + 20*i + 10
            - Gap lies between:
                    (bottom_y + bottom_height) and (top_y)

    ACTIONS:
        - A 5-dimensional continuous action:
        - action[0]: dx in [-0.05, 0.05]
        - action[1]: dy in [-0.05, 0.05]
        - action[2]: dtheta in [-pi/16, pi/16]
        - action[3]: darm in [-0.1, 0.1] (not needed, set to 0)
        - action[4]: vacuum in [0, 1] (not needed, set to 0)
        - The action array must be dtype float32.

    TASK:
        - Move the robot to the target region while avoiding obstacles.
        - Success occurs when the robot center lies inside the target
          region.

    REWARD:
        -1.0 per step until success.
"""


@pytest.fixture(scope="module", autouse=True)
def _register_envs() -> None:
    if _MOTION2D_P1 not in registry:
        register(
            id=_MOTION2D_P1,
            entry_point="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
            kwargs={"num_passages": 1},
        )
    if _MOTION2D_P2 not in registry:
        register(
            id=_MOTION2D_P2,
            entry_point="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
            kwargs={"num_passages": 2},
        )


def _make_env(env_id: str) -> gym.Env:
    return kinder.make(env_id, allow_state_access=True)


def _get_oc_state(inner_env: Any, obs: Any) -> Any:
    """Return ObjectCentricState for *obs*."""
    inner_env.set_state(obs)
    return inner_env._object_centric_env.get_state()


def _get_robot(inner_env: Any, obs: Any) -> Any:
    """Return the robot Object from an observation."""
    # pylint: disable=import-outside-toplevel
    from kinder.envs.kinematic2d.object_types import CRVRobotType

    state = _get_oc_state(inner_env, obs)
    robots = state.get_objects(CRVRobotType)
    if not robots:
        raise ValueError("No robot found.")
    return robots[0]


def _build_planner_context(env: gym.Env, obs: np.ndarray) -> dict[str, Any]:
    """Build the BiRRT planner context dict for the approach."""
    inner = env.unwrapped
    robot = _get_robot(inner, obs)
    return {
        "get_object_centric_state": (lambda o, _i=inner: _get_oc_state(_i, o)),
        "robot": robot,
        "action_space": env.action_space,
    }


def _make_score_fn(env_id: str, seed: int) -> partial:
    """Create a score_fn that runs a full gym episode."""

    def _env_factory(_env_id: str = env_id, _seed: int = seed) -> Any:
        env = kinder.make(_env_id, allow_state_access=True)
        env.reset(seed=_seed)
        return env

    return partial(
        score_policy_motion2d,
        env_factory=_env_factory,
    )


@pytest.fixture()
def p1_env() -> Iterator[tuple[gym.Env, np.ndarray, dict]]:
    """Create a 1-passage Motion2D env, reset, yield."""
    env = _make_env(_MOTION2D_P1)
    obs, info = env.reset(seed=42)
    yield env, obs, info
    env.close()  # type: ignore[no-untyped-call]


# ------------------------------------------------------------------
# Real LLM test
# ------------------------------------------------------------------


@runllms
def test_agentic_integrated_motion2d_with_real_llm() -> None:
    """AgenticIntegratedApproach generates and evaluates candidate policies on
    a Motion2D environment using a real LLM."""
    # Set up birrt metrics
    metrics_file = Path(tempfile.NamedTemporaryFile(suffix=".json", delete=False).name)
    os.environ["birrt_metrics_path"] = str(metrics_file)

    cache = SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
    llm = OpenAIModel("gpt-5.2", cache)

    # Training env: p=1
    train_env = _make_env(_MOTION2D_P1)
    train_obs, train_info = train_env.reset(seed=42)
    planner_context = _build_planner_context(train_env, train_obs)
    score_fn = _make_score_fn(_MOTION2D_P1, 42)

    approach = AgenticIntegratedApproach(
        MOTION2D_ENVIRONMENT_DESCRIPTION,
        train_env.observation_space,
        train_env.action_space,
        seed=0,
        llm=llm,
        planner_context=planner_context,
        planner_doc=BIRRT_PLANNER_DOC,
        init_doc=BIRRT_INIT_DOC,
        score_fn=score_fn,
        num_candidates=4,
        scoring_max_timesteps=300,
    )

    # TRAINING: synthesize + score candidates
    approach.reset(train_obs, train_info)
    train_env.close()  # type: ignore[no-untyped-call]

    assert approach._best_code, "No best code selected"
    assert len(approach._all_candidate_codes) == 4
    print(f"Best policy:\n{approach._best_code}")

    # EVALUATION: different env instance (p=1, different seed)
    eval_env = _make_env(_MOTION2D_P1)
    eval_obs, eval_info = eval_env.reset(seed=99)

    eval_context = _build_planner_context(eval_env, eval_obs)
    approach.update_planner_context(eval_context)

    # Clear metrics before evaluation
    metrics_file.write_text("", encoding="utf-8")
    approach.reset(eval_obs, eval_info)

    goal_reached = False
    for step in range(500):
        action = approach.step()
        assert eval_env.action_space.contains(action)
        eval_obs, reward, terminated, _, _ = eval_env.step(action)
        approach.update(eval_obs, float(reward), terminated, {})
        if terminated:
            goal_reached = True
            break

    # Read BiRRT metrics
    num_checks = 0
    num_nodes = 0
    if metrics_file.exists():
        with metrics_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    num_checks += entry["num_collision_checks"]
                    num_nodes += entry["num_nodes_extended"]

    print(f"Goal reached: {goal_reached}")
    print(f"Steps: {step + 1}")
    print(f"Collision checks: {num_checks}")
    print(f"Nodes extended: {num_nodes}")

    eval_env.close()  # type: ignore[no-untyped-call]
    metrics_file.unlink(missing_ok=True)
