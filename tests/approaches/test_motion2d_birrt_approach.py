"""Tests for motion2d_birrt_approach.py."""

from pathlib import Path
from typing import Any, Iterator

import gymnasium as gym
import kinder
import numpy as np
import pytest
from gymnasium.envs.registration import register, registry

from programmatic_policy_learning.approaches.motion2d_birrt_approach import (
    Motion2DBiRRTApproach,
)

_MOTION2D_P1 = "kinder/Motion2D-p1-v0"
_MOTION2D_P2 = "kinder/Motion2D-p2-v0"


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
    return kinder.make(env_id, render_mode="rgb_array", allow_state_access=True)


def _get_oc_state(inner_env: Any, obs: Any) -> Any:
    """Return the ObjectCentricState for *obs* by setting state on the inner
    env."""
    inner_env.set_state(obs)
    return inner_env._object_centric_env.get_state()


@pytest.fixture()
def p1_env() -> Iterator[tuple[gym.Env, np.ndarray, dict]]:
    env = _make_env(_MOTION2D_P1)
    obs, info = env.reset(seed=42)
    yield env, obs, info
    env.close()


# ---------------------------------------------------------------------------
# Construction and basic plan properties
# ---------------------------------------------------------------------------


def test_approach_constructs_and_plans(
    p1_env: tuple[gym.Env, np.ndarray, dict],
) -> None:
    """Approach builds a non-empty plan and metrics are populated after
    reset."""
    env, obs, info = p1_env
    inner = env.unwrapped
    approach = Motion2DBiRRTApproach(
        environment_description="test",
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=42,
        get_object_centric_state=lambda o: _get_oc_state(inner, o),
        num_attempts=20,
        num_iters=500,
        smooth_amt=50,
    )
    approach.reset(obs, info)

    assert len(approach._plan) > 0, "Expected a non-empty plan"
    assert approach.metrics is not None
    assert approach.metrics.num_collision_checks > 0
    assert approach.metrics.num_nodes_extended > 0


def test_first_action_is_valid(
    p1_env: tuple[gym.Env, np.ndarray, dict],
) -> None:
    """Each action from the plan is within the action space."""
    env, obs, info = p1_env
    inner = env.unwrapped
    approach = Motion2DBiRRTApproach(
        environment_description="test",
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=42,
        get_object_centric_state=lambda o: _get_oc_state(inner, o),
    )
    approach.reset(obs, info)
    action = approach.step()
    assert env.action_space.contains(action)


# ---------------------------------------------------------------------------
# Plan replay
# ---------------------------------------------------------------------------


def test_plan_is_consumed_in_order(
    p1_env: tuple[gym.Env, np.ndarray, dict],
) -> None:
    """Each call to step() pops the first action from the plan."""
    env, obs, info = p1_env
    inner = env.unwrapped
    approach = Motion2DBiRRTApproach(
        environment_description="test",
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=42,
        get_object_centric_state=lambda o: _get_oc_state(inner, o),
    )
    approach.reset(obs, info)

    initial_len = len(approach._plan)
    assert initial_len > 0
    approach.step()
    assert len(approach._plan) == initial_len - 1


def test_empty_plan_raises(
    p1_env: tuple[gym.Env, np.ndarray, dict],
) -> None:
    """Calling step() on an exhausted plan raises ValueError."""
    env, obs, info = p1_env
    inner = env.unwrapped
    approach = Motion2DBiRRTApproach(
        environment_description="test",
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=42,
        get_object_centric_state=lambda o: _get_oc_state(inner, o),
    )
    approach.reset(obs, info)
    approach._plan.clear()

    with pytest.raises(ValueError, match="Plan is empty"):
        approach.step()


def test_reset_clears_previous_plan() -> None:
    """Calling reset() a second time replaces the old plan."""
    env = _make_env(_MOTION2D_P1)
    inner = env.unwrapped
    obs, info = env.reset(seed=3)

    approach = Motion2DBiRRTApproach(
        environment_description="test",
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=3,
        get_object_centric_state=lambda o: _get_oc_state(inner, o),
    )
    approach.reset(obs, info)

    for _ in range(min(5, len(approach._plan))):
        approach.step()
    old_metrics = approach.metrics

    obs2, info2 = env.reset(seed=4)
    approach.reset(obs2, info2)

    assert len(approach._plan) > 0, "Plan should be repopulated after second reset"
    assert (
        approach.metrics is not old_metrics
    ), "Metrics should be refreshed after reset"
    env.close()


# ---------------------------------------------------------------------------
# Full rollout — goal reached
# ---------------------------------------------------------------------------


def test_birrt_reaches_goal_p1() -> None:
    """BiRRT approach reaches the target on a 1-passage instance."""
    env = _make_env(_MOTION2D_P1)
    inner = env.unwrapped
    obs, info = env.reset(seed=42)

    approach = Motion2DBiRRTApproach(
        environment_description="test",
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=42,
        get_object_centric_state=lambda o: _get_oc_state(inner, o),
        num_attempts=20,
        num_iters=500,
        smooth_amt=50,
    )
    approach.reset(obs, info)

    goal_reached = False
    for _ in range(1000):
        action = approach.step()
        assert env.action_space.contains(action)
        obs, reward, terminated, _, _ = env.step(action)
        approach.update(obs, float(reward), terminated, {})
        if terminated:
            goal_reached = True
            break

    assert goal_reached, "BiRRT did not reach the target within 1000 steps (p1)"
    env.close()


def test_birrt_reaches_goal_p2() -> None:
    """BiRRT approach reaches the target on a 2-passage instance."""
    env = _make_env(_MOTION2D_P2)
    inner = env.unwrapped
    obs, info = env.reset(seed=7)

    approach = Motion2DBiRRTApproach(
        environment_description="test",
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=7,
        get_object_centric_state=lambda o: _get_oc_state(inner, o),
        num_attempts=20,
        num_iters=500,
        smooth_amt=50,
    )
    approach.reset(obs, info)

    goal_reached = False
    for _ in range(1500):
        action = approach.step()
        assert env.action_space.contains(action)
        obs, reward, terminated, _, _ = env.step(action)
        approach.update(obs, float(reward), terminated, {})
        if terminated:
            goal_reached = True
            break

    assert goal_reached, "BiRRT did not reach the target within 1500 steps (p2)"
    env.close()


# ---------------------------------------------------------------------------
# Visual / metrics test  (run with:  pytest --runvisual -s)
# ---------------------------------------------------------------------------

runvisual = pytest.mark.skipif("not config.getoption('--runvisual')")


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    from moviepy import ImageSequenceClip  # type: ignore[import-untyped]

    clean = [f[:, :, :3] if f.ndim == 3 and f.shape[2] == 4 else f for f in frames]
    clip = ImageSequenceClip(clean, fps=fps)
    clip.write_videofile(str(path), codec="libx264", logger=None)
    print(f"Video saved → {path}  ({len(clean)} frames)")


@runvisual
@pytest.mark.parametrize("passages,seed", [(1, 42), (2, 7)])
def test_birrt_render_and_metrics(passages: int, seed: int) -> None:
    """Full rollout: prints metrics table and saves a video.

    Run with:  pytest tests/approaches/test_motion2d_birrt_approach.py --runvisual -s
    """
    env_id = f"kinder/Motion2D-p{passages}-v0"
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
            kwargs={"num_passages": passages},
        )

    env = kinder.make(env_id, render_mode="rgb_array", allow_state_access=True)
    inner = env.unwrapped
    obs, info = env.reset(seed=seed)

    approach = Motion2DBiRRTApproach(
        environment_description=f"Motion2D-p{passages}",
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=seed,
        get_object_centric_state=lambda o: _get_oc_state(inner, o),
        num_attempts=20,
        num_iters=500,
        smooth_amt=50,
    )
    approach.reset(obs, info)

    plan_len = len(approach._plan)
    metrics = approach.metrics

    frames: list[np.ndarray] = []
    total_reward = 0.0
    steps_taken = 0
    goal_reached = False

    for step in range(1000):
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

        action = approach.step()
        obs, reward, terminated, _, _ = env.step(action)
        approach.update(obs, float(reward), terminated, {})
        total_reward += float(reward)
        steps_taken = step + 1

        if terminated:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            goal_reached = True
            break

    env.close()

    print()
    print("=" * 50)
    print(f"  Env:               {env_id}  (seed={seed})")
    print(f"  BiRRT plan length: {plan_len} actions")
    print(f"  Collision checks:  {metrics.num_collision_checks}")
    print(f"  Nodes extended:    {metrics.num_nodes_extended}")
    print(f"  Goal reached:      {goal_reached}")
    print(f"  Steps taken:       {steps_taken}")
    print(f"  Total reward:      {total_reward:.1f}")
    print("=" * 50)

    if frames:
        video_dir = Path("videos")
        video_dir.mkdir(exist_ok=True)
        _save_video(frames, video_dir / f"birrt_p{passages}_s{seed}.mp4")

    assert goal_reached, f"BiRRT did not reach goal on {env_id} (seed={seed})"
