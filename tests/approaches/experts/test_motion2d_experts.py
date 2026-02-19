"""Tests for experts/motion2d_experts.py."""

# pylint: disable=redefined-outer-name

from typing import Iterator

import gymnasium as gym
import kinder
import numpy as np
import pytest

from programmatic_policy_learning.approaches.experts.motion2d_experts import (
    Motion2DRejectionSamplingExpert,
    _find_next_passage,
    _is_y_aligned,
    create_motion2d_expert,
    f_1,
    f_2,
)

kinder.register_all_environments()

EnvObs = tuple[gym.Env, np.ndarray]


@pytest.fixture()
def p1_env_and_obs() -> Iterator[EnvObs]:
    """Create Motion2D-p1 env and return (env, initial_obs)."""
    env = kinder.make("kinder/Motion2D-p1-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    yield env, obs
    env.close()


# -- _is_y_aligned -----------------------------------------------------------


def test_is_y_aligned_inside() -> None:
    """Robot centered in gap is aligned."""
    assert _is_y_aligned(robot_y=1.0, robot_radius=0.1, gap_bottom=0.5, gap_top=1.5)


def test_is_y_aligned_at_boundary() -> None:
    """Robot exactly at passable boundary is still aligned."""
    assert _is_y_aligned(robot_y=0.6, robot_radius=0.1, gap_bottom=0.5, gap_top=1.5)
    assert _is_y_aligned(robot_y=1.4, robot_radius=0.1, gap_bottom=0.5, gap_top=1.5)


def test_is_y_aligned_outside() -> None:
    """Robot outside passable range is not aligned."""
    assert not _is_y_aligned(robot_y=0.3, robot_radius=0.1, gap_bottom=0.5, gap_top=1.5)
    assert not _is_y_aligned(robot_y=1.7, robot_radius=0.1, gap_bottom=0.5, gap_top=1.5)


def test_is_y_aligned_gap_too_narrow() -> None:
    """Gap narrower than robot diameter means alignment is impossible."""
    assert not _is_y_aligned(robot_y=1.0, robot_radius=0.5, gap_bottom=0.8, gap_top=1.2)


# -- _find_next_passage -------------------------------------------------------


def test_find_next_passage_with_p1_env(p1_env_and_obs: EnvObs) -> None:
    """Passage is found when robot is left of the wall."""
    _env, obs = p1_env_and_obs
    result = _find_next_passage(obs)
    assert result is not None
    wall_x, passage_y, gap_bottom, gap_top = result
    assert wall_x > 0
    assert gap_bottom < passage_y < gap_top


def test_find_next_passage_no_obstacles() -> None:
    """No passage returned when observation has zero obstacles."""
    obs = np.zeros(19, dtype=np.float32)
    assert _find_next_passage(obs) is None


def test_find_next_passage_robot_past_wall() -> None:
    """No passage returned when robot is already past all walls."""
    obs = np.zeros(39, dtype=np.float32)
    obs[0] = 10.0  # robot far to the right
    obs[19] = 0.5  # wall at x=0.5
    assert _find_next_passage(obs) is None


# -- f_1 & f_2 ---------------------------------------------------------------


def test_f1_true_when_no_passage() -> None:
    """f_1 is always True when there is no wall ahead."""
    obs = np.zeros(19, dtype=np.float32)
    action = np.zeros(5, dtype=np.float32)
    assert f_1(obs, action) is True


def test_f2_reduces_distance_to_target_when_no_passage() -> None:
    """f_2 accepts actions that reduce distance to target, rejects
    otherwise."""
    obs = np.zeros(19, dtype=np.float32)
    obs[0], obs[1] = 0.0, 0.0  # robot at origin
    obs[9], obs[10] = 1.0, 1.0  # target at (1,1)

    good_action = np.array([0.05, 0.05, 0, 0, 0], dtype=np.float32)
    bad_action = np.array([-0.05, -0.05, 0, 0, 0], dtype=np.float32)

    assert f_2(obs, good_action) is True
    assert f_2(obs, bad_action) is False


def test_f1_accepts_corrective_dy(p1_env_and_obs: EnvObs) -> None:
    """When robot is misaligned, f_1 accepts dy toward the passage."""
    _env, obs = p1_env_and_obs
    passage = _find_next_passage(obs)
    if passage is None:
        pytest.skip("No passage found in initial obs")

    _wall_x, passage_y, _gb, _gt = passage
    robot_y = float(obs[1])
    direction = 1.0 if passage_y > robot_y else -1.0

    action = np.zeros(5, dtype=np.float32)
    action[1] = direction * 0.05
    assert f_1(obs, action) is True


# -- Motion2DRejectionSamplingExpert -----------------------------------------


def test_expert_returns_5d_action(p1_env_and_obs: EnvObs) -> None:
    """Expert produces a 5-D float32 action."""
    env, obs = p1_env_and_obs
    assert isinstance(env.action_space, gym.spaces.Box)
    expert = create_motion2d_expert(env.action_space, seed=42)
    action = expert(obs)
    assert action.shape == (5,)
    assert action.dtype == np.float32


def test_expert_is_instance_of_class(p1_env_and_obs: EnvObs) -> None:
    """Factory returns the correct class."""
    env, _obs = p1_env_and_obs
    assert isinstance(env.action_space, gym.spaces.Box)
    expert = create_motion2d_expert(env.action_space)
    assert isinstance(expert, Motion2DRejectionSamplingExpert)


def test_expert_reaches_target(p1_env_and_obs: EnvObs) -> None:
    """Full rollout with the expert should reach the target."""
    env, obs = p1_env_and_obs
    assert isinstance(env.action_space, gym.spaces.Box)
    expert = create_motion2d_expert(env.action_space, seed=42)
    max_steps = 500

    for _ in range(max_steps):
        action = expert(obs)
        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break

    assert terminated, "Robot did not reach the target within max_steps"
