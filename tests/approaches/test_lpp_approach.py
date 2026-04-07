"""Tests for LPP Approach."""

from pathlib import Path

import numpy as np
import pytest
from gymnasium.spaces import Box
from omegaconf import DictConfig, OmegaConf

from programmatic_policy_learning.approaches.expert_approach import ExpertApproach
from programmatic_policy_learning.approaches.experts.grid_experts import get_grid_expert
from programmatic_policy_learning.approaches.lpp_approach import (
    LogicProgrammaticPolicyApproach,
)
from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.envs.registry import EnvRegistry


class _DummyRecoveryExpert:
    def set_env(self, env) -> None:
        self._env = env

    def reset(self, obs, info) -> None:
        del obs, info

    def step(self):
        return np.asarray(self._env.expert_action, dtype=np.float32)


class _ConstantPolicy:
    def __init__(self, action: np.ndarray) -> None:
        self._action = np.asarray(action, dtype=np.float32)

    def __call__(self, obs):
        del obs
        return self._action.copy()


class _DummyRecoveryEnv:
    def __init__(
        self,
        *,
        obs_seq: list[np.ndarray],
        expert_action: np.ndarray,
    ) -> None:
        self._obs_seq = [np.asarray(obs, dtype=np.float32) for obs in obs_seq]
        self.expert_action = np.asarray(expert_action, dtype=np.float32)
        self._idx = 0

    def reset(self, seed=None):
        del seed
        self._idx = 0
        return self._obs_seq[0].copy(), {}

    def step(self, action):
        del action
        self._idx = min(self._idx + 1, len(self._obs_seq) - 1)
        terminated = self._idx >= len(self._obs_seq) - 1
        return self._obs_seq[self._idx].copy(), 0.0, terminated, False, {}

    def close(self) -> None:
        return None


def _make_motion2d_obs(
    robot_x: float,
    robot_y: float,
    *,
    target_x: float = 1.0,
    target_y: float = 1.0,
) -> np.ndarray:
    obs = np.zeros(19, dtype=np.float32)
    obs[0] = robot_x
    obs[1] = robot_y
    obs[9] = target_x
    obs[10] = target_y
    obs[17] = 0.0
    obs[18] = 0.0
    return obs


def _make_recovery_approach(env_factory):
    approach = LogicProgrammaticPolicyApproach(
        environment_description="Motion2D-p0-v0",
        observation_space=Box(
            low=-np.ones(19, dtype=np.float32),
            high=np.ones(19, dtype=np.float32) * 2.0,
            dtype=np.float32,
        ),
        action_space=Box(
            low=np.array([-0.05, -0.05, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([0.05, 0.05, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        ),
        seed=0,
        expert=_DummyRecoveryExpert(),  # type: ignore[arg-type]
        env_factory=env_factory,
        base_class_name="Motion2D-p0",
        demo_numbers=(0,),
        env_specs={"action_mode": "continuous"},
        recovery_augmentation={
            "enabled": True,
            "rounds": 1,
            "capture_deviation": True,
            "capture_stuck": True,
            "max_steps": 10,
            "stuck_window": 3,
            "min_progress_delta": 0.01,
            "stuck_position_delta": 0.01,
            "deviation_bucket_tolerance": 2,
            "deviation_require_sign_flip": False,
            "max_queries_per_env": 4,
        },
    )
    approach._negative_sampling_cfg = {
        "action_mode": "continuous",
        "action_low": np.array([-0.05, -0.05, -1.0, -1.0, -1.0], dtype=np.float32),
        "action_high": np.array([0.05, 0.05, 1.0, 1.0, 1.0], dtype=np.float32),
        "continuous": {
            "active_action_dims": [0, 1],
            "inactive_action_fill_value": 0.0,
            "bucket_edges": [
                [-0.05, -0.006, 0.0, 0.006, 0.05],
                [-0.05, -0.006, 0.0, 0.006, 0.05],
            ],
            "relaxed_labeling": {
                "enabled": True,
                "neighbor_radius": 2,
                "nearby_bucket_behavior": "ignore",
                "weak_negative_scale": 0.25,
            },
        },
    }
    return approach


@pytest.mark.skipif(
    not Path("src/programmatic_policy_learning/dsl/llm_primitives/outputs/*").exists(),
    reason="Required output files are missing",
)
def test_lpp_approach_real_data() -> None:
    """Test lpp approach with real_data."""
    cfg: DictConfig = OmegaConf.create(
        {
            "provider": "ggg",
            "make_kwargs": {
                "base_name": "TwoPileNim",
                "id": "TwoPileNim0-v0",
            },
            "instance_num": 0,
        }
    )
    registry = EnvRegistry()
    env = registry.load(cfg)
    env_id = cfg["make_kwargs"]["id"]
    expert_fn = get_grid_expert(env_id)
    expert = ExpertApproach(  # type: ignore
        env_id,  # env_description
        env.observation_space,
        env.action_space,
        seed=1,
        expert_fn=expert_fn,
    )
    base_class_name = cfg.make_kwargs.base_name

    # Define observation and action spaces
    observation_space = env.observation_space
    action_space = env.action_space

    # Environment specifications
    env_specs = {"object_types": env.get_object_types()}
    env_factory = lambda instance_num=None: env
    # Initialize the approach
    approach = LogicProgrammaticPolicyApproach(
        environment_description=env_id,
        observation_space=observation_space,
        action_space=action_space,
        seed=42,
        env_factory=env_factory,
        base_class_name=base_class_name,
        expert=expert,
        demo_numbers=(0, 1),
        num_programs=2,
        num_dts=1,
        max_num_particles=2,
        max_demo_length=100,
        env_specs=env_specs,
        start_symbol=0,
    )

    # Test reset and action
    obs = env.reset()[0]
    info = env.reset()[1]
    approach.reset(obs, info)
    # pylint: disable=protected-access
    action = approach._get_action()
    assert isinstance(action, tuple)
    assert len(action) == 2
    assert all(isinstance(x, int) for x in action)


def test_recovery_augmentation_adds_deviation_state_before_stuck() -> None:
    obs_seq = [
        _make_motion2d_obs(0.0, 0.0),
        _make_motion2d_obs(0.0, 0.0),
        _make_motion2d_obs(0.0, 0.0),
        _make_motion2d_obs(0.0, 0.0),
    ]
    expert_action = np.array([0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    learner_action = np.array([-0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    approach = _make_recovery_approach(
        env_factory=lambda instance_num=None: _DummyRecoveryEnv(
            obs_seq=obs_seq,
            expert_action=expert_action,
        )
    )

    aug_ids, aug_demo, aug_dict, changed = approach._augment_recovery_states(
        policy=_ConstantPolicy(learner_action),  # type: ignore[arg-type]
        train_demo_ids=(0,),
        demonstrations_train=Trajectory(steps=[]),
        demo_dict_train={0: Trajectory(steps=[])},
        negative_sampling_cfg=approach._negative_sampling_cfg,
    )

    assert changed is True
    assert aug_ids == (0, -1)
    assert len(aug_demo.steps) == 1
    np.testing.assert_allclose(aug_demo.steps[0][0], obs_seq[0])
    np.testing.assert_allclose(aug_demo.steps[0][1], expert_action)
    assert -1 in aug_dict
    assert len(aug_dict[-1].steps) == 1


def test_recovery_augmentation_ignores_small_bucket_difference() -> None:
    obs_seq = [
        _make_motion2d_obs(0.0, 0.0),
        _make_motion2d_obs(0.03, 0.0),
        _make_motion2d_obs(0.06, 0.0),
    ]
    expert_action = np.array([0.005, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    learner_action = np.array([0.004, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    approach = _make_recovery_approach(
        env_factory=lambda instance_num=None: _DummyRecoveryEnv(
            obs_seq=obs_seq,
            expert_action=expert_action,
        )
    )

    aug_ids, aug_demo, aug_dict, changed = approach._augment_recovery_states(
        policy=_ConstantPolicy(learner_action),  # type: ignore[arg-type]
        train_demo_ids=(0,),
        demonstrations_train=Trajectory(steps=[]),
        demo_dict_train={0: Trajectory(steps=[])},
        negative_sampling_cfg=approach._negative_sampling_cfg,
    )

    assert changed is False
    assert aug_ids == (0,)
    assert len(aug_demo.steps) == 0
    assert list(aug_dict.keys()) == [0]
