"""Tests for LPP Approach."""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from gymnasium.spaces import Box, MultiDiscrete
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import csr_matrix

from programmatic_policy_learning.approaches.expert_approach import ExpertApproach
from programmatic_policy_learning.approaches.experts.grid_experts import get_grid_expert
from programmatic_policy_learning.approaches.lpp_approach import (
    LogicProgrammaticPolicyApproach,
)
from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram
from programmatic_policy_learning.envs.registry import EnvRegistry


class _DummyRecoveryExpert:
    """Minimal expert stub for recovery-augmentation tests."""

    def __init__(self) -> None:
        self._env: Any | None = None

    def set_env(self, env: Any) -> None:
        """Attach the environment used to source expert actions."""
        self._env = env

    def reset(self, obs: Any, info: Any) -> None:
        """Reset hook matching the expert interface."""
        del obs, info

    def step(self) -> np.ndarray:
        """Return the environment-provided expert action."""
        if self._env is None:
            raise RuntimeError("Environment was not set on _DummyRecoveryExpert.")
        return np.asarray(self._env.expert_action, dtype=np.float32)


class _ConstantPolicy:
    """Policy stub that always returns the same action."""

    def __init__(self, action: np.ndarray) -> None:
        self._action = np.asarray(action, dtype=np.float32)

    def __call__(self, obs: Any) -> np.ndarray:
        """Ignore the observation and return the stored action."""
        del obs
        return self._action.copy()


class _DummyRecoveryEnv:
    """Small deterministic env stub for recovery-augmentation tests."""

    def __init__(
        self,
        *,
        obs_seq: list[np.ndarray],
        expert_action: np.ndarray,
    ) -> None:
        self._obs_seq = [np.asarray(obs, dtype=np.float32) for obs in obs_seq]
        self.expert_action = np.asarray(expert_action, dtype=np.float32)
        self._idx = 0

    def reset(self, seed: Any = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to the first observation in the scripted sequence."""
        del seed
        self._idx = 0
        return self._obs_seq[0].copy(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one step through the scripted observation sequence."""
        del action
        self._idx = min(self._idx + 1, len(self._obs_seq) - 1)
        terminated = self._idx >= len(self._obs_seq) - 1
        return self._obs_seq[self._idx].copy(), 0.0, terminated, False, {}

    def close(self) -> None:
        """Close hook matching the environment interface."""
        return None


def _make_motion2d_obs(
    robot_x: float,
    robot_y: float,
    *,
    target_x: float = 1.0,
    target_y: float = 1.0,
) -> np.ndarray:
    """Construct a compact Motion2D-style observation vector."""
    obs = np.zeros(19, dtype=np.float32)
    obs[0] = robot_x
    obs[1] = robot_y
    obs[9] = target_x
    obs[10] = target_y
    obs[17] = 0.0
    obs[18] = 0.0
    return obs


def _make_recovery_approach(
    env_factory: Callable[[int | None], _DummyRecoveryEnv],
) -> LogicProgrammaticPolicyApproach:
    """Build an LPP approach configured for recovery-augmentation tests."""
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
    # Tests intentionally seed the internal continuous sampling config directly.
    # pylint: disable=protected-access
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
    """Deviation recovery should add a new queried state before stuck logic."""
    obs_seq = [
        _make_motion2d_obs(0.0, 0.0),
        _make_motion2d_obs(0.0, 0.0),
        _make_motion2d_obs(0.0, 0.0),
        _make_motion2d_obs(0.0, 0.0),
    ]
    expert_action = np.array([0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    learner_action = np.array([-0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def env_factory(instance_num: int | None = None) -> _DummyRecoveryEnv:
        del instance_num
        return _DummyRecoveryEnv(
            obs_seq=obs_seq,
            expert_action=expert_action,
        )

    approach = _make_recovery_approach(env_factory=env_factory)

    # Tests intentionally exercise the internal recovery augmentation routine.
    # pylint: disable=protected-access
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
    """Tiny bucket differences should not trigger recovery augmentation."""
    obs_seq = [
        _make_motion2d_obs(0.0, 0.0),
        _make_motion2d_obs(0.03, 0.0),
        _make_motion2d_obs(0.06, 0.0),
    ]
    expert_action = np.array([0.005, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    learner_action = np.array([0.004, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def env_factory(instance_num: int | None = None) -> _DummyRecoveryEnv:
        del instance_num
        return _DummyRecoveryEnv(
            obs_seq=obs_seq,
            expert_action=expert_action,
        )

    approach = _make_recovery_approach(env_factory=env_factory)

    # Tests intentionally exercise the internal recovery augmentation routine.
    # pylint: disable=protected-access
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


def test_discrete_training_uses_grid_enumeration_not_empty_candidate_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discrete PLP scoring should enumerate grid cells, not an empty
    continuous catalog."""
    approach = LogicProgrammaticPolicyApproach(
        environment_description="DummyGrid-v0",
        observation_space=Box(low=0, high=1, shape=(2, 2), dtype=np.int64),
        action_space=MultiDiscrete([2, 2]),
        seed=0,
        expert=None,  # type: ignore[arg-type]
        env_factory=lambda instance_num=None: None,
        base_class_name="DummyGrid",
        demo_numbers=(0,),
        env_specs={"action_mode": "discrete"},
        max_num_particles=2,
        prior_version="uniform",
    )

    captured_candidate_actions: list[Any] = []

    def fake_compute_likelihood_plps(*args: Any, **kwargs: Any) -> list[float]:
        captured_candidate_actions.append(kwargs.get("candidate_actions"))
        return [0.0]

    def fake_log_plp_violation_counts(*args: Any, **kwargs: Any) -> None:
        captured_candidate_actions.append(kwargs.get("candidate_actions"))

    monkeypatch.setattr(
        "programmatic_policy_learning.approaches.lpp_approach.learn_plps",
        lambda *args, **kwargs: ([StateActionProgram("True")], [0.0]),
    )
    monkeypatch.setattr(
        "programmatic_policy_learning.approaches.lpp_approach.compute_likelihood_plps",
        fake_compute_likelihood_plps,
    )
    monkeypatch.setattr(
        "programmatic_policy_learning.approaches.lpp_approach.log_plp_violation_counts",
        fake_log_plp_violation_counts,
    )

    X = np.array([[True], [False]])
    y_bool = [True, False]
    demonstrations = Trajectory(steps=[(np.zeros((2, 2), dtype=int), (0, 0))])

    # pylint: disable=protected-access
    policy = approach._train_policy_from_matrix(
        X,
        y_bool,
        sample_weight=None,
        programs_sa=[StateActionProgram("True")],
        program_prior_log_probs_opt=[0.0],
        demonstrations=demonstrations,
        dsl_functions={},
    )

    assert policy.action_mode == "discrete"
    assert captured_candidate_actions
    assert all(value is None for value in captured_candidate_actions)


def test_discrete_train_matrix_passes_none_sample_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discrete LPP should let DecisionTreeClassifier use class_weight."""
    approach = LogicProgrammaticPolicyApproach(
        environment_description="DummyGrid-v0",
        observation_space=Box(low=0, high=1, shape=(2, 2), dtype=np.int64),
        action_space=MultiDiscrete([2, 2]),
        seed=0,
        expert=None,  # type: ignore[arg-type]
        env_factory=lambda instance_num=None: None,
        base_class_name="DummyGrid",
        demo_numbers=(0,),
        env_specs={"action_mode": "discrete"},
        prior_version="uniform",
        cross_demo_feature_filter={"enabled": False},
    )

    examples = [
        (np.zeros((2, 2), dtype=int), (0, 0)),
        (np.zeros((2, 2), dtype=int), (0, 1)),
    ]

    def fake_run_all_programs_on_demonstrations(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return (
            csr_matrix(np.array([[True], [False]])),
            np.array([1, 0], dtype=np.uint8),
            examples,
            np.ones(2, dtype=float),
            np.array([0, 0], dtype=int),
        )

    monkeypatch.setattr(
        "programmatic_policy_learning.approaches.lpp_approach.run_all_programs_on_demonstrations",
        fake_run_all_programs_on_demonstrations,
    )
    monkeypatch.setattr(
        "programmatic_policy_learning.approaches.lpp_approach._filter_redundant_features",
        lambda X, programs, priors: (X, programs, priors, np.asarray([1])),
    )
    monkeypatch.setattr(
        "programmatic_policy_learning.approaches.lpp_approach.log_feature_collisions",
        lambda *args, **kwargs: [],
    )

    # pylint: disable=protected-access
    _X, _y, _y_bool, sample_weights, _examples, _programs, _priors = (
        approach._build_and_process_train_matrix(
            train_demo_ids=(0,),
            val_demo_ids=(),
            demo_dict_train={0: Trajectory(steps=examples)},
            programs_sa=[StateActionProgram("True")],
            program_prior_log_probs_opt=[0.0],
            dsl_functions={},
            negative_sampling_cfg=None,
            offline_path_name=None,
            start_index=2,
        )
    )

    assert sample_weights is None
