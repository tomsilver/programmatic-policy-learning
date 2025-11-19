"""Tests for collect.py."""

from typing import Any, TypeVar

import numpy as np
from omegaconf import DictConfig, OmegaConf

from programmatic_policy_learning.approaches.expert_approach import ExpertApproach
from programmatic_policy_learning.approaches.experts.grid_experts import get_grid_expert
from programmatic_policy_learning.approaches.random_actions import RandomActionsApproach
from programmatic_policy_learning.data.collect import collect_demo, get_demonstrations
from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.envs.registry import EnvRegistry

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


class DummySpace:
    """DummySpace."""

    def sample(self) -> int:
        """Sample."""
        return 0

    def seed(self, _: int) -> None:
        """Seed."""
        pass  # pylint: disable=unnecessary-pass


class DummyEnv:
    """DummyEnv."""

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset."""
        return np.zeros((2, 2)), {}

    def step(self, _: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step."""
        return np.zeros((2, 2)), 1.0, True, False, {}

    @property
    def observation_space(self) -> DummySpace:
        """Observation space."""
        return DummySpace()

    @property
    def action_space(self) -> DummySpace:
        """Action space."""
        return DummySpace()


def test_collect_demo_returns_trajectory_DummyEnv() -> None:
    """Test that collect_demo returns a Trajectory with Demo steps,
    DummyEnv."""

    env = DummyEnv()
    env_factory = (
        lambda instance_num=None: env
    )  # returns a new environment each time you call

    expert: RandomActionsApproach = RandomActionsApproach(
        "TEST",
        env.observation_space,  # type: ignore
        env.action_space,  # type: ignore
        seed=1,
    )

    traj: Trajectory = collect_demo(env_factory, expert, max_demo_length=5, env_num=0)
    assert isinstance(traj, Trajectory)
    assert isinstance(traj.steps, list)
    assert isinstance(traj.steps[0], tuple)
    assert len(traj.steps) > 0


def test_collect_demo_with_real_env() -> None:
    """Test collect_demo with a real environment using EnvRegistry."""
    env_num = 2
    cfg: DictConfig = OmegaConf.create(
        {
            "provider": "ggg",
            "make_kwargs": {
                "base_name": "TwoPileNim",
                "id": f"TwoPileNim{env_num}-v0",
            },
        }
    )
    registry = EnvRegistry()
    env = registry.load(cfg)
    env_factory = lambda instance_num=None: registry.load(
        cfg, instance_num=instance_num
    )
    env: Any = env_factory(env_num)  # type: ignore
    expert = RandomActionsApproach(  # type: ignore
        "TEST", env.observation_space, env.action_space, seed=1
    )
    traj: Trajectory = collect_demo(
        env_factory, expert, max_demo_length=5, env_num=env_num
    )
    assert isinstance(traj, Trajectory)
    assert isinstance(traj.steps, list)
    assert isinstance(traj.steps[0], tuple)
    assert len(traj.steps) > 0


def test_collect_demo_with_real_env_and_expert() -> None:
    """Test collect_demo with a real environment and expert policy."""
    cfg: DictConfig = OmegaConf.create(
        {
            "provider": "ggg",
            "make_kwargs": {
                "base_name": "TwoPileNim",
                "id": "TwoPileNim1-v0",
            },
            "instance_num": 1,
        }
    )
    registry = EnvRegistry()
    env_factory = lambda instance_num=None: registry.load(
        cfg, instance_num=instance_num
    )
    env: Any = env_factory(1)  # type: ignore

    expert_fn = get_grid_expert("TwoPileNim1-v0")
    expert = ExpertApproach(  # type: ignore
        "TwoPileNim",
        env.observation_space,
        env.action_space,
        seed=1,
        expert_fn=expert_fn,
    )
    traj: Trajectory = collect_demo(env_factory, expert, max_demo_length=5, env_num=1)
    assert isinstance(traj, Trajectory)
    assert isinstance(traj.steps, list)
    assert isinstance(traj.steps[0], tuple)
    assert len(traj.steps) > 0


def test_get_demonstrations() -> None:
    """Test get_demonstrations collects multiple trajectories."""
    env = DummyEnv()
    env_factory = lambda instance_num=None: env

    expert: RandomActionsApproach = RandomActionsApproach(
        "TEST",
        env.observation_space,  # type: ignore
        env.action_space,  # type: ignore
        seed=1,
    )

    demo_numbers = (1, 2, 3)

    demonstrations, _ = get_demonstrations(
        env_factory, expert, demo_numbers=demo_numbers, max_demo_length=5
    )
    assert isinstance(demonstrations, Trajectory)
    assert len(demonstrations.steps) == len(demo_numbers)
    for each in demonstrations.steps:
        assert isinstance(each, tuple)
