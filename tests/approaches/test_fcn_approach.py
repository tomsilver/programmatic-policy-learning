"""Smoke tests for the FCN grid imitation baseline."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from programmatic_policy_learning.approaches.expert_approach import ExpertApproach
from programmatic_policy_learning.approaches.fcn_approach import FCNApproach


def _target_policy(obs: np.ndarray) -> tuple[int, int]:
    """Expert that selects the target token's coordinates."""
    row, col = np.argwhere(obs == "target")[0]
    return (int(row), int(col))


class _ToyGridEnv(gym.Env[np.ndarray, tuple[int, int]]):
    """Minimal one-step grid task for FCN behavioral cloning."""

    metadata = {"render_modes": []}

    def __init__(self, layouts: dict[int, np.ndarray], instance_num: int) -> None:
        super().__init__()
        self._layouts = layouts
        self._instance_num = int(instance_num)
        self._step_count = 0
        sample_layout = next(iter(layouts.values()))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=sample_layout.shape,
            dtype=np.int32,
        )
        self.action_space = gym.spaces.MultiDiscrete(sample_layout.shape)
        self._obs: np.ndarray | None = None

    def get_object_types(self) -> tuple[str, ...]:
        return ("empty", "target", "wall")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed, options
        self._step_count = 0
        self._obs = np.array(self._layouts[self._instance_num], copy=True)
        return self._obs, {}

    def step(
        self, action: tuple[int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._obs is not None
        row, col = action
        target_row, target_col = np.argwhere(self._obs == "target")[0]
        success = int(row) == int(target_row) and int(col) == int(target_col)
        self._step_count += 1
        terminated = bool(success and self._step_count >= 3)
        truncated = bool(not success)
        return self._obs, float(success), terminated, truncated, {"is_success": success}


def test_fcn_approach_learns_grid_cell_selection_from_demos() -> None:
    """FCN should fit the demonstration mapping and solve the toy rollouts."""
    layouts = {
        0: np.array(
            [
                ["empty", "empty", "empty", "empty"],
                ["empty", "target", "empty", "empty"],
                ["empty", "empty", "wall", "empty"],
                ["empty", "empty", "empty", "empty"],
            ],
            dtype=object,
        ),
        1: np.array(
            [
                ["empty", "wall", "empty", "empty"],
                ["empty", "empty", "empty", "target"],
                ["empty", "empty", "empty", "empty"],
                ["empty", "empty", "empty", "empty"],
            ],
            dtype=object,
        ),
        2: np.array(
            [
                ["empty", "empty", "empty", "empty"],
                ["target", "empty", "empty", "empty"],
                ["empty", "empty", "wall", "empty"],
                ["empty", "empty", "empty", "empty"],
            ],
            dtype=object,
        ),
    }

    def env_factory(instance_num: int) -> _ToyGridEnv:
        return _ToyGridEnv(layouts=layouts, instance_num=instance_num)

    env = env_factory(0)
    approach = FCNApproach(
        "toy-grid",
        env.observation_space,
        env.action_space,
        seed=123,
        expert=ExpertApproach(
            "toy-grid-expert",
            env.observation_space,
            env.action_space,
            seed=123,
            expert_fn=_target_policy,
        ),
        env_factory=env_factory,
        demo_numbers=(0, 1, 2),
        num_epochs=80,
        min_epochs=5,
        batch_size=2,
        learning_rate=1e-2,
        train_before_eval=False,
        object_types=env.get_object_types(),
    )

    approach.train_offline()
    assert approach.training_summary["num_training_steps"] > 0

    obs, info = env_factory(2).reset()
    approach.reset(obs, info)
    assert approach.step() == (1, 0)

    results = approach.test_policy_on_envs(
        test_env_nums=[0, 1, 2],
        max_num_steps=3,
        base_class_name="ToyGrid",
    )
    assert results == [True, True, True]
