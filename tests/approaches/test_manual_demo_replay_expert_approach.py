"""Tests for manual demo replay expert approach."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from programmatic_policy_learning.approaches.manual_demo_replay_expert_approach import (
    ManualDemoReplayExpertApproach,
)
from programmatic_policy_learning.data.demo_io import (
    DemoRecord,
    load_demo_record,
    save_demo_record,
)
from programmatic_policy_learning.data.demo_types import Trajectory


class _DummySpace:
    def seed(self, seed: int) -> None:
        del seed


class _DummyEnv:
    def __init__(self, seed: int) -> None:
        self.last_reset_seed = seed


def test_manual_demo_replay_expert_replays_actions_for_seed(tmp_path: Path) -> None:
    """Replay expert should load the matching seed and emit its saved
    actions."""
    demo_root = tmp_path / "manual_demos"
    seed = 7
    obs0 = np.array([1.0, 2.0], dtype=np.float32)
    obs1 = np.array([3.0, 4.0], dtype=np.float32)
    act0 = np.array([0.1, 0.2], dtype=np.float32)
    act1 = np.array([0.3, 0.4], dtype=np.float32)

    save_demo_record(
        demo_root / "seed_0007.pkl",
        DemoRecord(
            env_id="kinder/PushPullHook2D-v0",
            seed=seed,
            trajectory=Trajectory(steps=[(obs0, act0), (obs1, act1)]),
            rewards=[-1.0, 0.0],
            terminated=True,
            metadata={"num_actions": 2, "num_observations": 3},
        ),
    )

    approach = ManualDemoReplayExpertApproach(
        "PushPullHook2D",
        _DummySpace(),
        _DummySpace(),
        seed=0,
        demos_root=str(demo_root),
        env_id="kinder/PushPullHook2D-v0",
    )
    approach.set_env(_DummyEnv(seed))
    approach.reset(obs0.copy(), {})

    action0 = approach.step()
    np.testing.assert_allclose(action0, act0)
    approach.update(obs1, -1.0, False, {})
    action1 = approach.step()
    np.testing.assert_allclose(action1, act1)


def test_manual_demo_replay_expert_with_real_seed0_demo() -> None:
    """Load the real saved seed-0 demo and print a few replayed steps."""
    demo_path = Path("manual_demos/kinder__PushPullHook2D-v0/seed_0000.pkl")
    if not demo_path.exists():
        pytest.skip(f"Real manual demo not found at {demo_path}.")

    record = load_demo_record(demo_path)
    if not record.trajectory.steps:
        pytest.skip("Real manual demo exists but contains no steps.")

    approach = ManualDemoReplayExpertApproach(
        "PushPullHook2D",
        _DummySpace(),
        _DummySpace(),
        seed=0,
        demos_root="manual_demos",
        env_id="kinder/PushPullHook2D-v0",
    )
    approach.set_env(_DummyEnv(record.seed))

    first_obs, first_action = record.trajectory.steps[0]
    approach.reset(np.asarray(first_obs, dtype=np.float32).copy(), {})

    num_preview_steps = min(5, len(record.trajectory.steps))
    for step_idx in range(num_preview_steps):
        saved_obs, saved_action = record.trajectory.steps[step_idx]
        replay_action = approach.step()
        np.testing.assert_allclose(
            replay_action, np.asarray(saved_action, dtype=np.float32)
        )
        approach.update(np.asarray(saved_obs, dtype=np.float32), 0.0, False, {})
