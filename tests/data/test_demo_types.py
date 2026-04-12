"""Tests for demo_types.py."""

from pathlib import Path

import numpy as np

from programmatic_policy_learning.data.demo_io import (
    DemoRecord,
    load_demo_record,
    load_demo_records_from_dir,
    save_demo_record,
)
from programmatic_policy_learning.data.demo_types import Trajectory


def test_trajectory_dataclass() -> None:
    """Test Trajectory dataclass with steps as (obs, act) tuples."""
    obs = [np.ones((3, 3)), np.zeros((3, 3))]
    act = [42, 7]
    steps = list(zip(obs, act))
    traj = Trajectory(steps=steps)
    assert isinstance(traj, Trajectory)
    assert isinstance(traj.steps, list)
    assert isinstance(traj.steps[0], tuple)
    assert traj.steps[0][0].shape == (3, 3)
    assert traj.steps[0][1] == 42
    assert len(traj.steps) == 2


def test_demo_record_round_trip(tmp_path: Path) -> None:
    """Saved demo records should round-trip through pickle."""
    record = DemoRecord(
        env_id="kinder/PushPullHook2D-v0",
        seed=3,
        trajectory=Trajectory(
            steps=[
                (
                    np.array([1.0, 2.0], dtype=np.float32),
                    np.array([0.1, 0.2], dtype=np.float32),
                )
            ]
        ),
        rewards=[-1.0],
        terminated=True,
        truncated=False,
        metadata={"source": "manual"},
    )
    out_path = tmp_path / "seed_0003.pkl"
    save_demo_record(out_path, record)

    loaded = load_demo_record(out_path)
    assert loaded.env_id == record.env_id
    assert loaded.seed == record.seed
    assert loaded.rewards == record.rewards
    assert loaded.terminated is True
    assert loaded.truncated is False
    assert loaded.metadata == {"source": "manual"}
    np.testing.assert_allclose(loaded.trajectory.steps[0][0], record.trajectory.steps[0][0])
    np.testing.assert_allclose(loaded.trajectory.steps[0][1], record.trajectory.steps[0][1])


def test_load_demo_records_from_dir(tmp_path: Path) -> None:
    """Directory loader should return all saved demo records in sorted order."""
    for seed in (2, 0):
        save_demo_record(
            tmp_path / f"seed_{seed:04d}.pkl",
            DemoRecord(
                env_id="kinder/PushPullHook2D-v0",
                seed=seed,
                trajectory=Trajectory(steps=[]),
            ),
        )

    records = load_demo_records_from_dir(tmp_path)
    assert [record.seed for record in records] == [0, 2]
