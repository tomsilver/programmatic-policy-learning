"""Validation tests for manually collected KinDER demo pickle files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from programmatic_policy_learning.data.demo_io import (
    DemoRecord,
    load_demo_record,
    load_demo_records_from_dir,
)
from programmatic_policy_learning.envs.providers.kinder_provider import (
    KinderEnvWithTypes,
    create_kinder_env,
)


def _candidate_demo_roots() -> list[Path]:
    return [Path("demos"), Path("manual_demos")]


def _discover_manual_demo_paths() -> list[Path]:
    paths: list[Path] = []
    for root in _candidate_demo_roots():
        if root.exists():
            paths.extend(sorted(root.rglob("*.pkl")))
    return sorted(paths)


def _make_pushpullhook2d_env() -> KinderEnvWithTypes:
    cfg = OmegaConf.create(
        {
            "provider": "kinder",
            "make_kwargs": {
                "base_name": "PushPullHook2D",
                "id": "kinder/PushPullHook2D-v0",
                "render_mode": "rgb_array",
            },
        }
    )
    env = create_kinder_env(cfg)
    assert isinstance(env, KinderEnvWithTypes)
    return env


def test_pushpullhook2d_action_bounds() -> None:
    """Print and validate the PushPullHook2D action bounds."""
    env = _make_pushpullhook2d_env()
    try:
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)
        names = ["dx", "dy", "dtheta", "darm", "vac"]

        print("\nPushPullHook2D action bounds")
        for idx, name in enumerate(names):
            print(f"{name}: low={float(low[idx]):+.6f}, high={float(high[idx]):+.6f}")

        assert low.shape == (5,)
        assert high.shape == (5,)
        np.testing.assert_allclose(low, [-0.05, -0.05, -np.pi / 16, -0.1, 0.0])
        np.testing.assert_allclose(high, [0.05, 0.05, np.pi / 16, 0.1, 1.0])
    finally:
        env.close()


def test_saved_manual_demo_pickles_are_valid_and_executable() -> None:
    """Replay saved manual demo pickles in the real env."""
    demo_paths = _discover_manual_demo_paths()
    if not demo_paths:
        pytest.skip("No demo pickles found under demos/ or manual_demos/.")

    # Only validate PushPullHook2D demos here.
    demo_paths = [
        path
        for path in demo_paths
        if "PushPullHook2D" in str(path) or "pushpullhook2d" in str(path)
    ]
    if not demo_paths:
        pytest.skip("No PushPullHook2D demo pickles found.")

    for path in demo_paths:
        record = load_demo_record(path)
        assert isinstance(record, DemoRecord)
        assert record.env_id == "kinder/PushPullHook2D-v0"
        assert record.seed >= 0
        assert len(record.trajectory.steps) == len(record.rewards)
        assert record.metadata.get("num_actions") == len(record.trajectory.steps)
        assert (
            record.metadata.get("num_observations") == len(record.trajectory.steps) + 1
        )

        env = _make_pushpullhook2d_env()
        try:
            obs, info = env.reset(seed=record.seed)
            assert isinstance(info, dict)
            assert isinstance(obs, np.ndarray)

            low = np.asarray(env.action_space.low, dtype=np.float32)
            high = np.asarray(env.action_space.high, dtype=np.float32)

            for step_idx, ((saved_obs, action), expected_reward) in enumerate(
                zip(record.trajectory.steps, record.rewards)
            ):
                saved_obs_arr = np.asarray(saved_obs, dtype=np.float32)
                action_arr = np.asarray(action, dtype=np.float32)
                if step_idx < 5:
                    print(
                        f"step {step_idx}: "
                        f"obs[:6]={np.array2string(saved_obs_arr[:6], precision=3, suppress_small=True)} "
                        f"action={np.array2string(action_arr, precision=3, suppress_small=True)} "
                        f"expected_reward={expected_reward:+.1f}"
                    )
                assert saved_obs_arr.shape == obs.shape == (38,)
                assert action_arr.shape == low.shape == high.shape == (5,)
                np.testing.assert_allclose(
                    obs,
                    saved_obs_arr,
                    atol=1e-5,
                    err_msg=f"Observation mismatch at step {step_idx} for {path}",
                )
                assert np.all(
                    action_arr >= low
                ), f"Action below low at step {step_idx} in {path}"
                assert np.all(
                    action_arr <= high
                ), f"Action above high at step {step_idx} in {path}"

                obs, reward, terminated, truncated, info = env.step(action_arr)
                assert isinstance(obs, np.ndarray)
                assert isinstance(reward, float)
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)
                assert isinstance(info, dict)
                assert np.isclose(reward, expected_reward)

                if terminated or truncated:
                    assert step_idx == len(record.trajectory.steps) - 1
                    assert terminated == record.terminated
                    assert truncated == record.truncated
                    if terminated:
                        print(
                            f"Successfully terminated {path} in {step_idx + 1} steps."
                        )
                    elif truncated:
                        print(f"Replay truncated for {path} in {step_idx + 1} steps.")
                    break
            else:
                assert not record.terminated
                assert not record.truncated
                print(
                    f"Replay completed for {path} without termination in "
                    f"{len(record.trajectory.steps)} steps."
                )
        finally:
            env.close()


def test_manual_demo_directory_loader_finds_saved_pickles() -> None:
    """Directory loader should discover saved manual demo records."""
    roots = [root for root in _candidate_demo_roots() if root.exists()]
    if not roots:
        pytest.skip("No demos/ or manual_demos/ directory found.")

    loaded: list[DemoRecord] = []
    for root in roots:
        loaded.extend(load_demo_records_from_dir(root))

    if not loaded:
        pytest.skip("Demo directory exists but contains no pickle files.")
    assert any(record.env_id == "kinder/PushPullHook2D-v0" for record in loaded)
