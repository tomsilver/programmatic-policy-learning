"""End-to-end Motion2D smoke test with deterministic checks.

This test exercises:
- continuous demo collection with Motion2D expert
- quantized expansion + optional cost-sensitive weighting
- program evaluation matrix generation
- deterministic output across repeated runs (same seed/config)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from programmatic_policy_learning.approaches.motion2d_expert_approach import (
    Motion2DExpertApproach,
)
from programmatic_policy_learning.data.dataset import (
    run_all_programs_on_single_demonstration,
)
from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.envs.registry import EnvRegistry


@dataclass
class SmokeRunOutput:
    X: Any
    y: np.ndarray
    sample_weights: np.ndarray
    examples: list[tuple[Any, Any]]


def _collect_tiny_motion2d_demo(
    *,
    seed: int,
    max_steps: int,
) -> Trajectory[Any, Any]:
    cfg_env = OmegaConf.load("experiments/conf/env/kinder_motion2d.yaml")
    registry = EnvRegistry()
    env = registry.load(cfg_env, instance_num=seed)

    expert = Motion2DExpertApproach(
        environment_description=str(cfg_env.description),
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)
    expert.reset(obs, info)

    steps: list[tuple[Any, Any]] = []
    for t in range(max_steps):
        action = expert.step()
        steps.append((obs, action))
        obs, reward, terminated, truncated, info = env.step(action)
        expert.update(obs, reward, bool(terminated or truncated), info)
        if terminated or truncated:
            print(
                "[motion2d-smoke] rollout finished early",
                f"at step={t + 1}",
                f"reward={reward:.4f}",
                f"terminated={terminated}",
                f"truncated={truncated}",
            )
            break

    env.close()
    return Trajectory(steps=steps)


def _run_smoke_pipeline(
    *,
    seed: int,
    split_tag: str,
    max_steps: int,
) -> SmokeRunOutput:
    demo_traj = _collect_tiny_motion2d_demo(seed=seed, max_steps=max_steps)
    print("[motion2d-smoke] demo steps=", len(demo_traj.steps))

    # Tiny program set keeps runtime low while exercising matrix creation.
    programs = [
        "a[0] > 0.0",
        "a[1] > 0.0",
        "np.linalg.norm(a[:2]) > 1e-6",
    ]

    # Enable weighted continuous expansion with tiny bucket grid (3x3).
    cfg_neg = {
        "enabled": True,
        "action_low": [-1.0, -1.0, -1.0, -1.0, -1.0],
        "action_high": [1.0, 1.0, 1.0, 1.0, 1.0],
        "continuous": {
            "bucket_counts": 3,
            "weight_config": {
                "enabled": True,
                "beta_pos": 1.0,
                "beta_neg": 1.0,
                "alpha": 1.0,
                "lambda_per_dim": [1.0, 1.0],
            },
        },
    }

    X, y, examples, sample_weights = run_all_programs_on_single_demonstration(
        "Motion2D-Smoke",
        seed,
        programs,
        demo_traj,
        {},
        negative_sampling=cfg_neg,
        return_examples=True,
        split_tag=split_tag,
        action_mode="continuous",
        seed=seed,
    )

    assert examples is not None
    print(
        "[motion2d-smoke] matrix shape=",
        X.shape,
        "labels=",
        y.shape,
        "weights=",
        sample_weights.shape,
    )
    print(
        "[motion2d-smoke] y counts:",
        "pos=",
        int(np.sum(y)),
        "neg=",
        int(len(y) - np.sum(y)),
    )
    print(
        "[motion2d-smoke] weights summary:",
        "min=",
        float(np.min(sample_weights)),
        "max=",
        float(np.max(sample_weights)),
        "mean=",
        float(np.mean(sample_weights)),
    )

    return SmokeRunOutput(X=X, y=y, sample_weights=sample_weights, examples=examples)


def test_motion2d_end_to_end_smoke_deterministic() -> None:
    """Run a tiny Motion2D end-to-end pass twice and verify determinism."""
    seed = 0
    max_steps = 6

    run_1 = _run_smoke_pipeline(
        seed=seed,
        split_tag="motion2d_smoke_run1",
        max_steps=max_steps,
    )
    run_2 = _run_smoke_pipeline(
        seed=seed,
        split_tag="motion2d_smoke_run2",
        max_steps=max_steps,
    )

    # No regressions on output dimensions/alignment.
    assert run_1.X.shape[0] == len(run_1.y) == len(run_1.sample_weights)
    assert run_2.X.shape[0] == len(run_2.y) == len(run_2.sample_weights)

    # Determinism checks: recomputed runs with identical seed/config must match.
    assert run_1.X.shape == run_2.X.shape
    assert (run_1.X != run_2.X).nnz == 0
    assert np.array_equal(run_1.y, run_2.y)
    assert np.allclose(run_1.sample_weights, run_2.sample_weights)

    # Extra detail for quick visual confirmation in -s mode.
    print("[motion2d-smoke] deterministic check: PASS")
    print(
        "[motion2d-smoke] first 10 weights=",
        run_1.sample_weights[:10].tolist(),
    )
