"""Smoke tests for LPP Hydra config exposure of cost-sensitive weights."""

from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import OmegaConf

from programmatic_policy_learning.data.dataset import (
    extract_examples_from_demonstration_item,
)


def test_lpp_yaml_exposes_continuous_weight_config() -> None:
    """Hydra config should expose continuous weight hyperparameters."""
    cfg_path = Path("experiments/conf/approach/lpp.yaml")
    cfg = OmegaConf.load(cfg_path)

    weight_cfg = cfg.program_generation.negative_sampling.continuous.weight_config
    assert isinstance(bool(weight_cfg.enabled), bool)
    assert float(weight_cfg.beta_pos) == 1.0
    assert float(weight_cfg.beta_neg) == 1.0
    assert float(weight_cfg.alpha) == 1.0
    assert list(weight_cfg.lambda_per_dim) == [1.0, 1.0]


def test_lpp_yaml_weight_config_smoke_run() -> None:
    """Config-shaped weight settings should drive weighted expansion path."""
    cfg_path = Path("experiments/conf/approach/lpp.yaml")
    cfg = OmegaConf.load(cfg_path)

    neg_cfg = OmegaConf.to_container(
        cfg.program_generation.negative_sampling,
        resolve=True,
    )
    assert isinstance(neg_cfg, dict)
    neg_cfg_typed = cast(dict[str, Any], neg_cfg)

    # Flip on weighting for this smoke run without mutating the checked-in config.
    neg_cfg_typed["continuous"]["weight_config"]["enabled"] = True
    neg_cfg_typed["action_low"] = [-1.0, -1.0]
    neg_cfg_typed["action_high"] = [1.0, 1.0]
    neg_cfg_typed["continuous"]["bucket_counts"] = 3
    neg_cfg_typed["continuous"].pop("bucket_edges", None)
    neg_cfg_typed["continuous"]["relaxed_labeling"]["enabled"] = False

    obs = np.array([0.0, 0.0], dtype=np.float32)
    action = np.array([0.0, 0.0], dtype=np.float32)

    pos, neg, weights = extract_examples_from_demonstration_item(
        (obs, action),
        negative_sampling=neg_cfg_typed,
        action_mode="continuous",
        compute_sample_weights=bool(
            neg_cfg_typed["continuous"]["weight_config"].get("enabled", False)
        ),
    )

    assert len(weights) == len(pos) + len(neg)
    # Positive row is first by construction.
    assert np.isclose(
        weights[0],
        float(neg_cfg_typed["continuous"]["weight_config"]["beta_pos"]),
    )
    # Weighted negatives should not all collapse to one value in this setup.
    assert not np.allclose(weights[1:], weights[1])


def test_kinder_pushpullhook2d_env_config_loads() -> None:
    """PushPullHook2D env config should expose the registered KinDER id."""
    cfg_path = Path("experiments/conf/env/kinder_pushpullhook2d.yaml")
    cfg = OmegaConf.load(cfg_path)

    assert str(cfg.make_kwargs.base_name) == "PushPullHook2D"
    assert str(cfg.make_kwargs.id) == "kinder/PushPullHook2D-v0"
    assert str(cfg.provider) == "kinder"
    assert str(cfg.observation_mode) == "continuous"
    assert str(cfg.action_mode) == "continuous"
