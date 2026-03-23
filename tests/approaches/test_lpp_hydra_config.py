"""Smoke tests for LPP Hydra config exposure of cost-sensitive weights."""

from pathlib import Path

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
    assert weight_cfg.enabled is False
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

    # Flip on weighting for this smoke run without mutating the checked-in config.
    neg_cfg["continuous"]["weight_config"]["enabled"] = True
    neg_cfg["action_low"] = [-1.0, -1.0]
    neg_cfg["action_high"] = [1.0, 1.0]
    neg_cfg["continuous"]["bucket_counts"] = 3

    obs = np.array([0.0, 0.0], dtype=np.float32)
    action = np.array([0.0, 0.0], dtype=np.float32)

    pos, neg, weights = extract_examples_from_demonstration_item(
        (obs, action),
        negative_sampling=neg_cfg,
        action_mode="continuous",
        compute_sample_weights=bool(
            neg_cfg["continuous"]["weight_config"].get("enabled", False)
        ),
    )

    assert len(weights) == len(pos) + len(neg)
    # Positive row is first by construction.
    assert np.isclose(weights[0], float(neg_cfg["continuous"]["weight_config"]["beta_pos"]))
    # Weighted negatives should not all collapse to one value in this setup.
    assert not np.allclose(weights[1:], weights[1])
