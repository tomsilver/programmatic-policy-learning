"""Tests for ContinuousStateEncoder and ContinuousStateEncoderConfig."""

from __future__ import annotations

import numpy as np
import pytest

# pylint: disable=redefined-outer-name,line-too-long
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_encoder import (
    ContinuousStateEncoder,
    ContinuousStateEncoderConfig,
)

# pylint: enable=line-too-long


@pytest.fixture()
def basic_cfg() -> ContinuousStateEncoderConfig:
    """Config with 3 obs fields, 2 action fields, no salient filtering."""
    return ContinuousStateEncoderConfig(
        obs_field_names=["x", "y", "theta"],
        action_field_names=["dx", "dy"],
        precision=3,
    )


@pytest.fixture()
def salient_cfg() -> ContinuousStateEncoderConfig:
    """Config with salient_indices set to only show x and theta."""
    return ContinuousStateEncoderConfig(
        obs_field_names=["x", "y", "theta"],
        action_field_names=["dx", "dy"],
        precision=2,
        salient_indices=[0, 2],
    )


@pytest.fixture()
def encoder(basic_cfg: ContinuousStateEncoderConfig) -> ContinuousStateEncoder:
    """Encoder using the basic (no-filter) config."""
    return ContinuousStateEncoder(basic_cfg)


@pytest.fixture()
def salient_encoder(
    salient_cfg: ContinuousStateEncoderConfig,
) -> ContinuousStateEncoder:
    """Encoder using the salient-index config."""
    return ContinuousStateEncoder(salient_cfg)


# ── encode_obs ──────────────────────────────────────────────────────


class TestEncodeObs:
    """Tests for ContinuousStateEncoder.encode_obs."""

    def test_all_fields_shown_without_salient(
        self, encoder: ContinuousStateEncoder
    ) -> None:
        """All obs fields appear when salient_indices is None."""
        obs = np.array([1.0, 2.0, 3.0])
        result = encoder.encode_obs(obs, step_index=0)
        assert "obs[0] (x)=1.000" in result
        assert "obs[1] (y)=2.000" in result
        assert "obs[2] (theta)=3.000" in result
        assert "*** Step 0 ***" in result

    def test_salient_indices_filter(
        self, salient_encoder: ContinuousStateEncoder
    ) -> None:
        """Only salient indices appear in the output."""
        obs = np.array([1.0, 2.0, 3.0])
        result = salient_encoder.encode_obs(obs, step_index=5)
        assert "obs[0] (x)=1.00" in result
        assert "obs[2] (theta)=3.00" in result
        assert "(y)" not in result
        assert "*** Step 5 ***" in result

    def test_step_index_in_header(self, encoder: ContinuousStateEncoder) -> None:
        """Step index appears in both the header and the s_N label."""
        result = encoder.encode_obs(np.array([0.0, 0.0, 0.0]), step_index=42)
        assert "*** Step 42 ***" in result
        assert "s_42" in result

    def test_obs_longer_than_field_names_raises(
        self, encoder: ContinuousStateEncoder
    ) -> None:
        """Extra obs dimensions beyond named fields crash immediately."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(IndexError, match="Observation index 3 out of range"):
            encoder.encode_obs(obs, step_index=0)

    def test_precision_respected(self, salient_encoder: ContinuousStateEncoder) -> None:
        """Output uses the precision from the config."""
        obs = np.array([1.23456, 0.0, 9.87654])
        result = salient_encoder.encode_obs(obs, step_index=0)
        assert "1.23" in result
        assert "9.88" in result
        assert "1.235" not in result


# ── encode_action ───────────────────────────────────────────────────


class TestEncodeAction:
    """Tests for ContinuousStateEncoder.encode_action."""

    def test_none_action(self, encoder: ContinuousStateEncoder) -> None:
        """None action produces the terminal state string."""
        assert encoder.encode_action(None) == "Action: None (terminal state)."

    def test_named_action_fields(self, encoder: ContinuousStateEncoder) -> None:
        """Action fields use the configured names."""
        action = np.array([0.05, -0.03])
        result = encoder.encode_action(action)
        assert "dx=0.050" in result
        assert "dy=-0.030" in result

    def test_extra_action_dims_raises(self, encoder: ContinuousStateEncoder) -> None:
        """Extra action dimensions beyond named fields crash immediately."""
        action = np.array([0.1, 0.2, 0.3])
        with pytest.raises(IndexError, match="Action index 2 out of range"):
            encoder.encode_action(action)


# ── encode_step ─────────────────────────────────────────────────────


class TestEncodeStep:
    """Tests for ContinuousStateEncoder.encode_step."""

    def test_combines_obs_and_action(self, encoder: ContinuousStateEncoder) -> None:
        """encode_step contains both the observation and action blocks."""
        obs = np.array([1.0, 2.0, 3.0])
        action = np.array([0.01, 0.02])
        result = encoder.encode_step(obs, action, step_index=7)
        assert "*** Step 7 ***" in result
        assert "obs[0] (x)=1.000" in result
        assert "dx=0.010" in result

    def test_terminal_step(self, encoder: ContinuousStateEncoder) -> None:
        """encode_step with None action shows the terminal marker."""
        obs = np.array([1.0, 2.0, 3.0])
        result = encoder.encode_step(obs, None, step_index=99)
        assert "*** Step 99 ***" in result
        assert "terminal state" in result


# ── compute_deltas ──────────────────────────────────────────────────


class TestComputeDeltas:
    """Tests for ContinuousStateEncoder.compute_deltas."""

    def test_no_change(self, encoder: ContinuousStateEncoder) -> None:
        """Identical observations produce no deltas."""
        obs = np.array([1.0, 2.0, 3.0])
        assert encoder.compute_deltas(obs, obs) == []

    def test_all_fields_change(self, encoder: ContinuousStateEncoder) -> None:
        """Deltas are reported for every changed field."""
        obs_t = np.array([1.0, 2.0, 3.0])
        obs_t1 = np.array([1.5, 2.5, 3.5])
        deltas = encoder.compute_deltas(obs_t, obs_t1)
        assert len(deltas) == 3
        assert "obs[0] (x)" in deltas[0]
        assert "+0.500" in deltas[0]

    def test_threshold_filters_small_changes(
        self, encoder: ContinuousStateEncoder
    ) -> None:
        """Changes below the threshold are suppressed."""
        obs_t = np.array([1.0, 2.0, 3.0])
        obs_t1 = np.array([1.0 + 1e-5, 2.0, 3.0 + 0.5])
        deltas = encoder.compute_deltas(obs_t, obs_t1)
        assert len(deltas) == 1
        assert "obs[2] (theta)" in deltas[0]

    def test_custom_threshold(self, encoder: ContinuousStateEncoder) -> None:
        """A larger threshold suppresses more deltas."""
        obs_t = np.array([1.0, 2.0, 3.0])
        obs_t1 = np.array([1.05, 2.0, 3.5])
        deltas_default = encoder.compute_deltas(obs_t, obs_t1)
        deltas_large = encoder.compute_deltas(obs_t, obs_t1, threshold=0.1)
        assert len(deltas_default) == 2
        assert len(deltas_large) == 1

    def test_extra_dims_raises(self, encoder: ContinuousStateEncoder) -> None:
        """Dimensions beyond obs_field_names crash immediately."""
        obs_t = np.array([1.0, 2.0, 3.0, 4.0])
        obs_t1 = np.array([1.0, 2.0, 3.0, 5.0])
        with pytest.raises(IndexError, match="Observation index 3 out of range"):
            encoder.compute_deltas(obs_t, obs_t1)

    def test_negative_delta(self, encoder: ContinuousStateEncoder) -> None:
        """Negative deltas are shown with a minus sign."""
        obs_t = np.array([1.0, 2.0, 3.0])
        obs_t1 = np.array([0.5, 2.0, 3.0])
        deltas = encoder.compute_deltas(obs_t, obs_t1)
        assert len(deltas) == 1
        assert "-0.500" in deltas[0]
