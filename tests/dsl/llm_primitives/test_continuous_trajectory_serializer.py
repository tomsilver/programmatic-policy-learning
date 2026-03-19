"""Tests for the continuous trajectory serializer."""

from __future__ import annotations

import numpy as np

from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_encoder import (
    ContinuousStateEncoder,
    ContinuousStateEncoderConfig,
)
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_hint_config import (
    ACTION_FIELD_NAMES,
    obs_field_names_for_motion2d,
    salient_obs_indices_for_motion2d,
)
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_trajectory_serializer import (
    enc_5,
    trajectory_to_text,
)


def _make_motion2d_encoder(num_passages: int = 1) -> ContinuousStateEncoder:
    return ContinuousStateEncoder(
        ContinuousStateEncoderConfig(
            obs_field_names=obs_field_names_for_motion2d(num_passages),
            action_field_names=ACTION_FIELD_NAMES["Motion2D"],
            salient_indices=salient_obs_indices_for_motion2d(num_passages),
        )
    )


def test_enc_5_serializes_motion2d_vector_observation() -> None:
    """enc_5 should convert Motion2D vectors into object-centric text."""
    encoder = _make_motion2d_encoder(num_passages=1)

    obs0 = np.zeros(39, dtype=np.float32)
    obs0[0], obs0[1], obs0[3] = 0.3, 1.0, 0.1
    obs0[9], obs0[10], obs0[17], obs0[18] = 2.0, 0.5, 0.25, 0.25
    obs0[19], obs0[20], obs0[27], obs0[28] = 0.5, 0.0, 0.01, 0.8
    obs0[29], obs0[30], obs0[37], obs0[38] = 0.5, 1.7, 0.01, 0.8

    action = np.array([0.05, -0.02, 0.0, 0.0, 0.0], dtype=np.float32)
    obs1 = obs0.copy()
    obs1[0] += action[0]
    obs1[1] += action[1]

    text = trajectory_to_text(
        [(obs0, action, obs1)],
        encoder=encoder,
        num_passages=1,
        encoding_method="5",
    )

    assert "Trajectory summary:" in text
    assert "- object types present: obstacle, robot, target" in text
    assert "- initial robot pose: x=0.300, y=1.000, theta=0.000" in text
    assert "*** Step 0 ***" in text
    assert "- robot(type=robot, x=0.300, y=1.000" in text
    assert "- target(type=target, x=2.000, y=0.500, theta=0.000" in text
    assert "Action:" in text
    assert "- dx=0.050" in text
    assert "- dy=-0.020" in text
    assert "Object changes:" in text
    assert "- robot.x: 0.300 -> 0.350 (delta=+0.050)" in text
    assert "- robot.y: 1.000 -> 0.980 (delta=-0.020)" in text
    assert "Pairwise relations before action:" in text
    assert "rel(robot, target): center_dx=1.825, center_dy=-0.375" in text
    assert "Pairwise relations after action:" in text
    assert "Relation deltas:" in text
    assert "dist_change(robot, target)=-0.053" in text
    assert "moved_toward(robot, target)=true" in text
    assert "*** Step 1 ***" in text
    assert "Action: None (terminal state)" in text


def test_enc_5_supports_object_centric_mapping_observations() -> None:
    """enc_5 should also accept already object-centric observations."""
    encoder = ContinuousStateEncoder(
        ContinuousStateEncoderConfig(
            obs_field_names=[],
            action_field_names=["dx", "dy"],
        )
    )

    obs0 = {
        "robot": {"type": "robot", "x": 1.0, "y": 2.0, "theta": 0.5, "radius": 0.2},
        "obj0": {"type": "box", "x": 2.0, "y": 2.0, "width": 1.0, "height": 1.0},
    }
    obs1 = {
        "robot": {"type": "robot", "x": 1.2, "y": 2.0, "theta": 0.5, "radius": 0.2},
        "obj0": {"type": "box", "x": 2.0, "y": 2.0, "width": 1.0, "height": 1.0},
    }
    action = np.array([0.2, 0.0], dtype=np.float32)

    text = enc_5([(obs0, action, obs1)], encoder=encoder)

    assert "- object types present: box, robot" in text
    assert "- changed objects: robot" in text
    assert "- obj0(type=box, x=2.000, y=2.000, width=1.000, height=1.000)" in text
    assert "rel(robot, obj0): center_dx=1.500, center_dy=0.500" in text
    assert "dist_change(robot, obj0)=-0.188" in text
    assert "moved_toward(robot, obj0)=true" in text
