"""Tests for offline ARC-AGI-3 initial-state variants."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from arcengine import Level, Sprite

from programmatic_policy_learning.envs.arc_agi3.variants import (
    Ls20InitialStateVariant,
    add_random_ls20_shape,
    coerce_ls20_variant,
    prepare_local_initial_state,
    sample_ls20_variant,
)


def _level() -> Level:
    return Level(
        sprites=[
            Sprite([[12] * 5] * 5, x=34, y=45, tags=["sfqyzhzkij"]),
            Sprite([[1] * 5] * 5, x=19, y=30, tags=["rhsxkxzdjz"]),
            Sprite([[9] * 3] * 3, x=3, y=55, scale=2, tags=["wgmbtyhvbc"]),
            Sprite([[9] * 3] * 3, x=35, y=11, tags=["kvynsvxbpi"]),
            Sprite(
                [[0] * 5] * 5,
                x=34,
                y=10,
                collidable=False,
                tags=["rjlbuycveu"],
            ),
            Sprite([[5] * 5] * 5, x=29, y=30, tags=["ihdgageizm"]),
        ],
        grid_size=(64, 64),
        data={
            "StartShape": 5,
            "kvynsvxbpi": 5,
            "StartRotation": 270,
            "GoalRotation": 0,
        },
    )


def test_variant_changes_shared_shape_but_preserves_locations_and_rotations() -> None:
    baseline = [_level()]
    reference = baseline[0].get_sprites_by_tag("kvynsvxbpi")[0]
    game = SimpleNamespace(_clean_levels=[baseline[0].clone()])
    variant = Ls20InitialStateVariant(
        player_position=(24, 45),
        rotation_switch_position=(39, 30),
        shape_index=2,
    )

    prepare_local_initial_state(
        game_id="ls20",
        sdk_env=SimpleNamespace(_game=game),
        baseline_levels=baseline,
        variant=variant,
    )

    changed = game._clean_levels[0]
    changed_reference = changed.get_sprites_by_tag("kvynsvxbpi")[0]
    assert (changed_reference.x, changed_reference.y) == (reference.x, reference.y)
    assert changed_reference.pixels.tolist() == reference.pixels.tolist()
    assert changed.get_data("kvynsvxbpi") == 2
    assert changed.get_data("GoalRotation") == baseline[0].get_data("GoalRotation")
    assert changed.get_data("StartShape") == 2
    assert changed.get_data("StartRotation") == baseline[0].get_data("StartRotation")
    current = changed.get_sprites_by_tag("wgmbtyhvbc")[0]
    baseline_current = baseline[0].get_sprites_by_tag("wgmbtyhvbc")[0]
    assert (current.x, current.y) == (baseline_current.x, baseline_current.y)
    target = changed.get_sprites_by_tag("rjlbuycveu")[0]
    baseline_target = baseline[0].get_sprites_by_tag("rjlbuycveu")[0]
    assert (target.x, target.y) == (baseline_target.x, baseline_target.y)


def test_random_variant_preserves_goal_and_changes_only_initial_state() -> None:
    baseline = [_level()]
    assert sample_ls20_variant(baseline_levels=baseline, seed=0) is None

    for seed in range(1, 100):
        variant = sample_ls20_variant(baseline_levels=baseline, seed=seed)
        assert variant is not None

        assert variant.player_position != variant.rotation_switch_position
        assert variant.player_position != (34, 45)
        assert variant.rotation_switch_position != (19, 30)
        assert set(variant.to_dict()) == {
            "player_position",
            "rotation_switch_position",
        }


def test_random_shape_is_stage_b_only() -> None:
    baseline = [_level()]

    assert add_random_ls20_shape(None, seed=0) is None
    stage_a_variant = sample_ls20_variant(baseline_levels=baseline, seed=1)
    assert stage_a_variant is not None
    assert stage_a_variant.shape_index is None

    stage_b_variant = add_random_ls20_shape(stage_a_variant, seed=1)
    assert stage_b_variant is not None
    assert stage_b_variant.player_position == stage_a_variant.player_position
    assert (
        stage_b_variant.rotation_switch_position
        == stage_a_variant.rotation_switch_position
    )
    assert stage_b_variant.shape_index is not None
    assert set(stage_b_variant.to_dict()) == {
        "player_position",
        "rotation_switch_position",
        "shape_index",
    }


def test_shape_index_is_validated() -> None:
    with pytest.raises(ValueError, match="shape_index"):
        coerce_ls20_variant({"shape_index": 6})
