"""Tests for ARC-AGI-3 object-centric preprocessing."""

from __future__ import annotations

import numpy as np

from programmatic_policy_learning.envs.arc_agi3 import (
    extract_latest_grid,
    extract_objects_by_color,
    preprocess_arc_observation,
)


def test_extract_latest_grid_uses_last_animation_frame() -> None:
    first = np.zeros((3, 4), dtype=int)
    last = np.full((3, 4), 7, dtype=int)

    result = extract_latest_grid(np.stack([first, last]))

    assert np.array_equal(result, last)


def test_extract_objects_by_color_uses_four_connectivity() -> None:
    grid = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [2, 2, 0],
        ],
        dtype=int,
    )

    objects = extract_objects_by_color(grid)

    assert len(objects[1]) == 2
    assert len(objects[2]) == 1
    assert objects[2][0]["bbox"] == [0, 2, 1, 2]
    assert objects[2][0]["center"] == [0.5, 2.0]
    assert objects[2][0]["mask"] == [[1, 1]]


def test_preprocess_ls20_adds_generic_and_semantic_objects() -> None:
    grid = np.zeros((64, 64), dtype=int)
    grid[45:47, 34:39] = 12
    grid[32, 20] = 1
    grid[33, 21] = 1

    # Same L-shaped symbol at two display scales.
    reference_pixels = [(35, 11), (36, 11), (37, 11), (37, 12), (37, 13)]
    for x, y in reference_pixels:
        grid[y, x] = 9
    current_cells = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    for cell_x, cell_y in current_cells:
        x = 3 + 2 * cell_x
        y = 55 + 2 * cell_y
        grid[
            y : y + 2,
            x : x + 2,
        ] = 9

    raw = {
        "game_id": "ls20-test-version",
        "state": "NOT_FINISHED",
        "levels_completed": 0,
        "win_levels": 7,
        "available_actions": [1, 2, 3, 4],
        "frame": [grid.tolist()],
    }

    processed = preprocess_arc_observation(raw)

    assert processed["raw"] == raw
    assert processed["grid"] == grid.tolist()
    assert processed["player"]["bbox"] == [34, 45, 38, 46]
    assert processed["player"]["center"] == [36.0, 45.5]
    assert processed["rotation_switch"]["area"] == 2
    assert len(processed["objects_by_color"][1]) == 2
    assert (
        processed["current_shape"]["canonical_mask"]
        == processed["reference_shape"]["canonical_mask"]
    )
    assert processed["shape_matches_reference"] is True


def test_preprocess_is_idempotent() -> None:
    raw = {
        "game_id": "unknown-game",
        "frame": [[[0, 1], [1, 0]]],
    }
    processed = preprocess_arc_observation(raw)

    assert preprocess_arc_observation(processed) is processed
    assert "player" not in processed
