"""Object-centric preprocessing for ARC-AGI-3 observations."""

from __future__ import annotations

import os
import pdb
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

ProcessedState = dict[str, Any]
ObjectEnricher = Callable[[ProcessedState], dict[str, Any]]


def register_arc_object_enricher(
    game_id: str,
    enricher: ObjectEnricher,
    *,
    replace: bool = False,
) -> None:
    """Register semantic object extraction for one ARC game family."""
    key = game_id.split("-", maxsplit=1)[0].lower()
    if key in _GAME_ENRICHERS and not replace:
        raise ValueError(f"An ARC object enricher is already registered for {key!r}.")
    _GAME_ENRICHERS[key] = enricher


def preprocess_arc_observation(
    observation: Any,
    *,
    game_id: str | None = None,
    initial_state_variant: dict[str, Any] | None = None,
) -> ProcessedState:
    """Convert one ARC observation into reusable grid and object structures."""
    if _is_processed_state(observation):
        return observation

    raw = _observation_to_plain_dict(observation)
    grid = extract_latest_grid(raw.get("frame"))
    resolved_game_id = str(game_id or raw.get("game_id") or "")
    processed: ProcessedState = {
        "raw": raw,
        "grid": grid.tolist(),
        "objects_by_color": extract_objects_by_color(grid),
        "preprocessing": {
            "version": 1,
            "connectivity": 4,
            "selected_frame": "last",
        },
    }
    if initial_state_variant:
        processed["initial_state_variant"] = initial_state_variant

    # Preserve the old top-level observation interface during migration. New
    # features should prefer raw/grid/objects_by_color and semantic object keys.
    processed.update(raw)
    processed.update(_enrich_for_game(resolved_game_id, processed))
    # if os.getenv("ARC_AGI3_PDB_PREPROCESS", "").strip().lower() in {
    #     "1",
    #     "true",
    #     "yes",
    # }:
    #     print(
    #         "Entering pdb after ARC preprocessing. Inspect: observation, raw, "
    #         "grid, processed, processed['objects_by_color']."
    #     )
    return processed


def extract_latest_grid(frame: Any) -> np.ndarray:
    """Return the last 2D grid from an SDK frame or serialized frame list."""
    if frame is None:
        raise ValueError("ARC observation does not contain frame data.")
    grid = np.asarray(frame, dtype=int)
    while grid.ndim > 2:
        if grid.shape[0] == 0:
            raise ValueError("ARC observation contains an empty frame sequence.")
        grid = grid[-1]
    if grid.ndim != 2:
        raise ValueError(f"Expected a 2D ARC grid, got shape {grid.shape}.")
    return grid


def extract_objects_by_color(grid: np.ndarray) -> dict[int, list[dict[str, Any]]]:
    """Extract 4-connected same-color components from a 2D ARC grid."""
    if grid.ndim != 2:
        raise ValueError(f"Expected a 2D grid, got shape {grid.shape}.")

    objects_by_color: dict[int, list[dict[str, Any]]] = {}
    for color_value in np.unique(grid):
        color = int(color_value)
        components = _connected_components(grid == color)
        objects_by_color[color] = [
            _object_from_pixels(color, index, pixels)
            for index, pixels in enumerate(components)
        ]
    return objects_by_color


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for y in range(height):
        for x in range(width):
            if not bool(mask[y, x]) or bool(visited[y, x]):
                continue
            queue = deque([(x, y)])
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                px, py = queue.popleft()
                pixels.append((px, py))
                for nx, ny in (
                    (px - 1, py),
                    (px + 1, py),
                    (px, py - 1),
                    (px, py + 1),
                ):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and bool(mask[ny, nx])
                        and not bool(visited[ny, nx])
                    ):
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            components.append(pixels)
    return components


def _object_from_pixels(
    color: int,
    component_index: int,
    pixels: list[tuple[int, int]],
) -> dict[str, Any]:
    xs = [x for x, _ in pixels]
    ys = [y for _, y in pixels]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pixel_set = set(pixels)
    mask = [
        [1 if (x, y) in pixel_set else 0 for x in range(x0, x1 + 1)]
        for y in range(y0, y1 + 1)
    ]
    return {
        "color": color,
        "component_index": component_index,
        "area": len(pixels),
        "bbox": [x0, y0, x1, y1],
        "center": [
            sum(xs) / len(xs),
            sum(ys) / len(ys),
        ],
        "width": x1 - x0 + 1,
        "height": y1 - y0 + 1,
        "pixels": [[x, y] for x, y in pixels],
        "mask": mask,
    }


def _aggregate_color(
    processed: ProcessedState,
    color: int,
) -> dict[str, Any] | None:
    components = processed["objects_by_color"].get(color, [])
    pixels = [
        (int(x), int(y)) for component in components for x, y in component["pixels"]
    ]
    if not pixels:
        return None
    aggregate = _object_from_pixels(color, -1, pixels)
    aggregate["components"] = components
    return aggregate


def _region_color_object(
    processed: ProcessedState,
    *,
    color: int,
    bounds: tuple[int, int, int, int],
    name: str,
) -> dict[str, Any] | None:
    grid = np.asarray(processed["grid"], dtype=int)
    x0, y0, x1, y1 = bounds
    height, width = grid.shape
    pixels = [
        (x, y)
        for y in range(max(0, y0), min(height, y1 + 1))
        for x in range(max(0, x0), min(width, x1 + 1))
        if int(grid[y, x]) == color
    ]
    if not pixels:
        return None
    obj = _object_from_pixels(color, -1, pixels)
    obj["name"] = name
    obj["region"] = [x0, y0, x1, y1]
    obj["canonical_mask"] = _canonical_mask(
        pixels,
        rows=3,
        columns=3,
    )
    return obj


def _canonical_mask(
    pixels: list[tuple[int, int]],
    *,
    rows: int,
    columns: int,
) -> list[list[int]]:
    xs = [x for x, _ in pixels]
    ys = [y for _, y in pixels]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pixel_set = set(pixels)
    mask: list[list[int]] = []
    for row_idx in range(rows):
        cell_y0 = y0 + int(row_idx * (y1 - y0 + 1) / rows)
        cell_y1 = y0 + int((row_idx + 1) * (y1 - y0 + 1) / rows) - 1
        mask_row: list[int] = []
        for column_idx in range(columns):
            cell_x0 = x0 + int(column_idx * (x1 - x0 + 1) / columns)
            cell_x1 = x0 + int((column_idx + 1) * (x1 - x0 + 1) / columns) - 1
            occupied = any(
                (x, y) in pixel_set
                for y in range(cell_y0, max(cell_y0, cell_y1) + 1)
                for x in range(cell_x0, max(cell_x0, cell_x1) + 1)
            )
            mask_row.append(1 if occupied else 0)
        mask.append(mask_row)
    return mask


def _enrich_ls20(processed: ProcessedState) -> dict[str, Any]:
    player = _aggregate_color(processed, 12)
    rotation_switch = _aggregate_color(processed, 1)
    current_shape = _region_color_object(
        processed,
        color=9,
        bounds=(0, 52, 12, 63),
        name="current_shape",
    )
    reference_shape = _region_color_object(
        processed,
        color=9,
        bounds=(30, 8, 42, 16),
        name="reference_shape",
    )
    enriched: dict[str, Any] = {
        "player": player,
        "rotation_switch": rotation_switch,
        "current_shape": current_shape,
        "reference_shape": reference_shape,
    }
    if current_shape is not None and reference_shape is not None:
        enriched["shape_matches_reference"] = (
            current_shape["canonical_mask"] == reference_shape["canonical_mask"]
        )
    else:
        enriched["shape_matches_reference"] = None
    return enriched


_GAME_ENRICHERS: dict[str, ObjectEnricher] = {
    "ls20": _enrich_ls20,
}


def _enrich_for_game(
    game_id: str,
    processed: ProcessedState,
) -> dict[str, Any]:
    base_game_id = game_id.split("-", maxsplit=1)[0].lower()
    enricher = _GAME_ENRICHERS.get(base_game_id)
    if enricher is None:
        return {}
    return enricher(processed)


def _observation_to_plain_dict(observation: Any) -> dict[str, Any]:
    if isinstance(observation, dict):
        return _to_plain_data(observation)
    model_dump = getattr(observation, "model_dump", None)
    if callable(model_dump):
        data = model_dump(mode="json")
    else:
        as_dict = getattr(observation, "dict", None)
        if not callable(as_dict):
            raise TypeError(
                "ARC observations must be dictionaries or SDK/Pydantic models."
            )
        data = as_dict()
    frame = getattr(observation, "frame", None)
    if frame is not None:
        data["frame"] = _to_plain_data(frame)
    return _to_plain_data(data)


def _to_plain_data(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_plain_data(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(val) for val in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None and not isinstance(value, (str, int, float, bool)):
        return _to_plain_data(enum_value)
    return value


def _is_processed_state(observation: Any) -> bool:
    return (
        isinstance(observation, dict)
        and "raw" in observation
        and "grid" in observation
        and "objects_by_color" in observation
    )
