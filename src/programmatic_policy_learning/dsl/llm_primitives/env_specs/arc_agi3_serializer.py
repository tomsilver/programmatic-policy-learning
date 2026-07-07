"""Prompt serialization for ARC-AGI-3 demonstrations."""

from __future__ import annotations

from typing import Any

ACTION_NAMES = {
    0: "noop",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}

OBJECT_KEYS = (
    "player",
    "rotation_switch",
    "current_shape",
    "reference_shape",
)

_LARGE_STATE_KEYS = {
    "raw",
    "grid",
    "objects_by_color",
    "frame",
}


def trajectory_to_text(
    trajectory: list[tuple[Any, Any, Any]],
    *,
    max_steps: int = 50,
) -> str:
    """Render one ARC-AGI-3 trajectory into compact object-centric text."""
    lines: list[str] = []
    for step_idx, (obs, action, next_obs) in enumerate(trajectory[:max_steps]):
        state = _as_mapping(obs)
        next_state = _as_mapping(next_obs)
        action_id = _action_id(action)
        lines.extend(
            [
                f"step {step_idx}:",
                f"  expert_action: {action_id} ({ACTION_NAMES.get(action_id, 'unknown')})",
                f"  available_actions: {_available_actions(state)}",
                f"  previous_action: {_previous_action(state)}",
                f"  full_reset: {_format_scalar(state.get('full_reset'))}",
                f"  levels_completed: {_format_scalar(state.get('levels_completed'))}",
                "",
                "  objects:",
            ]
        )
        for object_key in OBJECT_KEYS:
            lines.append(
                f"    {object_key}: {_format_object(object_key, state.get(object_key))}"
            )

        flag_lines = _state_flag_lines(state)
        lines.extend(["", "  state_flags:"])
        lines.extend(flag_lines or ["    none"])

        relation_lines = _relation_lines(state)
        lines.extend(["", "  useful_relations:"])
        lines.extend(relation_lines or ["    none"])

        crop_lines = _local_crop_lines(state)
        lines.extend(["", "  local_crops:"])
        lines.extend(crop_lines or ["    omitted"])

        transition_lines = _transition_lines(state, next_state)
        lines.extend(["", "  transition:"])
        lines.extend(transition_lines or ["    omitted"])
        lines.append("")
    return "\n".join(lines).rstrip()


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _action_id(action: Any) -> int:
    raw_value = getattr(action, "value", action)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return -1


def _available_actions(state: dict[str, Any]) -> list[int]:
    actions = state.get("available_actions") or []
    return [_action_id(action) for action in actions]


def _previous_action(state: dict[str, Any]) -> int | None:
    action_input = state.get("action_input")
    if isinstance(action_input, dict):
        action_id = action_input.get("id")
        if action_id is not None:
            return _action_id(action_id)
    return None


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "None"
    return repr(value)


def _format_object(object_key: str, obj: Any) -> str:
    if not isinstance(obj, dict):
        return "None"

    fields: list[str] = []
    for key in ("color", "bbox", "center", "width", "height", "area"):
        if key in obj:
            fields.append(f"{key}={_compact_value(obj[key])}")

    mask = obj.get("canonical_mask")
    if mask is None and object_key != "player":
        mask = obj.get("mask")
    if mask is not None:
        fields.append(f"canonical_mask={_compact_value(mask)}")
    return "{" + ", ".join(fields) + "}"


def _compact_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return "[" + ", ".join(_compact_value(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "[" + ", ".join(_compact_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return str(value).lower()
    return repr(value)


def _state_flag_lines(state: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if "shape_matches_reference" in state:
        lines.append(
            "    shape_matches_reference: "
            f"{_format_scalar(state.get('shape_matches_reference'))}"
        )

    for key in sorted(state):
        if key in _LARGE_STATE_KEYS or key in OBJECT_KEYS:
            continue
        if key in {
            "shape_matches_reference",
            "available_actions",
            "action_input",
            "full_reset",
            "levels_completed",
            "preprocessing",
        }:
            continue
        value = state[key]
        if isinstance(value, bool) or value is None:
            lines.append(f"    {key}: {_format_scalar(value)}")
    return lines


def _relation_lines(state: dict[str, Any]) -> list[str]:
    player = _object_center(state.get("player"))
    switch = _object_center(state.get("rotation_switch"))
    current = _object_center(state.get("current_shape"))
    reference = _object_center(state.get("reference_shape"))
    lines: list[str] = []

    if player is not None and switch is not None:
        dx, dy = _dxdy(player, switch)
        lines.append(f"    player_to_rotation_switch_dxdy: [{dx:.3f}, {dy:.3f}]")
        lines.append(
            "    player_vs_rotation_switch: "
            f"{_relative_position_text(dx, dy, target_name='rotation_switch')}"
        )
    if player is not None and reference is not None:
        dx, dy = _dxdy(player, reference)
        lines.append(f"    player_to_reference_shape_dxdy: [{dx:.3f}, {dy:.3f}]")
        lines.append(
            "    player_vs_reference_shape: "
            f"{_relative_position_text(dx, dy, target_name='reference_shape')}"
        )
    if current is not None and reference is not None:
        dx, dy = _dxdy(current, reference)
        lines.append(f"    current_shape_to_reference_shape_dxdy: [{dx:.3f}, {dy:.3f}]")

    match = state.get("shape_matches_reference")
    if match is not None:
        verb = "matches" if bool(match) else "does_not_match"
        lines.append(f"    current_shape_matches_reference_shape: {verb}")
    return lines


def _object_center(obj: Any) -> tuple[float, float] | None:
    if not isinstance(obj, dict):
        return None
    center = obj.get("center")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return None
    try:
        return (float(center[0]), float(center[1]))
    except (TypeError, ValueError):
        return None


def _dxdy(
    source_center: tuple[float, float],
    target_center: tuple[float, float],
) -> tuple[float, float]:
    return (
        target_center[0] - source_center[0],
        target_center[1] - source_center[1],
    )


def _relative_position_text(dx: float, dy: float, *, target_name: str) -> str:
    vertical = ""
    horizontal = ""
    if abs(dy) > 1e-6:
        vertical = "above" if dy > 0 else "below"
    if abs(dx) > 1e-6:
        horizontal = "left_of" if dx > 0 else "right_of"
    if vertical and horizontal:
        return f"player_is_{vertical}_and_{horizontal}_{target_name}"
    if vertical:
        return f"player_is_{vertical}_{target_name}"
    if horizontal:
        return f"player_is_{horizontal}_{target_name}"
    return f"player_is_aligned_with_{target_name}"


def _local_crop_lines(state: dict[str, Any]) -> list[str]:
    grid = state.get("grid")
    if not isinstance(grid, list) or not grid:
        return []

    crops = [
        ("player_crop", state.get("player"), 2),
        ("switch_crop", state.get("rotation_switch"), 2),
        ("shape_crop", state.get("current_shape"), 1),
    ]
    lines: list[str] = []
    for name, obj, padding in crops:
        crop = _crop_around_object(grid, obj, padding=padding)
        if crop is not None:
            lines.append(f"    {name}: {_compact_value(crop)}")
    return lines


def _crop_around_object(
    grid: list[Any],
    obj: Any,
    *,
    padding: int,
) -> list[list[int]] | None:
    if not isinstance(obj, dict):
        return None
    bbox = obj.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    height = len(grid)
    width = len(grid[0]) if height else 0
    if width == 0:
        return None
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(width - 1, x1 + padding)
    y1 = min(height - 1, y1 + padding)
    return [[int(grid[y][x]) for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)]


def _transition_lines(
    state: dict[str, Any],
    next_state: dict[str, Any],
    *,
    max_changed_cells: int = 24,
) -> list[str]:
    if not next_state:
        return []

    lines: list[str] = []
    diff = _changed_cells(state.get("grid"), next_state.get("grid"))
    if diff is not None:
        preview = diff[:max_changed_cells]
        suffix = "" if len(diff) <= max_changed_cells else f" ... ({len(diff)} total)"
        lines.append(f"    changed_cells: {_compact_value(preview)}{suffix}")

    summary = _next_state_summary(next_state)
    if summary:
        lines.append(f"    s_next_summary: {summary}")
    return lines


def _changed_cells(grid_a: Any, grid_b: Any) -> list[dict[str, int]] | None:
    if not isinstance(grid_a, list) or not isinstance(grid_b, list):
        return None
    if not grid_a or not grid_b or len(grid_a) != len(grid_b):
        return None
    if any(
        not isinstance(row_a, list)
        or not isinstance(row_b, list)
        or len(row_a) != len(row_b)
        for row_a, row_b in zip(grid_a, grid_b)
    ):
        return None

    changed: list[dict[str, int]] = []
    for y, (row_a, row_b) in enumerate(zip(grid_a, grid_b)):
        for x, (cell_a, cell_b) in enumerate(zip(row_a, row_b)):
            if cell_a != cell_b:
                changed.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "from": int(cell_a),
                        "to": int(cell_b),
                    }
                )
    return changed


def _next_state_summary(next_state: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("levels_completed", "shape_matches_reference", "full_reset"):
        if key in next_state:
            parts.append(f"{key}={_format_scalar(next_state.get(key))}")
    player = _object_center(next_state.get("player"))
    if player is not None:
        parts.append(f"player_center=[{player[0]:.3f}, {player[1]:.3f}]")
    return ", ".join(parts)
