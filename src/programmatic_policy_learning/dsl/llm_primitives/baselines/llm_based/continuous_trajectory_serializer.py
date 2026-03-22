"""Serialize expert trajectories for continuous-action environments."""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

# pylint: disable=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_encoder import (
    ContinuousStateEncoder,
)

# ---------------------------------------------------------------------------
# Relational-fact extraction for continuous 2-D navigation
# ---------------------------------------------------------------------------


def extract_continuous_relational_facts(
    obs: np.ndarray,
    num_passages: int,
) -> list[str]:
    """Derive spatial relationships from a Motion2D observation vector.

    Extracts robot-target and robot-passage geometric facts that help
    the LLM understand the current scene layout.

    Parameters
    ----------
    obs : np.ndarray
        1-D float observation vector from a Motion2D environment.
    num_passages : int
        Number of wall passages; each produces 2 obstacle blocks
        (20 features) in *obs*.

    Returns
    -------
    list[str]
        Human-readable fact strings, e.g.
        ``"robot is left of passage_0 wall (robot_x=0.12 < wall_x=0.50)"``.

    Examples
    --------
    No passages

    >>> obs_p0 = np.zeros(19)
    >>> obs_p0[0], obs_p0[1] = 0.3, 0.5     # robot x, y
    >>> obs_p0[3] = 0.1                       # robot radius
    >>> obs_p0[9], obs_p0[10] = 2.0, 1.0     # target x, y
    >>> obs_p0[17], obs_p0[18] = 0.25, 0.25  # target w, h
    >>> facts = extract_continuous_relational_facts(obs_p0, num_passages=0)
    >>> len(facts)
    4
    >>> facts[0]
    'robot is left of target center (robot_x=0.300 < target_cx=2.125)'
    >>> facts[1]
    'robot is below target center (robot_y=0.500 < target_cy=1.125)'
    >>> facts[2]
    'Manhattan distance to target center = 2.450'
    >>> facts[3]
    'robot is NOT inside target region'

    With one passage (2 obstacles), 3 extra passage-related facts appear.

    >>> obs_p1 = np.zeros(39)
    >>> obs_p1[0], obs_p1[1] = 0.3, 1.0     # robot x, y
    >>> obs_p1[3] = 0.1                       # robot radius
    >>> obs_p1[9], obs_p1[10] = 2.0, 0.5     # target x, y
    >>> obs_p1[17], obs_p1[18] = 0.25, 0.25  # target w, h
    >>> obs_p1[19] = 0.5                      # bottom obstacle x
    >>> obs_p1[20] = 0.0                      # bottom obstacle y
    >>> obs_p1[27] = 0.01                     # bottom obstacle width
    >>> obs_p1[28] = 0.8                      # bottom obstacle height
    >>> obs_p1[29] = 0.5                      # top obstacle x
    >>> obs_p1[30] = 1.7                      # top obstacle y
    >>> obs_p1[37] = 0.01                     # top obstacle width
    >>> obs_p1[38] = 0.8                      # top obstacle height (reaches 2.5)
    >>> facts_p1 = extract_continuous_relational_facts(obs_p1, num_passages=1)
    >>> len(facts_p1)
    7
    >>> facts_p1[4]
    'robot is left of passage_0 wall (robot_x=0.300+r < wall_x=0.500)'
    >>> facts_p1[5]
    'robot y IS aligned with passage_0 gap [0.800+r, 1.700-r] (robot_y=1.000, radius=0.100)'
    >>> facts_p1[6]
    'robot y offset from passage_0 center = -0.250'
    """
    robot_x = float(obs[0])
    robot_y = float(obs[1])
    robot_radius = float(obs[3])
    target_x = float(obs[9])
    target_y = float(obs[10])
    target_width = float(obs[17])
    target_height = float(obs[18])

    target_cx = target_x + target_width / 2.0
    target_cy = target_y + target_height / 2.0

    facts: list[str] = []

    # 1. Robot vs. target center position
    # 1.1 Robot vs. target center x
    if robot_x < target_cx:
        facts.append(
            f"robot is left of target center "
            f"(robot_x={robot_x:.3f} < target_cx={target_cx:.3f})"
        )
    else:
        facts.append(
            f"robot is right of target center "
            f"(robot_x={robot_x:.3f} >= target_cx={target_cx:.3f})"
        )
    # 1.2 Robot vs. target center y
    if robot_y < target_cy:
        facts.append(
            f"robot is below target center "
            f"(robot_y={robot_y:.3f} < target_cy={target_cy:.3f})"
        )
    else:
        facts.append(
            f"robot is above target center "
            f"(robot_y={robot_y:.3f} >= target_cy={target_cy:.3f})"
        )

    dist_to_target = abs(robot_x - target_cx) + abs(robot_y - target_cy)
    facts.append(f"Manhattan distance to target center = {dist_to_target:.3f}")

    # 2. Robot vs. target region overlap check
    # (x, y) is the bottom-left corner of the rectangle.
    target_left = target_x
    target_right = target_x + target_width
    target_bottom = target_y
    target_top = target_y + target_height
    in_target = (
        target_left <= robot_x <= target_right
        and target_bottom <= robot_y <= target_top
    )
    facts.append(f"robot {'IS' if in_target else 'is NOT'} inside target region")

    # 3. Robot vs. each passage
    for i in range(num_passages):
        bot_base = 19 + 20 * i
        top_base = bot_base + 10

        wall_x = float(obs[bot_base])
        bot_y = float(obs[bot_base + 1])
        wall_width = float(obs[bot_base + 8])
        bot_height = float(obs[bot_base + 9])
        top_y = float(obs[top_base + 1])
        gap_bottom = bot_y + bot_height
        gap_top = top_y

        wall_right = wall_x + wall_width
        clear_x = wall_right + robot_radius
        if robot_x + robot_radius < wall_x:
            facts.append(
                f"robot is left of passage_{i} wall "
                f"(robot_x={robot_x:.3f}+r < wall_x={wall_x:.3f})"
            )
        elif robot_x < clear_x:
            facts.append(
                f"robot is inside passage_{i} zone "
                f"(wall_x={wall_x:.3f} <= robot_x={robot_x:.3f} "
                f"< wall_right={wall_right:.3f}+r)"
            )
        else:
            facts.append(
                f"robot has passed passage_{i} wall "
                f"(robot_x={robot_x:.3f} >= wall_right={wall_right:.3f}+r)"
            )

        y_aligned = (gap_bottom + robot_radius) <= robot_y <= (gap_top - robot_radius)
        facts.append(
            f"robot y {'IS' if y_aligned else 'is NOT'} aligned with passage_{i} gap "
            f"[{gap_bottom:.3f}+r, {gap_top:.3f}-r] "
            f"(robot_y={robot_y:.3f}, radius={robot_radius:.3f})"
        )

        passage_center_y = (gap_bottom + gap_top) / 2.0
        y_offset = robot_y - passage_center_y
        facts.append(f"robot y offset from passage_{i} center = {y_offset:+.3f}")

    return facts


# ---------------------------------------------------------------------------
# Trajectory → text  (analogous to trajectory_serializer.trajectory_to_text)
# ---------------------------------------------------------------------------

_CENTER_X_CANDIDATES = ("center_x", "cx", "x")
_CENTER_Y_CANDIDATES = ("center_y", "cy", "y")
_WIDTH_CANDIDATES = ("width", "w", "gripper_width")
_HEIGHT_CANDIDATES = ("height", "h", "gripper_height")
_RADIUS_CANDIDATES = ("radius", "r", "base_radius", "robot_base_radius")
_POSE_ATTRS = ("x", "y", "theta")
_CHANGE_THRESHOLD = 1e-4
_RELATION_DELTA_THRESHOLD = 1e-4


def _natural_sort_key(text: str) -> list[Any]:
    parts = re.split(r"(\d+)", text)
    return [int(part) if part.isdigit() else part for part in parts]


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_numeric_scalar(value: Any) -> bool:
    value = _to_python_scalar(value)
    return isinstance(value, (int, float, bool, np.number))


def _to_float(value: Any) -> float | None:
    value = _to_python_scalar(value)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return None


def _format_value(value: Any, precision: int = 3) -> str:
    value = _to_python_scalar(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{precision}f}"
    return str(value)


def _format_delta(value: float, precision: int = 3) -> str:
    return f"{value:+.{precision}f}"


def _normalize_attr_name(attr_name: str) -> str:
    if attr_name == "base_radius":
        return "radius"
    return attr_name


def _split_object_attr(field_name: str) -> tuple[str, str]:
    if "_" not in field_name:
        return "global", field_name
    object_name, attr_name = field_name.split("_", maxsplit=1)
    return object_name, _normalize_attr_name(attr_name)


def _infer_object_type(object_name: str, attrs: Mapping[str, Any]) -> str:
    if "type" in attrs:
        return str(attrs["type"])
    suffix_match = re.match(r"([a-zA-Z_]+)\d+$", object_name)
    if suffix_match:
        return suffix_match.group(1).rstrip("_")
    return object_name


def _ordered_object_names(objects: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted(
        objects,
        key=lambda name: (0 if name == "robot" else 1, _natural_sort_key(name)),
    )


def _mapping_looks_object_centric(obs: Mapping[str, Any]) -> bool:
    if not obs:
        return False
    return any(isinstance(value, Mapping) for value in obs.values())


def _extract_objects_from_flat_mapping(
    obs: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for field_name, value in obs.items():
        if not _is_numeric_scalar(value):
            continue
        object_name, attr_name = _split_object_attr(str(field_name))
        record = objects.setdefault(object_name, {})
        record[attr_name] = _to_python_scalar(value)
        record["type"] = _infer_object_type(object_name, record)
    return objects


def _extract_objects_from_vector(
    obs: np.ndarray,
    encoder: ContinuousStateEncoder | None,
) -> dict[str, dict[str, Any]]:
    field_names = list(encoder.cfg.obs_field_names) if encoder is not None else []
    objects: dict[str, dict[str, Any]] = {}
    flat_obs = np.asarray(obs).reshape(-1)
    for i, value in enumerate(flat_obs):
        field_name = field_names[i] if i < len(field_names) else f"obs_{i}"
        object_name, attr_name = _split_object_attr(field_name)
        record = objects.setdefault(object_name, {})
        record[attr_name] = _to_python_scalar(value)
        record["type"] = _infer_object_type(object_name, record)
    return objects


def _extract_objects_from_object_mapping(
    obs: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for object_name, value in obs.items():
        if not isinstance(value, Mapping):
            continue
        record = {
            str(attr_name): _to_python_scalar(attr_value)
            for attr_name, attr_value in value.items()
            if _is_numeric_scalar(attr_value) or attr_name in {"type", "name"}
        }
        stable_name = str(record.pop("name", object_name))
        record["type"] = _infer_object_type(stable_name, record)
        objects[stable_name] = record
    return objects


def _extract_objects_from_sequence(
    obs: Sequence[Any],
) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(obs):
        if not isinstance(value, Mapping):
            continue
        record = {
            str(attr_name): _to_python_scalar(attr_value)
            for attr_name, attr_value in value.items()
            if _is_numeric_scalar(attr_value) or attr_name in {"type", "name"}
        }
        stable_name = str(record.pop("name", f"obj{index}"))
        record["type"] = _infer_object_type(stable_name, record)
        objects[stable_name] = record
    return objects


def _extract_objects(
    obs: Any,
    encoder: ContinuousStateEncoder | None,
) -> dict[str, dict[str, Any]]:
    if isinstance(obs, np.ndarray):
        return _extract_objects_from_vector(obs, encoder)
    if isinstance(obs, Mapping):
        if _mapping_looks_object_centric(obs):
            objects = _extract_objects_from_object_mapping(obs)
            if objects:
                return objects
        return _extract_objects_from_flat_mapping(obs)
    if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes)):
        objects = _extract_objects_from_sequence(obs)
        if objects:
            return objects
    raise TypeError(f"Unsupported observation type for enc_5: {type(obs)!r}")


def _get_attr(attrs: Mapping[str, Any], candidates: Sequence[str]) -> Any | None:
    for candidate in candidates:
        if candidate in attrs:
            return attrs[candidate]
    return None


def _object_center(attrs: Mapping[str, Any]) -> tuple[float, float] | None:
    raw_x = _get_attr(attrs, _CENTER_X_CANDIDATES)
    raw_y = _get_attr(attrs, _CENTER_Y_CANDIDATES)
    x = _to_float(raw_x)
    y = _to_float(raw_y)
    if x is None or y is None:
        return None

    width = _to_float(_get_attr(attrs, _WIDTH_CANDIDATES))
    height = _to_float(_get_attr(attrs, _HEIGHT_CANDIDATES))
    radius = _to_float(_get_attr(attrs, _RADIUS_CANDIDATES))

    if radius is not None:
        return x, y
    if width is not None and height is not None:
        return x + width / 2.0, y + height / 2.0
    return x, y


def _object_bbox(attrs: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    raw_x = _get_attr(attrs, _CENTER_X_CANDIDATES)
    raw_y = _get_attr(attrs, _CENTER_Y_CANDIDATES)
    x = _to_float(raw_x)
    y = _to_float(raw_y)
    if x is None or y is None:
        return None

    width = _to_float(_get_attr(attrs, _WIDTH_CANDIDATES))
    height = _to_float(_get_attr(attrs, _HEIGHT_CANDIDATES))
    radius = _to_float(_get_attr(attrs, _RADIUS_CANDIDATES))

    if radius is not None:
        return x - radius, y - radius, x + radius, y + radius
    if width is not None and height is not None:
        return x, y, x + width, y + height
    return None


def _boxes_intersect(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> bool:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    return ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1


def _compute_relation(
    object_a: Mapping[str, Any],
    object_b: Mapping[str, Any],
) -> dict[str, Any]:
    relation: dict[str, Any] = {}
    center_a = _object_center(object_a)
    center_b = _object_center(object_b)
    if center_a is not None and center_b is not None:
        ax, ay = center_a
        bx, by = center_b
        dx = bx - ax
        dy = by - ay
        relation["center_dx"] = dx
        relation["center_dy"] = dy
        relation["euclidean_dist"] = float(np.hypot(dx, dy))
        relation["left_of"] = ax < bx
        relation["right_of"] = ax > bx
        relation["above"] = ay > by
        relation["below"] = ay < by

    box_a = _object_bbox(object_a)
    box_b = _object_bbox(object_b)
    if box_a is not None and box_b is not None:
        relation["intersects"] = _boxes_intersect(box_a, box_b)
    return relation


def _relation_line(
    name_a: str,
    name_b: str,
    relation: Mapping[str, Any],
    precision: int,
) -> str:
    ordered_keys = (
        "center_dx",
        "center_dy",
        "euclidean_dist",
        "left_of",
        "right_of",
        "above",
        "below",
        "intersects",
    )
    entries = [
        f"{key}={_format_value(relation[key], precision)}"
        for key in ordered_keys
        if key in relation
    ]
    return f"rel({name_a}, {name_b}): " + ", ".join(entries)


def _relation_delta_lines(
    name_a: str,
    name_b: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    precision: int,
) -> list[str]:
    lines: list[str] = []

    before_dist = _to_float(before.get("euclidean_dist"))
    after_dist = _to_float(after.get("euclidean_dist"))
    if before_dist is not None and after_dist is not None:
        dist_change = after_dist - before_dist
        lines.append(
            f"dist_change({name_a}, {name_b})={_format_delta(dist_change, precision)}"
        )
        lines.append(
            f"moved_toward({name_a}, {name_b})="
            f"{str(after_dist < before_dist - _RELATION_DELTA_THRESHOLD).lower()}"
        )

    for key in ("left_of", "right_of", "above", "below", "intersects"):
        if key in before and key in after and before[key] != after[key]:
            lines.append(
                f"{key}({name_a}, {name_b}): "
                f"{_format_value(before[key], precision)} -> "
                f"{_format_value(after[key], precision)}"
            )

    for key in ("center_dx", "center_dy"):
        before_value = _to_float(before.get(key))
        after_value = _to_float(after.get(key))
        if (
            before_value is not None
            and after_value is not None
            and abs(after_value - before_value) >= _RELATION_DELTA_THRESHOLD
        ):
            lines.append(
                f"{key}_change({name_a}, {name_b})="
                f"{_format_delta(after_value - before_value, precision)}"
            )
    return lines


def _format_object_line(
    object_name: str,
    attrs: Mapping[str, Any],
    precision: int,
) -> str:
    ordered_attrs: list[str] = []
    for attr_name in ("type", "x", "y", "theta", "radius", "width", "height"):
        if attr_name in attrs:
            ordered_attrs.append(attr_name)
    for attr_name in sorted(attrs, key=_natural_sort_key):
        if attr_name not in ordered_attrs:
            ordered_attrs.append(attr_name)
    entries = [
        f"{attr_name}={_format_value(attrs[attr_name], precision)}"
        for attr_name in ordered_attrs
    ]
    return f"{object_name}(" + ", ".join(entries) + ")"


def _format_action_lines(
    action: Any,
    encoder: ContinuousStateEncoder | None,
    precision: int,
) -> list[str]:
    if action is None:
        return ["Action: None (terminal state)"]
    if isinstance(action, np.ndarray):
        flat_action = np.asarray(action).reshape(-1)
        field_names = (
            list(encoder.cfg.action_field_names) if encoder is not None else []
        )
        entries = []
        for index, value in enumerate(flat_action):
            name = field_names[index] if index < len(field_names) else f"a{index}"
            entries.append(f"- {name}={_format_value(value, precision)}")
        return ["Action:"] + entries
    if isinstance(action, Mapping):
        entries = [
            f"- {name}={_format_value(value, precision)}"
            for name, value in sorted(
                action.items(),
                key=lambda item: _natural_sort_key(str(item[0])),
            )
        ]
        return ["Action:"] + entries
    return [f"Action: {_format_value(action, precision)}"]


def _object_change_lines(
    before_objects: Mapping[str, Mapping[str, Any]],
    after_objects: Mapping[str, Mapping[str, Any]],
    precision: int,
) -> tuple[list[str], set[str]]:
    lines: list[str] = []
    changed_objects: set[str] = set()
    all_object_names = sorted(
        set(before_objects) | set(after_objects),
        key=lambda name: (0 if name == "robot" else 1, _natural_sort_key(name)),
    )
    for object_name in all_object_names:
        before_attrs = before_objects.get(object_name)
        after_attrs = after_objects.get(object_name)
        if before_attrs is None or after_attrs is None:
            changed_objects.add(object_name)
            status = "added" if before_attrs is None else "removed"
            lines.append(f"{object_name}: {status}")
            continue
        attr_names = sorted(set(before_attrs) | set(after_attrs), key=_natural_sort_key)
        for attr_name in attr_names:
            if attr_name == "type":
                continue
            before_value = before_attrs.get(attr_name)
            after_value = after_attrs.get(attr_name)
            if before_value == after_value:
                continue
            before_float = _to_float(before_value)
            after_float = _to_float(after_value)
            if (
                before_float is not None
                and after_float is not None
                and abs(after_float - before_float) < _CHANGE_THRESHOLD
            ):
                continue
            changed_objects.add(object_name)
            line = (
                f"{object_name}.{attr_name}: "
                f"{_format_value(before_value, precision)} -> "
                f"{_format_value(after_value, precision)}"
            )
            if before_float is not None and after_float is not None:
                line += (
                    f" (delta={_format_delta(after_float - before_float, precision)})"
                )
            lines.append(line)
    return lines, changed_objects


def _select_relation_pairs(
    objects: Mapping[str, Mapping[str, Any]],
    changed_objects: set[str],
) -> list[tuple[str, str]]:
    object_names = _ordered_object_names(objects)
    if len(object_names) < 2:
        return []

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def maybe_add(pair: tuple[str, str]) -> None:
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    if "robot" in objects:
        for other_name in object_names:
            if other_name != "robot":
                maybe_add(("robot", other_name))

    non_robot = [name for name in object_names if name != "robot"]
    important_non_robot = [name for name in non_robot if name in changed_objects][:3]
    for name_a, name_b in combinations(important_non_robot, 2):
        maybe_add((name_a, name_b))

    if not pairs:
        for pair in combinations(object_names[:4], 2):
            maybe_add(pair)
    return pairs[:6]


def _robot_pose_summary(
    objects: Mapping[str, Mapping[str, Any]],
    precision: int,
) -> str | None:
    robot = objects.get("robot")
    if robot is None:
        return None
    pose_parts = [
        f"{attr}={_format_value(robot[attr], precision)}"
        for attr in _POSE_ATTRS
        if attr in robot
    ]
    if not pose_parts:
        return None
    return ", ".join(pose_parts)


def enc_5(
    trajectory: list[tuple[Any, Any, Any]],
    *,
    encoder: ContinuousStateEncoder | None,
) -> str:
    """Object-centric continuous trajectory encoding for KinDER-style tasks."""
    if not trajectory:
        raise ValueError("trajectory must be non-empty")

    steps_with_idx = list(enumerate(trajectory))
    precision = encoder.cfg.precision if encoder is not None else 3

    initial_objects = _extract_objects(steps_with_idx[0][1][0], encoder)
    final_objects = _extract_objects(steps_with_idx[-1][1][2], encoder)
    object_types = sorted(
        {
            str(attrs.get("type", _infer_object_type(name, attrs)))
            for name, attrs in {**initial_objects, **final_objects}.items()
        },
        key=_natural_sort_key,
    )

    trajectory_changed_objects: set[str] = set()
    for _, (obs_t, _, obs_t1) in steps_with_idx:
        _, changed_objects = _object_change_lines(
            _extract_objects(obs_t, encoder),
            _extract_objects(obs_t1, encoder),
            precision,
        )
        trajectory_changed_objects.update(changed_objects)

    summary_lines = [
        "Trajectory summary:",
        f"- number of shown steps: {len(steps_with_idx) + 1}",
        f"- object types present: {', '.join(object_types) if object_types else '(unknown)'}",
    ]
    initial_robot_pose = _robot_pose_summary(initial_objects, precision)
    final_robot_pose = _robot_pose_summary(final_objects, precision)
    if initial_robot_pose is not None:
        summary_lines.append(f"- initial robot pose: {initial_robot_pose}")
    if final_robot_pose is not None:
        summary_lines.append(f"- final robot pose: {final_robot_pose}")
    summary_lines.append(
        "- changed objects: "
        + (
            ", ".join(
                sorted(
                    trajectory_changed_objects,
                    key=lambda name: (
                        0 if name == "robot" else 1,
                        _natural_sort_key(name),
                    ),
                )
            )
            if trajectory_changed_objects
            else "(none)"
        )
    )

    blocks: list[str] = ["\n".join(summary_lines)]

    for original_idx, (obs_t, action, obs_t1) in steps_with_idx:
        before_objects = _extract_objects(obs_t, encoder)
        after_objects = _extract_objects(obs_t1, encoder)
        object_change_lines, changed_objects = _object_change_lines(
            before_objects,
            after_objects,
            precision,
        )
        relation_pairs = _select_relation_pairs(before_objects, changed_objects)

        block_lines = [f"*** Step {original_idx} ***", "", "Objects:"]
        for object_name in _ordered_object_names(before_objects):
            block_lines.append(
                f"- {_format_object_line(object_name, before_objects[object_name], precision)}"
            )
        block_lines.extend(["", *_format_action_lines(action, encoder, precision), ""])

        if object_change_lines:
            block_lines.append("Object changes:")
            block_lines.extend(f"- {line}" for line in object_change_lines)
        else:
            block_lines.extend(["Object changes:", "- (no changed attributes)"])

        if relation_pairs:
            before_relations = {
                pair: _compute_relation(
                    before_objects[pair[0]], before_objects[pair[1]]
                )
                for pair in relation_pairs
            }
            after_relations = {
                pair: _compute_relation(after_objects[pair[0]], after_objects[pair[1]])
                for pair in relation_pairs
                if pair[0] in after_objects and pair[1] in after_objects
            }

            block_lines.extend(["", "Pairwise relations before action:"])
            block_lines.extend(
                f"- {_relation_line(name_a, name_b, relation, precision)}"
                for (name_a, name_b), relation in before_relations.items()
                if relation
            )

            after_relation_lines = [
                f"- {_relation_line(name_a, name_b, relation, precision)}"
                for (name_a, name_b), relation in after_relations.items()
                if relation
            ]
            if after_relation_lines:
                block_lines.extend(["", "Pairwise relations after action:"])
                block_lines.extend(after_relation_lines)

            delta_lines: list[str] = []
            for pair in relation_pairs:
                if pair in before_relations and pair in after_relations:
                    delta_lines.extend(
                        _relation_delta_lines(
                            pair[0],
                            pair[1],
                            before_relations[pair],
                            after_relations[pair],
                            precision,
                        )
                    )
            if delta_lines:
                block_lines.extend(["", "Relation deltas:"])
                block_lines.extend(f"- {line}" for line in delta_lines)

        blocks.append("\n".join(block_lines))

    last_original_idx, (_, _, _final_obs) = steps_with_idx[-1]
    final_step_lines = [f"*** Step {last_original_idx + 1} ***", "", "Objects:"]
    for object_name in _ordered_object_names(final_objects):
        final_step_lines.append(
            f"- {_format_object_line(object_name, final_objects[object_name], precision)}"
        )
    final_step_lines.extend(["", "Action: None (terminal state)"])
    blocks.append("\n".join(final_step_lines))

    return "\n\n".join(blocks)


def trajectory_to_text(
    trajectory: list[tuple[Any, Any, Any]],
    *,
    encoder: ContinuousStateEncoder,
    num_passages: int,
    encoding_method: str,
    max_steps: int | None = None,
    skip_rate: int = 1,
) -> str:
    """Convert ``(obs, action, obs_next)`` tuples to structured text.

    Produces the trajectory portion of the LLM prompt.  A final
    observation with ``Action: None (terminal state).`` is always
    appended so the LLM can see the goal state.

    Parameters
    ----------
    trajectory : list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of ``(obs_t, action_t, obs_{t+1})`` transitions collected
        from an expert rollout.
    encoder : ContinuousStateEncoder
        Encoder used to render observations and actions as named fields.
    num_passages : int
        Number of wall passages in the environment, forwarded to
        :func:`extract_continuous_relational_facts`.
    encoding_method : str
        ``"1"`` - raw numeric dump.
        ``"2"`` - named-field listing.
        ``"3"`` - named-field + change summary.
        ``"4"`` - named-field + change summary + relational facts.
    max_steps : int | None, optional
        Maximum number of transitions to include (default is ``None``,
        meaning all steps).
    skip_rate : int, optional
        Sub-sampling rate.  When > 1, only every *skip_rate*-th frame
        is kept (plus the true final step), and a header is prepended
        to inform the LLM about the sub-sampling (default is 1).

    Returns
    -------
    str
        Multi-block text representation of the trajectory, with blocks
        separated by blank lines.

    Raises
    ------
    ValueError
        If ``trajectory`` is empty.

    Examples
    --------
    A single-step trajectory in a ``Motion2D-p1`` environment with
    ``encoding_method="4"`` (named fields + change summary + relational
    facts).  The text contains **two** blocks: Step 0 (the transition)
    and Step 1 (the terminal observation).

    >>> from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_encoder import (
    ...     ContinuousStateEncoder, ContinuousStateEncoderConfig,
    ... )
    >>> from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_hint_config import (
    ...     obs_field_names_for_motion2d, ACTION_FIELD_NAMES,
    ...     salient_obs_indices_for_motion2d,
    ... )
    >>> cfg = ContinuousStateEncoderConfig(
    ...     obs_field_names=obs_field_names_for_motion2d(1),
    ...     action_field_names=ACTION_FIELD_NAMES["Motion2D"],
    ...     salient_indices=salient_obs_indices_for_motion2d(1),
    ... )
    >>> enc = ContinuousStateEncoder(cfg)
    >>> obs0 = np.zeros(39)
    >>> obs0[0], obs0[1] = 0.3, 1.0        # robot x, y
    >>> obs0[3] = 0.1                        # robot radius
    >>> obs0[9], obs0[10] = 2.0, 0.5        # target x, y (bottom-left)
    >>> obs0[17], obs0[18] = 0.25, 0.25     # target w, h
    >>> obs0[19] = 0.5; obs0[20] = 0.0      # bottom obstacle x, y
    >>> obs0[27] = 0.01; obs0[28] = 0.8     # bottom obstacle w, h
    >>> obs0[29] = 0.5; obs0[30] = 1.7      # top obstacle x, y
    >>> obs0[37] = 0.01; obs0[38] = 0.8     # top obstacle w, h
    >>> act0 = np.zeros(5)
    >>> act0[0], act0[1] = 0.05, -0.02      # dx, dy
    >>> obs1 = obs0.copy()
    >>> obs1[0] += 0.05; obs1[1] -= 0.02
    >>> traj = [(obs0, act0, obs1)]
    >>> text = trajectory_to_text(
    ...     traj, encoder=enc, num_passages=1, encoding_method="4",
    ... )

    The output text has two blocks separated by blank lines.  Each block
    is printed below (long observation lines are wrapped by the terminal
    but are single lines in the actual string).

    >>> print(text)  # doctest: +NORMALIZE_WHITESPACE
    *** Step 0 ***
    Observation (s_0):
      obs[0] (robot_x)=0.300, obs[1] (robot_y)=1.000,
      obs[3] (robot_base_radius)=0.100,
      obs[9] (target_x)=2.000, obs[10] (target_y)=0.500,
      obs[17] (target_width)=0.250, obs[18] (target_height)=0.250,
      obs[19] (obstacle0_x)=0.500, obs[20] (obstacle0_y)=0.000,
      obs[27] (obstacle0_width)=0.010, obs[28] (obstacle0_height)=0.800,
      obs[29] (obstacle1_x)=0.500, obs[30] (obstacle1_y)=1.700,
      obs[37] (obstacle1_width)=0.010, obs[38] (obstacle1_height)=0.800
    <BLANKLINE>
    Action: [dx=0.050, dy=-0.020, dtheta=0.000, darm=0.000, vac=0.000]
    <BLANKLINE>
    CHANGES SUMMARY:
    - obs[0] (robot_x): 0.300 -> 0.350 (delta=+0.050)
    - obs[1] (robot_y): 1.000 -> 0.980 (delta=-0.020)
    <BLANKLINE>
    Relational Facts:
    - robot is left of target center (robot_x=0.300 < target_cx=2.125)
    - robot is above target center (robot_y=1.000 >= target_cy=0.625)
    - Manhattan distance to target center = 2.200
    - robot is NOT inside target region
    - robot is left of passage_0 wall (robot_x=0.300+r < wall_x=0.500)
    - robot y IS aligned with passage_0 gap
      [0.800+r, 1.700-r] (robot_y=1.000, radius=0.100)
    - robot y offset from passage_0 center = -0.250
    <BLANKLINE>
    *** Step 1 ***
    Observation (s_1):
      obs[0] (robot_x)=0.350, obs[1] (robot_y)=0.980,
      obs[3] (robot_base_radius)=0.100,
      obs[9] (target_x)=2.000, obs[10] (target_y)=0.500,
      obs[17] (target_width)=0.250, obs[18] (target_height)=0.250,
      obs[19] (obstacle0_x)=0.500, obs[20] (obstacle0_y)=0.000,
      obs[27] (obstacle0_width)=0.010, obs[28] (obstacle0_height)=0.800,
      obs[29] (obstacle1_x)=0.500, obs[30] (obstacle1_y)=1.700,
      obs[37] (obstacle1_width)=0.010, obs[38] (obstacle1_height)=0.800
    <BLANKLINE>
    Action: None (terminal state).
    <BLANKLINE>
    Relational Facts:
    - robot is left of target center (robot_x=0.350 < target_cx=2.125)
    - robot is above target center (robot_y=0.980 >= target_cy=0.625)
    - Manhattan distance to target center = 2.130
    - robot is NOT inside target region
    - robot is left of passage_0 wall (robot_x=0.350+r < wall_x=0.500)
    - robot y IS aligned with passage_0 gap
      [0.800+r, 1.700-r] (robot_y=0.980, radius=0.100)
    - robot y offset from passage_0 center = -0.270
    """
    del max_steps
    del skip_rate

    if not trajectory:
        raise ValueError("trajectory must be non-empty")

    if encoding_method == "5":
        return enc_5(trajectory, encoder=encoder)

    steps_with_idx = list(enumerate(trajectory))

    blocks: list[str] = []

    for original_idx, (obs_t, action, obs_t1) in steps_with_idx:
        if encoding_method == "1":
            blocks.append(
                f"*** Step {original_idx} ***\n"
                f"obs = {np.array2string(obs_t, precision=4, separator=', ')}\n"
                f"action = {np.array2string(action, precision=4, separator=', ')}"
            )

        elif encoding_method == "2":
            blocks.append(encoder.encode_step(obs_t, action, original_idx))

        elif encoding_method == "3":
            blocks.append(encoder.encode_step(obs_t, action, original_idx))
            changes = encoder.compute_deltas(obs_t, obs_t1)
            if changes:
                blocks.append(
                    "CHANGES SUMMARY:\n" + "\n".join(f"- {c}" for c in changes)
                )
            else:
                blocks.append("CHANGES SUMMARY:\n- (no changes)")

        elif encoding_method == "4":
            blocks.append(encoder.encode_step(obs_t, action, original_idx))

            changes = encoder.compute_deltas(obs_t, obs_t1)
            if changes:
                blocks.append(
                    "CHANGES SUMMARY:\n" + "\n".join(f"- {c}" for c in changes)
                )
            else:
                blocks.append("CHANGES SUMMARY:\n- (no changes)")

            rel_facts = extract_continuous_relational_facts(obs_t, num_passages)
            blocks.append(
                "Relational Facts:\n" + "\n".join(f"- {f}" for f in rel_facts)
            )

    # Final observation (obs_next of the last transition)
    last_original_idx, (_, _, final_obs) = steps_with_idx[-1]
    final_display_idx = last_original_idx + 1
    if encoding_method == "1":
        blocks.append(
            f"*** Step {final_display_idx} ***\n"
            f"obs = {np.array2string(final_obs, precision=4, separator=', ')}\n"
            f"action = None (terminal state)"
        )
    else:  # TODOO: later clean this up to assert if the encoding is not available
        blocks.append(encoder.encode_obs(final_obs, final_display_idx))
        blocks.append("Action: None (terminal state).")
        if encoding_method == "4":
            rel_facts = extract_continuous_relational_facts(final_obs, num_passages)
            blocks.append(
                "Relational Facts:\n" + "\n".join(f"- {f}" for f in rel_facts)
            )

    return "\n\n".join(blocks)
