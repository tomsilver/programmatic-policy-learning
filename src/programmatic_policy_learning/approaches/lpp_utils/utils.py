"""Utils for Testing LPP Approach."""

# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long

from __future__ import annotations

import ast
import json
import logging
import re
import signal
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Generator, TypeVar

import numpy as np
from scipy.sparse import csr_matrix

from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    continuous_hint_config,
    grid_hint_config,
)
from programmatic_policy_learning.dsl.llm_primitives.hint_generation.llm_based import (
    hint_extractor,
)
from programmatic_policy_learning.dsl.state_action_program import (
    StateActionProgram,
    set_dsl_functions,
)
from programmatic_policy_learning.utils.grid_validation import require_grid_state_action

GymnasiumEnvType: type | None = None
GymnasiumRecordVideoType: type | None = None
ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
try:
    from gymnasium import Env as GymnasiumEnvType
    from gymnasium.wrappers import RecordVideo as GymnasiumRecordVideoType
except Exception:  # pylint: disable=broad-exception-caught
    GymnasiumEnvType = None
    GymnasiumRecordVideoType = None


def _is_stateful_policy(policy: Any) -> bool:
    """Return whether *policy* exposes the agent-style rollout protocol."""
    return all(hasattr(policy, attr) for attr in ("reset", "step", "update"))


def run_single_episode(
    env: Any,
    policy: Any,
    record_video: bool = False,
    video_out_path: str | None = None,
    max_num_steps: int = 100,
    reset_seed: int | None = None,
) -> tuple[float, np.bool_]:
    """Run a single episode in the environment using the given policy.

    Returns
    -------
    tuple[float, np.bool_]
        ``(total_reward, terminated)`` — cumulative reward and whether the
        episode ended via the environment's termination signal (as opposed
        to reaching *max_num_steps* or being truncated).  ``terminated``
        is ``np.bool_`` (from the environment); callers that need native
        ``bool`` (e.g. for JSON serialization) should cast explicitly.
    """

    record_frames: list[Any] | None = None
    if record_video:
        if hasattr(env, "start_recording_video"):
            env.start_recording_video(video_out_path=video_out_path)
        else:
            if video_out_path is None:
                raise ValueError("video_out_path is required when using RecordVideo.")
            out_path = Path(video_out_path)
            base_env = env
            # Unwrap common wrapper chains (GGGEnvWithTypes -> GymToGymnasium -> gym env)
            while hasattr(base_env, "env"):
                base_env = getattr(base_env, "env")
            if hasattr(base_env, "_env"):
                base_env = getattr(base_env, "_env")

            if (
                GymnasiumEnvType is not None
                and GymnasiumRecordVideoType is not None
                and isinstance(base_env, GymnasiumEnvType)
            ):
                env = GymnasiumRecordVideoType(  # type: ignore[operator]
                    base_env,
                    video_folder=str(out_path.parent),
                    name_prefix=out_path.stem,
                    episode_trigger=lambda _: True,
                )
            else:
                # Fallback: manual frame capture for legacy gym envs.
                record_frames = []

    stateful_policy = _is_stateful_policy(policy)
    if reset_seed is None:
        reset_out = env.reset()
    else:
        try:
            reset_out = env.reset(seed=reset_seed)
        except TypeError:
            reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs = reset_out
        info = {}
    if stateful_policy:
        policy.reset(obs, info)
    if record_frames is not None and hasattr(env, "render"):
        try:
            frame = env.render()
            if frame is not None:
                record_frames.append(frame)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    total_reward = 0.0
    episode_terminated: np.bool_ = np.bool_(False)
    for _ in range(max_num_steps):
        action = policy.step() if stateful_policy else policy(obs)
        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 4:
            new_obs, reward, done, info = step_out
            terminated, truncated = done, False
        else:
            new_obs, reward, terminated, truncated, info = step_out
        total_reward += reward
        if stateful_policy:
            policy.update(new_obs, reward, terminated or truncated, info)

        obs = new_obs
        if record_frames is not None and hasattr(env, "render"):
            try:
                frame = env.render()
                if frame is not None:
                    record_frames.append(frame)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        if terminated or truncated:
            episode_terminated = terminated
            break
    env.close()

    if record_frames is not None and video_out_path is not None:
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception as e:  # pylint: disable=broad-exception-caught
            raise RuntimeError(
                "Manual video recording requires imageio. "
                "Install with `pip install imageio`."
            ) from e
        if record_frames:
            imageio.mimsave(video_out_path, record_frames, fps=10)

    return total_reward, episode_terminated


def load_hint_text(
    env_name: str, encoding_method: str, structured_hint: bool, hints_root: str | Path
) -> str:
    """Load the latest hint text for an environment/encoding pair."""
    hint_type = "structured" if structured_hint else "simple"
    hint_dir = Path(hints_root) / env_name / encoding_method / hint_type
    if not hint_dir.exists():
        raise FileNotFoundError(f"Missing hint directory: {hint_dir}")
    hint_files = sorted(hint_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not hint_files:
        raise FileNotFoundError(f"No hint files found in {hint_dir}")
    latest_file = hint_files[-1]

    raw_text = latest_file.read_text(encoding="utf-8").strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text

    if isinstance(data, list):
        return "\n".join(str(x) for x in data)
    if isinstance(data, dict):
        if "hints" in data:
            return "\n".join(str(x) for x in data["hints"])
        if "aggregated_hints" in data:
            return "\n".join(str(x) for x in data["aggregated_hints"])
    return raw_text


def load_unique_hint(env_name: str, hints_root: str | Path) -> str:
    """Load the hint file from hints_root/env_name and return its raw JSON."""
    hint_dir = Path(hints_root) / env_name
    hint_files = sorted(hint_dir.glob("*.json"))
    if not hint_files:
        raise FileNotFoundError(f"No hint files found in {hint_dir}")
    hint_file = hint_files[0]
    return hint_file.read_text(encoding="utf-8").strip()


def convert_dir_lists_to_tuples(programs: list[str]) -> list[str]:
    """Convert direction lists like [-1, -1] to tuples (-1, -1) inside program
    strings.

    Args:
        programs (list[str]): list of program strings

    Returns:
        list[str]: rewritten program strings
    """
    _DIR_LIST_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")

    def repl(match: re.Match[str]) -> str:
        x, y = match.groups()
        return f"({x}, {y})"

    return [_DIR_LIST_RE.sub(repl, p) for p in programs]


def log_feature_collisions(
    X: Any,
    y: np.ndarray | None,
    _examples: list[tuple[ObsT, ActT]] | None,
) -> list[dict[str, Any]]:
    """Log collisions where identical feature vectors have different labels."""
    if y is None:
        logging.info("Collision check skipped: y is None.")
        return []
    if X is None or X.shape[0] == 0:
        logging.info("Collision check skipped: empty X.")
        return []

    labels = y.astype(int).flatten().tolist()
    collisions: list[tuple[int, int, int]] = []  # (idx_prev, idx_cur, label_prev)
    seen: dict[bytes, tuple[int, int]] = {}  # key -> (index, label)

    for i in range(X.shape[0]):
        row = X.getrow(i)
        dense = row.toarray().ravel().astype(np.uint8)
        key = dense.tobytes()
        label = labels[i]
        if key in seen:
            prev_idx, prev_label = seen[key]
            if prev_label != label:
                collisions.append((prev_idx, i, prev_label))
        else:
            seen[key] = (i, label)

    # if collisions:
    #     logging.info("Feature collisions found: %d", len(collisions))
    #     for prev_idx, cur_idx, prev_label in collisions[:10]:
    #         logging.info(
    #             "Collision: row %d(label=%d) vs row %d(label=%d)",
    #             prev_idx,
    #             prev_label,
    #             cur_idx,
    #             labels[cur_idx],
    #         )
    #         if _examples is not None:
    #             try:
    #                 prev_example = _examples[prev_idx]
    #                 cur_example = _examples[cur_idx]
    #                 logging.info("  row %d example: %s", prev_idx, prev_example)
    #                 logging.info("  row %d example: %s", cur_idx, cur_example)
    #             except (IndexError, TypeError):
    #                 logging.info(
    #                     "  Example payload unavailable for rows %d and %d.",
    #                     prev_idx,
    #                     cur_idx,
    #                 )
    #     grouped = group_collision_indices(collisions, labels)
    #     if grouped:
    #         top_group = max(grouped, key=lambda g: int(g["max_occur"]))
    #         logging.info(
    #             "Top collision group: pos=%d neg=%d max_occur=%d",
    #             len(top_group["pos"]),
    #             len(top_group["neg"]),
    #             int(top_group["max_occur"]),
    #         )
    #     return grouped
    if collisions:
        logging.info("Feature collisions found: %d", len(collisions))
        grouped = group_collision_indices(collisions, labels)
        return grouped

    logging.info("No feature collisions found.")
    return []


def _example_value_to_hashable_bytes(value: Any) -> bytes:
    """Convert nested example values into a stable byte key."""
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return (
            b"np|"
            + str(arr.dtype).encode("utf-8")
            + b"|"
            + str(arr.shape).encode("utf-8")
            + b"|"
            + arr.tobytes()
        )
    if isinstance(value, tuple):
        return b"tuple|" + b"|".join(_example_value_to_hashable_bytes(v) for v in value)
    if isinstance(value, list):
        return b"list|" + b"|".join(_example_value_to_hashable_bytes(v) for v in value)
    return repr(value).encode("utf-8")


def _group_exact_example_labels(
    examples: list[tuple[ObsT, ActT]] | None,
    y: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Group exact duplicate (state, action) examples by labels seen."""
    if examples is None or y is None or len(examples) == 0:
        return []

    labels = y.astype(int).flatten().tolist()
    if len(labels) != len(examples):
        return []

    grouped: dict[bytes, dict[str, Any]] = {}
    for idx, ((state, action), label) in enumerate(zip(examples, labels)):
        key = (
            _example_value_to_hashable_bytes(state)
            + b"||"
            + _example_value_to_hashable_bytes(action)
        )
        entry = grouped.setdefault(
            key,
            {
                "example": (state, action),
                "pos": [],
                "neg": [],
            },
        )
        if label == 1:
            entry["pos"].append(idx)
        else:
            entry["neg"].append(idx)

    return [
        {
            "example": entry["example"],
            "pos": entry["pos"],
            "neg": entry["neg"],
            "max_occur": max(len(entry["pos"]), len(entry["neg"])),
        }
        for entry in grouped.values()
        if entry["pos"] and entry["neg"]
    ]


def log_exact_example_label_contradictions(
    examples: list[tuple[ObsT, ActT]] | None,
    y: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Log exact duplicate (state, action) examples with conflicting labels."""
    if examples is None or y is None:
        logging.info(
            "Exact example contradiction check skipped: examples or y is None."
        )
        return []
    if len(examples) == 0:
        logging.info("Exact example contradiction check skipped: no examples.")
        return []

    labels = y.astype(int).flatten().tolist()
    if len(labels) != len(examples):
        logging.info(
            "Exact example contradiction check skipped: labels/examples length mismatch (%d vs %d).",
            len(labels),
            len(examples),
        )
        return []
    contradictions = _group_exact_example_labels(examples, y)

    if not contradictions:
        logging.info("No exact (state, action) label contradictions found.")
        return []

    logging.info(
        "Exact (state, action) label contradictions found: %d groups",
        len(contradictions),
    )
    for contradiction in contradictions[:10]:
        logging.info(
            "Exact contradiction: pos=%d neg=%d max_occur=%d",
            len(contradiction["pos"]),
            len(contradiction["neg"]),
            int(contradiction["max_occur"]),
        )
        logging.info("  example: %s", contradiction["example"])

    top_group = max(contradictions, key=lambda g: int(g["max_occur"]))
    logging.info(
        "Top exact contradiction group: pos=%d neg=%d max_occur=%d",
        len(top_group["pos"]),
        len(top_group["neg"]),
        int(top_group["max_occur"]),
    )
    return contradictions


def drop_negative_exact_contradictions(
    X: Any,
    y: np.ndarray,
    examples: list[tuple[ObsT, ActT]] | None,
    sample_weights: np.ndarray | None,
) -> tuple[Any, np.ndarray, list[tuple[ObsT, ActT]] | None, np.ndarray | None]:
    """Remove negative rows for exact (state, action) contradictions.

    If an exact example appears as both positive and negative, keep all
    positive copies and drop the negative copies.
    """
    contradictions = _group_exact_example_labels(examples, y)
    if not contradictions:
        logging.info("No contradictory negative rows removed.")
        return X, y, examples, sample_weights

    drop_indices = sorted(
        {int(idx) for group in contradictions for idx in group["neg"]}
    )
    if not drop_indices:
        logging.info("No contradictory negative rows removed.")
        return X, y, examples, sample_weights

    keep_mask = np.ones(len(y), dtype=bool)
    keep_mask[np.asarray(drop_indices, dtype=int)] = False

    X_kept = X[keep_mask]
    y_kept = y[keep_mask]
    examples_kept = (
        [example for idx, example in enumerate(examples) if keep_mask[idx]]
        if examples is not None
        else None
    )
    sample_weights_kept = (
        sample_weights[keep_mask] if sample_weights is not None else None
    )

    logging.info(
        "Removed %d contradictory negative rows across %d exact contradiction groups. Remaining rows: %d",
        len(drop_indices),
        len(contradictions),
        int(y_kept.shape[0]),
    )
    return X_kept, y_kept, examples_kept, sample_weights_kept


def deduplicate_negative_examples(
    X: Any,
    y: np.ndarray,
    examples: list[tuple[ObsT, ActT]] | None,
    sample_weights: np.ndarray | None,
) -> tuple[Any, np.ndarray, list[tuple[ObsT, ActT]] | None, np.ndarray | None]:
    """Remove exact duplicate negative (state, action) rows, keeping first
    copy."""
    if examples is None or len(examples) == 0:
        logging.info("Negative dedup skipped: no examples available.")
        return X, y, examples, sample_weights

    labels = y.astype(int).flatten().tolist()
    if len(labels) != len(examples):
        logging.info(
            "Negative dedup skipped: labels/examples length mismatch (%d vs %d).",
            len(labels),
            len(examples),
        )
        return X, y, examples, sample_weights

    seen_negative_keys: set[bytes] = set()
    drop_indices: list[int] = []
    for idx, ((state, action), label) in enumerate(zip(examples, labels)):
        if label != 0:
            continue
        key = (
            _example_value_to_hashable_bytes(state)
            + b"||"
            + _example_value_to_hashable_bytes(action)
        )
        if key in seen_negative_keys:
            drop_indices.append(idx)
            continue
        seen_negative_keys.add(key)

    if not drop_indices:
        logging.info("No duplicate negative examples removed.")
        return X, y, examples, sample_weights

    keep_mask = np.ones(len(y), dtype=bool)
    keep_mask[np.asarray(drop_indices, dtype=int)] = False
    X_kept = X[keep_mask]
    y_kept = y[keep_mask]
    examples_kept = [example for idx, example in enumerate(examples) if keep_mask[idx]]
    sample_weights_kept = (
        sample_weights[keep_mask] if sample_weights is not None else None
    )
    logging.info(
        "Removed %d duplicate negative rows. Remaining rows: %d",
        len(drop_indices),
        int(y_kept.shape[0]),
    )
    return X_kept, y_kept, examples_kept, sample_weights_kept


def group_collision_indices(
    collisions: list[tuple[int, int, int]],
    labels: list[int],
) -> list[dict[str, Any]]:
    """Group collisions into pos/neg index lists and compute max_occur."""
    groups: dict[int, dict[str, set[int]]] = {}
    for prev_idx, cur_idx, prev_label in collisions:
        cur_label = labels[cur_idx]
        if prev_label == cur_label:
            continue
        if prev_label == 1:
            pos_idx, neg_idx = prev_idx, cur_idx
        else:
            pos_idx, neg_idx = cur_idx, prev_idx
        entry = groups.setdefault(pos_idx, {"pos": set(), "neg": set()})
        entry["pos"].add(pos_idx)
        entry["neg"].add(neg_idx)

    out: list[dict[str, Any]] = []
    for pos_idx, data in groups.items():
        pos_list = sorted(data["pos"])
        neg_list = sorted(data["neg"])
        max_occur = max(len(pos_list), len(neg_list))
        out.append({"pos": pos_list, "neg": neg_list, "max_occur": max_occur})
    return out


def _extract_policy_feature_names(policy_str: str) -> list[str]:
    return sorted(
        set(
            re.findall(
                r"\b(f\d+)\s*\(\s*s\s*,\s*a\s*\)", policy_str, flags=re.IGNORECASE
            )
        )
    )


def _split_top_level_and(expr: str) -> list[str]:
    expr = _strip_outer_parens(expr)
    parts: list[str] = []
    depth = 0
    last = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and expr[i : i + 3] == "and":
            prev_ok = (i == 0) or (not (expr[i - 1].isalnum() or expr[i - 1] == "_"))
            next_ok = (i + 3 == len(expr)) or (
                not (expr[i + 3].isalnum() or expr[i + 3] == "_")
            )
            if prev_ok and next_ok:
                part = expr[last:i].strip()
                if part:
                    parts.append(part)
                i += 3
                last = i
                continue
        i += 1
    tail = expr[last:].strip()
    if tail:
        parts.append(tail)
    return parts if parts else [expr]


def _extract_clause_literals(clause: str) -> list[str]:
    literals = []
    for part in _split_top_level_and(clause):
        stripped = _strip_outer_parens(part)
        if re.fullmatch(
            r"(?:not\s+)?f\d+\s*\(\s*s\s*,\s*a\s*\)",
            stripped,
            flags=re.IGNORECASE,
        ):
            literals.append(stripped)
    return literals


def _feature_fn_map(
    dsl_functions: dict[str, Any],
) -> dict[str, Callable[[Any, Any], bool]]:
    out: dict[str, Callable[[Any, Any], bool]] = {}
    for name, value in dsl_functions.items():
        if re.fullmatch(r"f\d+", str(name), flags=re.IGNORECASE) and callable(value):
            out[str(name)] = value
    return out


def _compute_feature_values(
    feature_names: list[str],
    feature_fns: dict[str, Callable[[Any, Any], bool]],
    s: Any,
    a: Any,
) -> dict[str, bool]:
    values: dict[str, bool] = {}
    for fname in feature_names:
        fn = feature_fns.get(fname)
        if fn is None:
            values[fname] = False
            continue
        try:
            values[fname] = bool(fn(s, a))
        except Exception:  # pylint: disable=broad-exception-caught
            values[fname] = False
    return values


def _literal_holds(literal: str, feature_values: dict[str, bool]) -> bool:
    match = re.fullmatch(
        r"(not\s+)?(f\d+)\s*\(\s*s\s*,\s*a\s*\)",
        literal.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return False
    negated = bool(match.group(1))
    fname = str(match.group(2))
    value = bool(feature_values.get(fname, False))
    return (not value) if negated else value


def _closest_clause_debug(
    policy_str: str,
    feature_values: dict[str, bool],
) -> tuple[str | None, list[str]]:
    """Return the nearest OR-clause and its failing feature literals.

    The nearest clause is the one with the fewest unsatisfied literals
    (breaking ties by preferring longer clauses).
    """
    clauses = _split_top_level_or(policy_str)
    best_clause_idx = None
    best_failed_literals: list[str] = []
    best_score = None

    for idx, clause in enumerate(clauses, start=1):
        literals = _extract_clause_literals(clause)
        failed_literals = [
            lit for lit in literals if not _literal_holds(lit, feature_values)
        ]
        score = (len(failed_literals), -len(literals))
        if best_score is None or score < best_score:
            best_score = score
            best_clause_idx = idx
            best_failed_literals = failed_literals

    if best_clause_idx is None:
        return None, []
    normalized_failed = []
    for literal in best_failed_literals:
        match = re.search(r"\b(f\d+)\b", literal, flags=re.IGNORECASE)
        normalized_failed.append(match.group(1) if match else literal)
    return f"clause_{best_clause_idx}", normalized_failed


def _format_action_short(action: Any) -> str:
    if isinstance(action, np.ndarray):
        return np.array2string(action, precision=3, separator=", ")
    if isinstance(action, (list, tuple)):
        try:
            arr = np.asarray(action, dtype=float).reshape(-1)
            return np.array2string(arr, precision=3, separator=", ")
        except Exception:  # pylint: disable=broad-exception-caught
            return repr(action)
    return repr(action)


def _format_state_short(state: Any) -> str:
    try:
        arr = np.asarray(state)
        return np.array2string(arr, precision=3, separator=", ")
    except Exception:  # pylint: disable=broad-exception-caught
        return repr(state)


def _enumerate_accepted_actions(
    plp: StateActionProgram,
    obs: Any,
    candidate_actions: list[Any] | None,
) -> list[Any]:
    accepted_actions: list[Any] = []
    if candidate_actions is not None:
        for action in candidate_actions:
            try:
                if plp(obs, action):
                    accepted_actions.append(action)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
        return accepted_actions

    if hasattr(obs, "shape") and len(getattr(obs, "shape")) == 2:
        for r in range(obs.shape[0]):  # type: ignore[attr-defined]
            for c in range(obs.shape[1]):  # type: ignore[attr-defined]
                action = (r, c)
                try:
                    if plp(obs, action):
                        accepted_actions.append(action)
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
    return accepted_actions


def _feature_hamming_distance(
    lhs: dict[str, bool],
    rhs: dict[str, bool],
    feature_names: list[str],
) -> int:
    return sum(
        int(bool(lhs.get(name, False)) != bool(rhs.get(name, False)))
        for name in feature_names
    )


def _action_distance(lhs: Any, rhs: Any) -> float:
    try:
        lhs_arr = np.asarray(lhs, dtype=float).reshape(-1)
        rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
        if lhs_arr.shape != rhs_arr.shape:
            return float("inf")
        return float(np.linalg.norm(lhs_arr - rhs_arr))
    except Exception:  # pylint: disable=broad-exception-caught
        return 0.0 if lhs == rhs else float("inf")


def _parse_motion2d_obstacles(
    s: np.ndarray,
) -> list[tuple[int, float, float, float, float]]:
    """Return Motion2D obstacles as (idx, x, y, w, h)."""
    num_obstacles = max(0, (len(s) - 19) // 10)
    obstacles: list[tuple[int, float, float, float, float]] = []
    for idx in range(num_obstacles):
        base = 19 + 10 * idx
        obstacles.append(
            (
                idx,
                float(s[base]),
                float(s[base + 1]),
                float(s[base + 8]),
                float(s[base + 9]),
            )
        )
    return obstacles


def _parse_motion2d_passages(
    s: np.ndarray,
    robot_radius: float,
) -> list[dict[str, float | int]]:
    """Return Motion2D passages parsed from obstacle pairs."""
    obstacles = _parse_motion2d_obstacles(s)
    passages: list[dict[str, float | int]] = []
    for passage_idx in range(len(obstacles) // 2):
        bot_idx, bot_x, bot_y, bot_w, bot_h = obstacles[2 * passage_idx]
        top_idx, top_x, top_y, top_w, top_h = obstacles[2 * passage_idx + 1]
        del top_x, top_w, top_h
        gap_lower = bot_y + bot_h + robot_radius
        gap_upper = top_y - robot_radius
        passages.append(
            {
                "passage_idx": passage_idx,
                "bottom_obstacle_idx": bot_idx,
                "top_obstacle_idx": top_idx,
                "wall_left": bot_x,
                "wall_right": bot_x + bot_w,
                "wall_width": bot_w,
                "gap_lower": gap_lower,
                "gap_upper": gap_upper,
                "gap_center": (gap_lower + gap_upper) / 2.0,
            }
        )
    return passages


def _select_motion2d_reference_passage(
    s: np.ndarray,
    robot_radius: float,
) -> dict[str, float | int] | None:
    """Select the next passage ahead, or the closest one if already past all."""
    passages = _parse_motion2d_passages(s, robot_radius)
    if not passages:
        return None
    robot_x = float(s[0])
    for passage in sorted(passages, key=lambda p: float(p["wall_left"])):
        if robot_x + robot_radius < float(passage["wall_right"]):
            return passage
    return min(
        passages,
        key=lambda p: abs(robot_x - float(p["gap_center"])),
    )


def _motion2d_mode_tags(obs: Any) -> set[str]:
    try:
        s = np.asarray(obs, dtype=float).reshape(-1)
    except Exception:  # pylint: disable=broad-exception-caught
        return set()
    if s.size < 39:
        return set()

    robot_x = float(s[0])
    robot_y = float(s[1])
    r = float(s[3])
    target_x = float(s[9])
    target_y = float(s[10])
    target_w = float(s[17])
    target_h = float(s[18])
    target_cx = target_x + target_w / 2.0
    target_cy = target_y + target_h / 2.0
    dist_target = float(np.hypot(target_cx - robot_x, target_cy - robot_y))
    ref_passage = _select_motion2d_reference_passage(s, r)

    tags: set[str] = set()
    if ref_passage is not None:
        wall_left = float(ref_passage["wall_left"])
        wall_right = float(ref_passage["wall_right"])
        gap_lower = float(ref_passage["gap_lower"])
        gap_upper = float(ref_passage["gap_upper"])
        if robot_x + r < wall_left:
            tags.add("before_passage")
        elif wall_left <= robot_x < wall_right + r:
            tags.add("inside_passage")
        else:
            tags.add("after_passage")

        if gap_lower <= robot_y <= gap_upper:
            tags.add("aligned")
        else:
            tags.add("not_aligned")
            if robot_y < gap_lower:
                tags.add("below_gap")
            if robot_y > gap_upper:
                tags.add("above_gap")

        if abs(robot_x - wall_left) <= 0.15 or abs(robot_x - wall_right) <= 0.15:
            tags.add("near_wall")
    if dist_target <= 0.35:
        tags.add("near_target")
    return tags


def _mode_bucket_label(tags: set[str]) -> str:
    if "near_target" in tags:
        return "near target"
    if "before_passage" in tags and "not_aligned" in tags:
        return "not aligned before passage"
    if "inside_passage" in tags and "aligned" in tags:
        return "inside passage aligned"
    if "inside_passage" in tags and "not_aligned" in tags:
        return "inside passage misaligned"
    if "before_passage" in tags and "aligned" in tags:
        return "before passage aligned"
    if "after_passage" in tags:
        return "after passage"
    return ", ".join(sorted(tags)) if tags else "uncategorized"


def _active_feature_names(feature_values: dict[str, bool]) -> list[str]:
    return sorted([name for name, value in feature_values.items() if value])


def log_plp_violation_counts(
    plps: list[StateActionProgram],
    demonstrations: Any,
    dsl_functions: dict[str, Any],
    *,
    candidate_actions: list[Any] | None = None,
    max_logged_plps: int = 10,
    max_debug_violations: int = 20,
    max_accepted_actions_to_show: int = 8,
    detailed_debug: bool = False,
) -> None:
    """Log how many demo steps each PLP fails (False on expert action)."""
    set_dsl_functions(dsl_functions)
    counts: list[tuple[int, StateActionProgram, list[dict[str, Any]]]] = []
    total_steps = len(demonstrations.steps)
    feature_fns = _feature_fn_map(dsl_functions)

    for plp in plps:
        violations = 0
        violation_debug_rows: list[dict[str, Any]] = []
        policy_str = str(plp)
        policy_feature_names = _extract_policy_feature_names(policy_str)
        for step_idx, (obs, action) in enumerate(demonstrations.steps):
            try:
                accepted_actions_for_step: list[Any] | None = None
                if detailed_debug:
                    accepted_actions_for_step = _enumerate_accepted_actions(
                        plp,
                        obs,
                        candidate_actions,
                    )
                    if candidate_actions is not None:
                        logging.info(
                            "State step=%d allowed candidate actions=%d/%d",
                            step_idx,
                            len(accepted_actions_for_step),
                            len(candidate_actions),
                        )
                        logging.info(
                            "State step=%d allowed candidates = %s",
                            step_idx,
                            [
                                _format_action_short(candidate_action)
                                for candidate_action in accepted_actions_for_step
                            ],
                        )

                if not plp(obs, action):
                    violations += 1
                    expert_feature_values = _compute_feature_values(
                        policy_feature_names,
                        feature_fns,
                        obs,
                        action,
                    )
                    closest_clause, failed_literals = _closest_clause_debug(
                        policy_str,
                        expert_feature_values,
                    )
                    accepted_actions = (
                        accepted_actions_for_step
                        if accepted_actions_for_step is not None
                        else _enumerate_accepted_actions(
                            plp,
                            obs,
                            candidate_actions,
                        )
                    )
                    best_action = None
                    best_action_feature_values: dict[str, bool] = {}
                    best_hamming = None
                    for accepted_action in accepted_actions:
                        accepted_feature_values = _compute_feature_values(
                            policy_feature_names,
                            feature_fns,
                            obs,
                            accepted_action,
                        )
                        hamming = _feature_hamming_distance(
                            expert_feature_values,
                            accepted_feature_values,
                            policy_feature_names,
                        )
                        if best_hamming is None:
                            best_action = accepted_action
                            best_action_feature_values = accepted_feature_values
                            best_hamming = hamming
                            continue
                        current_key = (
                            hamming,
                            _action_distance(accepted_action, action),
                        )
                        best_key = (
                            int(best_hamming),
                            _action_distance(best_action, action),
                        )
                        if current_key < best_key:
                            best_action = accepted_action
                            best_action_feature_values = accepted_feature_values
                            best_hamming = hamming
                    mode_tags = _motion2d_mode_tags(obs)
                    violation_debug_rows.append(
                        {
                            "step_idx": step_idx,
                            "state": obs,
                            "expert_action": action,
                            "accepted_actions": accepted_actions,
                            "closest_clause": closest_clause,
                            "failed_literals": failed_literals,
                            "expert_feature_values": expert_feature_values,
                            "best_action": best_action,
                            "best_action_feature_values": best_action_feature_values,
                            "best_hamming": best_hamming,
                            "mode_tags": mode_tags,
                            "mode_bucket": _mode_bucket_label(mode_tags),
                        }
                    )
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(e)
                print(plp)
                logging.info("EXCEPTION")
                violations += 1
        counts.append((violations, plp, violation_debug_rows))

    counts_sorted = sorted(counts, key=lambda item: item[0])
    logging.info("PLP violation counts (lower is better):")
    for violations, plp, debug_rows in counts_sorted[:max_logged_plps]:
        rate = (violations / total_steps) if total_steps else 0.0
        logging.info(
            "violations=%d/%d (%.2f%%) | plp=%s",
            violations,
            total_steps,
            100.0 * rate,
            plp,
        )
        if detailed_debug:
            mode_counter: Counter[str] = Counter()
            for violation_idx, row in enumerate(
                debug_rows[:max_debug_violations], start=1
            ):
                mode_counter.update([str(row["mode_bucket"])])
                accepted_preview = [
                    _format_action_short(action)
                    for action in row["accepted_actions"][:max_accepted_actions_to_show]
                ]
                if len(row["accepted_actions"]) > max_accepted_actions_to_show:
                    accepted_preview.append(
                        f"... (+{len(row['accepted_actions']) - max_accepted_actions_to_show} more)"
                    )
                logging.info("Violation #%d", violation_idx)
                logging.info("demo=%s, step=%d", "unknown", int(row["step_idx"]))
                logging.info("state = %s", _format_state_short(row["state"]))
                logging.info(
                    "chosen_action = %s", _format_action_short(row["best_action"])
                )
                logging.info(
                    "expert_action = %s", _format_action_short(row["expert_action"])
                )
                if candidate_actions is not None:
                    logging.info(
                        "allowed candidate actions = %d/%d",
                        len(row["accepted_actions"]),
                        len(candidate_actions),
                    )
                logging.info("accepted_actions = %s", accepted_preview)
                logging.info("closest_matching_clause = %s", row["closest_clause"])
                logging.info("failed_literals = %s", row["failed_literals"])
                logging.info(
                    "active_features(expert) = %s",
                    _active_feature_names(row["expert_feature_values"]),
                )
                logging.info(
                    "active_features(accepted_best) = %s",
                    _active_feature_names(row["best_action_feature_values"]),
                )
                logging.info(
                    "feature_hamming_distance(expert, accepted_best) = %s",
                    row["best_hamming"],
                )
                logging.info("mode tags = %s", sorted(row["mode_tags"]))
            for row in debug_rows[max_debug_violations:]:
                mode_counter.update([str(row["mode_bucket"])])
            if len(debug_rows) > max_debug_violations:
                logging.info(
                    "... omitted %d additional violations for this PLP",
                    len(debug_rows) - max_debug_violations,
                )
            for mode_bucket, count in mode_counter.most_common():
                logging.info("%d violations in %s", count, mode_bucket)


def assert_features_fire(X: Any, programs: list[StateActionProgram]) -> None:
    """Assert that every feature fires at least once across all examples."""
    if X is None:
        raise AssertionError("X is None; cannot validate feature coverage.")
    if X.shape[1] == 0:
        raise AssertionError("No features found in X.")
    totals = np.asarray(X.sum(axis=0)).ravel()
    dead_idxs = np.where(totals == 0)[0].tolist()
    if dead_idxs:
        dead = [str(programs[i]) for i in dead_idxs[:10]]  #:20
        logging.info(f"{len(dead_idxs)} features never fire. Examples: {dead}")


def _format_one_example(
    s: Any,
    a: Any,
    *,
    label: int,
    idx: int,
) -> str:
    """Format one labeled (state, action) example for prompt text."""
    s, (r, c) = require_grid_state_action(s, a, context="_format_one_example")
    rows = []
    for rr in range(s.shape[0]):
        row = ", ".join(repr(str(x)) for x in s[rr])
        rows.append(f"[{row}]")
    grid = "\n".join(f"  {row}" for row in rows)
    cell = s[r, c] if 0 <= r < s.shape[0] and 0 <= c < s.shape[1] else "OOB"
    return f"- idx={idx} label={label} action=({r}, {c}) cell={cell}\n[\n{grid}\n]"


def _format_ascii_legend(symbol_map: dict[str, str]) -> str:
    lines = ["LEGEND:"]
    for token, char in symbol_map.items():
        lines.append(f"  {char} = {token}")
    lines.append("  (token_code)! = action cell")
    return "\n".join(lines)


def _format_one_example_ascii(
    s: Any,
    a: Any,
    *,
    label: int,
    idx: int,
    symbol_map: dict[str, str],
) -> str:
    """Format one labeled (state, action) example using ASCII token codes."""
    s, (r, c) = require_grid_state_action(s, a, context="_format_one_example_ascii")
    h, w = s.shape[0], s.shape[1]
    code_width = max((len(code) for code in symbol_map.values()), default=1)

    in_bounds = 0 <= r < h and 0 <= c < w
    cell = s[r, c] if in_bounds else "OOB"
    rows = []
    for rr in range(h):
        row_codes = []
        for cc in range(w):
            tok = str(s[rr, cc])
            code = symbol_map.get(tok, "?").rjust(code_width)
            if in_bounds and rr == r and cc == c:
                code = f"{code}!"
            row_codes.append(f"'{code}'")
        rows.append("  [" + ", ".join(row_codes) + "]")
    grid = "\n".join(rows)

    return f"- idx={idx} label={label} action=({r}, {c}) cell={cell}\n[\n{grid}\n]"


def _format_one_example_coords(
    s: Any,
    a: Any,
    *,
    label: int,
    idx: int,
) -> str:
    """Format one labeled (state, action) example using coordinate lists."""
    s, (r, c) = require_grid_state_action(s, a, context="_format_one_example_coords")
    s_str = s.astype(str)
    h, w = s_str.shape[0], s_str.shape[1]
    tokens, counts = np.unique(s_str, return_counts=True)
    bg_idx = int(np.argmax(counts)) if tokens.size else -1
    background = tokens[bg_idx] if bg_idx >= 0 else ""

    in_bounds = 0 <= r < h and 0 <= c < w
    cell = s_str[r, c] if in_bounds else "OOB"

    lines = [f"- idx={idx} label={label} action=({r}, {c}) cell={cell}"]
    if background:
        lines.append(f"background={background}")

    for tok in tokens:
        if tok == background:
            continue
        coords = np.argwhere(s_str == tok)
        coord_text = ", ".join(f"({rr}, {cc})" for rr, cc in coords)
        lines.append(f"{tok}: {coord_text}")

    return "\n".join(lines)


SMALL_LIST_THRESHOLD = 12
PATCH_K = 7
DIFF_LIMIT = 10


def _maybe_map_ascii_tokens(s: Any, symbol_map: dict[str, str]) -> Any:
    """Map ASCII tokens to canonical names if symbol_map is provided."""
    if not isinstance(s, np.ndarray):
        raise ValueError("Expected s to be a numpy array.")
    s_str = s.astype(str)
    if not symbol_map:
        return s_str
    inverse = {v: k for k, v in symbol_map.items()}
    if not np.isin(s_str, list(inverse.keys())).any():
        return s_str
    mapper = np.vectorize(lambda tok: inverse.get(tok, tok))
    return mapper(s_str)


def _compress_ranges(cols: list[int]) -> list[str]:
    if not cols:
        return []
    ranges: list[str] = []
    start = prev = cols[0]
    for col in cols[1:]:
        if col == prev + 1:
            prev = col
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = col
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ranges


def _find_token_positions(grid: np.ndarray, token_name: str) -> list[tuple[int, int]]:
    coords = np.argwhere(grid == token_name)
    if coords.size == 0:
        return []
    coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
    return [(int(r), int(c)) for r, c in coords]


def _compress_rows(positions: list[tuple[int, int]]) -> dict[int, list[str]]:
    rows: dict[int, list[int]] = {}
    for r, c in positions:
        rows.setdefault(int(r), []).append(int(c))
    compressed: dict[int, list[str]] = {}
    for r in sorted(rows):
        cols = sorted(rows[r])
        compressed[r] = _compress_ranges(cols)
    return compressed


def _compute_patch(
    grid: np.ndarray, center: tuple[int, int], window: int
) -> dict[str, Any]:
    h, w = grid.shape[0], grid.shape[1]
    r, c = center
    half = window // 2
    top_left = (r - half, c - half)
    patch_grid: list[list[str]] = []
    counts: dict[str, int] = {}
    for dr in range(-half, half + 1):
        row: list[str] = []
        for dc in range(-half, half + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                tok = str(grid[rr, cc])
            else:
                tok = "OOB"
            row.append(tok)
            if tok != "OOB":
                counts[tok] = counts.get(tok, 0) + 1
        patch_grid.append(row)
    return {
        "window": window,
        "top_left": top_left,
        "grid": patch_grid,
        "counts": {k: counts[k] for k in sorted(counts)},
    }


def _raycast(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    background_token: str,
) -> tuple[str | None, int | None]:
    h, w = grid.shape[0], grid.shape[1]
    r, c = start
    dr, dc = direction
    steps = 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        steps += 1
        tok = str(grid[rr, cc])
        if tok != background_token:
            return tok, steps
        rr += dr
        cc += dc
    return None, None


def _manhattan_nearest(
    positions: list[tuple[int, int]],
    point: tuple[int, int],
) -> int | None:
    if not positions:
        return None
    r, c = point
    return min(abs(r - rr) + abs(c - cc) for rr, cc in positions)


def _connected_components(grid: np.ndarray, token_name: str) -> tuple[int, int]:
    h, w = grid.shape[0], grid.shape[1]
    visited = np.zeros((h, w), dtype=bool)
    comps = 0
    largest = 0
    for r in range(h):
        for c in range(w):
            if visited[r, c] or str(grid[r, c]) != token_name:
                continue
            comps += 1
            stack = [(r, c)]
            visited[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if not visited[nr, nc] and str(grid[nr, nc]) == token_name:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
            largest = max(largest, size)
    return comps, largest


def _diff_cells(
    pos_grid: np.ndarray,
    neg_grid: np.ndarray,
    limit: int = DIFF_LIMIT,
) -> list[tuple[int, int, str, str]]:
    h = min(pos_grid.shape[0], neg_grid.shape[0])
    w = min(pos_grid.shape[1], neg_grid.shape[1])
    diffs: list[tuple[int, int, str, str]] = []
    for r in range(h):
        for c in range(w):
            a = str(pos_grid[r, c])
            b = str(neg_grid[r, c])
            if a != b:
                diffs.append((r, c, a, b))
            if len(diffs) >= limit:
                return diffs
    return diffs


def _diag_path_summary(
    grid: np.ndarray,
    star_pos: tuple[int, int] | None,
    agent_pos: tuple[int, int] | None,
    drawn_token_name: str,
    limit_steps: int | None = None,
) -> dict[str, Any]:
    if star_pos is None or agent_pos is None:
        return {
            "diag_dir": None,
            "diag_steps_considered": 0,
            "first_mismatch_on_diag": None,
            "num_mismatches_on_diag": 0,
        }
    h, w = grid.shape[0], grid.shape[1]
    sr, sc = star_pos
    ar, ac = agent_pos
    dc = 1 if (ac - sc) >= 0 else -1
    dr = 1
    steps = max(ar - sr, 0)
    if limit_steps is not None:
        steps = min(steps, limit_steps)
    mismatches = 0
    first_mismatch = None
    rr, cc = sr, sc
    for _ in range(steps):
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            break
        tok = str(grid[rr, cc])
        if tok != drawn_token_name:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = (rr, cc)
    return {
        "diag_dir": (dr, dc),
        "diag_steps_considered": steps,
        "first_mismatch_on_diag": first_mismatch,
        "num_mismatches_on_diag": mismatches,
    }


def _select_diverse_examples(
    pos_indices: list[int],
    neg_indices: list[int],
    examples: list[tuple[ObsT, ActT]],
    symbol_map: dict[str, str],
    max_pos: int = 2,
    max_neg: int = 3,
) -> tuple[list[int], list[int]]:
    if not pos_indices or not neg_indices:
        return pos_indices[:max_pos], neg_indices[:max_neg]
    pair_scores: list[tuple[int, int, int]] = []
    for p in pos_indices:
        s_pos, _ = examples[p]
        s_pos_tok = _maybe_map_ascii_tokens(s_pos, symbol_map)
        for n in neg_indices:
            s_neg, _ = examples[n]
            s_neg_tok = _maybe_map_ascii_tokens(s_neg, symbol_map)
            score = len(_diff_cells(s_pos_tok, s_neg_tok, limit=DIFF_LIMIT))
            pair_scores.append((score, p, n))
    pair_scores.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, best_p, best_n = pair_scores[0]
    selected_pos = [best_p]
    selected_neg = [best_n]

    remaining_pos = [p for p in pos_indices if p not in selected_pos]
    remaining_neg = [n for n in neg_indices if n not in selected_neg]

    if remaining_pos and len(selected_pos) < max_pos:
        pos_scores: list[tuple[int, int]] = []
        for p in remaining_pos:
            s_pos, _ = examples[p]
            s_pos_tok = _maybe_map_ascii_tokens(s_pos, symbol_map)
            best = 0
            for n in selected_neg:
                s_neg, _ = examples[n]
                s_neg_tok = _maybe_map_ascii_tokens(s_neg, symbol_map)
                best = max(
                    best, len(_diff_cells(s_pos_tok, s_neg_tok, limit=DIFF_LIMIT))
                )
            pos_scores.append((best, p))
        pos_scores.sort(key=lambda item: (-item[0], item[1]))
        selected_pos.append(pos_scores[0][1])
        remaining_pos = [p for p in remaining_pos if p not in selected_pos]

    while remaining_neg and len(selected_neg) < max_neg:
        neg_scores: list[tuple[int, int]] = []
        for n in remaining_neg:
            s_neg, _ = examples[n]
            s_neg_tok = _maybe_map_ascii_tokens(s_neg, symbol_map)
            best = 0
            for p in selected_pos:
                s_pos, _ = examples[p]
                s_pos_tok = _maybe_map_ascii_tokens(s_pos, symbol_map)
                best = max(
                    best, len(_diff_cells(s_pos_tok, s_neg_tok, limit=DIFF_LIMIT))
                )
            neg_scores.append((best, n))
        neg_scores.sort(key=lambda item: (-item[0], item[1]))
        chosen = neg_scores[0][1]
        selected_neg.append(chosen)
        remaining_neg = [n for n in remaining_neg if n != chosen]

    return selected_pos[:max_pos], selected_neg[:max_neg]


def _build_diff_hints(
    pos_idx: int,
    neg_idx: int,
    examples: list[tuple[ObsT, ActT]],
    symbol_map: dict[str, str],
) -> str:
    s_pos, a_pos = examples[pos_idx]
    s_neg, a_neg = examples[neg_idx]
    _, action_pos = require_grid_state_action(s_pos, a_pos, context="_build_diff_hints")
    _, action_neg = require_grid_state_action(s_neg, a_neg, context="_build_diff_hints")
    s_pos_tok = _maybe_map_ascii_tokens(s_pos, symbol_map)
    s_neg_tok = _maybe_map_ascii_tokens(s_neg, symbol_map)

    uniq_pos, counts_pos = np.unique(s_pos_tok, return_counts=True)
    uniq_neg, _ = np.unique(s_neg_tok, return_counts=True)
    background = (
        str(uniq_pos[int(np.argmax(counts_pos))]) if len(uniq_pos) > 0 else "EMPTY"
    )

    pos_r, pos_c = action_pos
    neg_r, neg_c = action_neg
    in_bounds_pos = 0 <= pos_r < s_pos_tok.shape[0] and 0 <= pos_c < s_pos_tok.shape[1]
    in_bounds_neg = 0 <= neg_r < s_neg_tok.shape[0] and 0 <= neg_c < s_neg_tok.shape[1]
    action_cell_pos = str(s_pos_tok[pos_r, pos_c]) if in_bounds_pos else "OOB"
    action_cell_neg = str(s_neg_tok[neg_r, neg_c]) if in_bounds_neg else "OOB"

    common_non_bg = sorted(
        set(str(t) for t in uniq_pos if str(t) != background)
        & set(str(t) for t in uniq_neg if str(t) != background)
    )
    component_diffs: list[dict[str, int | str]] = []
    nearest_dist_deltas: list[tuple[str, int]] = []
    for tok in common_non_bg:
        comps_pos, largest_pos = _connected_components(s_pos_tok, tok)
        comps_neg, largest_neg = _connected_components(s_neg_tok, tok)
        if comps_pos != comps_neg or largest_pos != largest_neg:
            component_diffs.append(
                {
                    "token": tok,
                    "num_components_pos": int(comps_pos),
                    "num_components_neg": int(comps_neg),
                    "largest_component_pos": int(largest_pos),
                    "largest_component_neg": int(largest_neg),
                }
            )
        d_pos = _manhattan_nearest(_find_token_positions(s_pos_tok, tok), action_pos)
        d_neg = _manhattan_nearest(_find_token_positions(s_neg_tok, tok), action_neg)
        if d_pos is not None and d_neg is not None and d_pos != d_neg:
            nearest_dist_deltas.append((tok, int(d_pos - d_neg)))
    component_diffs = component_diffs[:8]
    nearest_dist_deltas.sort(key=lambda item: (-abs(item[1]), item[0]))
    nearest_dist_deltas = nearest_dist_deltas[:8]

    patch = _compute_patch(s_pos_tok, action_pos, PATCH_K)
    rays = {
        name: _raycast(s_pos_tok, action_pos, direction, background)
        for name, direction in {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }.items()
    }
    diffs = _diff_cells(s_pos_tok, s_neg_tok, limit=min(6, DIFF_LIMIT))

    lines = []
    lines.append("DIFF HINTS:")
    lines.append(f"- POS_EXAMPLE_IDX: {pos_idx}")
    lines.append(f"- NEG_EXAMPLE_IDX: {neg_idx}")
    lines.append(f"- BACKGROUND_TOKEN_POS: {background}")
    lines.append(f"- ACTION_POS: {action_pos}")
    lines.append(f"- ACTION_NEG: {action_neg}")
    lines.append(f"- ACTION_CELL_POS_TOKEN: {action_cell_pos}")
    lines.append(f"- ACTION_CELL_NEG_TOKEN: {action_cell_neg}")
    lines.append(f"- TOP_K_CELL_DIFFS(pos_vs_neg): {diffs}")
    lines.append("- STRUCTURAL_CANDIDATES:")
    lines.append(f"  COMPONENT_DIFFS: {component_diffs}")
    lines.append(f"  ACTION_NEAREST_DIST_DELTAS_POS_MINUS_NEG: {nearest_dist_deltas}")
    lines.append(f"- ACTION_PATCH_POS(window={PATCH_K}): {patch}")
    lines.append(f"- ACTION_RAYS_POS(first_non_bg_token,dist): {rays}")
    return "\n".join(lines)


def _format_one_example_enc2(
    s: Any,
    a: Any,
    *,
    label: int,
    idx: int,
    symbol_map: dict[str, str],
    small_list_threshold: int = SMALL_LIST_THRESHOLD,
    patch_k: int = PATCH_K,
) -> str:
    """Format one labeled (state, action) example using enc_2."""
    s, (r, c) = require_grid_state_action(s, a, context="_format_one_example_enc2")
    s_tok = _maybe_map_ascii_tokens(s, symbol_map)
    h, w = s_tok.shape[0], s_tok.shape[1]
    tokens, counts = np.unique(s_tok, return_counts=True)
    bg_idx = int(np.argmax(counts)) if tokens.size else -1
    background = tokens[bg_idx] if bg_idx >= 0 else ""

    in_bounds = 0 <= r < h and 0 <= c < w
    cell = s_tok[r, c] if in_bounds else "OOB"

    lines = [f"- idx={idx} label={label} action=({r}, {c}) cell={cell}"]
    lines.append(f"BOARD: {h}x{w}")
    lines.append(f"BACKGROUND_TOKEN: {background}")

    non_bg_tokens = sorted([tok for tok in tokens if tok != background])
    lines.append(f"TOKENS: [{', '.join(non_bg_tokens)}]")
    global_counts = {tok: int((s_tok == tok).sum()) for tok in non_bg_tokens}
    counts_text = ", ".join(f"'{tok}': {global_counts[tok]}" for tok in non_bg_tokens)
    lines.append(f"GLOBAL_COUNTS: {{{counts_text}}}")

    for tok in non_bg_tokens:
        coords = np.argwhere(s_tok == tok)
        if coords.size:
            order = np.lexsort((coords[:, 1], coords[:, 0]))
            coords = coords[order]
        if coords.shape[0] <= small_list_threshold:
            coord_text = ", ".join(f"({rr}, {cc})" for rr, cc in coords)
            lines.append(f"{tok}: [{coord_text}]")
        else:
            coord_row_map: dict[int, list[int]] = {}
            for rr, cc in coords:
                coord_row_map.setdefault(int(rr), []).append(int(cc))
            row_parts: list[str] = []
            for rr in sorted(coord_row_map):
                cols = sorted(coord_row_map[rr])
                ranges = _compress_ranges(cols)
                row_parts.append(f"{rr}:[{', '.join(ranges)}]")
            lines.append(f"{tok}: rows {{{', '.join(row_parts)}}}")
            positions = _find_token_positions(s_tok, tok)
            if positions:
                rmin = min(r for r, _ in positions)
                rmax = max(r for r, _ in positions)
                cmin = min(c for _, c in positions)
                cmax = max(c for _, c in positions)
                tok_bbox = (rmin, cmin, rmax, cmax)
                comps, largest = _connected_components(s_tok, tok)
            else:
                tok_bbox = None
                comps, largest = 0, 0
            lines.append(f"{tok}_bbox: {tok_bbox}")
            lines.append(
                f"{tok}_components_summary: {{'num_components': {comps}, 'largest_component_size': {largest}}}"
            )

    patch = _compute_patch(s_tok, (r, c), patch_k)
    lines.append("LOCAL_PATCH:")
    lines.append(f"- window: {patch['window']}")
    lines.append(f"- top_left: {patch['top_left']}")
    lines.append(f"- grid: {patch['grid']}")
    lines.append(f"- counts: {patch['counts']}")

    rays = {
        name: _raycast(s_tok, (r, c), direction, background)
        for name, direction in {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }.items()
    }
    ray_text = ", ".join(
        f"{name}: (first={tok}, dist={dist})" for name, (tok, dist) in rays.items()
    )
    lines.append(f"RAYS_FROM_ACTION: {{{ray_text}}}")

    dist_tokens = non_bg_tokens
    dist_parts = []
    for name in dist_tokens:
        positions = _find_token_positions(s_tok, name)
        dist = _manhattan_nearest(positions, (r, c))
        dist_parts.append(f"{name}: {dist}")
    lines.append(f"DISTANCES_MANHATTAN: {{{', '.join(dist_parts)}}}")

    return "\n".join(lines)


def _format_one_example_kinder_enc2(
    s: Any,
    a: Any,
    *,
    label: int,
    idx: int,
) -> str:
    s = np.asarray(s, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float).reshape(-1)

    robot_x = float(s[0])
    robot_y = float(s[1])
    robot_theta = float(s[2])
    r = float(s[3])
    target_x = float(s[9])
    target_y = float(s[10])
    target_w = float(s[17])
    target_h = float(s[18])

    dx = float(a[0])
    dy = float(a[1])
    dtheta = float(a[2])
    darm = float(a[3])
    vac = float(a[4])

    target_cx = target_x + target_w / 2.0
    target_cy = target_y + target_h / 2.0
    obstacles = _parse_motion2d_obstacles(s)
    passages = _parse_motion2d_passages(s, r)
    ref_passage = _select_motion2d_reference_passage(s, r)

    x_err = target_cx - robot_x
    y_err = target_cy - robot_y
    gap_y_err = (
        float(ref_passage["gap_center"]) - robot_y if ref_passage is not None else None
    )

    next_x = robot_x + dx
    next_y = robot_y + dy

    dist_target_l1 = abs(x_err) + abs(y_err)
    next_dist_target_l1 = abs(target_cx - next_x) + abs(target_cy - next_y)

    obstacle_lines = [
        (
            f"OBSTACLE{obs_idx}: x={obs_x:.3f}, y={obs_y:.3f}, "
            f"w={obs_w:.3f}, h={obs_h:.3f}, right={obs_x + obs_w:.3f}, "
            f"top={obs_y + obs_h:.3f}"
        )
        for obs_idx, obs_x, obs_y, obs_w, obs_h in obstacles
    ]

    if ref_passage is not None:
        wall_left = float(ref_passage["wall_left"])
        wall_right = float(ref_passage["wall_right"])
        gap_lower = float(ref_passage["gap_lower"])
        gap_upper = float(ref_passage["gap_upper"])
        gap_center = float(ref_passage["gap_center"])
        passage_idx = int(ref_passage["passage_idx"])
        left_of_wall = robot_x + r < wall_left
        inside_wall_x_band = wall_left <= robot_x < wall_right + r
        passed_wall = robot_x >= wall_right + r
        y_aligned = gap_lower <= robot_y <= gap_upper
        y_below_gap = robot_y < gap_lower
        y_above_gap = robot_y > gap_upper
        passage_line = (
            f"REFERENCE_PASSAGE{passage_idx}: wall_left={wall_left:.3f}, "
            f"wall_right={wall_right:.3f}, gap_lower={gap_lower:.3f}, "
            f"gap_upper={gap_upper:.3f}, gap_center={gap_center:.3f}"
        )
        regime_line = (
            "REGIME: "
            f"left_of_wall={left_of_wall}, inside_wall_x_band={inside_wall_x_band}, "
            f"passed_wall={passed_wall}, y_aligned={y_aligned}, "
            f"y_below_gap={y_below_gap}, y_above_gap={y_above_gap}"
        )
        gap_err_text = f"{gap_y_err:.3f}"
    else:
        passage_line = "REFERENCE_PASSAGE: none"
        regime_line = "REGIME: no_passage_structure=True"
        gap_err_text = "n/a"

    lines = [
        f"- idx={idx} label={label}",
        f"ROBOT: x={robot_x:.3f}, y={robot_y:.3f}, theta={robot_theta:.3f}, r={r:.3f}",
        f"ACTION: dx={dx:.3f}, dy={dy:.3f}, dtheta={dtheta:.3f}, darm={darm:.3f}, vac={vac:.3f}",
        f"TARGET: x={target_x:.3f}, y={target_y:.3f}, w={target_w:.3f}, h={target_h:.3f}, cx={target_cx:.3f}, cy={target_cy:.3f}",
        f"PASSAGE_COUNT: {len(passages)}",
        *obstacle_lines,
        passage_line,
        regime_line,
        f"ERRORS: x_err_to_target={x_err:.3f}, y_err_to_target={y_err:.3f}, y_err_to_gap_center={gap_err_text}",
        f"NEXT_STATE_ESTIMATE: next_x={next_x:.3f}, next_y={next_y:.3f}, target_l1_now={dist_target_l1:.3f}, target_l1_next={next_dist_target_l1:.3f}",
        f"ACTION_SHAPE: mostly_x={abs(dx) > abs(dy)}, mostly_y={abs(dy) > abs(dx)}, dx_pos={dx > 0}, dx_neg={dx < 0}, dy_pos={dy > 0}, dy_neg={dy < 0}",
    ]
    return "\n".join(lines)


def _format_one_example_kinder_enc2_delta(
    s: Any,
    a: Any,
    *,
    label: int,
    idx: int,
) -> str:
    s = np.asarray(s, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float).reshape(-1)

    robot_x = float(s[0])
    robot_y = float(s[1])
    r = float(s[3])
    target_x = float(s[9])
    target_y = float(s[10])
    target_w = float(s[17])
    target_h = float(s[18])

    dx = float(a[0])
    dy = float(a[1])

    target_cx = target_x + target_w / 2.0
    target_cy = target_y + target_h / 2.0
    ref_passage = _select_motion2d_reference_passage(s, r)

    if ref_passage is not None:
        wall_left = float(ref_passage["wall_left"])
        wall_right = float(ref_passage["wall_right"])
        gap_lower = float(ref_passage["gap_lower"])
        gap_upper = float(ref_passage["gap_upper"])
        gap_center = float(ref_passage["gap_center"])
        passage_text = f"ref_passage={int(ref_passage['passage_idx'])}"
        left_of_wall = robot_x + r < wall_left
        passed_wall = robot_x >= wall_right + r
        y_aligned = gap_lower <= robot_y <= gap_upper
        gap_err_text = f"{gap_center - robot_y:.3f}"
    else:
        passage_text = "ref_passage=none"
        left_of_wall = False
        passed_wall = False
        y_aligned = False
        gap_err_text = "n/a"

    parts = [
        f"NEG idx={idx} label={label}",
        f"ROBOT(x={robot_x:.3f}, y={robot_y:.3f})",
        f"ACTION(dx={dx:.3f}, dy={dy:.3f})",
        f"REGIME({passage_text}, left_of_wall={left_of_wall}, passed_wall={passed_wall}, y_aligned={y_aligned})",
        f"ERR(target_x={target_cx - robot_x:.3f}, target_y={target_cy - robot_y:.3f}, gap_y={gap_err_text})",
    ]
    return "; ".join(parts)


def _format_one_example_enc2_delta(
    s: Any,
    a: Any,
    *,
    label: int,
    idx: int,
    symbol_map: dict[str, str],
    small_list_threshold: int = SMALL_LIST_THRESHOLD,
) -> str:
    """Compact delta-only summary for enc_2 collision evidence."""
    s, (r, c) = require_grid_state_action(
        s, a, context="_format_one_example_enc2_delta"
    )
    s_tok = _maybe_map_ascii_tokens(s, symbol_map)
    tokens, counts = np.unique(s_tok, return_counts=True)
    bg_idx = int(np.argmax(counts)) if tokens.size else -1
    background = tokens[bg_idx] if bg_idx >= 0 else ""

    in_bounds = 0 <= r < s_tok.shape[0] and 0 <= c < s_tok.shape[1]
    cell = s_tok[r, c] if in_bounds else "OOB"

    non_bg_tokens = sorted([tok for tok in tokens if tok != background])
    global_counts = {tok: int((s_tok == tok).sum()) for tok in non_bg_tokens}
    counts_text = ", ".join(f"'{tok}': {global_counts[tok]}" for tok in non_bg_tokens)

    parts = [
        f"NEG idx={idx} label={label} action=({r}, {c}) cell={cell}",
        f"GLOBAL_COUNTS: {{{counts_text}}}",
    ]
    for tok in non_bg_tokens:
        if global_counts.get(tok, 0) <= small_list_threshold:
            continue
        positions = _find_token_positions(s_tok, tok)
        row_map = _compress_rows(positions)
        row_parts: list[str] = []
        for rr in sorted(row_map):
            row_parts.append(f"{rr}:[{', '.join(row_map[rr])}]")
        tok_rows = f"rows {{{', '.join(row_parts)}}}"
        if positions:
            rmin = min(r for r, _ in positions)
            rmax = max(r for r, _ in positions)
            cmin = min(c for _, c in positions)
            cmax = max(c for _, c in positions)
            tok_bbox = (rmin, cmin, rmax, cmax)
        else:
            tok_bbox = None
        parts.append(f"{tok}: {tok_rows}")
        parts.append(f"{tok}_bbox: {tok_bbox}")
    return "; ".join(parts)


def is_kinder_env(env_name: str | None) -> bool:
    """Return whether the environment name refers to a KinDER/Motion2D task."""
    if env_name is None:
        return False
    env = env_name.lower()
    return "motion2d" in env or "kinder" in env


def _build_continuous_observation_field_guide(env_name: str | None) -> str:
    """Return a prompt-friendly observation/action field guide."""
    if env_name is None:
        return "- Observation fields are environment-specific continuous values."

    base_env_name = env_name.split("-p", maxsplit=1)[0]
    canonical_name = continuous_hint_config.canonicalize_env_name(base_env_name)

    match = re.search(r"-p(\d+)", env_name)
    num_passages = int(match.group(1)) if match else 0
    try:
        obs_fields = continuous_hint_config.obs_field_names_for_kinder(
            canonical_name,
            num_passages,
        )
    except ValueError:
        return (
            "- Observation fields are object-centric continuous attributes.\n"
            "- Use the serialized object names and attributes shown in the "
            "demonstrations as the source of truth."
        )
    action_fields = continuous_hint_config.ACTION_FIELD_NAMES[canonical_name]

    obs_lines = [
        f"- obs[{idx}] = {field_name}" for idx, field_name in enumerate(obs_fields)
    ]
    action_lines = [
        f"- a[{idx}] = {field_name}" for idx, field_name in enumerate(action_fields)
    ]

    return "\n".join(
        [
            (
                f"- Environment variant: {canonical_name}-p{num_passages}"
                if canonical_name == "Motion2D"
                else f"- Environment variant: {canonical_name}"
            ),
            "- When raw arrays are used, index them with the following schema:",
            *obs_lines,
            "- Action dimensions:",
            *action_lines,
            "- These raw fields are also summarized into stable "
            "object-centric names like robot, target, obstacle0, obstacle1, etc.",
        ]
    )


def build_collision_repair_prompt(
    pos_indices: list[int],
    neg_indices: list[int],
    examples: list[tuple[ObsT, ActT]],
    *,
    env_name: str | None = None,
    existing_feature_summary: str | None = None,
    max_per_label: int = 5,
    collision_feedback_enc: str = "enc_1",
    pos_indices_2: list[int] | None = None,
    neg_indices_2: list[int] | None = None,
    seed: int | None = None,
    collision_template_feedback: bool = True,
    failed_attempt_summaries: str | None = None,
) -> str:
    """Build an LLM prompt that proposes features to resolve label
    collisions."""
    if not pos_indices or not neg_indices:
        raise ValueError(
            "Need at least 1 positive and 1 negative from the SAME feature-key bucket."
        )

    use_ascii = collision_feedback_enc == "enc_1"
    if collision_feedback_enc not in {"enc_1", "enc_2"}:
        raise ValueError("collision_feedback_enc must be 'enc_1' or 'enc_2'")
    use_enc2 = collision_feedback_enc == "enc_2"

    symbol_map: dict[str, str] = {}
    legend_block = ""
    is_kinder = is_kinder_env(env_name)

    if use_ascii and env_name and not is_kinder:
        symbol_map = grid_hint_config.get_symbol_map(env_name)
        legend_block = _format_ascii_legend(symbol_map) + "\n\n"
    elif env_name and not is_kinder:
        symbol_map = grid_hint_config.get_symbol_map(env_name)
    # FORMAT2 note: uses richer evidence + deterministic example selection.
    if use_enc2:
        if not is_kinder:
            pos_indices, neg_indices = _select_diverse_examples(
                pos_indices, neg_indices, examples, symbol_map, max_pos=1, max_neg=3
            )
            if pos_indices_2 and neg_indices_2:
                pos_indices_2, neg_indices_2 = _select_diverse_examples(
                    pos_indices_2,
                    neg_indices_2,
                    examples,
                    symbol_map,
                    max_pos=1,
                    max_neg=3,
                )
        else:
            pos_indices = pos_indices[:1]
            neg_indices = neg_indices[:3]
            if pos_indices_2 and neg_indices_2:
                pos_indices_2 = pos_indices_2[:1]
                neg_indices_2 = neg_indices_2[:3]
    else:
        rng = np.random.default_rng(seed)
        if len(pos_indices) > max_per_label:
            pos_indices = rng.choice(
                pos_indices, size=max_per_label, replace=False
            ).tolist()
        if len(neg_indices) > max_per_label:
            neg_indices = rng.choice(
                neg_indices, size=max_per_label, replace=False
            ).tolist()
    pos_blocks = []
    for idx in pos_indices:
        s, a = examples[idx]
        if use_ascii:
            pos_blocks.append(
                _format_one_example_ascii(s, a, label=1, idx=idx, symbol_map=symbol_map)
            )
        elif use_enc2 and is_kinder:
            pos_blocks.append(_format_one_example_kinder_enc2(s, a, label=1, idx=idx))
        elif use_enc2:
            pos_blocks.append(
                _format_one_example_enc2(s, a, label=1, idx=idx, symbol_map=symbol_map)
            )

        else:
            pos_blocks.append(_format_one_example_coords(s, a, label=1, idx=idx))

    neg_blocks = []
    for i, idx in enumerate(neg_indices):
        s, a = examples[idx]
        if use_ascii:
            neg_blocks.append(
                _format_one_example_ascii(s, a, label=0, idx=idx, symbol_map=symbol_map)
            )
        elif use_enc2 and is_kinder:
            if i == 0:
                neg_blocks.append(
                    _format_one_example_kinder_enc2(s, a, label=0, idx=idx)
                )
            else:
                neg_blocks.append(
                    _format_one_example_kinder_enc2_delta(s, a, label=0, idx=idx)
                )

        elif use_enc2:
            if i == 0:
                neg_blocks.append(
                    _format_one_example_enc2(
                        s, a, label=0, idx=idx, symbol_map=symbol_map
                    )
                )
            else:
                neg_blocks.append(
                    _format_one_example_enc2_delta(
                        s, a, label=0, idx=idx, symbol_map=symbol_map
                    )
                )

        else:
            neg_blocks.append(_format_one_example_coords(s, a, label=0, idx=idx))

    pos_blocks_2: list[str] = []
    neg_blocks_2: list[str] = []
    if pos_indices_2 and neg_indices_2:
        for idx in pos_indices_2:
            s, a = examples[idx]
            if use_ascii:
                pos_blocks_2.append(
                    _format_one_example_ascii(
                        s, a, label=1, idx=idx, symbol_map=symbol_map
                    )
                )
            elif use_enc2 and is_kinder:
                pos_blocks_2.append(
                    _format_one_example_kinder_enc2(s, a, label=1, idx=idx)
                )
            elif use_enc2:
                pos_blocks_2.append(
                    _format_one_example_enc2(
                        s, a, label=1, idx=idx, symbol_map=symbol_map
                    )
                )
            else:
                pos_blocks_2.append(_format_one_example_coords(s, a, label=1, idx=idx))

        for i, idx in enumerate(neg_indices_2):
            s, a = examples[idx]
            if use_ascii:
                neg_blocks_2.append(
                    _format_one_example_ascii(
                        s, a, label=0, idx=idx, symbol_map=symbol_map
                    )
                )
            elif use_enc2 and is_kinder:
                if i == 0:
                    neg_blocks_2.append(
                        _format_one_example_kinder_enc2(s, a, label=0, idx=idx)
                    )
                else:
                    neg_blocks_2.append(
                        _format_one_example_kinder_enc2_delta(s, a, label=0, idx=idx)
                    )
            elif use_enc2:
                if i == 0:
                    neg_blocks_2.append(
                        _format_one_example_enc2(
                            s, a, label=0, idx=idx, symbol_map=symbol_map
                        )
                    )
                else:
                    neg_blocks_2.append(
                        _format_one_example_enc2_delta(
                            s, a, label=0, idx=idx, symbol_map=symbol_map
                        )
                    )
            else:
                neg_blocks_2.append(_format_one_example_coords(s, a, label=0, idx=idx))
    diff_hints_1 = ""
    if use_enc2 and pos_indices and neg_indices:
        try:
            diff_hints_1 = _build_diff_hints(
                pos_indices[0], neg_indices[0], examples, symbol_map
            )
        except Exception:  # pylint: disable=broad-exception-caught
            diff_hints_1 = ""

    bucket2_block = ""
    if pos_blocks_2 or neg_blocks_2:
        diff_hints_2 = ""
        if use_enc2 and pos_indices_2 and neg_indices_2:
            try:
                diff_hints_2 = _build_diff_hints(
                    pos_indices_2[0], neg_indices_2[0], examples, symbol_map
                )
            except Exception:  # pylint: disable=broad-exception-caught
                diff_hints_2 = ""
        bucket2_block = f"""

BUCKET 2
POSITIVE EXAMPLES (label = 1):
{(chr(10) * 2).join(pos_blocks_2[:2])}

NEGATIVE EXAMPLES (label = 0):
{(chr(10) * 2).join(neg_blocks_2[:3])}

{diff_hints_2}
"""

    raw_token_examples = ""
    final_check_tokens = ""
    if env_name and symbol_map:
        token_samples = list(symbol_map.values())
        raw_token_examples = ", ".join(repr(s) for s in token_samples[:6])
        final_check_tokens = ", ".join(repr(s) for s in token_samples[:6])
    token_constants_block = "- {ENV_TOKEN_CONSTANTS}"
    if env_name and not is_kinder:
        try:
            token_map = hint_extractor.build_token_map(env_name)
            ordered_constants: list[str] = []
            for raw_char in sorted(token_map.keys()):
                const = token_map[raw_char]
                if const not in ordered_constants:
                    ordered_constants.append(const)
            if ordered_constants:
                token_constants_block = "\n".join(
                    f"- {const}" for const in ordered_constants
                )
        except Exception:  # pylint: disable=broad-exception-caught
            token_constants_block = "- ENV token constants from this domain."

    # FORMAT2 dev note: enc_2 collision evidence now includes explicit token anchors,
    # global counts, drawn compression + component stats, expanded local patch,
    # rays/distances, and per-bucket DIFF HINTS. These extra fields help cheaper
    # models separate positives/negatives while keeping code constraints strict.
    if use_enc2 and is_kinder:
        feature_prompt_filename = "featured_collision_feedback_enc2_kinder.txt"
    elif use_enc2:
        feature_prompt_filename = "featured_collision_feedback_enc2.txt"
    else:
        feature_prompt_filename = "featured_collision_feedback_enc1.txt"

    feature_prompt_path = (
        Path(__file__).resolve().parents[2]
        / "dsl"
        / "llm_primitives"
        / "prompts"
        / "py_feature_gen"
        / "collision_prompts"
        / feature_prompt_filename
    )
    prompt_feature = feature_prompt_path.read_text(encoding="utf-8")
    prompt_feature = prompt_feature.replace(
        "${OBSERVATION_FIELD_GUIDE}",
        _build_continuous_observation_field_guide(env_name),
    )
    prompt_feature = prompt_feature.replace(
        "${existing_relevant_features}",
        existing_feature_summary or "- None provided.",
    )
    prompt_feature = prompt_feature.replace(
        "${failed_attempt_summaries}",
        failed_attempt_summaries or "- None yet.",
    )
    prompt_feature = prompt_feature.replace(
        "${token_constants_block}", token_constants_block
    )
    prompt_feature = prompt_feature.replace(
        "${bucket1_pos}", legend_block + (chr(10) * 2).join(pos_blocks[:2])
    )
    prompt_feature = prompt_feature.replace(
        "${bucket1_neg}", (chr(10) * 2).join(neg_blocks[:3])
    )
    prompt_feature = prompt_feature.replace("${diff_hints_1}", diff_hints_1)
    prompt_feature = prompt_feature.replace("${bucket2_block}", bucket2_block)
    prompt_feature = prompt_feature.replace("{raw_token_examples}", raw_token_examples)
    prompt_feature = prompt_feature.replace("{final_check_tokens}", final_check_tokens)

    if collision_template_feedback:
        template_prompt_filename = (
            "template_collision_feedback_enc2.txt"
            if use_enc2
            else "template_collision_feedback_enc1.txt"
        )
        template_prompt_path = (
            Path(__file__).resolve().parents[2]
            / "dsl"
            / "llm_primitives"
            / "prompts"
            / "py_feature_gen"
            / "collision_prompts"
            / template_prompt_filename
        )
        prompt_template = template_prompt_path.read_text(encoding="utf-8")
        prompt_template = prompt_template.replace(
            "${OBSERVATION_FIELD_GUIDE}",
            _build_continuous_observation_field_guide(env_name),
        )
        prompt_template = prompt_template.replace(
            "${existing_relevant_features}",
            existing_feature_summary or "- None provided.",
        )
        prompt_template = prompt_template.replace(
            "${failed_attempt_summaries}",
            failed_attempt_summaries or "- None yet.",
        )
        prompt_template = prompt_template.replace(
            "${bucket1_pos}", legend_block + (chr(10) * 2).join(pos_blocks[:2])
        )
        prompt_template = prompt_template.replace(
            "${bucket1_neg}", (chr(10) * 2).join(neg_blocks[:3])
        )
        prompt_template = prompt_template.replace("${diff_hints_1}", diff_hints_1)
        prompt_template = prompt_template.replace("${bucket2_block}", bucket2_block)
        prompt_template = prompt_template.replace(
            "{raw_token_examples}", raw_token_examples
        )
        prompt_template = prompt_template.replace(
            "${final_check_tokens}", final_check_tokens
        )
        prompt_template = prompt_template.replace(
            "{final_check_tokens}", final_check_tokens
        )
    # input(prompt_feature)
    return prompt_template if collision_template_feedback else prompt_feature


def _strip_outer_parens(expr: str) -> str:
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        valid_wrap = True
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(expr) - 1:
                    valid_wrap = False
                    break
        if depth != 0 or not valid_wrap:
            break
        expr = expr[1:-1].strip()
    return expr


def _split_top_level_or(expr: str) -> list[str]:
    expr = _strip_outer_parens(expr)
    parts: list[str] = []
    depth = 0
    last = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and expr[i : i + 2] == "or":
            prev_ok = (i == 0) or (not (expr[i - 1].isalnum() or expr[i - 1] == "_"))
            next_ok = (i + 2 == len(expr)) or (
                not (expr[i + 2].isalnum() or expr[i + 2] == "_")
            )
            if prev_ok and next_ok:
                part = expr[last:i].strip()
                if part:
                    parts.append(part)
                i += 2
                last = i
                continue
        i += 1
    tail = expr[last:].strip()
    if tail:
        parts.append(tail)
    return parts if parts else [expr]


def _extract_clause_features(clause: str) -> list[str]:
    return sorted(
        set(re.findall(r"\b(f\d+)\s*\(\s*s\s*,\s*a\s*\)", clause, flags=re.IGNORECASE))
    )


def _safe_eval_bool(expr: str, locals_map: dict[str, bool]) -> bool:
    """Safely evaluate a boolean expression over known local variables."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.Name,
        ast.Constant,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Load,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return False
        if isinstance(node, ast.Name) and node.id not in locals_map:
            return False
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
            return False
    try:
        return bool(
            eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, locals_map)
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def build_dnf_failure_payload(
    policy_str: str,
    examples: list[tuple[ObsT, ActT]],
    y: list[bool] | np.ndarray,
    feature_fns: dict[str, Callable[[Any, Any], bool]],
    max_bad_clauses: int = 5,
    max_examples: int = 5,
    include_feature_values: bool = True,
) -> dict[str, Any]:
    """Build structured diagnostics for false positive/negative DNF
    behavior."""
    clauses = _split_top_level_or(policy_str)
    clause_features = [_extract_clause_features(c) for c in clauses]
    all_features = sorted({f for feats in clause_features for f in feats})

    y_arr = np.asarray(y).astype(bool).ravel()
    n = min(len(examples), y_arr.size)
    if n == 0:
        return {"policy": policy_str, "error": "no examples"}

    feature_values: dict[int, dict[str, bool]] = {}
    for idx in range(n):
        s, a = examples[idx]
        values: dict[str, bool] = {}
        for fname in all_features:
            fn = feature_fns.get(fname)
            if fn is None:
                values[fname] = False
                continue
            try:
                values[fname] = bool(fn(s, a))
            except Exception:  # pylint: disable=broad-exception-caught
                values[fname] = False
        feature_values[idx] = values

    fires = np.zeros((n, len(clauses)), dtype=bool)
    for ci, clause in enumerate(clauses):
        clause_expr = re.sub(
            r"\b(f\d+)\s*\(\s*s\s*,\s*a\s*\)", r"\1", clause, flags=re.IGNORECASE
        )
        clause_features_set = set(clause_features[ci])
        for i in range(n):
            local_vals = {
                k: v for k, v in feature_values[i].items() if k in clause_features_set
            }
            fires[i, ci] = _safe_eval_bool(clause_expr, local_vals)

    pred = fires.any(axis=1)
    fp_indices = np.where((pred == 1) & (y_arr[:n] == 0))[0].tolist()
    fn_indices = np.where((pred == 0) & (y_arr[:n] == 1))[0].tolist()

    clause_stats: list[dict[str, Any]] = []
    for ci, clause in enumerate(clauses):
        fired = fires[:, ci]
        fires_pos = int(np.sum(fired & y_arr[:n]))
        fires_neg = int(np.sum(fired & (~y_arr[:n])))
        denom = fires_pos + fires_neg
        precision = float(fires_pos / denom) if denom > 0 else 0.0
        tp_ids = [int(i) for i in np.where(fired & y_arr[:n])[0][:max_examples]]
        fp_ids = [int(i) for i in np.where(fired & (~y_arr[:n]))[0][:max_examples]]
        clause_stats.append(
            {
                "clause_id": ci,
                "clause": clause,
                "fires_pos": fires_pos,
                "fires_neg": fires_neg,
                "precision": precision,
                "num_fires": denom,
                "tp_example_ids": tp_ids,
                "fp_example_ids": fp_ids,
                "features": clause_features[ci],
            }
        )

    clause_stats.sort(key=lambda d: (-d["fires_neg"], d["precision"]))
    bad_clauses = clause_stats[:max_bad_clauses]

    fp_examples = []
    for idx in fp_indices[:max_examples]:
        fired_clause_ids = np.where(fires[idx])[0].tolist()
        fp_examples.append(
            {
                "example_id": int(idx),
                "label": 0,
                "fired_clause_ids": fired_clause_ids,
            }
        )

    fn_examples = [
        {"example_id": int(idx), "label": 1} for idx in fn_indices[:max_examples]
    ]

    payload: dict[str, Any] = {
        "policy": policy_str,
        "num_examples": int(n),
        "num_clauses": int(len(clauses)),
        "fp_count": int(len(fp_indices)),
        "fn_count": int(len(fn_indices)),
        "bad_clauses": bad_clauses,
        "fp_examples": fp_examples,
        "fn_examples": fn_examples,
    }

    if include_feature_values and (fp_examples or fn_examples):
        bad_features = sorted({f for c in bad_clauses for f in c["features"]})
        example_ids = [e["example_id"] for e in fp_examples] + [
            e["example_id"] for e in fn_examples
        ]
        feature_table: dict[int, dict[str, bool]] = {}
        for idx in example_ids:
            feature_table[idx] = {
                f: feature_values[idx].get(f, False) for f in bad_features
            }
        payload["feature_values"] = feature_table

    return payload


def build_dnf_failure_prompt(
    payload: dict[str, Any],
    examples: list[tuple[ObsT, ActT]],
    max_examples: int = 5,
) -> str:
    """Create a prompt for repairing a DNF policy given failure payload."""
    policy = payload.get("policy", "")
    bad_clauses = payload.get("bad_clauses", [])[:max_examples]
    fp_examples = payload.get("fp_examples", [])[:max_examples]
    fn_examples = payload.get("fn_examples", [])[:max_examples]
    feature_values = payload.get("feature_values", {})

    def _format_example(idx: int) -> str:
        if idx < 0 or idx >= len(examples):
            return f"[example_id={idx}] <missing>"
        obs, action = examples[idx]
        return f"[example_id={idx}] action={action} obs={obs}"

    lines: list[str] = []
    lines.append("You are an expert feature-library designer for LPP (DNF policies).")
    lines.append("We learned the following DNF policy that fails on training data.")
    lines.append("")
    lines.append("POLICY:")
    lines.append(policy)
    lines.append("")
    lines.append("TASK:")
    lines.append(
        "Propose new boolean feature functions fNN(s, a) or edits to existing "
        "features to fix the failures. You may add, modify, or replace features."
    )
    lines.append("")

    if bad_clauses:
        lines.append("WORST CLAUSES (by false positives / low precision):")
        for c in bad_clauses:
            lines.append(
                f"- clause_id={c['clause_id']} fires_pos={c['fires_pos']} "
                f"fires_neg={c['fires_neg']} precision={c['precision']:.3f} "
                f"num_fires={c['num_fires']} clause={c['clause']}"
            )
        lines.append("")

    if fp_examples:
        lines.append("FALSE POSITIVES (policy predicts 1 but label=0):")
        for e in fp_examples:
            idx = int(e["example_id"])
            lines.append(_format_example(idx))
            lines.append(f"  fired_clauses={e.get('fired_clause_ids', [])}")
            if feature_values:
                lines.append(f"  feature_values={feature_values.get(idx, {})}")
        lines.append("")

    if fn_examples:
        lines.append("FALSE NEGATIVES (policy predicts 0 but label=1):")
        for e in fn_examples:
            idx = int(e["example_id"])
            lines.append(_format_example(idx))
            if feature_values:
                lines.append(f"  feature_values={feature_values.get(idx, {})}")
        lines.append("")

    lines.append("Return a compact list of proposed feature functions.")
    return "\n".join(lines)


class CustomTimeoutError(Exception):
    """Custom exception raised when a time limit is exceeded.

    This exception is used to indicate that a block of code has exceeded
    the allowed time limit set by the `time_limit` context manager.
    """


@contextmanager
def time_limit(seconds: int) -> Generator[None, None, None]:
    """Context manager to enforce a time limit on a block of code.

    Args:
        seconds (int): The maximum number of seconds to allow the block to run.

    Raises:
        CustomTimeoutError: If the time limit is exceeded.
    """

    def handler(signum: int, frame: FrameType | None) -> None:
        """Signal handler to raise a CustomTimeoutError when the alarm is
        triggered.

        Args:
            signum (int): The signal number.
            frame (FrameType | None): The current stack frame (unused).
        """
        raise CustomTimeoutError(f"Timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def _gini_from_counts(pos: int, n: int) -> float:
    if n <= 0:
        return 0.0
    p = pos / n
    q = 1.0 - p
    return 1.0 - (p * p + q * q)


def gini_gain_per_feature(X: csr_matrix, y_bool: list[bool]) -> np.ndarray:
    """Returns gain[j] for each feature column j, assuming X is binary (0/1).

    Works efficiently for CSR.
    """
    y = np.asarray(y_bool, dtype=np.int8)  # 1 for pos, 0 for neg
    N = X.shape[0]
    P = int(y.sum())

    g_parent = _gini_from_counts(P, N)

    # N1: number of ones in each column (since X is 0/1)
    N1 = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)  # shape (F,)

    # P1: number of positives among those ones in each column
    # Multiply each row by y, then sum columns -> counts of pos where feature fires.
    Xy = X.multiply(y[:, None])
    P1 = np.asarray(Xy.sum(axis=0)).ravel().astype(np.int64)

    N0 = N - N1
    P0 = P - P1

    # Compute child ginis (avoid divide-by-zero cases)
    g1 = np.zeros_like(N1, dtype=np.float64)
    g0 = np.zeros_like(N1, dtype=np.float64)

    mask1 = N1 > 0
    mask0 = N0 > 0

    g1[mask1] = 1.0 - (
        (P1[mask1] / N1[mask1]) ** 2 + ((N1[mask1] - P1[mask1]) / N1[mask1]) ** 2
    )
    g0[mask0] = 1.0 - (
        (P0[mask0] / N0[mask0]) ** 2 + ((N0[mask0] - P0[mask0]) / N0[mask0]) ** 2
    )

    g_split = (N0 / N) * g0 + (N1 / N) * g1
    gain = g_parent - g_split

    return gain
