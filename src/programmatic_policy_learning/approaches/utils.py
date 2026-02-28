"""Utils for Testing LPP Approach."""

# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long

from __future__ import annotations

import ast
import json
import logging
import re
import signal
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Generator

import numpy as np
from scipy.sparse import csr_matrix

from programmatic_policy_learning.approaches.experts.grid_experts import (
    get_grid_expert,
)
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    grid_encoder,
    grid_hint_config,
)
from programmatic_policy_learning.dsl.llm_primitives.hint_generation.llm_based import (
    hint_extractor,
)
from programmatic_policy_learning.dsl.state_action_program import (
    StateActionProgram,
    set_dsl_functions,
)

GymnasiumEnvType: type | None = None
GymnasiumRecordVideoType: type | None = None
try:
    from gymnasium import Env as GymnasiumEnvType
    from gymnasium.wrappers import RecordVideo as GymnasiumRecordVideoType
except Exception:  # pylint: disable=broad-exception-caught
    GymnasiumEnvType = None
    GymnasiumRecordVideoType = None


def run_single_episode(
    env: Any,
    policy: Callable[[Any], Any],
    record_video: bool = False,
    video_out_path: str | None = None,
    max_num_steps: int = 50,
) -> float:
    """Run a single episode in the environment using the given policy."""

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

    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, _ = reset_out
    else:
        obs = reset_out
    if record_frames is not None and hasattr(env, "render"):
        try:
            frame = env.render()
            if frame is not None:
                record_frames.append(frame)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    total_reward = 0.0
    for _ in range(max_num_steps):
        action = policy(obs)
        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 4:
            new_obs, reward, done, _ = step_out
            terminated, truncated = done, False
        else:
            new_obs, reward, terminated, truncated, _ = step_out
        total_reward += reward

        obs = new_obs
        if record_frames is not None and hasattr(env, "render"):
            try:
                frame = env.render()
                if frame is not None:
                    record_frames.append(frame)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        if terminated or truncated:
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

    return total_reward


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
    hint_file = sorted(hint_dir.glob("*.json"))[0]
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


def sample_transition_example(
    env_factory: Callable[[int], Any],
    env_name: str,
    encoding_method: str,
    max_steps: int = 50,
) -> tuple[str, str, str]:
    """Sample a single (s_t, a_t, s_t1) example and format with encoding."""
    expert = get_grid_expert(env_name)
    env = env_factory(0)
    traj = hint_extractor.collect_full_episode(
        env, expert, max_steps=max_steps, sample_count=None
    )
    env.close()
    if not traj:
        raise ValueError("No trajectory data collected.")
    obs_t, action, obs_t1 = traj[0]

    symbol_map = grid_hint_config.get_symbol_map(env_name)
    encoder = grid_encoder.GridStateEncoder(
        grid_encoder.GridStateEncoderConfig(
            symbol_map=symbol_map,
            empty_token="empty",
            coordinate_style="rc",
        )
    )
    salient_tokens = grid_hint_config.SALIENT_TOKENS[env_name]

    def list_literal(obs: Any) -> str:
        rows: list[str] = []
        for r in range(obs.shape[0]):
            entries: list[str] = []
            for c in range(obs.shape[1]):
                token = obs[r, c]
                char = symbol_map.get(token, "?")
                entries.append(f"'{char}'")
            rows.append(f"[{', '.join(entries)}]")
        return "[\n" + "\n".join(f"  {row}" for row in rows) + "\n]"

    def listing(obs: Any) -> str:
        tokens = list(dict.fromkeys([*salient_tokens, encoder.cfg.empty_token]))
        objs = encoder.extract_objects(obs, tokens)
        entries: list[tuple[tuple[int, int], str]] = []
        for token, coords in objs.items():
            if not coords:
                continue
            for coord in coords:
                entries.append((coord, token))
        entries.sort(key=lambda item: item[0])
        return "\n".join(f"{coord} - '{label}'" for coord, label in entries)

    if encoding_method == "1":
        state_t = list_literal(obs_t)
        state_t1 = list_literal(obs_t1)
        state_t1 += f"\nToken meanings: {grid_hint_config.SYMBOL_MAPS[env_name]}"
    else:
        state_t = listing(obs_t)
        state_t1 = listing(obs_t1)

    action_token = obs_t[action]
    action_text = f"{action}: '{action_token}'"
    return state_t, action_text, state_t1


def log_feature_collisions(
    X: Any,
    y: np.ndarray | None,
    _examples: list[tuple[np.ndarray, tuple[int, int]]] | None,
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

    if collisions:
        logging.info("Feature collisions found: %d", len(collisions))
        for prev_idx, cur_idx, prev_label in collisions:
            logging.info(
                "Collision: row %d(label=%d) vs row %d(label=%d)",
                prev_idx,
                prev_label,
                cur_idx,
                labels[cur_idx],
            )
        return group_collision_indices(collisions, labels)

    logging.info("No feature collisions found.")
    return []


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


def log_plp_violation_counts(
    plps: list[StateActionProgram],
    demonstrations: Any,
    dsl_functions: dict[str, Any],
) -> list[StateActionProgram]:
    """Log how many demo steps each PLP fails (False on expert action)."""
    set_dsl_functions(dsl_functions)
    counts: list[tuple[int, StateActionProgram, list[Any], list[Any]]] = []
    total_steps = len(demonstrations.steps)

    for plp in plps:
        violations = 0
        all_obs = []
        all_acts = []
        for obs, action in demonstrations.steps:
            try:
                if not plp(obs, action):
                    violations += 1
                    all_obs.append(obs)
                    all_acts.append(action)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(e)
                logging.info("EXCEPTION")
                violations += 1
        counts.append((violations, plp, all_obs, all_acts))

    counts.sort(key=lambda item: item[0])
    logging.info("PLP violation counts (lower is better):")
    for violations, plp, _, _ in counts[:10]:
        rate = (violations / total_steps) if total_steps else 0.0
        logging.info(
            "violations=%d/%d (%.2f%%) | plp=%s",
            violations,
            total_steps,
            100.0 * rate,
            plp,
        )
        # for idx, item in enumerate(obs_all):
        #     logging.info(item)
        #     actionn = act_all[idx]
        #     logging.info(actionn)
        #     logging.info(item[actionn])
    return [plp for _, plp, _, _ in counts]


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
    s: np.ndarray,
    a: tuple[int, int],
    *,
    label: int,
    idx: int,
) -> str:
    """Format one labeled (state, action) example for prompt text."""
    rows = []
    for r in range(s.shape[0]):
        row = ", ".join(repr(str(x)) for x in s[r])
        rows.append(f"[{row}]")
    grid = "\n".join(f"  {row}" for row in rows)
    r, c = int(a[0]), int(a[1])
    cell = s[r, c] if 0 <= r < s.shape[0] and 0 <= c < s.shape[1] else "OOB"
    return f"- idx={idx} label={label} action=({r}, {c}) cell={cell}\n[\n{grid}\n]"


def _format_ascii_legend(symbol_map: dict[str, str]) -> str:
    lines = ["LEGEND:"]
    for token, char in symbol_map.items():
        lines.append(f"  {char} = {token}")
    lines.append("  (token_code)! = action cell")
    return "\n".join(lines)


def _format_one_example_ascii(
    s: np.ndarray,
    a: tuple[int, int],
    *,
    label: int,
    idx: int,
    symbol_map: dict[str, str],
) -> str:
    """Format one labeled (state, action) example using ASCII token codes."""
    h, w = s.shape[0], s.shape[1]
    code_width = max((len(code) for code in symbol_map.values()), default=1)

    r, c = int(a[0]), int(a[1])
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
    s: np.ndarray,
    a: tuple[int, int],
    *,
    label: int,
    idx: int,
) -> str:
    """Format one labeled (state, action) example using coordinate lists."""
    s_str = s.astype(str)
    h, w = s_str.shape[0], s_str.shape[1]
    tokens, counts = np.unique(s_str, return_counts=True)
    bg_idx = int(np.argmax(counts)) if tokens.size else -1
    background = tokens[bg_idx] if bg_idx >= 0 else ""

    r, c = int(a[0]), int(a[1])
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


def _maybe_map_ascii_tokens(s: np.ndarray, symbol_map: dict[str, str]) -> np.ndarray:
    """Map ASCII tokens to canonical names if symbol_map is provided."""
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
    examples: list[tuple[np.ndarray, tuple[int, int]]],
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
    examples: list[tuple[np.ndarray, tuple[int, int]]],
    symbol_map: dict[str, str],
) -> str:
    s_pos, a_pos = examples[pos_idx]
    s_neg, _ = examples[neg_idx]
    s_pos_tok = _maybe_map_ascii_tokens(s_pos, symbol_map)
    s_neg_tok = _maybe_map_ascii_tokens(s_neg, symbol_map)
    agent_pos = _find_token_positions(s_pos_tok, "agent")
    star_pos = _find_token_positions(s_pos_tok, "star")
    agent_anchor = agent_pos[0] if agent_pos else None
    star_anchor = star_pos[0] if star_pos else None
    action_pos = (int(a_pos[0]), int(a_pos[1]))
    agent_minus_star = None
    if agent_anchor and star_anchor:
        agent_minus_star = (
            agent_anchor[0] - star_anchor[0],
            agent_anchor[1] - star_anchor[1],
        )
    diffs = _diff_cells(s_pos_tok, s_neg_tok, limit=DIFF_LIMIT)
    diag_summary = _diag_path_summary(
        s_pos_tok, star_anchor, agent_anchor, "drawn", limit_steps=None
    )
    lines = []
    lines.append("DIFF HINTS:")
    lines.append(f"- AGENT_POS: {agent_anchor}")
    lines.append(f"- STAR_POS: {star_anchor}")
    lines.append(f"- ACTION_POS: {action_pos}")
    lines.append(f"- agent_minus_star: {agent_minus_star}")
    lines.append(f"- top_k_diffs: {diffs}")
    lines.append(f"- diag_path_from_star_toward_agent: {diag_summary}")
    return "\n".join(lines)


def _format_one_example_enc2(
    s: np.ndarray,
    a: tuple[int, int],
    *,
    label: int,
    idx: int,
    symbol_map: dict[str, str],
    small_list_threshold: int = SMALL_LIST_THRESHOLD,
    patch_k: int = PATCH_K,
) -> str:
    """Format one labeled (state, action) example using enc_2."""
    s_tok = _maybe_map_ascii_tokens(s, symbol_map)
    h, w = s_tok.shape[0], s_tok.shape[1]
    tokens, counts = np.unique(s_tok, return_counts=True)
    bg_idx = int(np.argmax(counts)) if tokens.size else -1
    background = tokens[bg_idx] if bg_idx >= 0 else ""

    r, c = int(a[0]), int(a[1])
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


def _format_one_example_enc2_delta(
    s: np.ndarray,
    a: tuple[int, int],
    *,
    label: int,
    idx: int,
    symbol_map: dict[str, str],
    small_list_threshold: int = SMALL_LIST_THRESHOLD,
) -> str:
    """Compact delta-only summary for enc_2 collision evidence."""
    s_tok = _maybe_map_ascii_tokens(s, symbol_map)
    tokens, counts = np.unique(s_tok, return_counts=True)
    bg_idx = int(np.argmax(counts)) if tokens.size else -1
    background = tokens[bg_idx] if bg_idx >= 0 else ""

    r, c = int(a[0]), int(a[1])
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


def build_collision_repair_prompt(
    pos_indices: list[int],
    neg_indices: list[int],
    examples: list[tuple[np.ndarray, tuple[int, int]]],
    *,
    env_name: str | None = None,
    existing_feature_summary: str | None = None,  # pylint: disable=unused-argument
    max_per_label: int = 5,
    collision_feedback_enc: str = "enc_1",
    pos_indices_2: list[int] | None = None,
    neg_indices_2: list[int] | None = None,
    seed: int | None = None,
    collision_template_feedback: bool = True,
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
    if use_ascii and env_name:
        symbol_map = grid_hint_config.get_symbol_map(env_name)
        legend_block = _format_ascii_legend(symbol_map) + "\n\n"
    elif env_name:
        symbol_map = grid_hint_config.get_symbol_map(env_name)
    # FORMAT2 note: uses richer evidence + deterministic example selection.
    if use_enc2:
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

    N_NEW = "N_NEW"
    TOKEN_A = "TOKEN_A"
    TOKEN_B = "TOKEN_B"
    DRs = "DRs"
    K = "K"
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
    if env_name:
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
    feature_prompt_filename = (
        "featured_collision_feedback_enc2.txt"
        if use_enc2
        else "featured_collision_feedback_enc1.txt"
    )
    feature_prompt_path = (
        Path(__file__).resolve().parent.parent
        / "dsl"
        / "llm_primitives"
        / "prompts"
        / "py_feature_gen"
        / feature_prompt_filename
    )
    prompt_feature = feature_prompt_path.read_text(encoding="utf-8")
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
    prompt_template = f"""
## SYSTEM

You are an expert feature-library repair agent for Logical Programmatic Policies in grid-based games.

Your job:
Given evidence of feature collisions, propose NEW boolean feature templates f_i(s, a) that can distinguish positive vs negative examples within the collision bucket.

A feature collision means:
All provided examples have IDENTICAL feature vectors under the current feature set (same feature-key),
yet labels differ. Therefore, the current features are provably insufficient.

You must propose new feature families that break this indistinguishability.

---

## ENVIRONMENT

- Grid observation s is a 2D list of tokens (strings).
- Action a is a clicked cell coordinate: a = (row, col).
- a may contain numpy integers.

Every feature MUST begin with this exact validation logic:

    try:
        r = int(a[0]); c = int(a[1])
    except:
        return False
    h = len(s); w = len(s[0]) if h else 0
    if not (0 <= r < h and 0 <= c < w): return False

No imports. Only Python built-ins.

---

## TOKEN RESTRICTIONS (CRITICAL)

You MUST NOT use:
- any raw token characters like {raw_token_examples}

You MUST use ONLY these placeholders inside feature code:
- ${TOKEN_A}  (token placeholder A)
- ${TOKEN_B}  (token placeholder B)
- ${DRs}    (list of (dr, dc) tuples placeholder)
- ${K}      (small positive integer placeholder)

No other placeholders. No token lists. Optional third token placeholder is NOT allowed.
Each feature may use at most 4 placeholders total.
Specific token names may appear in the COLLISION EVIDENCE text; the prohibition applies only to feature CODE, which must use placeholders.

---

## COLLISION EVIDENCE

All examples below produce IDENTICAL feature vectors under the current feature set,
yet the expert labels differ. Therefore, the current features are provably insufficient.

NOTE: Each bucket below may require different features to resolve.

BUCKET 1
POSITIVE EXAMPLES (label = 1):
{legend_block}{(chr(10) * 2).join(pos_blocks[:2])}

NEGATIVE EXAMPLES (label = 0):
{(chr(10) * 2).join(neg_blocks[:3])}

{diff_hints_1}

{bucket2_block}

---

## TASK

1) Carefully compare the POSITIVE vs NEGATIVE examples.
2) Identify structural distinctions that are NOT captured by typical local templates.
3) Propose NEW feature templates that can separate (at least some of) the positives from negatives in THIS bucket.

Important:
- Do NOT generate near-duplicates of the same adjacency/scan/count family.
- Prefer features that capture *relational structure*.
- Do NOT overfit to board size. No hard-coded row numbers.

---

## OUTPUT FORMAT (STRICT)

You MUST output ONLY valid JSON, directly parsable by json.loads().
No markdown. No prose.

Return exactly this structure:

{{
  "features": [
    {{
      "id": "f101",
      "name": "f101",
      "description": "What structural distinction this feature captures and why it helps separate the bucket.",
      "source": "def f101(s, a):\\n    ...\\n"
    }},
    ...
  ]
}}

Rules:
- ids/names must be sequential: f101, f102, ...
- Provide EXACTLY {N_NEW} new features.
- Each "source" must be a valid Python function string with \\n escaped newlines.
- Every feature must return True/False.
- Every feature must include the exact action validation boilerplate shown above.
- Use only ${TOKEN_A}, ${TOKEN_B}, ${DRs}, ${K} placeholders.

FINAL CHECK:
- No raw tokens {final_check_tokens} appear in code.
- Output is valid JSON only.

"""
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
    examples: list[tuple[np.ndarray, tuple[int, int]]],
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
    examples: list[tuple[np.ndarray, tuple[int, int]]],
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
