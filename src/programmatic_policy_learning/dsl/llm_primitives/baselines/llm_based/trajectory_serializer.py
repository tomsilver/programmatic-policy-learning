"""Serialize expert trajectories into textual hint blocks."""

from typing import Sequence

import numpy as np

from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.grid_encoder import (
    GridStateEncoder,
)
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.transition_analyzer import (
    GenericTransitionAnalyzer,
    extract_relational_facts,
)

Coord = tuple[int, int]


def _to_token_grid(obs: np.ndarray) -> list[list[str]]:
    return [[str(cell) for cell in row] for row in obs.tolist()]


def _token_counts(grid: list[list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in grid:
        for tok in row:
            counts[tok] = counts.get(tok, 0) + 1
    return counts


def _background_token(counts: dict[str, int]) -> str:
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _coords_by_token(grid: list[list[str]]) -> dict[str, list[Coord]]:
    by_token: dict[str, list[Coord]] = {}
    for r, row in enumerate(grid):
        for c, tok in enumerate(row):
            by_token.setdefault(tok, []).append((int(r), int(c)))
    return by_token


def _sorted_tokens_by_rarity(counts: dict[str, int], *, exclude: set[str]) -> list[str]:
    items = [(tok, cnt) for tok, cnt in counts.items() if tok not in exclude]
    items.sort(key=lambda kv: (kv[1], kv[0]))
    return [tok for tok, _ in items]


def _format_coord_list(coords: list[Coord], cap: int = 40) -> str:
    coords_sorted = sorted(coords)
    if len(coords_sorted) <= cap:
        return f"{coords_sorted}"
    shown = coords_sorted[:cap]
    remaining = len(coords_sorted) - cap
    return f"{shown} ... +{remaining} more"


def _row_range_compression(coords: list[Coord]) -> str | None:
    rows: dict[int, list[int]] = {}
    for r, c in coords:
        rows.setdefault(r, []).append(c)
    if not rows:
        return None

    max_cols_per_row = max(len(set(cols)) for cols in rows.values())
    if len(rows) > 20 and max_cols_per_row <= 1:
        return None

    def compress(cols: list[int]) -> str:
        cols_sorted = sorted(set(cols))
        ranges: list[str] = []
        start = cols_sorted[0]
        prev = cols_sorted[0]
        for c in cols_sorted[1:]:
            if c == prev + 1:
                prev = c
                continue
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = c
            prev = c
        if start == prev:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{prev}")
        return ",".join(ranges)

    row_items = []
    for r in sorted(rows):
        row_items.append(f"{r}:[{compress(rows[r])}]")
    return "rows {" + ", ".join(row_items) + "}"


def _connected_components_count(
    grid: list[list[str]],
    token: str,
) -> tuple[int, int]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    visited = [[False] * w for _ in range(h)]
    comps = 0
    largest = 0

    def neighbors(r: int, c: int) -> list[Coord]:
        out = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
        return out

    for r in range(h):
        for c in range(w):
            if visited[r][c] or grid[r][c] != token:
                continue
            comps += 1
            stack = [(r, c)]
            visited[r][c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for nr, nc in neighbors(rr, cc):
                    if not visited[nr][nc] and grid[nr][nc] == token:
                        visited[nr][nc] = True
                        stack.append((nr, nc))
            largest = max(largest, size)
    return comps, largest


def _local_view_3x3(grid: list[list[str]], r: int, c: int) -> list[list[str]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    view: list[list[str]] = []
    for dr in (-1, 0, 1):
        row: list[str] = []
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                row.append(grid[rr][cc])
            else:
                row.append("OOB")
        view.append(row)
    return view


def _local_view_3x3_marked(grid: list[list[str]], r: int, c: int) -> list[list[str]]:
    view = _local_view_3x3(grid, r, c)
    view[1][1] = f"{view[1][1]}[*]"
    return view


def _ray_feature(
    grid: list[list[str]],
    background: str,
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> tuple[str | None, int | None]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    steps = 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        steps += 1
        tok = grid[rr][cc]
        if tok != background:
            return tok, steps
        rr += dr
        cc += dc
    return None, None


def _nearest_distance(coords: list[Coord], r: int, c: int) -> int | None:
    if not coords:
        return None
    return min(abs(r - rr) + abs(c - cc) for rr, cc in coords)


def _effect_summary(
    grid_t: list[list[str]],
    grid_t1: list[list[str]] | None,
    background: str,
    action: Coord,
) -> list[str]:
    if grid_t1 is None:
        return [
            "EFFECT_SUMMARY:",
            "- num_cells_changed: 0",
            "- token_at_target_after: None",
            "- created: []",
            "- removed: []",
            "- moved: []",
            "- terminal: yes",
        ]

    h = len(grid_t)
    w = len(grid_t[0]) if h else 0
    created: list[tuple[str, Coord]] = []
    removed: list[tuple[str, Coord]] = []
    overwritten: list[tuple[Coord, str, str]] = []
    changed = 0
    for r in range(h):
        for c in range(w):
            tok0 = grid_t[r][c]
            tok1 = grid_t1[r][c]
            if tok0 != tok1:
                changed += 1
                if background not in (tok0, tok1):
                    overwritten.append(((r, c), tok0, tok1))
                if tok0 == background and tok1 != background:
                    created.append((tok1, (r, c)))
                elif tok0 != background and tok1 == background:
                    removed.append((tok0, (r, c)))

    moved: list[str] = []
    moved_tokens: set[str] = set()
    tokens = sorted(
        {tok for row in grid_t for tok in row if tok != background}
        | {tok for row in grid_t1 for tok in row if tok != background}
    )
    for tok in tokens:
        old_positions = {
            (r, c) for r in range(h) for c in range(w) if grid_t[r][c] == tok
        }
        new_positions = {
            (r, c) for r in range(h) for c in range(w) if grid_t1[r][c] == tok
        }
        disappeared = old_positions - new_positions
        appeared = new_positions - old_positions
        if len(disappeared) == 1 and len(appeared) == 1:
            r0, c0 = next(iter(disappeared))
            r1, c1 = next(iter(appeared))
            moved.append(f"{tok}: ({r0},{c0})->({r1},{c1})")
            moved_tokens.add(tok)

    if moved_tokens:
        created = [item for item in created if item[0] not in moved_tokens]
        removed = [item for item in removed if item[0] not in moved_tokens]

    created_sorted = sorted(created, key=lambda x: (x[0], x[1]))
    removed_sorted = sorted(removed, key=lambda x: (x[0], x[1]))
    overwritten_sorted = sorted(
        overwritten, key=lambda x: (x[0][0], x[0][1], x[1], x[2])
    )

    def fmt_items(items: list[tuple[str, Coord]], cap: int = 20) -> str:
        shown = items[:cap]
        formatted = [f"{tok}@({r},{c})" for tok, (r, c) in shown]
        if len(items) > cap:
            formatted.append(f"... +{len(items) - cap} more")
        return f"[{', '.join(formatted)}]"

    def fmt_overwritten(items: list[tuple[Coord, str, str]], cap: int = 20) -> str:
        shown = items[:cap]
        formatted = [f"{old}@({r},{c})->{new}" for (r, c), old, new in shown]
        if len(items) > cap:
            formatted.append(f"... +{len(items) - cap} more")
        return f"[{', '.join(formatted)}]"

    moved_list = moved[:20]
    if len(moved) > 20:
        moved_list.append(f"... +{len(moved) - 20} more")

    return [
        "EFFECT_SUMMARY:",
        f"- num_cells_changed: {changed}",
        f"- token_at_target_after: {grid_t1[action[0]][action[1]] if grid_t1 else None}",
        f"- created: {fmt_items(created_sorted)}",
        f"- removed: {fmt_items(removed_sorted)}",
        f"- overwritten: {fmt_overwritten(overwritten_sorted)}",
        f"- moved: [{', '.join(moved_list)}]",
        "- terminal: no",
    ]

def trajectory_to_text(
    trajectory: Sequence[tuple[np.ndarray, tuple[int, int], np.ndarray]],
    *,
    encoder: GridStateEncoder,
    analyzer: GenericTransitionAnalyzer,
    salient_tokens: list[str],
    encoding_method: str,
    max_steps: int | None = None,
) -> str:
    """Convert (obs, action, next_obs) tuples to structured text."""
    blocks: list[str] = []
    steps = trajectory[:max_steps] if max_steps else trajectory

    if steps and encoding_method == "5":
        first_obs = steps[0][0]
        h = int(first_obs.shape[0])
        w = int(first_obs.shape[1])
        blocks.append(f"BOARD: {h}x{w}")

    for i, (obs_t, action, obs_t1) in enumerate(steps):

        if encoding_method == "1":
            ascii_t = encoder.to_ascii_list_literal(obs_t, action, i)
            block = ascii_t
            blocks.append(block)
        elif encoding_method == "5":
            grid_t = _to_token_grid(obs_t)
            counts = _token_counts(grid_t)
            background = _background_token(counts)
            by_token = _coords_by_token(grid_t)

            non_bg_tokens = _sorted_tokens_by_rarity(counts, exclude={background})
            if non_bg_tokens:
                most_common_non_bg = sorted(
                    [(t, counts[t]) for t in non_bg_tokens],
                    key=lambda kv: (-kv[1], kv[0]),
                )[0][0]
                focus_tokens = _sorted_tokens_by_rarity(
                    counts, exclude={background, most_common_non_bg}
                )[:8]
            else:
                focus_tokens = []

            lines: list[str] = []
            h = len(grid_t)
            w = len(grid_t[0]) if h else 0
            lines.append(f"== STATE s_{i} ==")
            lines.append("")
            lines.append("BACKGROUND_TOKEN: " + background)
            lines.append("")
            lines.append("NON_EMPTY_BY_TOKEN:")
            for tok in non_bg_tokens:
                coords = by_token.get(tok, [])
                coord_list = _format_coord_list(coords, cap=40)
                compress_trigger = len(coords) >= max(12, w // 2)
                if compress_trigger:
                    coord_str = _row_range_compression(coords)
                    if coord_str is None or len(coord_str) > len(coord_list):
                        coord_str = coord_list
                else:
                    coord_str = coord_list
                lines.append(f"- {tok} (count={len(coords)}): {coord_str}")

            lines.append("")
            lines.append(f"FOCUS_TOKENS: {focus_tokens}")
            lines.append("")
            lines.append("GLOBAL_RELATIONS:")
            token_counts = {t: counts[t] for t in focus_tokens}
            lines.append(f"- token_counts: {token_counts}")
            cc_lines: list[str] = []
            for tok in focus_tokens:
                cnt = counts.get(tok, 0)
                if 2 <= cnt <= 200:
                    comps, largest = _connected_components_count(grid_t, tok)
                    cc_lines.append(
                        f"  - token={tok}: num_components={comps}, largest_size={largest}"
                    )
            if cc_lines:
                lines.append("- connected_components:")
                lines.extend(cc_lines)
            else:
                lines.append("- connected_components: none")

            r, c = int(action[0]), int(action[1])
            token_at = grid_t[r][c]
            lines.append("")
            lines.append(f"== ACTION a_{i} ==")
            lines.append("")
            lines.append("ACTION:")
            lines.append("- type: click")
            lines.append(f"- target: ({r},{c})")
            lines.append(f"- token_at_target_before: {token_at}")
            lines.append("")
            lines.append("LOCAL_VIEW:")
            lines.append("- neighborhood_3x3_tokens:")
            lines.append(f"  {_local_view_3x3_marked(grid_t, r, c)}")
            lines.append("")
            lines.append("RAY_FEATURES_FROM_TARGET:")
            for name, dr, dc in (
                ("up", -1, 0),
                ("down", 1, 0),
                ("left", 0, -1),
                ("right", 0, 1),
            ):
                tok, dist = _ray_feature(grid_t, background, r, c, dr, dc)
                lines.append(f"- {name}: first_nonbackground={tok}, dist={dist}")
            lines.append("")
            lines.append("DISTANCES_FROM_TARGET (Manhattan):")
            for tok in focus_tokens:
                dist = _nearest_distance(by_token.get(tok, []), r, c)
                lines.append(f"- to_nearest_{tok}: {dist}")

            grid_t1 = _to_token_grid(obs_t1) if obs_t1 is not None else None
            lines.append("")
            lines.append(f"== TRANSITION (s_{i} -> s_{i+1}) ==")
            lines.append("")
            lines.extend(_effect_summary(grid_t, grid_t1, background, (r, c)))

            blocks.append("\n".join(lines))
        else:
            tokens = list(dict.fromkeys([*salient_tokens, encoder.cfg.empty_token]))
            objs = encoder.extract_objects(obs_t, tokens)
            # objs_next = encoder.extract_objects(obs_t1, tokens)
            listing = encoder.format_cell_value_listing(
                objs,
                i,
                action=action,
            )

            if encoding_method == "2":
                block = listing
                blocks.append(block)

            elif encoding_method == "3":
                events = analyzer.analyze(
                    obs_t,
                    action,
                    obs_t1,
                    encoder=encoder,
                )
                change_summary = "\n".join(f"- {event}" for event in events)

                block = listing
                blocks.append(block)
                blocks.append(f"\nCHANGES SUMMARY:\n{change_summary}")
            elif encoding_method == "4":
                events = analyzer.analyze(
                    obs_t,
                    action,
                    obs_t1,
                    encoder=encoder,
                )
                change_summary = "\n".join(f"- {event}" for event in events)

                relational_facts = extract_relational_facts(
                    obs_t,
                    objs,
                    symbol_map=encoder.cfg.symbol_map,
                    reference_tokens=salient_tokens,
                    empty_tokens=(encoder.cfg.empty_token,),
                )

                block = listing
                blocks.append(block)
                blocks.append(f"\nCHANGES SUMMARY:\n{change_summary}\n\n")
                blocks.append(
                    f"Relational Facts:\n{', '.join(str(each) for each in relational_facts)}"
                )
    if steps:
        final_obs = steps[-1][2]
        final_step_idx = len(steps)
        if encoding_method == "1":
            blocks.append(
                encoder.to_ascii_list_literal(
                    final_obs, action=None, step_index=final_step_idx
                )
            )
        elif encoding_method == "5":
            grid_t = _to_token_grid(final_obs)
            counts = _token_counts(grid_t)
            background = _background_token(counts)
            by_token = _coords_by_token(grid_t)
            non_bg_tokens = _sorted_tokens_by_rarity(counts, exclude={background})
            if non_bg_tokens:
                most_common_non_bg = sorted(
                    [(t, counts[t]) for t in non_bg_tokens],
                    key=lambda kv: (-kv[1], kv[0]),
                )[0][0]
                focus_tokens = _sorted_tokens_by_rarity(
                    counts, exclude={background, most_common_non_bg}
                )[:8]
            else:
                focus_tokens = []

            lines: list[str] = []
            h = len(grid_t)
            w = len(grid_t[0]) if h else 0
            lines.append(f"== STATE s_{final_step_idx} ==")
            lines.append("")
            lines.append("BACKGROUND_TOKEN: " + background)
            lines.append("")
            lines.append("NON_EMPTY_BY_TOKEN:")
            for tok in non_bg_tokens:
                coords = by_token.get(tok, [])
                coord_list = _format_coord_list(coords, cap=40)
                compress_trigger = len(coords) >= max(12, w // 2)
                if compress_trigger:
                    coord_str = _row_range_compression(coords)
                    if coord_str is None or len(coord_str) > len(coord_list):
                        coord_str = coord_list
                else:
                    coord_str = coord_list
                lines.append(f"- {tok} (count={len(coords)}): {coord_str}")
            lines.append("")
            lines.append(f"FOCUS_TOKENS: {focus_tokens}")
            lines.append("")
            lines.append("GLOBAL_RELATIONS:")
            token_counts = {t: counts[t] for t in focus_tokens}
            lines.append(f"- token_counts: {token_counts}")
            cc_lines: list[str] = []
            for tok in focus_tokens:
                cnt = counts.get(tok, 0)
                if 2 <= cnt <= 200:
                    comps, largest = _connected_components_count(grid_t, tok)
                    cc_lines.append(
                        f"  - token={tok}: num_components={comps}, largest_size={largest}"
                    )
            if cc_lines:
                lines.append("- connected_components:")
                lines.extend(cc_lines)
            else:
                lines.append("- connected_components: none")
            blocks.append("\n".join(lines))
        else:
            tokens = list(dict.fromkeys([*salient_tokens, encoder.cfg.empty_token]))
            objs = encoder.extract_objects(final_obs, tokens)
            blocks.append(
                encoder.format_cell_value_listing(
                    objs,
                    final_step_idx,
                    action=None,
                )
            )
    return "\n\n".join(blocks)


def trajectory_to_diff_text(
    trajectory: Sequence[tuple[np.ndarray, tuple[int, int], np.ndarray]],
    *,
    encoder: GridStateEncoder,
    max_steps: int | None = None,
) -> str:
    """Convert (obs, action, next_obs) tuples to diff-style text."""
    steps = trajectory[:max_steps] if max_steps else trajectory
    if not steps:
        return ""

    symbol_map = encoder.cfg.symbol_map

    def token_to_symbol(token: str) -> str:
        return symbol_map.get(token, "?")

    def grid_to_literal(obs: np.ndarray) -> str:
        row_blocks: list[str] = []
        for r in range(obs.shape[0]):
            entries: list[str] = []
            for c in range(obs.shape[1]):
                entries.append(f"'{token_to_symbol(obs[r, c])}'")
            row_blocks.append(f"[{', '.join(entries)}];")
        indented_rows = "\n".join(f"  {row}" for row in row_blocks)
        return f"[\n{indented_rows}\n]"

    blocks: list[str] = []

    first_obs = steps[0][0]
    blocks.append(f"Observation (s_0):\n{grid_to_literal(first_obs)}")

    for i, (obs_t, action, obs_t1) in enumerate(steps):
        r, c = action
        token_name = obs_t[r, c]
        token_symbol = token_to_symbol(token_name)
        token_label = str(token_name)

        step_lines: list[str] = []
        step_lines.append(f"### Step {i}")
        step_lines.append(
            f"Action: Click `({r}, {c})`, cell contained `{token_symbol}` ({token_label}).  "
        )

        changes: list[str] = []
        for rr in range(obs_t.shape[0]):
            for cc in range(obs_t.shape[1]):
                if obs_t[rr, cc] != obs_t1[rr, cc]:
                    before = token_to_symbol(obs_t[rr, cc])
                    after = token_to_symbol(obs_t1[rr, cc])
                    changes.append(f"- `({rr}, {cc}): {before} -> {after}`")

        step_lines.append(f"Diff (`s_{i} → s_{i + 1}`):")
        if changes:
            step_lines.extend(changes)
        else:
            step_lines.append("- (no changes)")
        step_lines.append(f"Unchanged: all other cells same as `s_{i}`.")
        blocks.append("\n".join(step_lines))

    final_obs = steps[-1][2]
    final_step_idx = len(steps)
    final_lines = [
        f"### Step {final_step_idx}",
        f"Observation (s_{final_step_idx}):\n{grid_to_literal(final_obs)}",
        "Action: None (terminal state).",
    ]
    blocks.append("\n".join(final_lines))

    return "\n\n".join(blocks)
