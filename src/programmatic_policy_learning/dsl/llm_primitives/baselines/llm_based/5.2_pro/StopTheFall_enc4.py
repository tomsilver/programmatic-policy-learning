import numpy as np


def policy(obs):
    """
    Infer-and-act policy:

    1) Identify the column containing the single falling token.
    2) Find the nearest red token to the falling token by Manhattan distance
       (tie-breaker: prefer the red token with larger row index, then smaller col).
       Let r_red be that red token's row.
    3) In the falling token's column, find the first static token from top; let
       r_base be the row just above it (or the last row if there is no static token).
    4) "Dig" a contiguous vertical shaft by clicking, one per call, the lowest row
       (largest row index) that is still an EMPTY cell within rows [r_red, r_base]
       in the falling token's column.
    5) Once there are no EMPTY cells left in that interval, click the advance token.

    This reproduces the demonstrated behavior of repeatedly clicking empty cells in the
    falling token's column from the bottom of the target interval upward, then clicking
    the advance token.
    """
    obs = np.asarray(obs)
    nrows, ncols = obs.shape

    def is_string_grid(a):
        return a.dtype.kind in ("U", "S", "O")

    # -------- Token masks (string case) --------
    if is_string_grid(obs):
        empty_mask = obs == "empty"
        red_mask = obs == "red_token"
        static_mask = obs == "static_token"
        advance_mask = obs == "advance_token"
        falling_mask = obs == "falling_token"
    else:
        # -------- Token id inference (numeric/other case) --------
        vals, counts = np.unique(obs, return_counts=True)
        # Empty: most frequent
        empty_val = vals[int(np.argmax(counts))]

        # Collect occurrence info
        positions = {v: np.argwhere(obs == v) for v in vals}
        non_empty_vals = [v for v in vals if v != empty_val]

        # Rare candidates for falling/advance (typically count 1)
        rare = [v for v in non_empty_vals if positions[v].shape[0] <= 2]
        falling_val = None
        advance_val = None
        if rare:
            # Falling: rare value with smallest min row
            falling_val = min(
                rare,
                key=lambda v: (
                    int(np.min(positions[v][:, 0])),
                    int(np.min(positions[v][:, 1])),
                ),
            )
            # Advance: rare value with largest max row
            advance_val = max(
                rare,
                key=lambda v: (
                    int(np.max(positions[v][:, 0])),
                    -int(np.min(positions[v][:, 1])),
                ),
            )
            # If only one rare token, decide by its row location
            if falling_val == advance_val:
                r0 = int(np.min(positions[falling_val][:, 0]))
                if r0 < nrows / 2:
                    advance_val = None
                else:
                    falling_val = None

        remaining = [v for v in non_empty_vals if v not in (falling_val, advance_val)]

        # Static: tends to be abundant and concentrated at larger row indices
        static_val = None
        if remaining:

            def static_score(v):
                pos = positions[v]
                return float(pos.shape[0]) * (float(np.mean(pos[:, 0])) + 1.0)

            static_val = max(remaining, key=static_score)
            remaining2 = [v for v in remaining if v != static_val]
        else:
            remaining2 = []

        # Red: pick the most frequent among remaining
        red_val = None
        if remaining2:
            red_val = max(remaining2, key=lambda v: positions[v].shape[0])

        # Build masks
        empty_mask = obs == empty_val
        red_mask = (
            (obs == red_val)
            if red_val is not None
            else np.zeros_like(empty_mask, dtype=bool)
        )
        static_mask = (
            (obs == static_val)
            if static_val is not None
            else np.zeros_like(empty_mask, dtype=bool)
        )
        advance_mask = (
            (obs == advance_val)
            if advance_val is not None
            else np.zeros_like(empty_mask, dtype=bool)
        )
        falling_mask = (
            (obs == falling_val)
            if falling_val is not None
            else np.zeros_like(empty_mask, dtype=bool)
        )

    # -------- Locate key tokens --------
    fall_pos = np.argwhere(falling_mask)
    adv_pos = np.argwhere(advance_mask)
    red_pos = np.argwhere(red_mask)

    # Fallbacks to always return a valid coordinate
    if fall_pos.size == 0:
        if adv_pos.size != 0:
            r, c = adv_pos[0]
            return (int(r), int(c))
        # Any non-empty cell else (0,0)
        non_empty = np.argwhere(~empty_mask)
        if non_empty.size != 0:
            r, c = non_empty[0]
            return (int(r), int(c))
        return (0, 0)

    r_f, c_f = (int(fall_pos[0, 0]), int(fall_pos[0, 1]))

    # Determine r_red from nearest red token
    if red_pos.size == 0:
        r_red = None
    else:
        d = np.abs(red_pos[:, 0] - r_f) + np.abs(red_pos[:, 1] - c_f)
        min_d = np.min(d)
        cand = red_pos[d == min_d]
        # tie-break: larger row, then smaller col
        idx = np.lexsort((cand[:, 1], -cand[:, 0]))[0]
        r_red = int(cand[idx, 0])

    # Determine r_base (row just above first static in falling column)
    col_static_rows = np.where(static_mask[:, c_f])[0]
    if col_static_rows.size == 0:
        r_base = nrows - 1
    else:
        r_base = int(np.min(col_static_rows) - 1)
        if r_base < 0:
            r_base = 0

    if r_red is None:
        # No red: nothing to dig for this heuristic; advance immediately if possible
        if adv_pos.size != 0:
            r, c = adv_pos[0]
            return (int(r), int(c))
        return (r_f, c_f)

    r_start = max(0, min(r_red, r_base))
    r_end = max(0, max(r_red, r_base))

    # Dig: choose lowest still-empty cell in [r_start, r_end] at column c_f
    segment_empty_rows = np.where(empty_mask[r_start : r_end + 1, c_f])[0]
    if segment_empty_rows.size != 0:
        r_click = int(r_start + np.max(segment_empty_rows))
        return (r_click, int(c_f))

    # Otherwise click advance token
    if adv_pos.size != 0:
        r, c = adv_pos[0]
        return (int(r), int(c))

    return (r_f, c_f)
