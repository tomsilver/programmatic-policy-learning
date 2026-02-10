def policy(obs):
    """
    Infer-and-act policy:

    1) Locate the unique 'agent' and 'star' tokens.
    2) If the agent is below the star, build a filled right-triangle "ramp" of 'drawn' cells that starts
       directly adjacent to the star (row = star_row+1, col = star_col) and extends for H columns where
       H = agent_row - star_row. The triangle is oriented to the side opposite the agent:
         - if agent_col < star_col: ramp extends to decreasing columns (star_col, star_col-1, ...)
         - if agent_col > star_col: ramp extends to increasing columns (star_col, star_col+1, ...)
       For ramp column i (0-indexed), the highest required filled cell is at:
         (star_row + 1 + i, star_col + dir*i)
       Click that cell repeatedly until it becomes 'drawn'. Proceed to the next i.
    3) Once the entire ramp is complete (all those highest-required cells are 'drawn'), click the arrow
       control that moves the agent horizontally toward the star:
         - click the 'right_arrow' cell if agent_col < star_col
         - click the 'left_arrow' cell if agent_col > star_col
       Continue doing so until termination.
    4) Fallbacks: if required tokens are missing, click any arrow if present, else any empty cell, else (0,0).
    """
    nrows = len(obs)
    ncols = len(obs[0]) if nrows else 0

    def s(x):
        try:
            return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
        except Exception:
            return str(x)

    EMPTY = {".", "empty"}
    AGENT = {"A", "agent"}
    STAR = {"*", "star"}
    DRAWN = {"#", "drawn"}
    LARR = {"<", "left_arrow"}
    RARR = {">", "right_arrow"}

    def find_first(token_set):
        for r in range(nrows):
            row = obs[r]
            for c in range(ncols):
                if s(row[c]) in token_set:
                    return (r, c)
        return None

    def is_drawn(r, c):
        return s(obs[r][c]) in DRAWN

    agent_pos = find_first(AGENT)
    star_pos = find_first(STAR)

    left_arrow_pos = find_first(LARR)
    right_arrow_pos = find_first(RARR)

    # Hard fallbacks if key tokens missing
    if agent_pos is None or star_pos is None:
        if right_arrow_pos is not None:
            return right_arrow_pos
        if left_arrow_pos is not None:
            return left_arrow_pos
        # any empty
        for r in range(nrows):
            for c in range(ncols):
                if s(obs[r][c]) in EMPTY:
                    return (r, c)
        return (0, 0)

    ra, ca = agent_pos
    rs, cs = star_pos

    # Decide which control direction would move agent toward the star horizontally
    if ca < cs:
        move_dir = 1
        move_arrow = right_arrow_pos
    elif ca > cs:
        move_dir = -1
        move_arrow = left_arrow_pos
    else:
        move_dir = 1
        move_arrow = right_arrow_pos if right_arrow_pos is not None else left_arrow_pos

    # Build ramp if agent is below star
    H = ra - rs
    if H > 0:
        # Use ramp direction based on agent being on one side of star
        ramp_dir = 1 if ca > cs else -1 if ca < cs else 1

        # Find first ramp "top cell" that is not yet drawn; click it.
        for i in range(H):
            tr = rs + 1 + i
            tc = cs + ramp_dir * i
            if not (0 <= tr < nrows and 0 <= tc < ncols):
                break
            if not is_drawn(tr, tc):
                return (tr, tc)

    # Ramp is complete (or not needed): move using arrow
    if move_arrow is not None:
        return move_arrow

    # If arrow missing, click an empty cell close to agent-star line as a generic fallback
    # (prefer any empty cell, otherwise any cell).
    for r in range(nrows):
        for c in range(ncols):
            if s(obs[r][c]) in EMPTY:
                return (r, c)
    return (0, 0)
