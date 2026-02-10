def policy(obs):
    """
    Policy:
    1) Locate the 'agent' and the 'star'.
    2) Choose a movement control based on the star's column relative to the agent:
       - If star_col >= agent_col: use the 'right_arrow'
       - Else: use the 'left_arrow'
    3) Before using the arrow, build a diagonal "staircase" of 'drawn' cells (#) beneath the star:
       Let delta = agent_row - star_row. For i = 1..delta, define a target cell:
         target_row = star_row + i
         target_col = star_col - (i-1)  if using 'right_arrow'
                      star_col + (i-1)  if using 'left_arrow'
       Repeatedly click the earliest target (smallest i) that is not yet 'drawn'.
    4) Once all such targets are 'drawn' (or if delta <= 0), click the chosen arrow cell.
    The function always returns some valid (row, col) within bounds.
    """
    nrows = len(obs)
    ncols = len(obs[0]) if nrows else 0

    def is_tok(x, names):
        return x in names

    AGENT = {"agent", "A"}
    STAR = {"star", "*"}
    DRAWN = {"drawn", "#"}
    LARR = {"left_arrow", "<"}
    RARR = {"right_arrow", ">"}

    ar = ac = None
    sr = sc = None
    left_pos = None
    right_pos = None

    for r in range(nrows):
        row = obs[r]
        for c in range(ncols):
            v = row[c]
            if ar is None and is_tok(v, AGENT):
                ar, ac = r, c
            if sr is None and is_tok(v, STAR):
                sr, sc = r, c
            if left_pos is None and is_tok(v, LARR):
                left_pos = (r, c)
            if right_pos is None and is_tok(v, RARR):
                right_pos = (r, c)

    if ar is None or ac is None:
        if right_pos is not None:
            return right_pos
        if left_pos is not None:
            return left_pos
        return (0, 0) if nrows and ncols else (0, 0)

    if sr is None or sc is None:
        if right_pos is not None:
            return right_pos
        if left_pos is not None:
            return left_pos
        return (ar, ac)

    use_right = sc >= ac
    arrow_pos = right_pos if use_right else left_pos
    if arrow_pos is None:
        arrow_pos = right_pos if right_pos is not None else left_pos

    delta = ar - sr
    if delta > 0:
        for i in range(1, delta + 1):
            tr = sr + i
            tc = sc - (i - 1) if use_right else sc + (i - 1)
            if 0 <= tr < nrows and 0 <= tc < ncols:
                if not is_tok(obs[tr][tc], DRAWN):
                    return (tr, tc)

    if arrow_pos is not None:
        return arrow_pos

    return (ar, ac)
