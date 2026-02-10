def policy(obs):
    """
    Inferred policy:

    1) Locate the column containing the falling_token.
    2) In that same column, identify all cells strictly below the falling_token that are:
         - empty, and
         - horizontally adjacent (col-1 or col+1) to at least one red_token in the same row.
       Click these target cells in descending row order (largest row index first), effectively
       building a vertical stack of drawn_token cells aligned with nearby red_token rows.
    3) Once no such empty target cells remain, click the advance_token (to advance the fall).

    Fallbacks to ensure an action is always returned:
    - If no falling_token is present, click an advance_token if present; otherwise click the
      first empty cell found (row-major).
    - If falling_token is present but no target cells exist and no advance_token exists,
      click the first empty cell in the falling_token column below it; otherwise any empty cell.
    """
    nrows, ncols = obs.shape

    # Support either object-type identifiers or their single-character renderings.
    sym = {
        "empty": ".",
        "falling_token": "F",
        "red_token": "R",
        "static_token": "S",
        "advance_token": "A",
        "drawn_token": "D",
    }

    def _norm(v):
        if isinstance(v, bytes):
            try:
                return v.decode("utf-8")
            except Exception:
                return str(v)
        return v

    def is_type(v, tname):
        v = _norm(v)
        return v == tname or v == sym[tname]

    def find_first(tname):
        for r in range(nrows):
            for c in range(ncols):
                if is_type(obs[r, c], tname):
                    return (r, c)
        return None

    fpos = find_first("falling_token")
    apos = find_first("advance_token")

    if fpos is not None:
        fr, fc = fpos

        def has_adjacent_red(r, c):
            if c - 1 >= 0 and is_type(obs[r, c - 1], "red_token"):
                return True
            if c + 1 < ncols and is_type(obs[r, c + 1], "red_token"):
                return True
            return False

        # Click lowest (largest r) eligible empty cell in falling token column.
        for r in range(nrows - 1, fr, -1):
            if is_type(obs[r, fc], "empty") and has_adjacent_red(r, fc):
                return (r, fc)

        if apos is not None:
            return apos

        # Fallback: click any empty cell below the falling token in the same column.
        for r in range(fr + 1, nrows):
            if is_type(obs[r, fc], "empty"):
                return (r, fc)

    # If no falling token (or no other option), prefer advancing.
    if apos is not None:
        return apos

    # Last fallback: click the first empty cell found (row-major), else (0,0).
    for r in range(nrows):
        for c in range(ncols):
            if is_type(obs[r, c], "empty"):
                return (r, c)
    return (0, 0)
