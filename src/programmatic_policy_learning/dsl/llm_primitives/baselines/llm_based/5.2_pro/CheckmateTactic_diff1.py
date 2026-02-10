def policy(obs):
    """Policy inferred from demonstrations (K+Q vs K endgame interface with
    two-click moves):

    - If no white queen is currently selected, click the cell containing the (unhighlighted) white_queen.
      (If the queen is already highlighted, treat it as selected.)

    - If the white queen is selected (highlighted), click a destination cell for the queen with priority:
        1) If white_king and black_king are exactly two squares apart on the same row, column, or diagonal,
           and the middle cell is empty and reachable by a legal queen move (clear path), click that middle cell.
        2) Otherwise, among legal queen moves to empty cells, prefer a cell that is adjacent (king-neighborhood)
           to both kings (so it is protected by the white king while directly constraining the black king).
        3) Otherwise, prefer a legal move to an empty cell adjacent to the black king.
        4) Otherwise, choose any legal queen move to an empty cell that minimizes distance to the black king.

    - If no legal queen move exists (or pieces missing), fall back to clicking (0, 0).
    """
    # Support both object-identifiers and ASCII-like renderings.
    EMPTY = {"empty", "."}
    WQ = {"white_queen", "Q"}
    WQH = {"highlighted_white_queen", "Q*"}
    WK = {"white_king", "k"}
    WKH = {"highlighted_white_king", "k*"}
    BK = {"black_king", "K"}

    nrows, ncols = obs.shape

    def find_first(typeset):
        for r in range(nrows):
            for c in range(ncols):
                if obs[r, c] in typeset:
                    return (r, c)
        return None

    def is_empty(rc):
        r, c = rc
        return obs[r, c] in EMPTY

    def cheb(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def queen_legal_moves_from(qr, qc):
        moves = []
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        for dr, dc in directions:
            r, c = qr + dr, qc + dc
            while 0 <= r < nrows and 0 <= c < ncols:
                cell = obs[r, c]
                if cell in EMPTY:
                    moves.append((r, c))
                else:
                    # Blocked by any piece (including kings). Do not allow capturing the king.
                    break
                r += dr
                c += dc
        return moves

    def legal_queen_move(qpos, dest):
        qr, qc = qpos
        dr = dest[0] - qr
        dc = dest[1] - qc
        if dr == 0 and dc == 0:
            return False
        # Determine if aligned
        if dr == 0:
            step_r, step_c = 0, 1 if dc > 0 else -1
        elif dc == 0:
            step_r, step_c = 1 if dr > 0 else -1, 0
        elif abs(dr) == abs(dc):
            step_r = 1 if dr > 0 else -1
            step_c = 1 if dc > 0 else -1
        else:
            return False
        # Path clear (excluding dest)
        r, c = qr + step_r, qc + step_c
        while (r, c) != dest:
            if not (0 <= r < nrows and 0 <= c < ncols):
                return False
            if obs[r, c] not in EMPTY:
                return False
            r += step_r
            c += step_c
        # Destination must be empty
        return is_empty(dest)

    wq_pos = find_first(WQH)  # selected queen
    selected = wq_pos is not None
    if not selected:
        wq_pos = find_first(WQ)
        if wq_pos is not None:
            return wq_pos
        # If only highlighted queen exists, click it (already selected)
        wq_pos = find_first(WQH)
        if wq_pos is not None:
            return wq_pos
        return (0, 0)

    wk_pos = find_first(WKH) or find_first(WK)
    bk_pos = find_first(BK)

    # Compute legal moves (empty only)
    q_moves = queen_legal_moves_from(wq_pos[0], wq_pos[1])

    # Priority 1: middle square between kings if exactly 2 apart on row/col/diag
    if wk_pos is not None and bk_pos is not None:
        dr = bk_pos[0] - wk_pos[0]
        dc = bk_pos[1] - wk_pos[1]
        if (
            (dr == 0 and abs(dc) == 2)
            or (dc == 0 and abs(dr) == 2)
            or (abs(dr) == 2 and abs(dc) == 2)
        ):
            mid = (wk_pos[0] + dr // 2, wk_pos[1] + dc // 2)
            if is_empty(mid) and legal_queen_move(wq_pos, mid):
                return mid

    # Helper to pick deterministically from candidates
    def pick_best(cands):
        # Sort by (dist to black king, dist to queen, row, col)
        if bk_pos is None:
            cands_sorted = sorted(cands, key=lambda p: (cheb(p, wq_pos), p[0], p[1]))
        else:
            cands_sorted = sorted(
                cands, key=lambda p: (cheb(p, bk_pos), cheb(p, wq_pos), p[0], p[1])
            )
        return cands_sorted[0] if cands_sorted else None

    # Priority 2: adjacent to both kings (protected + constraining)
    if wk_pos is not None and bk_pos is not None:
        cands = []
        for m in q_moves:
            if cheb(m, wk_pos) == 1 and cheb(m, bk_pos) == 1:
                cands.append(m)
        best = pick_best(cands)
        if best is not None:
            return best

    # Priority 3: adjacent to black king
    if bk_pos is not None:
        cands = [m for m in q_moves if cheb(m, bk_pos) == 1]
        best = pick_best(cands)
        if best is not None:
            return best

    # Priority 4: any legal move minimizing distance to black king
    best = pick_best(q_moves)
    if best is not None:
        return best

    return (0, 0)
