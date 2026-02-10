def policy(obs):
    """
    Strategy:
    1) If the target is not in a corner cell (adjacent to two perpendicular walls), click any wall cell
       (prefer (0,0) when it is a wall) to advance the target.
    2) Once the target is in a corner, if there is no drawn cell adjacent to the target, click an
       adjacent empty cell (prefer same-row neighbor with smaller column, else larger column, then
       vertical neighbors) to create a drawn marker.
    3) Otherwise, move the agent toward the target along a shortest path through non-wall cells by
       clicking the appropriate arrow control cell (found by scanning the grid).
    """
    h, w = obs.shape

    def to_str(x):
        try:
            return x.item() if hasattr(x, "item") else x
        except Exception:
            return x

    def norm(x):
        x = to_str(x)
        if x == "#":
            return "w"
        if x == ".":
            return "e"
        if x == "A":
            return "a"
        if x == "T":
            return "t"
        if x == "*":
            return "s"
        if x == "<":
            return ("arr", "l")
        if x == ">":
            return ("arr", "r")
        if x == "^":
            return ("arr", "u")
        if x == "v":
            return ("arr", "d")

        if isinstance(x, str):
            if x == "wall":
                return "w"
            if x == "empty":
                return "e"
            if x == "agent":
                return "a"
            if x == "target":
                return "t"
            if x == "drawn":
                return "s"
            if x.endswith("arrow") and len(x) > 0:
                ch = x[0].lower()
                if ch in ("l", "r", "u", "d"):
                    return ("arr", ch)
        return x

    ag = None
    tg = None
    wall0 = None
    arr_pos = {}
    s_cells = []

    for r in range(h):
        for c in range(w):
            v = norm(obs[r, c])
            if v == "a":
                ag = (r, c)
            elif v == "t":
                tg = (r, c)
            elif v == "w" and wall0 is None:
                wall0 = (r, c)
            elif isinstance(v, tuple) and v[0] == "arr":
                arr_pos[v[1]] = (r, c)
            elif v == "s":
                s_cells.append((r, c))

    if wall0 is None:
        wall0 = (0, 0)

    if norm(obs[0, 0]) == "w":
        wall_click = (0, 0)
    else:
        wall_click = wall0

    if ag is None or tg is None:
        return wall_click

    def is_w(rr, cc):
        if rr < 0 or rr >= h or cc < 0 or cc >= w:
            return True
        return norm(obs[rr, cc]) == "w"

    tr, tc = tg
    corner = (
        (is_w(tr - 1, tc) and is_w(tr, tc - 1))
        or (is_w(tr - 1, tc) and is_w(tr, tc + 1))
        or (is_w(tr + 1, tc) and is_w(tr, tc - 1))
        or (is_w(tr + 1, tc) and is_w(tr, tc + 1))
    )

    if not corner:
        return wall_click

    def adj_cells(rr, cc):
        return [(rr, cc - 1), (rr, cc + 1), (rr - 1, cc), (rr + 1, cc)]

    has_adj_s = False
    for rr, cc in adj_cells(tr, tc):
        if 0 <= rr < h and 0 <= cc < w and norm(obs[rr, cc]) == "s":
            has_adj_s = True
            break

    if not has_adj_s:
        for rr, cc in adj_cells(tr, tc):
            if 0 <= rr < h and 0 <= cc < w and norm(obs[rr, cc]) == "e":
                return (rr, cc)
        return wall_click

    if ag == tg:
        return wall_click

    def passable(rr, cc):
        v = norm(obs[rr, cc])
        if v == "w":
            return False
        if isinstance(v, tuple) and v[0] == "arr":
            return False
        return True

    # BFS for a shortest path
    q = [ag]
    head = 0
    par = {ag: None}
    found = False
    moves = [(-1, 0, "u"), (1, 0, "d"), (0, -1, "l"), (0, 1, "r")]

    while head < len(q):
        cr, cc = q[head]
        head += 1
        if (cr, cc) == tg:
            found = True
            break
        for dr, dc, code in moves:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in par and passable(nr, nc):
                par[(nr, nc)] = (cr, cc)
                q.append((nr, nc))

    step = None
    if found:
        cur = tg
        while par.get(cur) is not None and par[cur] != ag:
            cur = par[cur]
        if par.get(cur) == ag:
            step = cur

    # Fallback: greedy preference on columns then rows
    if step is None:
        ar, ac = ag
        best = None
        for dr, dc, code in moves:
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < h and 0 <= nc < w and passable(nr, nc):
                best = (nr, nc)
                break
        step = best if best is not None else ag

    if step is None or step == ag:
        return wall_click

    dr = step[0] - ag[0]
    dc = step[1] - ag[1]
    if dr == -1 and "u" in arr_pos:
        return arr_pos["u"]
    if dr == 1 and "d" in arr_pos:
        return arr_pos["d"]
    if dc == -1 and "l" in arr_pos:
        return arr_pos["l"]
    if dc == 1 and "r" in arr_pos:
        return arr_pos["r"]

    # If the intended control is missing, click any available control; else wall.
    for k in ("r", "l", "u", "d"):
        if k in arr_pos:
            return arr_pos[k]
    return wall_click
