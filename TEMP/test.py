# stop_the_fall_feature_debug.py

from typing import Any, Iterable, List

from programmatic_policy_learning.approaches.experts.grid_experts import (
    get_grid_expert,
)
from programmatic_policy_learning.dsl.llm_primitives.hint_generation.llm_based.hint_extractor import (
    collect_full_episode,
    env_factory,
)


# ---------------------------------------------------------------------
# Minimal "stf" namespace + token normalization
# ---------------------------------------------------------------------
class stf:
    # Canonical tokens (your note)
    EMPTY = "empty"
    FALLING = "falling_token"
    RED = "red_token"
    STATIC = "static_token"
    ADVANCE = "advance_token"
    DRAWN = "drawn_token"


CHAR_TO_CANON = {
    ".": stf.EMPTY,
    "F": stf.FALLING,
    "R": stf.RED,
    "S": stf.STATIC,
    "A": stf.ADVANCE,
    "D": stf.DRAWN,
    # If your grid already contains canonical strings, we'll leave them as-is.
}


def normalize_grid(s: List[List[Any]]) -> List[List[str]]:
    """Convert a grid of chars ('.','F',...) OR already-canonical strings into
    canonical strings."""
    out: List[List[str]] = []
    for row in s:
        out_row: List[str] = []
        for x in row:
            if isinstance(x, str) and x in CHAR_TO_CANON:
                out_row.append(CHAR_TO_CANON[x])
            else:
                # assume it's already canonical like "empty", "red_token", etc.
                out_row.append(str(x))
        out.append(out_row)
    return out


def f2(s, a):
    if not (isinstance(a, (tuple, list)) and len(a) == 2):
        return False
    try:
        r = int(a[0])
        c = int(a[1])
    except:
        return False
    h = len(s)
    w = len(s[0]) if h else 0
    if not (0 <= r < h and 0 <= c < w):
        return False
    return s[r][c] == stf.ADVANCE


def f20(s, a):
    if not (isinstance(a, (tuple, list)) and len(a) == 2):
        return False
    try:
        r = int(a[0])
        c = int(a[1])
    except:
        return False
    h = len(s)
    w = len(s[0]) if h else 0
    if not (0 <= r < h and 0 <= c < w):
        return False
    return r < h - 1 and s[r + 1][c] == stf.DRAWN


def f23(s, a):
    if not (isinstance(a, (tuple, list)) and len(a) == 2):
        return False
    try:
        r = int(a[0])
        c = int(a[1])
    except:
        return False
    h = len(s)
    w = len(s[0]) if h else 0
    if not (0 <= r < h and 0 <= c < w):
        return False
    return r > 0 and s[r - 1][c] == stf.RED


def f25(s, a):
    if not (isinstance(a, (tuple, list)) and len(a) == 2):
        return False
    try:
        r = int(a[0])
        c = int(a[1])
    except:
        return False
    h = len(s)
    w = len(s[0]) if h else 0
    if not (0 <= r < h and 0 <= c < w):
        return False
    return c > 0 and s[r][c - 1] == stf.RED


def f26(s, a):
    if not (isinstance(a, (tuple, list)) and len(a) == 2):
        return False
    try:
        r = int(a[0])
        c = int(a[1])
    except:
        return False
    h = len(s)
    w = len(s[0]) if h else 0
    if not (0 <= r < h and 0 <= c < w):
        return False
    return c < w - 1 and s[r][c + 1] == stf.RED


def f41(s, a):
    if not (isinstance(a, (tuple, list)) and len(a) == 2):
        return False
    try:
        r = int(a[0])
        c = int(a[1])
    except:
        return False
    h = len(s)
    w = len(s[0]) if h else 0
    if not (0 <= r < h and 0 <= c < w):
        return False
    return any(stf.DRAWN in row for row in s)


def f44(s, a):
    if not (isinstance(a, (tuple, list)) and len(a) == 2):
        return False
    try:
        r = int(a[0])
        c = int(a[1])
    except:
        return False
    h = len(s)
    w = len(s[0]) if h else 0
    if not (0 <= r < h and 0 <= c < w):
        return False
    return s[r][c] == stf.EMPTY and r < h - 1 and s[r + 1][c] != stf.EMPTY


def f46(s, a):
    if not (isinstance(a, (tuple, list)) and len(a) == 2):
        return False
    try:
        r = int(a[0])
        c = int(a[1])
    except:
        return False
    h = len(s)
    w = len(s[0]) if h else 0
    if not (0 <= r < h and 0 <= c < w):
        return False
    return s[r][c] == stf.EMPTY and r > 0 and s[r - 1][c] == stf.EMPTY


def f56(s, a):
    if not (isinstance(a, (tuple, list)) and len(a) == 2):
        return False
    try:
        r = int(a[0])
        c = int(a[1])
    except:
        return False
    h = len(s)
    w = len(s[0]) if h else 0
    if not (0 <= r < h and 0 <= c < w):
        return False
    for rr in range(r - 1, -1, -1):
        if s[rr][c] == stf.FALLING:
            return True
        if s[rr][c] != stf.EMPTY:
            break
    return False


FEATURES: list[tuple[str, callable]] = [
    ("f2", f2),
    ("f20", f20),
    ("f23", f23),
    ("f25", f25),
    ("f26", f26),
    ("f41", f41),
    ("f44", f44),
    ("f46", f46),
    ("f56", f56),
]


# removedf23
def plp(s, a):
    return (
        (f20(s, a) and f46(s, a) and f25(s, a))
        or (f20(s, a) and f46(s, a) and (not f25(s, a)) and f26(s, a))
        or ((not f20(s, a)) and f2(s, a) and f41(s, a))
        or ((not f20(s, a)) and (not f2(s, a)) and f56(s, a) and f44(s, a))
    )


def _iter_feature_truths(s: List[List[str]], a: tuple[int, int]) -> Iterable[str]:
    for name, fn in FEATURES:
        yield f"{name}: {fn(s, a)}"


def _print_feature_checks(
    sample_steps: list[tuple[List[List[str]], tuple[int, int]]],
) -> None:
    feature_map = {name: fn for name, fn in FEATURES}
    target_names = ["f2", "f20", "f25", "f26", "f41", "f44", "f46", "f56"]

    for name in target_names:
        fn = feature_map.get(name)
        print(f"Feature {name} present={fn is not None} callable={callable(fn)}")
    print(len(sample_steps))
    for name in target_names:
        fn = feature_map.get(name)
        if not callable(fn):
            continue
        vals = []
        for s, a in sample_steps:
            try:
                vals.append(bool(fn(s, a)))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                vals.append(f"ERR:{type(exc).__name__}")
        print(f"Feature {name} sample outputs={vals}")


# ---------------------------------------------------------------------
# Main: collect 10 demos and evaluate features for each (s, a)
# ---------------------------------------------------------------------
def main():
    env_name = "StopTheFall"
    num_demos = 20
    max_steps = 200

    expert = get_grid_expert(env_name)

    for demo_idx in range(num_demos):
        env = env_factory(demo_idx, env_name)
        traj = collect_full_episode(env, expert, max_steps=max_steps, sample_count=None)
        env.close()

        if demo_idx == 0:
            sample_steps = [
                (normalize_grid(obs), tuple(action)) for obs, action, _ in traj[:5]
            ]
            _print_feature_checks(sample_steps)

        for step_idx, (obs, action, _obs_next) in enumerate(traj):
            s = normalize_grid(obs)
            a = tuple(action)
            token = (
                s[a[0]][a[1]]
                if 0 <= a[0] < len(s) and 0 <= a[1] < len(s[0])
                else "out_of_bounds"
            )
            # print(plp(s, a))

            # print(f"\n--- Demo {demo_idx} Step {step_idx}: action={a}, token={token} ---")
            # for line in _iter_feature_truths(s, a):
            #     print(line)
        # print("NEXT")

    print("\nDone.")


if __name__ == "__main__":
    main()
