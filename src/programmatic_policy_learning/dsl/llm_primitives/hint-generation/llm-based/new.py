"""Hint-extraction pipeline from demonstrations only.

Input:
- demonstrations: list of episodes
- each episode: list of transitions (s_t, a_t, s_tp1)
  - s_t, s_tp1: 2D numpy arrays of ints (grid)
  - a_t: tuple[int,int] clicked cell (row, col)

Output:
- role discovery (object-agnostic)
- control classes (action-causes-effect groupings)
- ranked decision cues (relational predicates)
- a compact JSON-able "hint" dict you can feed to another LLM

This code does NOT use any prior DSL primitives or any object names.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.CaP_baseline import *

Coord = Tuple[int, int]
Grid = np.ndarray  # shape (H,W), dtype int


# ---------------------------
# Utilities
# ---------------------------


def in_bounds(rc: Coord, h: int, w: int) -> bool:
    r, c = rc
    return 0 <= r < h and 0 <= c < w


def neighbors8() -> List[Coord]:
    return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def argwhere_one(grid: Grid, value: int) -> Optional[Coord]:
    """Return coord if exactly one occurrence, else None."""
    coords = np.argwhere(grid == value)
    if coords.shape[0] == 1:
        r, c = coords[0]
        return (int(r), int(c))
    return None


def unique_values(grid: Grid) -> np.ndarray:
    return np.unique(grid)


def diff_cells(s: Grid, sp: Grid) -> np.ndarray:
    """Return array of coords where s != sp, shape (K,2)."""
    return np.argwhere(s != sp)


# ---------------------------
# Data structures
# ---------------------------


@dataclass(frozen=True)
class Transition:
    s: Grid
    a: Coord
    sp: Grid


@dataclass
class ValueStats:
    value: int
    appear_count: int = 0  # total cell occurrences across all states
    state_count: int = 0  # number of states where value appears at least once
    unique_in_state_count: int = 0  # number of states where it appears exactly once
    changed_cell_count: int = 0  # number of changed cells that had this value in s
    changed_to_count: int = 0  # number of changed cells that became this value in sp
    clicked_count: int = 0  # number of times action clicked on this value (in s)
    click_caused_change_count: int = (
        0  # number of clicks on this value that caused any grid change
    )
    click_caused_agent_move_count: int = (
        0  # clicks on this value where inferred agent moved
    )
    inferred_moves_as_entity_count: int = (
        0  # value observed "moving" (disappear+appear nearby)
    )
    border_fraction_sum: float = (
        0.0  # fraction of occurrences on border (sum over states)
    )
    border_fraction_n: int = 0  # number of states contributing
    # displacement signature when clicked: multiset of inferred agent displacements
    click_agent_displacements: Counter = None

    def __post_init__(self):
        if self.click_agent_displacements is None:
            self.click_agent_displacements = Counter()


@dataclass(frozen=True)
class ControlClass:
    control_id: str
    members: List[int]  # raw values v that behave similarly when clicked
    effect_mode: Tuple[
        int, int
    ]  # most common inferred agent displacement (dr, dc) or (0,0)
    success_rate: float  # P(grid changes | clicked this class)
    supports: int  # number of clicks seen


@dataclass
class RoleAssignment:
    value_to_role: Dict[int, str]
    role_descriptions: Dict[str, Dict[str, Any]]


def _grid_to_int_array(
    obs: Any, token_mapping: Optional[Dict[Any, int]] = None
) -> np.ndarray:
    """Convert arbitrary grid tokens to a consistent integer array."""
    arr = np.array(obs, copy=True)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(int, copy=False)
    try:
        return arr.astype(int)
    except (TypeError, ValueError):
        pass
    if token_mapping is None:
        token_mapping = {}
    mapped = np.zeros(arr.shape, dtype=int)
    unique_tokens = np.unique(arr)
    for token in unique_tokens:
        key = token.item() if isinstance(token, np.generic) else token
        if key not in token_mapping:
            token_mapping[key] = len(token_mapping)
        mapped[arr == token] = token_mapping[key]
    return mapped


def trajectory_to_demo(
    trajectory,
    *,
    max_steps: int | None = None,
    token_mapping: Optional[Dict[Any, int]] = None,
) -> List[Transition]:
    """Convert a trajectory (whatever its concrete type is) into a
    list[Transition] for hint extraction.

    This mirrors trajectory_to_text but keeps raw arrays.
    """
    steps = trajectory[:max_steps] if max_steps else trajectory
    demo: List[Transition] = []

    token_mapping = {} if token_mapping is None else token_mapping

    for obs_t, action, obs_t1 in steps:
        s_grid = _grid_to_int_array(obs_t, token_mapping)
        sp_grid = _grid_to_int_array(obs_t1, token_mapping)
        demo.append(
            Transition(
                s=s_grid,
                a=(int(action[0]), int(action[1])),
                sp=sp_grid,
            )
        )

    return demo


# ---------------------------
# Core pipeline
# ---------------------------


class HintExtractor:
    """
    End-to-end pipeline:
      1) compute per-value statistics from demos
      2) infer "agent-like" value and (optional) "goal-like" value
      3) create control classes (values whose click has consistent effect)
      4) induce roles (R0..Rn) from stats + heuristics (no object names)
      5) build relational predicates and rank them contrastively
      6) emit JSON-able hint dict
    """

    def __init__(
        self,
        offsets: Optional[List[Coord]] = None,
        scan_dirs: Optional[List[Coord]] = None,
        max_scan_len: int = 20,
        top_k_cues: int = 25,
        seed: int = 0,
    ):
        self.offsets = (
            offsets
            if offsets is not None
            else neighbors8() + [(-2, 0), (2, 0), (0, -2), (0, 2)]
        )
        self.scan_dirs = (
            scan_dirs if scan_dirs is not None else [(-1, 0), (1, 0), (0, -1), (0, 1)]
        )
        self.max_scan_len = max_scan_len
        self.top_k_cues = top_k_cues
        self.rng = np.random.default_rng(seed)

    # ---------- Stage 1: Stats ----------
    def compute_value_stats(
        self, demos: List[List[Transition]]
    ) -> Dict[int, ValueStats]:
        stats: Dict[int, ValueStats] = {}

        def get(v: int) -> ValueStats:
            if v not in stats:
                stats[v] = ValueStats(value=v)
            return stats[v]

        for ep in demos:
            for tr in ep:
                s, a, sp = tr.s, tr.a, tr.sp
                h, w = s.shape

                # per-state: appearances, uniqueness, border fraction
                vals_in_state = unique_values(s)
                for v in vals_in_state:
                    vs = get(int(v))
                    vs.state_count += 1
                    coords = np.argwhere(s == v)
                    vs.appear_count += int(coords.shape[0])
                    if coords.shape[0] == 1:
                        vs.unique_in_state_count += 1
                    # border fraction in this state
                    if coords.shape[0] > 0:
                        border = 0
                        for rr, cc in coords:
                            rr = int(rr)
                            cc = int(cc)
                            if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                                border += 1
                        vs.border_fraction_sum += border / float(coords.shape[0])
                        vs.border_fraction_n += 1

                # change stats
                diffs = diff_cells(s, sp)
                if diffs.shape[0] > 0:
                    for rr, cc in diffs:
                        rr = int(rr)
                        cc = int(cc)
                        v_old = int(s[rr, cc])
                        v_new = int(sp[rr, cc])
                        get(v_old).changed_cell_count += 1
                        get(v_new).changed_to_count += 1

                # click stats
                ar, ac = a
                v_clicked = int(s[ar, ac])
                get(v_clicked).clicked_count += 1
                if diffs.shape[0] > 0:
                    get(v_clicked).click_caused_change_count += 1

        # init counters properly
        for v, vs in stats.items():
            if vs.click_agent_displacements is None:
                vs.click_agent_displacements = Counter()

        return stats

    # ---------- Stage 2: Agent & goal inference (optional but useful) ----------
    def infer_agent_value(
        self, demos: List[List[Transition]], stats: Dict[int, ValueStats]
    ) -> Optional[int]:
        """
        Heuristic: agent-like value tends to be unique-in-state often AND participates in changes/motion.
        """
        candidates = []
        for v, vs in stats.items():
            if vs.state_count == 0:
                continue
            uniq_rate = vs.unique_in_state_count / max(1, vs.state_count)
            change_rate = vs.changed_cell_count / max(1, vs.appear_count)
            # prefer values that are frequently unique and change
            score = 2.0 * uniq_rate + 1.0 * change_rate
            candidates.append((score, v))
        candidates.sort(reverse=True)
        if not candidates:
            return None
        # return best, but sanity check: must be unique reasonably often
        best_score, best_v = candidates[0]
        best_vs = stats[best_v]
        uniq_rate = best_vs.unique_in_state_count / max(1, best_vs.state_count)
        if uniq_rate < 0.3:
            return None
        return int(best_v)

    def infer_goal_value(
        self,
        demos: List[List[Transition]],
        stats: Dict[int, ValueStats],
        agent_v: Optional[int],
    ) -> Optional[int]:
        """
        Heuristic: goal-like value is often unique, mostly static (rarely changes), and not the agent.
        """
        candidates = []
        for v, vs in stats.items():
            if agent_v is not None and v == agent_v:
                continue
            if vs.state_count == 0:
                continue
            uniq_rate = vs.unique_in_state_count / max(1, vs.state_count)
            # staticness proxy: low changed_cell_count relative to appearances
            change_rate = vs.changed_cell_count / max(1, vs.appear_count)
            score = 2.0 * uniq_rate - 1.5 * change_rate
            candidates.append((score, v))
        candidates.sort(reverse=True)
        if not candidates:
            return None
        best_score, best_v = candidates[0]
        best_vs = stats[best_v]
        uniq_rate = best_vs.unique_in_state_count / max(1, best_vs.state_count)
        change_rate = best_vs.changed_cell_count / max(1, best_vs.appear_count)
        if uniq_rate < 0.3 or change_rate > 0.05:
            return None
        return int(best_v)

    # ---------- Stage 3: Control classes ----------
    def _infer_agent_displacement(
        self, s: Grid, sp: Grid, agent_v: int
    ) -> Optional[Tuple[int, int]]:
        a0 = argwhere_one(s, agent_v)
        a1 = argwhere_one(sp, agent_v)
        if a0 is None or a1 is None:
            return None
        return (a1[0] - a0[0], a1[1] - a0[1])

    def build_control_classes(
        self,
        demos: List[List[Transition]],
        stats: Dict[int, ValueStats],
        agent_v: Optional[int],
        min_clicks: int = 3,
        mode_threshold: float = 0.6,
    ) -> List[ControlClass]:
        """Group raw values by their most common inferred agent displacement
        when clicked.

        If agent is not inferable, control classes are based on "click
        causes any change" only.
        """
        # accumulate per clicked value:
        click_disp = defaultdict(Counter)  # v -> Counter[disp]
        click_success = Counter()  # v -> number of clicks causing any change
        click_total = Counter()  # v -> clicks
        for ep in demos:
            for tr in ep:
                s, a, sp = tr.s, tr.a, tr.sp
                v = int(s[a[0], a[1]])
                click_total[v] += 1
                if np.any(s != sp):
                    click_success[v] += 1
                if agent_v is not None:
                    disp = self._infer_agent_displacement(s, sp, agent_v)
                    if disp is not None:
                        click_disp[v][disp] += 1

        # assign each v to a signature
        sig_to_values: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        sig_success: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        sig_supports: Dict[Tuple[int, int], int] = defaultdict(int)

        for v, tot in click_total.items():
            if tot < min_clicks:
                continue

            succ_rate = click_success[v] / max(1, tot)

            if agent_v is None or len(click_disp[v]) == 0:
                # no displacement info; treat as (0,0)
                sig = (0, 0)
                sig_to_values[sig].append(v)
                sig_success[sig].append(succ_rate)
                sig_supports[sig] += tot
                continue

            disp_counter = click_disp[v]
            disp_mode, disp_mode_ct = disp_counter.most_common(1)[0]
            mode_rate = disp_mode_ct / max(1, sum(disp_counter.values()))
            if mode_rate < mode_threshold:
                # too ambiguous; bucket into (0,0)
                sig = (0, 0)
            else:
                sig = tuple(map(int, disp_mode))

            sig_to_values[sig].append(v)
            sig_success[sig].append(succ_rate)
            sig_supports[sig] += tot

            # update stats object too
            stats[v].click_agent_displacements.update(disp_counter)

        controls: List[ControlClass] = []
        idx = 0
        for sig, members in sorted(
            sig_to_values.items(), key=lambda kv: (-len(kv[1]), kv[0])
        ):
            avg_succ = float(np.mean(sig_success[sig])) if sig_success[sig] else 0.0
            controls.append(
                ControlClass(
                    control_id=f"C{idx}",
                    members=sorted(members),
                    effect_mode=(int(sig[0]), int(sig[1])),
                    success_rate=avg_succ,
                    supports=int(sig_supports[sig]),
                )
            )
            idx += 1
        return controls

    # ---------- Stage 4: Role induction ----------
    def induce_roles(
        self,
        stats: Dict[int, ValueStats],
        agent_v: Optional[int],
        goal_v: Optional[int],
        controls: List[ControlClass],
    ) -> RoleAssignment:
        """
        Assign each raw value v to an abstract role label:
          - R_AGENT / R_GOAL (if inferred)
          - R_CONTROL_k (if in control class with nontrivial effect or high click-success)
          - R_BLOCKER-like (static + border-ish + rarely clicked)
          - R_BACKGROUND-like (very frequent + rarely clicked + mostly static)
          - R_DYNAMIC_OTHER (changes often but not agent)
          - R_OTHER
        """
        value_to_role: Dict[int, str] = {}

        control_members = {}
        for cc in controls:
            for v in cc.members:
                control_members[v] = cc.control_id

        # helper measures
        def uniq_rate(vs: ValueStats) -> float:
            return vs.unique_in_state_count / max(1, vs.state_count)

        def change_rate(vs: ValueStats) -> float:
            return vs.changed_cell_count / max(1, vs.appear_count)

        def click_rate(vs: ValueStats) -> float:
            return vs.clicked_count / max(1, vs.state_count)

        def border_rate(vs: ValueStats) -> float:
            if vs.border_fraction_n == 0:
                return 0.0
            return vs.border_fraction_sum / vs.border_fraction_n

        # assign special inferred roles
        if agent_v is not None:
            value_to_role[agent_v] = "R_AGENT"
        if goal_v is not None:
            value_to_role[goal_v] = "R_GOAL"

        # assign control roles
        for v, cid in control_members.items():
            if v in value_to_role:
                continue
            value_to_role[v] = f"R_{cid}"

        # assign rest
        for v, vs in stats.items():
            if v in value_to_role:
                continue
            ur = uniq_rate(vs)
            cr = change_rate(vs)
            br = border_rate(vs)
            clk = click_rate(vs)
            # heuristic buckets
            if vs.appear_count > 0 and br > 0.6 and clk < 0.05 and cr < 0.01:
                value_to_role[v] = "R_BORDER_STATIC"
            elif vs.appear_count > 0 and clk < 0.02 and cr < 0.005 and ur < 0.05:
                value_to_role[v] = "R_BACKGROUND"
            elif cr > 0.02 and ur < 0.2:
                value_to_role[v] = "R_DYNAMIC_MANY"
            elif ur > 0.5 and cr < 0.02:
                value_to_role[v] = "R_UNIQUE_STATIC_OTHER"
            else:
                value_to_role[v] = "R_OTHER"

        # role descriptions
        role_to_values: Dict[str, List[int]] = defaultdict(list)
        for v, r in value_to_role.items():
            role_to_values[r].append(int(v))

        role_descriptions: Dict[str, Dict[str, Any]] = {}
        for role, vs_list in role_to_values.items():
            agg = {
                "values": sorted(vs_list),
                "count_values": len(vs_list),
            }
            # aggregate a few stats
            if vs_list:
                appear = sum(stats[v].appear_count for v in vs_list)
                clicked = sum(stats[v].clicked_count for v in vs_list)
                changed = sum(stats[v].changed_cell_count for v in vs_list)
                states = sum(stats[v].state_count for v in vs_list)
                uniq_states = sum(stats[v].unique_in_state_count for v in vs_list)
                agg.update(
                    {
                        "appear_total": int(appear),
                        "clicked_total": int(clicked),
                        "changed_total": int(changed),
                        "unique_state_rate": float(uniq_states / max(1, states)),
                        "change_rate": float(changed / max(1, appear)),
                    }
                )
            role_descriptions[role] = agg

        return RoleAssignment(
            value_to_role=value_to_role, role_descriptions=role_descriptions
        )

    # ---------- Stage 5: Relational predicates + ranking ----------
    def _role_at(self, grid: Grid, rc: Coord, value_to_role: Dict[int, str]) -> str:
        return value_to_role.get(int(grid[rc[0], rc[1]]), "R_UNKNOWN")

    def _find_unique_role_pos(
        self, grid: Grid, value_to_role: Dict[int, str], role: str
    ) -> Optional[Coord]:
        """If role corresponds to exactly one cell, return it."""
        # brute force since grids are small
        coords = []
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if self._role_at(grid, (r, c), value_to_role) == role:
                    coords.append((r, c))
                    if len(coords) > 1:
                        return None
        return coords[0] if len(coords) == 1 else None

    def _scan_sees_role_before(
        self,
        grid: Grid,
        start: Coord,
        direction: Coord,
        value_to_role: Dict[int, str],
        see_role: str,
        stop_role: Optional[str],
    ) -> bool:
        h, w = grid.shape
        r, c = start
        dr, dc = direction
        for _ in range(self.max_scan_len):
            r += dr
            c += dc
            if not in_bounds((r, c), h, w):
                return False
            role_here = self._role_at(grid, (r, c), value_to_role)
            if stop_role is not None and role_here == stop_role:
                return False
            if role_here == see_role:
                return True
        return False

    def _action_candidates(self, grid: Grid) -> List[Coord]:
        """Default: any cell is clickable. Override if you have env click mask."""
        h, w = grid.shape
        return [(r, c) for r in range(h) for c in range(w)]

    def _predicate_features(
        self,
        grid: Grid,
        action: Coord,
        value_to_role: Dict[int, str],
        agent_pos: Optional[Coord],
        goal_pos: Optional[Coord],
        stop_role_for_scan: Optional[str] = "R_BORDER_STATIC",
    ) -> Dict[str, int]:
        """Produce boolean predicate features as {feature_name: 0/1}.

        Keep these "close" to what a DSL might need, but not using your
        old primitives.
        """
        h, w = grid.shape
        feats: Dict[str, int] = {}

        # role of action cell
        a_role = self._role_at(grid, action, value_to_role)
        feats[f"ACT_ROLE=={a_role}"] = 1

        # neighbor roles around action
        ar, ac = action
        for dr, dc in self.offsets:
            nr, nc = ar + dr, ac + dc
            if not in_bounds((nr, nc), h, w):
                continue
            r = self._role_at(grid, (nr, nc), value_to_role)
            feats[f"NBR({dr},{dc})=={r}"] = 1

        # relative offsets agent/action and goal/action (only if unique positions known)
        if agent_pos is not None:
            feats[f"AGENT_OFFSET_TO_ACT({ar-agent_pos[0]},{ac-agent_pos[1]})"] = 1
        if goal_pos is not None:
            feats[f"GOAL_OFFSET_TO_ACT({ar-goal_pos[0]},{ac-goal_pos[1]})"] = 1

        # scan features: from action cell sees goal before blocker
        if goal_pos is not None:
            for d in self.scan_dirs:
                # treat "see role" as R_GOAL if present
                feats[f"SCAN_ACT_SEES_R_GOAL_DIR({d[0]},{d[1]})"] = int(
                    self._scan_sees_role_before(
                        grid, action, d, value_to_role, "R_GOAL", stop_role_for_scan
                    )
                )

        # scan from agent to see goal (if agent pos known)
        if agent_pos is not None:
            for d in self.scan_dirs:
                feats[f"SCAN_AGENT_SEES_R_GOAL_DIR({d[0]},{d[1]})"] = int(
                    self._scan_sees_role_before(
                        grid, agent_pos, d, value_to_role, "R_GOAL", stop_role_for_scan
                    )
                )

        return feats

    def _effect_label(
        self,
        s: Grid,
        sp: Grid,
        agent_v: Optional[int],
        action: Coord,
        value_to_role: Dict[int, str],
    ) -> Dict[str, Any]:
        diffs = diff_cells(s, sp)
        label = {
            "grid_changed": bool(diffs.shape[0] > 0),
            "changed_cells": int(diffs.shape[0]),
        }
        if agent_v is not None:
            disp = self._infer_agent_displacement(s, sp, agent_v)
            label["agent_disp"] = disp if disp is not None else None
        else:
            label["agent_disp"] = None
        # created/destroyed counts by role (optional)
        return label

    def rank_cues_contrastive(
        self,
        demos: List[List[Transition]],
        value_to_role: Dict[int, str],
        agent_v: Optional[int],
        goal_v: Optional[int],
        max_negatives_per_state: int = 30,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Contrastive scoring:
          For each state:
            positives = chosen action
            negatives = sampled other cells
          Score features by log-odds difference:
            score = log( P(f=1|pos) / P(f=1|neg) )  (with smoothing)
        """
        pos_counts = Counter()
        neg_counts = Counter()
        pos_total = 0
        neg_total = 0

        # figure unique positions for agent/goal using inferred values (mapped to roles)
        # we use roles R_AGENT / R_GOAL if present.
        for ep in demos:
            for tr in ep:
                s, a = tr.s, tr.a

                agent_pos = self._find_unique_role_pos(s, value_to_role, "R_AGENT")
                goal_pos = self._find_unique_role_pos(s, value_to_role, "R_GOAL")

                # positive
                feats_pos = self._predicate_features(
                    s, a, value_to_role, agent_pos, goal_pos
                )
                for k, v in feats_pos.items():
                    if v:
                        pos_counts[k] += 1
                pos_total += 1

                # negatives
                candidates = self._action_candidates(s)
                # remove the chosen action
                candidates = [c for c in candidates if c != a]
                if len(candidates) > max_negatives_per_state:
                    idxs = self.rng.choice(
                        len(candidates), size=max_negatives_per_state, replace=False
                    )
                    candidates = [candidates[i] for i in idxs]

                for na in candidates:
                    feats_neg = self._predicate_features(
                        s, na, value_to_role, agent_pos, goal_pos
                    )
                    for k, v in feats_neg.items():
                        if v:
                            neg_counts[k] += 1
                    neg_total += 1

        # score features
        scored: List[Tuple[str, float, Dict[str, Any]]] = []
        alpha = 1.0  # Laplace smoothing
        for feat in set(list(pos_counts.keys()) + list(neg_counts.keys())):
            p_pos = (pos_counts[feat] + alpha) / (pos_total + 2 * alpha)
            p_neg = (neg_counts[feat] + alpha) / (neg_total + 2 * alpha)
            score = math.log(p_pos / p_neg)
            meta = {
                "pos_rate": float(p_pos),
                "neg_rate": float(p_neg),
                "pos_support": int(pos_counts[feat]),
                "neg_support": int(neg_counts[feat]),
                "pos_total": int(pos_total),
                "neg_total": int(neg_total),
            }
            scored.append((feat, float(score), meta))

        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return scored[: self.top_k_cues]

    # ---------- Stage 6: Emit hint dict ----------
    def extract(self, demos: List[List[Transition]]) -> Dict[str, Any]:
        stats = self.compute_value_stats(demos)
        agent_v = self.infer_agent_value(demos, stats)
        goal_v = self.infer_goal_value(demos, stats, agent_v=agent_v)

        controls = self.build_control_classes(demos, stats, agent_v=agent_v)

        roles = self.induce_roles(
            stats, agent_v=agent_v, goal_v=goal_v, controls=controls
        )

        cues = self.rank_cues_contrastive(
            demos,
            value_to_role=roles.value_to_role,
            agent_v=agent_v,
            goal_v=goal_v,
        )

        # make stats JSON-friendly (Counters -> dict)
        stats_out = {}
        for v, vs in stats.items():
            d = asdict(vs)
            d["click_agent_displacements"] = dict(vs.click_agent_displacements)
            stats_out[int(v)] = d

        hint = {
            "inferred": {
                "agent_value": agent_v,
                "goal_value": goal_v,
                "agent_role": "R_AGENT" if agent_v is not None else None,
                "goal_role": "R_GOAL" if goal_v is not None else None,
            },
            "roles": roles.role_descriptions,  # role -> summary + values
            "value_to_role": roles.value_to_role,  # raw v -> role label
            "controls": [asdict(c) for c in controls],  # control classes
            "top_cues": [
                {"predicate": feat, "score": score, **meta}
                for feat, score, meta in cues
            ],
            "value_stats": stats_out,
        }
        return hint


# ---------------------------
# Example usage
# ---------------------------


def wrap_demo(
    raw_episode: Sequence[Tuple[np.ndarray, Tuple[int, int], np.ndarray]],
) -> List[Transition]:
    ep: List[Transition] = []
    for s, a, sp in raw_episode:
        # Ensure np arrays are copied/contiguous for safety
        s_arr = np.array(s, dtype=int, copy=True)
        sp_arr = np.array(sp, dtype=int, copy=True)
        ep.append(Transition(s=s_arr, a=(int(a[0]), int(a[1])), sp=sp_arr))
    return ep


def summarize_hint_for_llm(hint: dict) -> dict:
    """Reduce the raw hint dict into a compact, LLM-facing summary.

    No raw values, no grids, no DSL assumptions.
    """
    summary = {}

    # --- roles ---
    roles = {}
    for role, info in hint["roles"].items():
        roles[role] = {
            "count_values": info["count_values"],
            "unique_state_rate": round(info["unique_state_rate"], 3),
            "change_rate": round(info["change_rate"], 3),
            "clicked_total": info["clicked_total"],
        }
    summary["roles"] = roles

    # --- controls ---
    controls = []
    for c in hint["controls"]:
        controls.append(
            {
                "control_id": c["control_id"],
                "effect": c["effect_mode"],
                "supports": c["supports"],
            }
        )
    summary["controls"] = controls

    # --- top decision cues (truncate + simplify) ---
    cues = []
    for c in hint["top_cues"][:10]:
        cues.append(
            {
                "predicate": c["predicate"],
                "score": round(c["score"], 3),
            }
        )
    summary["top_cues"] = cues

    # --- inferred anchors ---
    summary["anchors"] = hint["inferred"]

    return summary


def build_dsl_invention_prompt(hint: dict) -> str:
    """
    Build a prompt that asks a second LLM to:
      1) Invent DSL primitives
      2) Propose a solution program
    using ONLY the learned hint information.
    """

    summary = summarize_hint_for_llm(hint)

    prompt = f"""
You are designing a domain-specific language (DSL) for expressing
expert policies in a grid-based decision problem.

You are NOT given object names, grid semantics, or any prior DSL.
You must invent abstractions purely from behavioral evidence.

============================================================
LEARNED ABSTRACT ROLES
============================================================
Each role is an equivalence class of grid values discovered from
demonstrations. The descriptions below summarize their behavior.

{summary["roles"]}

Notes:
- R_AGENT is the unique moving entity.
- R_GOAL is a unique static reference entity.
- Other roles may represent controls, constraints, or environment structure.

============================================================
LEARNED CONTROL EFFECTS
============================================================
Clicking certain cells reliably produces effects on the agent.

Each control class has a consistent effect:

{summary["controls"]}

The effect is given as a relative displacement of the agent (row_delta, col_delta).

============================================================
DECISION CUES USED BY THE EXPERT
============================================================
The following predicates strongly distinguish chosen actions
from non-chosen actions:

{summary["top_cues"]}

Each predicate is a boolean relation over:
- the action cell
- the agent
- the goal
- nearby or aligned roles

============================================================
YOUR TASK
============================================================

1) INVENT DSL PRIMITIVES
------------------------
Propose a minimal set of reusable DSL functions that could express
the expert policy.

Requirements:
- Functions must be generic and reusable.
- Functions should operate over roles, positions, or actions.
- Do NOT hard-code numeric offsets (e.g., (9,4)).
- Prefer relational or comparative abstractions (e.g., distance decreases).

For each function, provide:
- name
- type signature
- one-line semantic description

2) WRITE A POLICY PROGRAM
-------------------------
Using ONLY the primitives you proposed, write a high-level policy
that selects an action.

The policy should:
- choose among candidate action cells
- prefer actions that advance the agent toward the goal
- respect constraints implied by the cues

============================================================
OUTPUT FORMAT (STRICT)
============================================================

PRIMITIVES:
- name: ...
  signature: ...
  description: ...

PROGRAM:
<single policy expression or structured pseudo-code>

Do NOT include explanations outside these sections.
"""

    return prompt


def build_hint_section(hint: dict) -> str:
    """Convert the learned hint dict into an abstract, DSL-safe feature
    inventory to inject into the prompt."""

    lines = []

    # --- Anchors ---
    lines.append("### Entity Structure")
    lines.append("- There exists a unique moving entity (the agent).")
    lines.append("- There exists a unique static reference entity (the goal).")
    lines.append(
        "- Policies reason about the spatial relationship between the agent, the goal, and the chosen action cell."
    )

    # --- Controls ---
    lines.append("\n### Action Effects")
    lines.append(
        "- Certain action cells act as controls that deterministically move the agent."
    )
    lines.append("- Different controls induce different movement directions.")
    lines.append(
        "- Valid actions typically belong to a small subset of control-like cells."
    )

    # --- Spatial & relational cues ---
    lines.append("\n### Spatial Reasoning Patterns")
    lines.append(
        "- The expert prefers actions whose position is favorably aligned with the goal."
    )
    lines.append(
        "- The expert considers the relative offset between the agent and the action cell."
    )
    lines.append("- Local neighborhood structure around the action cell is relevant.")
    lines.append(
        "- Decisions depend on whether nearby cells satisfy certain role-based properties."
    )

    # --- Directional & distance cues ---
    lines.append("\n### Directional / Distance Abstractions")
    lines.append("- Useful abstractions compare distances before and after an action.")
    lines.append(
        "- Reasoning often depends on whether an action reduces the agent–goal distance."
    )
    lines.append(
        "- Directional relations (e.g., ahead, behind, aligned) are important, not exact offsets."
    )

    # --- Constraints ---
    lines.append("\n### Constraints")
    lines.append("- Avoid encoding absolute coordinates or fixed numeric offsets.")
    lines.append("- Prefer relational, comparative, or monotonic properties.")

    return "\n".join(lines)


if __name__ == "__main__":
    # Replace this with your real data loader
    # demos = [wrap_demo(ep0), wrap_demo(ep1), ...]
    # demos: List[List[Transition]] = []

    # # tiny fake demo just to show shape; REMOVE in real use
    # # values are opaque ints
    # s0 = np.array([[9, 9, 9],
    #                [9, 3, 0],
    #                [9, 7, 9]])
    # # click something that moves "agent"(3) right (pretend)
    # a0 = (1, 2)
    # sp0 = np.array([[9, 9, 9],
    #                 [9, 0, 3],
    #                 [9, 7, 9]])
    num_initial_states = 4
    env_name = "Chase"
    encoding_method = "4"
    max_steps_per_traj = 40
    seed = 0
    # _configure_rng(seed)

    expert = get_grid_expert(env_name)
    trajectories: list[list[tuple[Any, Any, Any]]] = []
    for init_idx in range(num_initial_states):
        env = env_factory(init_idx, env_name)
        traj = collect_full_episode(env, expert, sample_count=None)
        env.close()
        trajectories.append(traj)

    symbol_map = grid_hint_config.get_symbol_map(env_name)

    enc_cfg = grid_encoder.GridStateEncoderConfig(
        symbol_map=symbol_map,
        empty_token="empty",
        coordinate_style="rc",
    )
    encoder = grid_encoder.GridStateEncoder(enc_cfg)
    analyzer = transition_analyzer.GenericTransitionAnalyzer()

    all_traj_texts = []
    demos: List[List[Transition]] = []
    token_mapping: Dict[Any, int] = {}

    for i, traj in enumerate(trajectories):

        # 1) text view (unchanged)
        text = trajectory_serializer.trajectory_to_text(
            traj,
            encoder=encoder,
            analyzer=analyzer,
            salient_tokens=grid_hint_config.SALIENT_TOKENS[env_name],
            encoding_method=encoding_method,
            max_steps=max_steps_per_traj,
        )
        all_traj_texts.append(f"\n---[TRAJECTORY {i}]---\n{text}\n\n")

        # 2) behavioral view (NEW)
        demo = trajectory_to_demo(
            traj,
            max_steps=max_steps_per_traj,
            token_mapping=token_mapping,
        )
        demos.append(demo)

    combined_text = "\n\n".join(all_traj_texts)
    print(combined_text)

    extractor = HintExtractor(top_k_cues=20)
    hint = extractor.extract(demos)
    # print(hint)
    new_hint = build_hint_section(hint)
    print(new_hint)
    # dsl_prompt = build_dsl_invention_prompt(hint)
    # cache_path = Path("cache.db")
    # cache_path.parent.mkdir(parents=True, exist_ok=True)
    # cache = SQLite3PretrainedLargeModelCache(cache_path)
    # llm_client = OpenAIModel("gpt-4.1", cache)
    # query = Query(dsl_prompt)
    # reprompt_checks: list[RepromptCheck] = []
    # response = query_with_reprompts(
    #     llm_client,
    #     query,
    #     reprompt_checks=reprompt_checks,
    #     max_attempts=5,
    # )
    # print(response)

    # print(json.dumps(hint, indent=2, sort_keys=True))
