"""Simple heuristics for describing grid transitions."""

from collections import deque
from collections.abc import Callable, Iterable, Mapping

import numpy as np

# pylint: disable=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.grid_encoder import (
    GridStateEncoder,
)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return the Manhattan distance between two cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GenericTransitionAnalyzer:
    """Produce text descriptions of transitions without domain knowledge."""

    def analyze(
        self,
        obs_t: np.ndarray,
        action: tuple[int, int],
        obs_t1: np.ndarray,
        *,
        encoder: "GridStateEncoder",
        agent_token: str = "agent",
        target_token: str | None = "target",
    ) -> list[str]:
        """Return human-readable events summarizing the transition."""
        del encoder  # unused for now but kept for interface symmetry
        events: list[str] = []

        def find(token: str | None, obs: np.ndarray) -> tuple[int, int] | None:
            """Locate the first cell containing the given token."""
            if token is None:
                return None
            locs = np.argwhere(obs == token)
            return tuple(locs[0]) if len(locs) > 0 else None

        a0 = find(agent_token, obs_t)
        a1 = find(agent_token, obs_t1)
        t0 = find(target_token, obs_t)
        t1 = find(target_token, obs_t1)

        if a0 and a1:
            if a0 != a1:
                events.append(f"agent moved from {a0} to {a1}")
            else:
                events.append("agent did not move")

        if t0 and t1 and t0 != t1:
            events.append(f"target moved from {t0} to {t1}")

        if t0 and t1 and a0 and a1:
            d0 = manhattan(a0, t0)
            d1 = manhattan(a1, t1)
            if d1 < d0:
                events.append("agent reduced distance to target")
            elif d1 > d0:
                events.append("agent increased distance to target")
            else:
                events.append("agent kept same distance to target")

        r, c = action
        token_at_action = obs_t[r, c]
        events.append(f"action cell contained '{token_at_action}'")

        return events


ObjectsArg = Mapping[str, Iterable[tuple[int, int]]] | list[tuple[str, tuple[int, int]]]


def _normalize_token(name: str) -> str:
    """Return a normalised version of a token name for map lookups."""

    return name.lower()


def _format_label(name: str, *, capitalize: bool) -> str:
    """Convert a token name into a human-readable label."""

    text = _normalize_token(name).replace("_", " ")
    return text.title() if capitalize else text


def extract_relational_facts(
    grid: np.ndarray,
    objects: ObjectsArg,
    *,
    max_distance: int = 3,
    symbol_map: Mapping[str, str] | None = None,
    reference_tokens: Iterable[str] | None = None,
    empty_tokens: Iterable[str] | None = None,
    cluster_tokens: Iterable[str] | None = None,
) -> list[str]:
    """Return relational facts describing a grid state.

    Args:
        grid: Raw grid array (kept for compatibility; not currently used).
        objects: Either a list of (type, (row, col)) pairs or a dict mapping each
            type to a list of coordinates.
        max_distance: Maximum manhattan distance for directional facts.
        symbol_map: Optional symbol map used to order/whitelist tokens.
        reference_tokens: Optional ordered iterable describing which token types
            should be prioritised as the relational anchor.
        empty_tokens: Tokens that should be ignored for fact generation (defaults
            to {"empty"} if not provided).
        cluster_tokens: Which tokens should be inspected for connected-component
            clustering (defaults to tokens whose name contains "token").
    """

    del grid  # The current implementation relies on the parsed objects only.

    type_to_positions: dict[str, list[tuple[int, int]]] = {}
    insertion_order: list[str] = []
    for obj_type, pos in _iter_objects(objects):
        norm = _normalize_token(obj_type)
        position = (int(pos[0]), int(pos[1]))
        type_to_positions.setdefault(norm, []).append(position)
        if norm not in insertion_order:
            insertion_order.append(norm)

    facts: list[str] = []
    seen_facts: set[str] = set()

    ignored_tokens = {_normalize_token(tok) for tok in (empty_tokens or {"empty"})}

    ordered_types = _ordered_types(
        type_to_positions,
        symbol_map=symbol_map,
        reference_tokens=reference_tokens,
        insertion_order=insertion_order,
    )

    focus_type = next(
        (
            token
            for token in ordered_types
            if token not in ignored_tokens and type_to_positions.get(token)
        ),
        None,
    )

    if focus_type is None:
        return facts

    focus_positions = sorted(type_to_positions[focus_type])

    def _label(name: str, *, capitalize: bool) -> str:
        return _format_label(name, capitalize=capitalize)

    focus_title = _label(focus_type, capitalize=True)
    focus_lower = _label(focus_type, capitalize=False)

    # 1) Adjacency relations
    for obj_type in ordered_types:
        if (
            obj_type == focus_type
            or obj_type in ignored_tokens
            or not type_to_positions.get(obj_type)
        ):
            continue
        positions = type_to_positions[obj_type]
        if any(
            manhattan(agent_cell, obj_cell) == 1
            for agent_cell in focus_positions
            for obj_cell in positions
        ):
            fact = f"{focus_title} adjacent_to {_label(obj_type, capitalize=False)}"
            if fact not in seen_facts:
                facts.append(fact)
                seen_facts.add(fact)

    # 2) Directional relations (agent-centric)
    directions: tuple[tuple[str, Callable[[int, int], bool]], ...] = (
        ("left_of", lambda dr, dc: dr == 0 and dc < 0),
        ("right_of", lambda dr, dc: dr == 0 and dc > 0),
        ("above", lambda dr, dc: dc == 0 and dr < 0),
        ("below", lambda dr, dc: dc == 0 and dr > 0),
    )

    for obj_type in ordered_types:
        if (
            obj_type == focus_type
            or obj_type in ignored_tokens
            or not type_to_positions.get(obj_type)
        ):
            continue
        positions = type_to_positions[obj_type]
        for direction_name, predicate in directions:
            found = False
            for anchor in focus_positions:
                for pos in positions:
                    if predicate(pos[0] - anchor[0], pos[1] - anchor[1]) and (
                        manhattan(anchor, pos) <= max_distance
                    ):
                        found = True
                        break
                if found:
                    break
            if found:
                fact = (
                    f"{_label(obj_type, capitalize=True)} "
                    f"{direction_name} {focus_lower}"
                )
                if fact not in seen_facts:
                    facts.append(fact)
                    seen_facts.add(fact)

    # 3) Nearest distance per type
    for obj_type in ordered_types:
        if (
            obj_type == focus_type
            or obj_type in ignored_tokens
            or not type_to_positions.get(obj_type)
        ):
            continue
        min_dist = min(
            manhattan(anchor, pos)
            for anchor in focus_positions
            for pos in type_to_positions[obj_type]
        )
        fact = f"Nearest {_label(obj_type, capitalize=False)} distance = {min_dist}"
        if fact not in seen_facts:
            facts.append(fact)
            seen_facts.add(fact)

    # 4) Token connectivity facts
    if cluster_tokens is not None:
        cluster_candidates = {_normalize_token(tok) for tok in cluster_tokens}
    else:
        cluster_candidates = {tok for tok in type_to_positions if "token" in tok}

    for cluster_type in ordered_types:
        if cluster_type not in cluster_candidates:
            continue
        cluster_positions = type_to_positions.get(cluster_type)
        if not cluster_positions:
            continue

        token_clusters = _find_token_clusters(cluster_positions)
        touching_clusters: list[tuple[int, list[tuple[int, int]]]] = []
        for cluster in token_clusters:
            best = min(
                manhattan(anchor, cell)
                for anchor in focus_positions
                for cell in cluster
            )
            if best <= 1:
                touching_clusters.append((best, cluster))

        if touching_clusters:
            touching_clusters.sort(key=lambda item: (item[0], item[1][0], len(item[1])))
            _, chosen_cluster = touching_clusters[0]
            cluster_label = (
                f"{_label(cluster_type, capitalize=True).replace(' ', '_')}_cluster"
            )
            fact = f"{focus_title} adjacent_to {cluster_label}"
            if fact not in seen_facts:
                facts.append(fact)
                seen_facts.add(fact)
            size_fact = f"{cluster_label} size = {len(chosen_cluster)}"
            if size_fact not in seen_facts:
                facts.append(size_fact)
                seen_facts.add(size_fact)

    return facts


def _find_token_clusters(
    positions: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    """Return sorted connected components (4-neighborhood) of token
    positions."""

    tokens: set[tuple[int, int]] = {(int(r), int(c)) for (r, c) in positions}
    visited: set[tuple[int, int]] = set()
    clusters: list[list[tuple[int, int]]] = []
    for start in sorted(tokens):
        if start in visited:
            continue
        queue: deque[tuple[int, int]] = deque([start])
        visited.add(start)
        cluster: list[tuple[int, int]] = []
        while queue:
            cell = queue.popleft()
            cluster.append(cell)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (cell[0] + dr, cell[1] + dc)
                if neighbor in tokens and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        cluster.sort()
        clusters.append(cluster)

    clusters.sort(key=lambda c: c[0])
    return clusters


def _iter_objects(objects: ObjectsArg) -> Iterable[tuple[str, tuple[int, int]]]:
    """Yield (type, position) pairs from dict- or list-based grid objects."""

    if isinstance(objects, Mapping):
        for obj_type, positions in objects.items():
            for pos in positions:
                yield obj_type, pos
    else:
        for obj_type, pos in objects:
            yield obj_type, pos


def _ordered_types(
    type_to_positions: dict[str, list[tuple[int, int]]],
    *,
    symbol_map: Mapping[str, str] | None,
    reference_tokens: Iterable[str] | None,
    insertion_order: Iterable[str],
) -> list[str]:
    """Return token names ordered by user preference and availability."""

    sequences: list[Iterable[str]] = []
    if reference_tokens:
        sequences.append(reference_tokens)
    if symbol_map:
        sequences.append(symbol_map.keys())
    sequences.append(insertion_order)
    sequences.append(sorted(type_to_positions.keys()))

    ordered: list[str] = []
    seen: set[str] = set()
    for seq in sequences:
        for token in seq:
            norm = _normalize_token(token)
            if norm in type_to_positions and norm not in seen:
                ordered.append(norm)
                seen.add(norm)
    return ordered
