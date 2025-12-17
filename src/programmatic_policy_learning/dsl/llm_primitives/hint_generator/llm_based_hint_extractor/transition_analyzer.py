# transition_analyzer.py

from typing import Any, Dict, List, Tuple

import numpy as np
from grid_encoder import GridStateEncoder


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GenericTransitionAnalyzer:
    def analyze(
        self,
        obs_t: np.ndarray,
        action: Tuple[int, int],
        obs_t1: np.ndarray,
        *,
        encoder: GridStateEncoder,
        agent_token: str = "agent",
        target_token: str | None = "target",
    ) -> List[str]:
        events: List[str] = []

        # locate agent / target
        def find(token, obs):
            locs = np.argwhere(obs == token)
            return tuple(locs[0]) if len(locs) > 0 else None

        a0 = find(agent_token, obs_t)
        a1 = find(agent_token, obs_t1)
        t0 = find(target_token, obs_t) if target_token else None
        t1 = find(target_token, obs_t1) if target_token else None

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
