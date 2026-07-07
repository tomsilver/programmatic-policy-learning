"""Handwritten feature policy for ARC-AGI-3 ls20 level 0."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

Feature = Callable[[str, Any, int], bool]


def accepts(state: Any, action: int, feature: Feature) -> bool:
    """Return whether an action is allowed by the handwritten policy."""
    if not feature("f12", state, action):
        return False
    if not feature("f6", state, action):
        return False

    if feature("f14", state, action):
        # The shape is not aligned yet, so route the player to the switch.
        if not feature("f19", state, action):
            # Below the switch, both demonstrated routes are acceptable:
            # move Up or Left when that reduces switch distance.
            return feature("f15", state, action)
        if not feature("f20", state, action):
            # Above the switch but not at the upper corridor: continue Up.
            return action == 1
        if feature("f21", state, action):
            # Directly above the switch: move Down onto it.
            return action == 2
        # At the upper corridor but not switch-aligned: move Left.
        return action == 3

    if feature("f13", state, action):
        # Shape is aligned. Finish moving Up out of the switch before turning.
        if not feature("f20", state, action):
            return action == 1
        if feature("f21", state, action):
            return action == 4
        # Cross the middle corridor, then move Up the goal column.
        return feature("f3", state, action) or feature("f4", state, action)

    return False


def choose_action(
    state: Any,
    accepted_actions: Sequence[int],
    feature: Feature,
) -> int:
    """Choose one action deterministically when several are acceptable."""
    if not accepted_actions:
        raise ValueError("The policy did not accept any action.")

    # Prefer the first demonstration's lower route when both initial routes
    # are valid. This makes live rollout deterministic without declaring the
    # second demonstration's Up action invalid.
    if feature("f14", state, accepted_actions[0]):
        for action in accepted_actions:
            if feature("f1", state, action):
                return action

    return accepted_actions[0]
