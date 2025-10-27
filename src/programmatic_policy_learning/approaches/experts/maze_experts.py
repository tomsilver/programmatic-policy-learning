"""Experts for maze environments."""

from typing import Any, Callable
import numpy as np

Obs = tuple[int, int]
Act = int

def create_expert_maze_with_outer_world_policy(
    grid: np.ndarray,
    get_actions: Callable[[], list[Act]],
    get_next_state: Callable[[Obs, Act], Obs],
    get_cost: Callable[[Obs, Act, Obs], float],
    check_goal: Callable[[Obs, Any], bool]
) -> Callable[[Obs], Act]:
    """Create an expert maze policy."""

    # Create a plan list. TODO later: maybe refactor this into a class so that the
    # plan is actually maintained as part of the class.
    plan: list[Act] = []

    def expert_maze_with_outer_world_policy(obs: Obs) -> Act:
        """A policy that uses a simple rule to get to the entrance of the maze,
        and then uses search to navigate within the maze."""
        nonlocal plan

        # If we are not yet at the entrance of the maze, use a simple rule to determine
        # what action to take to get closer to it.
        # The rule is: if the column of the agent is not already equal to the column
        # of the entrance of the maze, move left or right towards that column. Otherwise
        # move down (assuming that we start above the entrance) until reaching.
        import ipdb; ipdb.set_trace()

        # Otherwise, we are in the maze, so determine if we need to get a new plan,
        # and then follow the plan.
        if plan is None:
            # TODO: Use get_actions, etc. to run astar search to populate plan.
            import ipdb; ipdb.set_trace()

        return plan.pop(0)


    return expert_maze_with_outer_world_policy
