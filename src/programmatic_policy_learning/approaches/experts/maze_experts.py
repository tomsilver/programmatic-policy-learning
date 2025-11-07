"""Experts for maze environments."""

from typing import Any, Callable
from prpl_utils.search import run_astar
from typing import Any, Callable, Iterator, Tuple
import numpy as np
from programmatic_policy_learning.approaches.base_approach import (
    _ActType,
)

Obs = tuple[int, int]
Act = int


def create_expert_maze_with_outer_world_policy(
    grid: np.ndarray,
    get_actions: Callable[[], list[Act]],
    get_next_state: Callable[[Obs, Act], Obs],
    get_cost: Callable[[Obs, Act, Obs], float],
    check_goal: Callable[[Obs, Any], bool],
) -> Callable[[Obs], Act]:
    """Create an expert maze policy."""

    # Create a plan list. TODO later: maybe refactor this into a class so that the
    # plan is actually maintained as part of the class.
    plan: list[Act] = []
    # import ipdb

    # ipdb.set_trace()

    def expert_maze_with_outer_world_policy(obs: Obs) -> Act:
        """A policy that uses a simple rule to get to the entrance of the maze,
        and then uses search to navigate within the maze."""
        nonlocal plan

        inner_h, inner_w = grid.shape
        entrance: Obs = (-1, 0)
        goal: Obs = (inner_h - 1, inner_w - 1)
        # If we are not yet at the entrance of the maze, use a simple rule to determine
        # what action to take to get closer to it.
        # The rule is: if the column of the agent is not already equal to the column
        # of the entrance of the maze, move left or right towards that column. Otherwise
        # move down (assuming that we start above the entrance) until reaching.
        def plan_to_entrance(start: tuple[int, int], inner_h: int, inner_w: int) -> list[int]:
            """
            Plan a simple path from any outer-world position to the entrance (-1, 0),
            avoiding the boxed maze (which spans 0..inner_h-1 and 0..inner_w-1).

            Actions:
                0 = North (-1, 0)
                1 = South (+1, 0)
                2 = East  (0, +1)
                3 = West  (0, -1)
            """
            r, c = start
            actions = []

            def move_north(k: int): actions.extend([0] * k)
            def move_south(k: int): actions.extend([1] * k)
            def move_east(k: int):  actions.extend([2] * k)
            def move_west(k: int):  actions.extend([3] * k)

            # 1. If below the maze, go sideways to nearest outer corridor, then up above the maze.
            if r > inner_h:
                left_corridor, right_corridor = -2, inner_w + 1
                target_c = left_corridor if abs(c - left_corridor) <= abs(c - right_corridor) else right_corridor
                if c < target_c:
                    move_east(target_c - c)
                    c = target_c
                elif c > target_c:
                    move_west(c - target_c)
                    c = target_c
                move_north(r - (-2))  # move up to row = -2
                r = -2

            # 2. If left of the maze, go above it first if necessary.
            if c < -1 and r >= -1:
                move_north(r - (-2))
                r = -2

            # 3. If right of the maze, same logic.
            if c > inner_w and r >= -1:
                move_north(r - (-2))
                r = -2

            # 4. If we’re still not above the maze, move up to above the top wall.
            if r >= -1:
                move_north(r - (-2))
                r = -2

            # 5. Move horizontally until aligned with entrance column (0)
            if c < 0:
                move_east(-c)
                c = 0
            elif c > 0:
                move_west(c)
                c = 0

            # 6. Finally, move down to the entrance row (-1)
            move_south((-1) - r)
            r = -1

            # Should now be exactly at (-1, 0)
            assert (r, c) == (-1, 0), f"Ended at {(r, c)} instead of entrance (-1, 0)"
            return actions


        # Otherwise, we are in the maze, so determine if we need to get a new plan,
        # and then follow the plan.
        if plan is None:
            # TODO: Use get_actions, etc. to run astar search to populate plan.
            def generate_plan(self, start: np.ndarray, goal: np.ndarray) -> list[_ActType]:
                """Generate a plan using A* search."""

                def get_successors(
                    state: Tuple[int, int],
                ) -> Iterator[Tuple[_ActType, Tuple[int, int], float]]:
                    """Generate successors for the current state."""
                    for action in get_actions():
                        next_state = get_next_state(state, action)
                        if not np.array_equal(next_state, state):  # Valid transition
                            cost = get_cost(state, action, next_state)
                            yield (action, next_state, cost)

                _, plan = run_astar(
                    initial_state=tuple(start),
                    check_goal=lambda state: check_goal(state, goal),
                    get_successors=get_successors,
                    heuristic=lambda s: abs(s[0] - goal[0]) + abs(s[1] - goal[1]),
                )
                return plan
            plan = generate_plan(obs, goal)

        return plan.pop(0)

    return expert_maze_with_outer_world_policy
