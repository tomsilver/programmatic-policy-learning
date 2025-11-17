"""Experts for maze environments."""

from typing import Any, Callable, Iterator, Tuple

import numpy as np
from prpl_utils.search import run_astar

from programmatic_policy_learning.approaches.base_approach import _ActType

_State = Tuple[int, int]
Obs = Tuple[int, int]
Act = int


class ExpertMazeWithOuterWorldPolicy:
    """Expert maze policy that keeps its plan as part of the class."""

    def __init__(
        self,
        grid: np.ndarray,
        goal: Obs,
        get_actions: Callable[[], list[Act]],
        get_next_state: Callable[[Obs, Act], Obs],
        get_cost: Callable[[], float],
        check_goal: Callable[[Obs, Any], bool],
    ) -> None:
        self.grid = grid
        self.get_actions = get_actions
        self.get_next_state = get_next_state
        self.get_cost = get_cost
        self.check_goal = check_goal

        self.inner_h, self.inner_w = grid.shape
        self.entrance: Obs = (-1, 0)
        self.goal: Obs = goal

        # Plan is maintained as part of the class.
        self.plan: list[Act] = []

    # ---------- Helper methods ----------

    def _is_inside_maze(self, s: Obs) -> bool:
        r, c = s
        return 0 <= r < self.inner_h and 0 <= c < self.inner_w

    def _plan_to_entrance(self, start: Obs) -> list[Act]:
        """Plan a simple path from any outer-world position to the entrance
        (-1, 0), avoiding the boxed maze (which spans 0..inner_h-1 and
        0..inner_w-1).

        Actions:
            0 = North (-1, 0)
            1 = South (+1, 0)
            2 = East  (0, +1)
            3 = West  (0, -1)
        """
        r, c = start
        actions: list[Act] = []

        def move_north(k: int) -> None:
            actions.extend([0] * k)

        def move_south(k: int) -> None:
            actions.extend([1] * k)

        def move_east(k: int) -> None:
            actions.extend([2] * k)

        def move_west(k: int) -> None:
            actions.extend([3] * k)

        inner_h, inner_w = self.inner_h, self.inner_w

        # 1. If below the maze, go sideways to nearest outer corridor,
        # then up above the maze.
        if r > inner_h:
            left_corridor, right_corridor = -2, inner_w + 1
            target_c = (
                left_corridor
                if abs(c - left_corridor) <= abs(c - right_corridor)
                else right_corridor
            )
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

        # 4. If we’re still not above the maze, move up to above the top
        # wall.
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

    def _generate_plan(self, start: Obs, goal: Obs | None = None) -> list[Act]:
        """Generate a plan using A* search."""
        if goal is None:
            goal = self.goal

        def get_successors(
            state: Obs,
        ) -> Iterator[Tuple[Act, Obs, float]]:
            for action in self.get_actions():
                next_state = self.get_next_state(state, action)
                if not np.array_equal(next_state, state):  # Valid transition
                    cost = self.get_cost()
                    yield (action, next_state, cost)

        _, search_plan = run_astar(
            initial_state=tuple(start),
            check_goal=lambda state: self.check_goal(state, goal),
            get_successors=get_successors,
            heuristic=lambda s: abs(s[0] - goal[0]) + abs(s[1] - goal[1]),
        )
        return search_plan

    # ---------- Main policy interface ----------

    def __call__(self, obs: Obs) -> Act:
        """A policy that uses a simple rule to get to the entrance of the maze,
        and then uses search to navigate within the maze."""
        # Refill plan if empty.
        if not self.plan:
            if (not self._is_inside_maze(obs)) and (obs != self.entrance):
                # We are in the outer world and not at the entrance yet.
                self.plan = self._plan_to_entrance(obs)
            else:
                # We are in the maze (or at the entrance), so generate a plan to the goal.
                self.plan = self._generate_plan(obs, self.goal)

        # Follow the plan one action at a time.
        return self.plan.pop(0)


def create_expert_maze_with_outer_world_policy(
    grid: np.ndarray,
    goal: Obs,
    get_actions: Callable[[], list[Act]],
    get_next_state: Callable[[Obs, Act], Obs],
    get_cost: Callable[[], float],
    check_goal: Callable[[Obs, Any], bool],
) -> Callable[[Obs], Act]:
    """Factory to keep the original functional interface."""
    expert = ExpertMazeWithOuterWorldPolicy(
        grid=grid,
        goal=goal,
        get_actions=get_actions,
        get_next_state=get_next_state,
        get_cost=get_cost,
        check_goal=check_goal,
    )
    # Return a callable policy with the same signature as before
    return expert
