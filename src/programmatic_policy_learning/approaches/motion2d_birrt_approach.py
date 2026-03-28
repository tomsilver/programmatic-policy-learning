"""An approach that uses BiRRT motion planning for the Motion2D environment."""

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.utils.motion_planning_utils import (
    crv_pose_plan_to_action_plan,
    run_motion_planning_for_crv_robot,
)

Obs = NDArray[np.float32]
Act = NDArray[np.float32]


class Motion2DBiRRTApproach(BaseApproach[Obs, Act]):
    """An approach that uses Bidirectional RRT (BiRRT) to plan a collision-free
    path for the Motion2D environment.

    Modelled after ``SearchApproach``: on ``reset`` a full action plan is
    computed and stored; ``_get_action`` replays it step by step.  If planning
    fails the plan is empty and ``_get_action`` raises.

    Parameters
    ----------
    environment_description, observation_space, action_space, seed:
        Standard ``BaseApproach`` arguments.
    get_object_centric_state:
        Callable that maps a flat numpy observation to an
        ``ObjectCentricState``.  Typically
        ``lambda obs: inner.set_state(obs) or inner._object_centric_env.get_state()``.
    num_attempts:
        Number of independent BiRRT trees to try during planning.
    num_iters:
        RRT iterations per tree.
    smooth_amt:
        Post-processing smoothing passes.
    """

    def __init__(
        self,
        environment_description: str,
        observation_space: Any,
        action_space: Any,
        seed: int,
        get_object_centric_state: Callable[[Obs], Any],
        num_attempts: int = 10,
        num_iters: int = 200,
        smooth_amt: int = 50,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        self._seed = seed
        self._get_object_centric_state = get_object_centric_state
        self._num_attempts = num_attempts
        self._num_iters = num_iters
        self._smooth_amt = smooth_amt
        self._plan: list[Act] = []
        self.metrics: Any = None  # MotionPlanningMetrics from last reset

    # ------------------------------------------------------------------
    # BaseApproach interface
    # ------------------------------------------------------------------

    def reset(self, obs: Obs, info: dict[str, Any]) -> None:
        """Plan a collision-free path from the current robot pose to the
        target."""
        super().reset(obs, info)
        self._plan, self.metrics = self._generate_plan(obs)

    def _get_action(self) -> Act:
        """Return the next action from the plan."""
        if not self._plan:
            raise ValueError(
                "Plan is empty. Ensure reset() was called and planning succeeded."
            )
        return self._plan.pop(0)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _generate_plan(self, obs: Obs) -> tuple[list[Act], Any]:
        """Run BiRRT and return (action_plan, metrics)."""
        from kinder.envs.kinematic2d.object_types import CRVRobotType
        from kinder.envs.kinematic2d.structs import SE2Pose

        state = self._get_object_centric_state(obs)

        robots = state.get_objects(CRVRobotType)
        if not robots:
            raise ValueError("No robot found in ObjectCentricState.")
        robot = robots[0]

        # Target region centre: bottom-left corner + half the extents.
        target_x = float(obs[9]) + float(obs[17]) / 2.0
        target_y = float(obs[10]) + float(obs[18]) / 2.0
        target_pose = SE2Pose(target_x, target_y, 0.0)

        pose_plan, metrics = run_motion_planning_for_crv_robot(
            state,
            robot,
            target_pose,
            self._action_space,
            seed=self._seed,
            num_attempts=self._num_attempts,
            num_iters=self._num_iters,
            smooth_amt=self._smooth_amt,
        )

        if pose_plan is None:
            return [], metrics

        return (
            list(crv_pose_plan_to_action_plan(pose_plan, self._action_space)),
            metrics,
        )
