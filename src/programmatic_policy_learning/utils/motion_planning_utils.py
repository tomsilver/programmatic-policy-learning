"""Motion-planning utilities for CRV robots.

This mirrors ``kinder.envs.kinematic2d.utils.run_motion_planning_for_crv_robot``
and ``crv_pose_plan_to_action_plan``, but uses the updated ``prpl_utils``
``BiRRT.query()`` API that returns ``(path, MotionPlanningMetrics)`` instead of
just the path.  The kindergarden package is kept untouched so that ongoing
experiments are not affected.
"""

from typing import Any, Iterable

import numpy as np
from kinder.envs.kinematic2d.structs import MultiBody2D, SE2Pose
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    snap_suctioned_objects,
)
from kinder.envs.utils import get_se2_pose, state_2d_has_collision
from prpl_utils.motion_planning import BiRRT

try:
    from prpl_utils.motion_planning import MotionPlanningMetrics
except ImportError:
    MotionPlanningMetrics = None  # type: ignore[assignment,misc]
from prpl_utils.utils import get_signed_angle_distance, wrap_angle
from relational_structs import Array, Object, ObjectCentricState


def run_motion_planning_for_crv_robot(
    state: ObjectCentricState,
    robot: Object,
    target_pose: SE2Pose,
    action_space: CRVRobotActionSpace,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
) -> tuple[list[SE2Pose] | None, Any]:
    """Run BiRRT motion planning for a CRV robot and return the pose plan and
    metrics.

    Identical in logic to ``kinder.envs.kinematic2d.utils.run_motion_planning_for_crv_robot``
    except that it returns ``(pose_plan, metrics)`` using the updated
    ``prpl_utils.BiRRT.query()`` API.

    Parameters
    ----------
    state:
        Current ``ObjectCentricState`` of the environment.
    robot:
        The robot ``Object`` to plan for.
    target_pose:
        Desired ``SE2Pose`` goal for the robot.
    action_space:
        ``CRVRobotActionSpace`` used to bound interpolation step sizes.
    static_object_body_cache:
        Optional cache of pre-built ``MultiBody2D`` for static objects.
    seed:
        RNG seed for reproducibility.
    num_attempts:
        Number of independent BiRRT trees to try.
    num_iters:
        RRT iterations per tree.
    smooth_amt:
        Post-processing smoothing passes.

    Returns
    -------
    tuple[list[SE2Pose] | None, MotionPlanningMetrics]
        ``(pose_plan, metrics)`` where ``pose_plan`` is ``None`` if planning
        failed.
    """
    if static_object_body_cache is None:
        static_object_body_cache = {}

    rng = np.random.default_rng(seed)

    # World bounds from object positions.
    x_lb, x_ub, y_lb, y_ub = np.inf, -np.inf, np.inf, -np.inf
    for obj in state:
        pose = get_se2_pose(state, obj)
        x_lb = min(x_lb, pose.x)
        x_ub = max(x_ub, pose.x)
        y_lb = min(y_lb, pose.y)
        y_ub = max(y_ub, pose.y)

    # Static copy of state for collision checking (don't pollute the caller's cache).
    static_object_body_cache = static_object_body_cache.copy()
    suctioned_objects = get_suctioned_objects(state, robot)
    moving_objects = {robot} | {o for o, _ in suctioned_objects}
    static_state = state.copy()
    for o in static_state:
        if o not in moving_objects:
            static_state.set(o, "static", 1.0)

    def sample_fn(_: SE2Pose) -> SE2Pose:
        x = rng.uniform(x_lb, x_ub)
        y = rng.uniform(y_lb, y_ub)
        theta = rng.uniform(-np.pi, np.pi)
        return SE2Pose(x, y, theta)

    def extend_fn(pt1: SE2Pose, pt2: SE2Pose) -> Iterable[SE2Pose]:
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
        assert isinstance(action_space, CRVRobotActionSpace)
        abs_x = action_space.high[0] if dx > 0 else action_space.low[0]
        abs_y = action_space.high[1] if dy > 0 else action_space.low[1]
        abs_theta = action_space.high[2] if dtheta > 0 else action_space.low[2]
        x_num_steps = int(dx / abs_x) + 1
        assert x_num_steps > 0
        y_num_steps = int(dy / abs_y) + 1
        assert y_num_steps > 0
        theta_num_steps = int(dtheta / abs_theta) + 1
        assert theta_num_steps > 0
        num_steps = max(x_num_steps, y_num_steps, theta_num_steps)
        x, y, theta = pt1.x, pt1.y, pt1.theta
        yield SE2Pose(x, y, theta)
        for _ in range(num_steps):
            x += dx / num_steps
            y += dy / num_steps
            theta = wrap_angle(theta + dtheta / num_steps)
            yield SE2Pose(x, y, theta)

    def collision_fn(pt: SE2Pose) -> bool:
        static_state.set(robot, "x", pt.x)
        static_state.set(robot, "y", pt.y)
        static_state.set(robot, "theta", pt.theta)
        snap_suctioned_objects(static_state, robot, suctioned_objects)
        obstacle_objects = set(static_state) - moving_objects
        return state_2d_has_collision(
            static_state, moving_objects, obstacle_objects, static_object_body_cache
        )

    def distance_fn(pt1: SE2Pose, pt2: SE2Pose) -> float:
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
        return float(np.sqrt(dx**2 + dy**2) + abs(dtheta))

    birrt = BiRRT(
        sample_fn,
        extend_fn,
        collision_fn,
        distance_fn,
        rng,
        num_attempts,
        num_iters,
        smooth_amt,
    )

    initial_pose = get_se2_pose(state, robot)
    result = birrt.query(initial_pose, target_pose)

    # prpl-utils >= 0.0.5 (local) returns (path, MotionPlanningMetrics).
    # prpl-utils == 0.1.0 (PyPI/git) returns path | None directly.
    if isinstance(result, tuple):
        pose_plan, metrics = result
    else:
        pose_plan = result
        try:
            from prpl_utils.motion_planning import MotionPlanningMetrics
            metrics = MotionPlanningMetrics()
        except ImportError:
            metrics = None

    return pose_plan, metrics


def crv_pose_plan_to_action_plan(
    pose_plan: list[SE2Pose],
    action_space: CRVRobotActionSpace,
    vacuum_while_moving: bool = False,
) -> list[Array]:
    """Convert a CRV robot pose plan into environment actions.

    Identical to ``kinder.envs.kinematic2d.utils.crv_pose_plan_to_action_plan``;
    re-exported here so callers can import everything from one place.
    """
    action_plan: list[Array] = []
    for pt1, pt2 in zip(pose_plan[:-1], pose_plan[1:]):
        action = np.zeros_like(action_space.high)
        action[0] = pt2.x - pt1.x
        action[1] = pt2.y - pt1.y
        action[2] = get_signed_angle_distance(pt2.theta, pt1.theta)
        action[4] = 1.0 if vacuum_while_moving else 0.0
        action_plan.append(action)
    return action_plan
