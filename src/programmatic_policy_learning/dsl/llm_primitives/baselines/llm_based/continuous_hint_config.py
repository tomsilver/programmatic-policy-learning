"""Domain-specific observation/action metadata for continuous environments.

Observation layout sourced from the PRBench Motion2D documentation:
https://prpl.group/prbench-site/environments/motion2d/motion2d-p1.html
"""

from __future__ import annotations

# pylint: disable=line-too-long
# ---------------------------------------------------------------------------
# Canonical environment-name mapping (case-insensitive → registered name)
# ---------------------------------------------------------------------------
_CANONICAL_ENV_NAME: dict[str, str] = {
    "balancebeam3d": "BalanceBeam3D",
    "basemotion3d": "BaseMotion3D",
    "clutteredretrieval2d": "ClutteredRetrieval2D",
    "clutteredstorage2d": "ClutteredStorage2D",
    "constrainedcupboard3d": "ConstrainedCupboard3D",
    "dynobstruction2d": "DynObstruction2D",
    "dynpushpullhook2d": "DynPushPullHook2D",
    "dynpusht2d": "DynPushT2D",
    "dynscooppour2d": "DynScoopPour2D",
    "dynamo3d": "Dynamo3D",
    "motion2d": "Motion2D",
    "obstruction2d": "Obstruction2D",
    "obstruction3d": "Obstruction3D",
    "packing3d": "Packing3D",
    "pushpullhook2d": "PushPullHook2D",
    "rearrange3d": "Rearrange3D",
    "scooppour3d": "ScoopPour3D",
    "shelf3d": "Shelf3D",
    "sortclutteredblocks3d": "SortClutteredBlocks3D",
    "stickbutton2d": "StickButton2D",
    "sweepintodrawer3d": "SweepIntoDrawer3D",
    "sweepsimple3d": "SweepSimple3D",
    "table3d": "Table3D",
    "tossing3d": "Tossing3D",
    "transport3d": "Transport3D",
}


def canonicalize_env_name(env_name: str) -> str:
    """Return the canonical (case-correct) KinDER environment name.

    If *env_name* (lowered) is in the mapping, return the canonical
    form; otherwise return the original string unchanged.

    Parameters
    ----------
    env_name: str
        Environment name to canonicalize.

    Returns
    -------
    str: The canonicalized environment name.

    Examples
    --------
    >>> canonicalize_env_name("motion2d")
    'Motion2D'
    >>> canonicalize_env_name("Motion2D")
    'Motion2D'
    """
    return _CANONICAL_ENV_NAME.get(env_name.lower(), env_name)


# Robot (9 features, indices 0-8)
_ROBOT_FIELDS = [
    "robot_x",
    "robot_y",
    "robot_theta",
    "robot_base_radius",
    "robot_arm_joint",
    "robot_arm_length",
    "robot_vacuum",
    "robot_gripper_height",
    "robot_gripper_width",
]

# Target region (10 features, indices 9-18)
_TARGET_FIELDS = [
    "target_x",
    "target_y",
    "target_theta",
    "target_static",
    "target_color_r",
    "target_color_g",
    "target_color_b",
    "target_z_order",
    "target_width",
    "target_height",
]

# Each obstacle has 10 features
_OBSTACLE_FEATURE_NAMES = [
    "x",
    "y",
    "theta",
    "static",
    "color_r",
    "color_g",
    "color_b",
    "z_order",
    "width",
    "height",
]

_HOOK_FIELDS = [
    "hook_x",
    "hook_y",
    "hook_theta",
    "hook_static",
    "hook_color_r",
    "hook_color_g",
    "hook_color_b",
    "hook_z_order",
    "hook_width",
    "hook_length_side1",
    "hook_length_side2",
]

_BUTTON_FIELDS = [
    "x",
    "y",
    "theta",
    "static",
    "color_r",
    "color_g",
    "color_b",
    "z_order",
    "radius",
]


def obs_field_names_for_motion2d(num_passages: int) -> list[str]:
    """Build the full observation field-name list for Motion2D-p{num_passages}.

    Each passage is made of two rectangular obstacles (bottom wall, top wall),
    so the total number of obstacles is ``2 * num_passages``.

    Parameters
    ----------
    num_passages: int
        The number of passages in the environment.

    Returns
    -------
    list[str]: The full observation field-name list.

    Examples
    --------
    >>> obs_field_names_for_motion2d(0)
    # no passages
    ['robot_x', 'robot_y', 'robot_theta', 'robot_base_radius', 'robot_arm_joint',
    'robot_arm_length', 'robot_vacuum', 'robot_gripper_height', 'robot_gripper_width',
    'target_x', 'target_y', 'target_theta', 'target_static', 'target_color_r',
    'target_color_g', 'target_color_b', 'target_z_order', 'target_width', 'target_height']
    >>> obs_field_names_for_motion2d(1)
    # 1 passage, 2 obstacles
    ['robot_x', 'robot_y', 'robot_theta', 'robot_base_radius', 'robot_arm_joint',
    'robot_arm_length', 'robot_vacuum', 'robot_gripper_height', 'robot_gripper_width',
    'target_x', 'target_y', 'target_theta', 'target_static', 'target_color_r',
    'target_color_g', 'target_color_b', 'target_z_order', 'target_width', 'target_height',
    'obstacle0_x', 'obstacle0_y', 'obstacle0_theta', 'obstacle0_static',
    'obstacle0_color_r', 'obstacle0_color_g', 'obstacle0_color_b', 'obstacle0_z_order',
    'obstacle0_width', 'obstacle0_height', 'obstacle1_x', 'obstacle1_y', 'obstacle1_theta',
    'obstacle1_static', 'obstacle1_color_r', 'obstacle1_color_g', 'obstacle1_color_b',
    'obstacle1_z_order', 'obstacle1_width', 'obstacle1_height']
    """
    fields = list(_ROBOT_FIELDS) + list(_TARGET_FIELDS)
    for i in range(2 * num_passages):
        for feat in _OBSTACLE_FEATURE_NAMES:
            fields.append(f"obstacle{i}_{feat}")
    return fields


def obs_field_names_for_pushpullhook2d() -> list[str]:
    """Build the full observation field-name list for PushPullHook2D.

    The observation is object-centric and concatenates:
    - robot (`crv_robot`): 9 features
    - hook (`lobject`): 11 features
    - movable button (`circle`): 9 features
    - target button (`circle`): 9 features

    Returns
    -------
    list[str]
        The full 38-name observation schema aligned with the env's
        flattened object-centric observation vector.
    """
    fields = list(_ROBOT_FIELDS)
    fields.extend(_HOOK_FIELDS)
    fields.extend(f"movable_button_{name}" for name in _BUTTON_FIELDS)
    fields.extend(f"target_button_{name}" for name in _BUTTON_FIELDS)
    return fields


def obs_field_names_for_kinder(env_name: str, num_passages: int = 0) -> list[str]:
    """Return the continuous observation schema for a supported KinDER env."""
    canonical_name = canonicalize_env_name(env_name)
    if canonical_name == "Motion2D":
        return obs_field_names_for_motion2d(num_passages)
    if canonical_name == "PushPullHook2D":
        return obs_field_names_for_pushpullhook2d()
    raise ValueError(f"Unsupported KinDER env for obs field names: {env_name}")


ACTION_FIELD_NAMES: dict[str, list[str]] = {
    "Motion2D": ["dx", "dy", "dtheta", "darm", "vac"],
    "PushPullHook2D": ["dx", "dy", "dtheta", "darm", "vac"],
}


def get_env_description(env_name: str, num_passages: int = 0) -> str:
    """Return a task description tailored to the environment variant.

    For Motion2D the description changes depending on whether there are wall
    obstacles (``num_passages > 0``) or not.

    Parameters
    ----------
    env_name: str
        The name of the environment.
    num_passages: int
        The number of passages in the environment.

    Returns
    -------
    str: The task description.

    Examples
    --------
    # no passages
    >>> get_env_description("Motion2D", 0)
    'A 2-D navigation task. A circular robot must travel from its
    starting position to a rectangular target region.'
    # 1 passage, 2 obstacles
    >>> get_env_description("Motion2D", 1)
    'A 2-D navigation task. A circular robot must travel from its
    starting position to a rectangular target region.
    There is 1 rectangular wall obstacle between the robot
    and the target, each with a narrow gap (passage)
    the robot must fit through.'
    """
    if env_name == "Motion2D":
        base = (
            "A 2-D navigation task. A circular robot must travel from its "
            "starting position to a rectangular target region."
        )
        if num_passages > 0:
            num_obstacles = 2 * num_passages
            base += (
                f" There are {num_obstacles} rectangular wall obstacles "
                "between the robot and the target.  Two obstacles that "
                "share the same x-coordinate form a passage (narrow gap) "
                "the robot must fit through."
                f" Therefore, there {'is' if num_passages == 1 else 'are'} "
                f"{num_passages} passage"
                f"{'s' if num_passages > 1 else ''} between the robot and "
                "the target in total."
            )
        base += (
            " Coordinate convention: robot_x and robot_y give the CENTER of "
            "the circular robot.  For all rectangles (target region and "
            "obstacles), x and y give the BOTTOM-LEFT corner; the rectangle "
            "extends rightward by its width and upward by its height."
            " The robot has a movable circular base, a retractable arm, and "
            "a vacuum end effector (the arm and vacuum are not needed for this "
            "task).  The action is a 5-D vector [dx, dy, dtheta, darm, vac]: "
            "dx/dy move the base (positive dx = rightward, positive dy = upward), "
            "dtheta rotates it, darm extends/retracts the arm, and vac toggles "
            "the vacuum.  For pure navigation, only dx and dy matter; set the "
            "others to 0."
        )
        return base
    if env_name == "PushPullHook2D":
        return (
            "A 2-D manipulation task with a mobile circular robot, an L-shaped hook, "
            "a movable button, and a fixed target button. The goal is to move the "
            "movable button until it presses the target button. The robot can use "
            "its base motion, orientation, and arm reach to contact the hook, then "
            "use the hook to push or pull the movable button toward the target. "
            "The action is a 5-D vector [dx, dy, dtheta, darm, vac]: dx/dy move "
            "the robot base, dtheta rotates it, darm extends/retracts the arm, "
            "and vac toggles the vacuum. The task is primarily contact-based; the "
            "hook and button positions are usually most important."
        )
    raise ValueError(f"Unknown environment name: {env_name}")


_BASE_SALIENT_INDICES = [
    0,  # robot_x
    1,  # robot_y
    3,  # robot_base_radius
    9,  # target_x
    10,  # target_y
    17,  # target_width
    18,  # target_height
]

# obstacle x, obstacle y, obstacle width, obstacle height
_OBSTACLE_SALIENT_OFFSETS = [0, 1, 8, 9]

_NUM_ROBOT_FIELDS = len(_ROBOT_FIELDS)
_NUM_TARGET_FIELDS = len(_TARGET_FIELDS)
_OBSTACLE_START = _NUM_ROBOT_FIELDS + _NUM_TARGET_FIELDS
_FEATURES_PER_OBSTACLE = len(_OBSTACLE_FEATURE_NAMES)


def salient_obs_indices_for_motion2d(num_passages: int) -> list[int]:
    """Return decision-critical observation indices for Motion2D.

    Always includes the base robot/target indices.  For each obstacle
    (``2 * num_passages`` total), appends the x, y, width and height
    indices while skipping cosmetic features (colour, z-order, etc.).

    Parameters
    ----------
    num_passages : int
        Number of narrow passages; each produces 2 wall obstacles.

    Returns
    -------
    list[int]
        Sorted observation indices.

    Examples
    --------
    >>> salient_obs_indices_for_motion2d(0)
    [0, 1, 3, 9, 10, 17, 18]
    >>> salient_obs_indices_for_motion2d(1)
    [0, 1, 3, 9, 10, 17, 18, 19, 20, 27, 28, 29, 30, 37, 38]
    """
    indices = list(_BASE_SALIENT_INDICES)
    for i in range(2 * num_passages):
        base = _OBSTACLE_START + i * _FEATURES_PER_OBSTACLE
        for offset in _OBSTACLE_SALIENT_OFFSETS:
            indices.append(base + offset)
    return indices


def salient_obs_indices_for_pushpullhook2d() -> list[int]:
    """Return decision-critical observation indices for PushPullHook2D."""
    return [
        0,   # robot_x
        1,   # robot_y
        2,   # robot_theta
        3,   # robot_base_radius
        4,   # robot_arm_joint
        5,   # robot_arm_length
        6,   # robot_vacuum
        9,   # hook_x
        10,  # hook_y
        11,  # hook_theta
        17,  # hook_width
        18,  # hook_length_side1
        19,  # hook_length_side2
        20,  # movable_button_x
        21,  # movable_button_y
        28,  # movable_button_radius
        29,  # target_button_x
        30,  # target_button_y
        37,  # target_button_radius
    ]


def salient_obs_indices_for_kinder(env_name: str, num_passages: int = 0) -> list[int]:
    """Return decision-critical observation indices for a supported KinDER env."""
    canonical_name = canonicalize_env_name(env_name)
    if canonical_name == "Motion2D":
        return salient_obs_indices_for_motion2d(num_passages)
    if canonical_name == "PushPullHook2D":
        return salient_obs_indices_for_pushpullhook2d()
    raise ValueError(f"Unsupported KinDER env for salient obs indices: {env_name}")
