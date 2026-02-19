"""Expert policies for Motion2D environments.

Provides:

- ``Motion2DRejectionSamplingExpert``: a callable expert that uses rejection
  sampling with feature predicates ``f_1 ∧ f_2`` to navigate through wall
  passages.
- ``create_motion2d_expert``: factory that builds the above given an
  action space.
- ``f_1``, ``f_2``: boolean feature predicates used by the expert and by
  LPP program evaluation.
"""

from typing import Callable

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

Obs = NDArray[np.float32]
Act = NDArray[np.float32]
FeatureFn = Callable[[Obs, Act], bool]

_PROGRESS_THRESHOLD = 0.005

# ---------------------------------------------------------------------------
# Feature helpers  (used by f_1 / f_2)
# ---------------------------------------------------------------------------


def _is_y_aligned(
    robot_y: float,
    robot_radius: float,
    gap_bottom: float,
    gap_top: float,
) -> bool:
    """Check whether the robot can physically fit through the passage gap.

    The robot is a circle of ``robot_radius``.  For it to pass through
    the gap ``[gap_bottom, gap_top]`` without collision, its center must
    satisfy ``gap_bottom + robot_radius <= robot_y <= gap_top - robot_radius``.
    """
    return (gap_bottom + robot_radius) <= robot_y <= (gap_top - robot_radius)


def _find_next_passage(
    s: Obs,
) -> tuple[float, float, float, float] | None:
    """Find the next wall/passage the robot has not yet passed.

    Parses all passages from the observation vector and returns the
    first wall (by x-coordinate) that the robot is still to the left of.

    Obs layout per obstacle (10 features each, starting at index 19)::

        [x, y, theta, static, color_r, color_g, color_b, z_order, width, height]

    Each passage ``i`` consists of two obstacles::

        bottom = obstacle ``2*i``  (obs index ``19 + 20*i``)
        top    = obstacle ``2*i+1`` (obs index ``19 + 20*i + 10``)

    ``(x, y)`` is the bottom-edge center of each rectangle, so the gap
    runs from ``bottom.y + bottom.height`` up to ``top.y``.

    Returns
    -------
    ``(wall_x, passage_y, gap_bottom, gap_top)`` for the next wall ahead,
    or ``None`` if the robot has passed all walls (or there are none).
    """
    robot_x = float(s[0])
    num_obstacles = (len(s) - 19) // 10
    num_passages = num_obstacles // 2

    if num_passages == 0:
        return None

    passages: list[tuple[float, float, float, float]] = []
    for i in range(num_passages):
        bot_base = 19 + 20 * i
        top_base = bot_base + 10
        wall_x = float(s[bot_base])
        gap_bottom = float(s[bot_base + 1] + s[bot_base + 9])
        gap_top = float(s[top_base + 1])
        passage_y = (gap_bottom + gap_top) / 2.0
        passages.append((wall_x, passage_y, gap_bottom, gap_top))

    passages.sort(key=lambda p: p[0])

    for wall_x, passage_y, gap_bottom, gap_top in passages:
        if robot_x < wall_x:
            return (wall_x, passage_y, gap_bottom, gap_top)

    return None


# ---------------------------------------------------------------------------
# Feature predicates  (obs, action → bool) for LPP rejection sampling
# ---------------------------------------------------------------------------


def f_1(s: Obs, a: Act) -> bool:
    """Align y with the next passage gap before approaching the wall.

    - If no wall ahead: ``True`` (no alignment needed).
    - If robot y is already within the passable range
      ``[gap_bottom + radius, gap_top - radius]``: ``True`` (aligned).
    - Otherwise: accept only actions whose dy component moves the robot
      y closer to the passage center.  No constraint on dx so the robot
      can back away from the wall if needed.
    """
    passage = _find_next_passage(s)
    if passage is None:
        return True

    _wall_x, passage_y, gap_bottom, gap_top = passage
    robot_y = float(s[1])
    robot_radius = float(s[3])

    if _is_y_aligned(robot_y, robot_radius, gap_bottom, gap_top):
        return True

    dy = float(a[1])
    next_y = robot_y + dy
    return abs(next_y - passage_y) < abs(robot_y - passage_y)


def f_2(s: Obs, a: Act) -> bool:
    """Make forward progress through the current passage or toward the target.

    - If no wall ahead: accept actions that reduce Manhattan distance to
      the target.
    - If wall ahead but y is NOT aligned with the gap: ``True``
      (let ``f_1`` handle alignment).
    - If wall ahead and y IS aligned: accept actions that reduce
      Manhattan distance to a waypoint just past the passage.
    """
    robot_x = float(s[0])
    robot_y = float(s[1])
    robot_radius = float(s[3])
    dx = float(a[0])
    dy = float(a[1])
    next_x = robot_x + dx
    next_y = robot_y + dy

    passage = _find_next_passage(s)

    if passage is None:
        target_x = float(s[9])
        target_y = float(s[10])
        dist_before = abs(robot_x - target_x) + abs(robot_y - target_y)
        dist_after = abs(next_x - target_x) + abs(next_y - target_y)
        return dist_after < dist_before - _PROGRESS_THRESHOLD

    wall_x, passage_y, gap_bottom, gap_top = passage

    if not _is_y_aligned(robot_y, robot_radius, gap_bottom, gap_top):
        return True

    waypoint_x = wall_x + robot_radius + 0.05
    waypoint_y = passage_y
    dist_before = abs(robot_x - waypoint_x) + abs(robot_y - waypoint_y)
    dist_after = abs(next_x - waypoint_x) + abs(next_y - waypoint_y)
    return dist_after < dist_before - _PROGRESS_THRESHOLD


# ---------------------------------------------------------------------------
# Expert policy  (obs → action via rejection sampling)
# ---------------------------------------------------------------------------


class Motion2DRejectionSamplingExpert:
    """Expert that samples random actions and accepts those satisfying f_1 ∧ f_2.

    Parameters
    ----------
    action_space
        The continuous Box action space to sample from.
    seed
        RNG seed for reproducible action sampling.
    max_samples
        Maximum rejection-sampling attempts per step before falling back
        to the last sampled action.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        seed: int = 0,
        max_samples: int = 2000,
    ) -> None:
        assert isinstance(action_space, gym.spaces.Box)
        self._action_space = action_space
        self._action_space.seed(seed)
        self._max_samples = max_samples

    def __call__(self, obs: Obs) -> Act:
        action = self._action_space.sample()
        for _ in range(self._max_samples):
            action = self._action_space.sample()
            if f_1(obs, action) and f_2(obs, action):
                return action
        return action


def create_motion2d_expert(
    action_space: gym.spaces.Box,
    seed: int = 0,
    max_samples: int = 2000,
) -> Motion2DRejectionSamplingExpert:
    """Factory that builds a rejection-sampling expert for Motion2D."""
    return Motion2DRejectionSamplingExpert(action_space, seed, max_samples)
