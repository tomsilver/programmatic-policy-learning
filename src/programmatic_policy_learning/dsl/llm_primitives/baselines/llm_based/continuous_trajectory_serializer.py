"""Serialize expert trajectories for continuous-action environments."""

from __future__ import annotations

import numpy as np

# pylint: disable=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_encoder import (
    ContinuousStateEncoder,
)

# ---------------------------------------------------------------------------
# Relational-fact extraction for continuous 2-D navigation
# ---------------------------------------------------------------------------


def extract_continuous_relational_facts(
    obs: np.ndarray,
    num_passages: int,
) -> list[str]:
    """Derive spatial relationships from a Motion2D observation vector.

    Extracts robot-target and robot-passage geometric facts that help
    the LLM understand the current scene layout.

    Parameters
    ----------
    obs : np.ndarray
        1-D float observation vector from a Motion2D environment.
    num_passages : int
        Number of wall passages; each produces 2 obstacle blocks
        (20 features) in *obs*.

    Returns
    -------
    list[str]
        Human-readable fact strings, e.g.
        ``"robot is left of passage_0 wall (robot_x=0.12 < wall_x=0.50)"``.

    Examples
    --------
    No passages

    >>> obs_p0 = np.zeros(19)
    >>> obs_p0[0], obs_p0[1] = 0.3, 0.5     # robot x, y
    >>> obs_p0[3] = 0.1                       # robot radius
    >>> obs_p0[9], obs_p0[10] = 2.0, 1.0     # target x, y
    >>> obs_p0[17], obs_p0[18] = 0.25, 0.25  # target w, h
    >>> facts = extract_continuous_relational_facts(obs_p0, num_passages=0)
    >>> len(facts)
    4
    >>> facts[0]
    'robot is left of target center (robot_x=0.300 < target_cx=2.125)'
    >>> facts[1]
    'robot is below target center (robot_y=0.500 < target_cy=1.125)'
    >>> facts[2]
    'Manhattan distance to target center = 2.450'
    >>> facts[3]
    'robot is NOT inside target region'

    With one passage (2 obstacles), 3 extra passage-related facts appear.

    >>> obs_p1 = np.zeros(39)
    >>> obs_p1[0], obs_p1[1] = 0.3, 1.0     # robot x, y
    >>> obs_p1[3] = 0.1                       # robot radius
    >>> obs_p1[9], obs_p1[10] = 2.0, 0.5     # target x, y
    >>> obs_p1[17], obs_p1[18] = 0.25, 0.25  # target w, h
    >>> obs_p1[19] = 0.5                      # bottom obstacle x
    >>> obs_p1[20] = 0.0                      # bottom obstacle y
    >>> obs_p1[27] = 0.01                     # bottom obstacle width
    >>> obs_p1[28] = 0.8                      # bottom obstacle height
    >>> obs_p1[29] = 0.5                      # top obstacle x
    >>> obs_p1[30] = 1.7                      # top obstacle y
    >>> obs_p1[37] = 0.01                     # top obstacle width
    >>> obs_p1[38] = 0.8                      # top obstacle height (reaches 2.5)
    >>> facts_p1 = extract_continuous_relational_facts(obs_p1, num_passages=1)
    >>> len(facts_p1)
    7
    >>> facts_p1[4]
    'robot is left of passage_0 wall (robot_x=0.300+r < wall_x=0.500)'
    >>> facts_p1[5]
    'robot y IS aligned with passage_0 gap [0.800+r, 1.700-r] (robot_y=1.000, radius=0.100)'
    >>> facts_p1[6]
    'robot y offset from passage_0 center = -0.250'
    """
    robot_x = float(obs[0])
    robot_y = float(obs[1])
    robot_radius = float(obs[3])
    target_x = float(obs[9])
    target_y = float(obs[10])
    target_width = float(obs[17])
    target_height = float(obs[18])

    target_cx = target_x + target_width / 2.0
    target_cy = target_y + target_height / 2.0

    facts: list[str] = []

    # 1. Robot vs. target center position
    # 1.1 Robot vs. target center x
    if robot_x < target_cx:
        facts.append(
            f"robot is left of target center "
            f"(robot_x={robot_x:.3f} < target_cx={target_cx:.3f})"
        )
    else:
        facts.append(
            f"robot is right of target center "
            f"(robot_x={robot_x:.3f} >= target_cx={target_cx:.3f})"
        )
    # 1.2 Robot vs. target center y
    if robot_y < target_cy:
        facts.append(
            f"robot is below target center "
            f"(robot_y={robot_y:.3f} < target_cy={target_cy:.3f})"
        )
    else:
        facts.append(
            f"robot is above target center "
            f"(robot_y={robot_y:.3f} >= target_cy={target_cy:.3f})"
        )

    dist_to_target = abs(robot_x - target_cx) + abs(robot_y - target_cy)
    facts.append(f"Manhattan distance to target center = {dist_to_target:.3f}")

    # 2. Robot vs. target region overlap check
    # (x, y) is the bottom-left corner of the rectangle.
    target_left = target_x
    target_right = target_x + target_width
    target_bottom = target_y
    target_top = target_y + target_height
    in_target = (
        target_left <= robot_x <= target_right
        and target_bottom <= robot_y <= target_top
    )
    facts.append(f"robot {'IS' if in_target else 'is NOT'} inside target region")

    # 3. Robot vs. each passage
    for i in range(num_passages):
        bot_base = 19 + 20 * i
        top_base = bot_base + 10

        wall_x = float(obs[bot_base])
        bot_y = float(obs[bot_base + 1])
        wall_width = float(obs[bot_base + 8])
        bot_height = float(obs[bot_base + 9])
        top_y = float(obs[top_base + 1])
        gap_bottom = bot_y + bot_height
        gap_top = top_y

        wall_right = wall_x + wall_width
        clear_x = wall_right + robot_radius
        if robot_x + robot_radius < wall_x:
            facts.append(
                f"robot is left of passage_{i} wall "
                f"(robot_x={robot_x:.3f}+r < wall_x={wall_x:.3f})"
            )
        elif robot_x < clear_x:
            facts.append(
                f"robot is inside passage_{i} zone "
                f"(wall_x={wall_x:.3f} <= robot_x={robot_x:.3f} "
                f"< wall_right={wall_right:.3f}+r)"
            )
        else:
            facts.append(
                f"robot has passed passage_{i} wall "
                f"(robot_x={robot_x:.3f} >= wall_right={wall_right:.3f}+r)"
            )

        y_aligned = (gap_bottom + robot_radius) <= robot_y <= (gap_top - robot_radius)
        facts.append(
            f"robot y {'IS' if y_aligned else 'is NOT'} aligned with passage_{i} gap "
            f"[{gap_bottom:.3f}+r, {gap_top:.3f}-r] "
            f"(robot_y={robot_y:.3f}, radius={robot_radius:.3f})"
        )

        passage_center_y = (gap_bottom + gap_top) / 2.0
        y_offset = robot_y - passage_center_y
        facts.append(f"robot y offset from passage_{i} center = {y_offset:+.3f}")

    return facts


# ---------------------------------------------------------------------------
# Trajectory → text  (analogous to trajectory_serializer.trajectory_to_text)
# ---------------------------------------------------------------------------


def trajectory_to_text(
    trajectory: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    encoder: ContinuousStateEncoder,
    num_passages: int,
    encoding_method: str,
    max_steps: int | None = None,
    skip_rate: int = 1,
) -> str:
    """Convert ``(obs, action, obs_next)`` tuples to structured text.

    Produces the trajectory portion of the LLM prompt.  A final
    observation with ``Action: None (terminal state).`` is always
    appended so the LLM can see the goal state.

    Parameters
    ----------
    trajectory : list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of ``(obs_t, action_t, obs_{t+1})`` transitions collected
        from an expert rollout.
    encoder : ContinuousStateEncoder
        Encoder used to render observations and actions as named fields.
    num_passages : int
        Number of wall passages in the environment, forwarded to
        :func:`extract_continuous_relational_facts`.
    encoding_method : str
        ``"1"`` - raw numeric dump.
        ``"2"`` - named-field listing.
        ``"3"`` - named-field + change summary.
        ``"4"`` - named-field + change summary + relational facts.
    max_steps : int | None, optional
        Maximum number of transitions to include (default is ``None``,
        meaning all steps).
    skip_rate : int, optional
        Sub-sampling rate.  When > 1, only every *skip_rate*-th frame
        is kept (plus the true final step), and a header is prepended
        to inform the LLM about the sub-sampling (default is 1).

    Returns
    -------
    str
        Multi-block text representation of the trajectory, with blocks
        separated by blank lines.

    Raises
    ------
    ValueError
        If ``trajectory`` is empty.

    Examples
    --------
    A single-step trajectory in a ``Motion2D-p1`` environment with
    ``encoding_method="4"`` (named fields + change summary + relational
    facts).  The text contains **two** blocks: Step 0 (the transition)
    and Step 1 (the terminal observation).

    >>> from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_encoder import (
    ...     ContinuousStateEncoder, ContinuousStateEncoderConfig,
    ... )
    >>> from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_hint_config import (
    ...     obs_field_names_for_motion2d, ACTION_FIELD_NAMES,
    ...     salient_obs_indices_for_motion2d,
    ... )
    >>> cfg = ContinuousStateEncoderConfig(
    ...     obs_field_names=obs_field_names_for_motion2d(1),
    ...     action_field_names=ACTION_FIELD_NAMES["Motion2D"],
    ...     salient_indices=salient_obs_indices_for_motion2d(1),
    ... )
    >>> enc = ContinuousStateEncoder(cfg)
    >>> obs0 = np.zeros(39)
    >>> obs0[0], obs0[1] = 0.3, 1.0        # robot x, y
    >>> obs0[3] = 0.1                        # robot radius
    >>> obs0[9], obs0[10] = 2.0, 0.5        # target x, y (bottom-left)
    >>> obs0[17], obs0[18] = 0.25, 0.25     # target w, h
    >>> obs0[19] = 0.5; obs0[20] = 0.0      # bottom obstacle x, y
    >>> obs0[27] = 0.01; obs0[28] = 0.8     # bottom obstacle w, h
    >>> obs0[29] = 0.5; obs0[30] = 1.7      # top obstacle x, y
    >>> obs0[37] = 0.01; obs0[38] = 0.8     # top obstacle w, h
    >>> act0 = np.zeros(5)
    >>> act0[0], act0[1] = 0.05, -0.02      # dx, dy
    >>> obs1 = obs0.copy()
    >>> obs1[0] += 0.05; obs1[1] -= 0.02
    >>> traj = [(obs0, act0, obs1)]
    >>> text = trajectory_to_text(
    ...     traj, encoder=enc, num_passages=1, encoding_method="4",
    ... )

    The output text has two blocks separated by blank lines.  Each block
    is printed below (long observation lines are wrapped by the terminal
    but are single lines in the actual string).

    >>> print(text)  # doctest: +NORMALIZE_WHITESPACE
    *** Step 0 ***
    Observation (s_0):
      obs[0] (robot_x)=0.300, obs[1] (robot_y)=1.000,
      obs[3] (robot_base_radius)=0.100,
      obs[9] (target_x)=2.000, obs[10] (target_y)=0.500,
      obs[17] (target_width)=0.250, obs[18] (target_height)=0.250,
      obs[19] (obstacle0_x)=0.500, obs[20] (obstacle0_y)=0.000,
      obs[27] (obstacle0_width)=0.010, obs[28] (obstacle0_height)=0.800,
      obs[29] (obstacle1_x)=0.500, obs[30] (obstacle1_y)=1.700,
      obs[37] (obstacle1_width)=0.010, obs[38] (obstacle1_height)=0.800
    <BLANKLINE>
    Action: [dx=0.050, dy=-0.020, dtheta=0.000, darm=0.000, vac=0.000]
    <BLANKLINE>
    CHANGES SUMMARY:
    - obs[0] (robot_x): 0.300 -> 0.350 (delta=+0.050)
    - obs[1] (robot_y): 1.000 -> 0.980 (delta=-0.020)
    <BLANKLINE>
    Relational Facts:
    - robot is left of target center (robot_x=0.300 < target_cx=2.125)
    - robot is above target center (robot_y=1.000 >= target_cy=0.625)
    - Manhattan distance to target center = 2.200
    - robot is NOT inside target region
    - robot is left of passage_0 wall (robot_x=0.300+r < wall_x=0.500)
    - robot y IS aligned with passage_0 gap
      [0.800+r, 1.700-r] (robot_y=1.000, radius=0.100)
    - robot y offset from passage_0 center = -0.250
    <BLANKLINE>
    *** Step 1 ***
    Observation (s_1):
      obs[0] (robot_x)=0.350, obs[1] (robot_y)=0.980,
      obs[3] (robot_base_radius)=0.100,
      obs[9] (target_x)=2.000, obs[10] (target_y)=0.500,
      obs[17] (target_width)=0.250, obs[18] (target_height)=0.250,
      obs[19] (obstacle0_x)=0.500, obs[20] (obstacle0_y)=0.000,
      obs[27] (obstacle0_width)=0.010, obs[28] (obstacle0_height)=0.800,
      obs[29] (obstacle1_x)=0.500, obs[30] (obstacle1_y)=1.700,
      obs[37] (obstacle1_width)=0.010, obs[38] (obstacle1_height)=0.800
    <BLANKLINE>
    Action: None (terminal state).
    <BLANKLINE>
    Relational Facts:
    - robot is left of target center (robot_x=0.350 < target_cx=2.125)
    - robot is above target center (robot_y=0.980 >= target_cy=0.625)
    - Manhattan distance to target center = 2.130
    - robot is NOT inside target region
    - robot is left of passage_0 wall (robot_x=0.350+r < wall_x=0.500)
    - robot y IS aligned with passage_0 gap
      [0.800+r, 1.700-r] (robot_y=0.980, radius=0.100)
    - robot y offset from passage_0 center = -0.270
    """
    if not trajectory:
        raise ValueError("trajectory must be non-empty")

    # e.g. trajectory = [(obs0, act0, obs1), (obs1, act1, obs2), ...]
    # -> indexed_traj = [(0, (obs0, act0, obs1)), (1, (obs1, act1, obs2)), ...]
    indexed_traj = list(enumerate(trajectory))

    if skip_rate > 1:
        sampled = indexed_traj[::skip_rate]
        if max_steps:
            sampled = sampled[: max_steps - 1]
        if sampled[-1][0] != indexed_traj[-1][0]:
            sampled.append(indexed_traj[-1])
        steps_with_idx = sampled
        header = (
            f"[NOTE: Trajectory sub-sampled. Showing 1 in every {skip_rate} steps, "
            f"plus the final terminal step.]\n\n"
        )
    else:
        steps_with_idx = indexed_traj[:max_steps] if max_steps else indexed_traj
        header = ""

    blocks: list[str] = []

    for original_idx, (obs_t, action, obs_t1) in steps_with_idx:
        if encoding_method == "1":
            blocks.append(
                f"*** Step {original_idx} ***\n"
                f"obs = {np.array2string(obs_t, precision=4, separator=', ')}\n"
                f"action = {np.array2string(action, precision=4, separator=', ')}"
            )

        elif encoding_method == "2":
            blocks.append(encoder.encode_step(obs_t, action, original_idx))

        elif encoding_method == "3":
            blocks.append(encoder.encode_step(obs_t, action, original_idx))
            changes = encoder.compute_deltas(obs_t, obs_t1)
            if changes:
                blocks.append(
                    "CHANGES SUMMARY:\n" + "\n".join(f"- {c}" for c in changes)
                )
            else:
                blocks.append("CHANGES SUMMARY:\n- (no changes)")

        elif encoding_method == "4":
            blocks.append(encoder.encode_step(obs_t, action, original_idx))

            changes = encoder.compute_deltas(obs_t, obs_t1)
            if changes:
                blocks.append(
                    "CHANGES SUMMARY:\n" + "\n".join(f"- {c}" for c in changes)
                )
            else:
                blocks.append("CHANGES SUMMARY:\n- (no changes)")

            rel_facts = extract_continuous_relational_facts(obs_t, num_passages)
            blocks.append(
                "Relational Facts:\n" + "\n".join(f"- {f}" for f in rel_facts)
            )

    # Final observation (obs_next of the last transition)
    last_original_idx, (_, _, final_obs) = steps_with_idx[-1]
    final_display_idx = last_original_idx + 1
    if encoding_method == "1":
        blocks.append(
            f"*** Step {final_display_idx} ***\n"
            f"obs = {np.array2string(final_obs, precision=4, separator=', ')}\n"
            f"action = None (terminal state)"
        )
    else:  # TODOO: later clean this up to assert if the encoding is not available
        blocks.append(encoder.encode_obs(final_obs, final_display_idx))
        blocks.append("Action: None (terminal state).")
        if encoding_method == "4":
            rel_facts = extract_continuous_relational_facts(final_obs, num_passages)
            blocks.append(
                "Relational Facts:\n" + "\n".join(f"- {f}" for f in rel_facts)
            )

    return header + "\n\n".join(blocks)
