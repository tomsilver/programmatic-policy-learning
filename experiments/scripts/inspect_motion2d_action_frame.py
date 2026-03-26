"""Probe Motion2D action semantics in a real KinDER environment.

This script helps answer:

1. Are ``dx`` and ``dy`` interpreted in the world frame or relative to
   the robot heading?
2. Where is the robot "head" given ``theta``?
3. Do we need to change ``theta`` just to move from A to B?

It does this by applying the same translational action before and after a
pure rotation from the same reset state, then comparing the resulting
position deltas.

Usage::

    python experiments/scripts/inspect_motion2d_action_frame.py
    python experiments/scripts/inspect_motion2d_action_frame.py --passages 1 --seed 0
    python experiments/scripts/inspect_motion2d_action_frame.py --move-step 0.05 --turn-step 0.5
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import gymnasium as gym
import kinder
import numpy as np


ACTION_FIELD_NAMES = ("dx", "dy", "dtheta", "darm", "vac")


@dataclass(frozen=True)
class PoseSnapshot:
    """Robot pose summary extracted from the observation."""

    x: float
    y: float
    theta: float
    radius: float

    @property
    def inferred_head(self) -> tuple[float, float]:
        """Approximate a head/tip point from center + radius * heading vector."""
        return (
            self.x + self.radius * math.cos(self.theta),
            self.y + self.radius * math.sin(self.theta),
        )


@dataclass(frozen=True)
class StepProbe:
    """Result of one probe rollout."""

    name: str
    pre_action: np.ndarray | None
    action: np.ndarray
    before: PoseSnapshot
    after: PoseSnapshot
    terminated: bool
    truncated: bool

    @property
    def delta(self) -> np.ndarray:
        return np.array(
            [
                self.after.x - self.before.x,
                self.after.y - self.before.y,
                self.after.theta - self.before.theta,
            ],
            dtype=float,
        )


def _pose_from_obs(obs: np.ndarray) -> PoseSnapshot:
    return PoseSnapshot(
        x=float(obs[0]),
        y=float(obs[1]),
        theta=float(obs[2]),
        radius=float(obs[3]),
    )


def _format_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):+.4f}" for v in vec.tolist()) + "]"


def _format_pose(pose: PoseSnapshot) -> str:
    head_x, head_y = pose.inferred_head
    return (
        f"x={pose.x:+.4f}, y={pose.y:+.4f}, theta={pose.theta:+.4f}, "
        f"radius={pose.radius:.4f}, inferred_head=({head_x:+.4f}, {head_y:+.4f})"
    )


def _movement_angle(delta_xy: np.ndarray) -> float | None:
    if np.linalg.norm(delta_xy) < 1e-8:
        return None
    return float(math.atan2(float(delta_xy[1]), float(delta_xy[0])))


def _angle_diff(a: float, b: float) -> float:
    raw = a - b
    return float(math.atan2(math.sin(raw), math.cos(raw)))


def _build_action(
    dx: float = 0.0,
    dy: float = 0.0,
    dtheta: float = 0.0,
    darm: float = 0.0,
    vac: float = 0.0,
) -> np.ndarray:
    return np.array([dx, dy, dtheta, darm, vac], dtype=np.float32)


def _clip_action(action: np.ndarray, action_space: gym.spaces.Box) -> np.ndarray:
    return np.clip(action, action_space.low, action_space.high).astype(np.float32)


def _single_probe(
    env: gym.Env,
    *,
    seed: int,
    name: str,
    action: np.ndarray,
    pre_action: np.ndarray | None = None,
) -> StepProbe:
    obs, _ = env.reset(seed=seed)
    if pre_action is not None:
        obs, _reward, terminated, truncated, _info = env.step(pre_action)
        if terminated or truncated:
            raise RuntimeError(
                f"Probe {name!r} terminated during pre_action; choose a smaller step."
            )

    before = _pose_from_obs(np.asarray(obs, dtype=float))
    next_obs, _reward, terminated, truncated, _info = env.step(action)
    after = _pose_from_obs(np.asarray(next_obs, dtype=float))
    return StepProbe(
        name=name,
        pre_action=pre_action,
        action=action,
        before=before,
        after=after,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def _default_step(bounds_low: np.ndarray, bounds_high: np.ndarray, idx: int) -> float:
    limit = min(abs(float(bounds_low[idx])), abs(float(bounds_high[idx])))
    return 0.25 * limit


def _infer_frame(
    rotate_probe: StepProbe,
    dx_probe: StepProbe,
    dx_after_turn_probe: StepProbe,
    dy_probe: StepProbe,
    dy_after_turn_probe: StepProbe,
) -> str:
    turn_delta = rotate_probe.delta[2]
    dx_base = dx_probe.delta[:2]
    dx_rot = dx_after_turn_probe.delta[:2]
    dy_base = dy_probe.delta[:2]
    dy_rot = dy_after_turn_probe.delta[:2]

    dx_base_angle = _movement_angle(dx_base)
    dx_rot_angle = _movement_angle(dx_rot)
    dy_base_angle = _movement_angle(dy_base)
    dy_rot_angle = _movement_angle(dy_rot)

    if None in (dx_base_angle, dx_rot_angle, dy_base_angle, dy_rot_angle):
        return "Could not infer frame because one of the translation probes produced ~zero motion."

    dx_angle_change = abs(_angle_diff(dx_rot_angle, dx_base_angle))
    dy_angle_change = abs(_angle_diff(dy_rot_angle, dy_base_angle))
    turn_mag = abs(float(turn_delta))

    if dx_angle_change < 0.15 and dy_angle_change < 0.15:
        return (
            "Inference: dx/dy look absolute in the world frame. The motion "
            "direction stayed nearly unchanged after rotating theta first."
        )

    if abs(dx_angle_change - turn_mag) < 0.2 and abs(dy_angle_change - turn_mag) < 0.2:
        return (
            "Inference: dx/dy look relative to the robot heading. The motion "
            "direction changed by about the same amount as the prior rotation."
        )

    return (
        "Inference: mixed or ambiguous result. Inspect the printed deltas; the "
        "environment may clip, constrain, or partially couple translation and rotation."
    )


def _print_probe(probe: StepProbe) -> None:
    print(f"\n== {probe.name} ==")
    if probe.pre_action is not None:
        print(f"pre_action: {_format_vec(probe.pre_action)}")
    print(f"action:     {_format_vec(probe.action)}")
    print(f"before:     {_format_pose(probe.before)}")
    print(f"after:      {_format_pose(probe.after)}")
    print(
        "delta:      "
        f"dx_world={probe.delta[0]:+.4f}, dy_world={probe.delta[1]:+.4f}, "
        f"dtheta_obs={probe.delta[2]:+.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect whether Motion2D dx/dy are absolute or heading-relative."
    )
    parser.add_argument("--passages", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--move-step",
        type=float,
        default=None,
        help="Translation magnitude for dx/dy probes. Defaults to 25%% of bound.",
    )
    parser.add_argument(
        "--turn-step",
        type=float,
        default=None,
        help="Rotation magnitude for dtheta probe. Defaults to 25%% of bound.",
    )
    args = parser.parse_args()

    env_id = f"kinder/Motion2D-p{args.passages}-v0"
    kinder.register_all_environments()
    env = kinder.make(env_id, render_mode="rgb_array")

    try:
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError(f"Expected Box action space, got {type(env.action_space)!r}")

        action_space = env.action_space
        move_step = (
            float(args.move_step)
            if args.move_step is not None
            else _default_step(action_space.low, action_space.high, 0)
        )
        turn_step = (
            float(args.turn_step)
            if args.turn_step is not None
            else _default_step(action_space.low, action_space.high, 2)
        )

        print(f"Env: {env_id}")
        print(f"Action space: {action_space}")
        for idx, name in enumerate(ACTION_FIELD_NAMES):
            print(
                f"  [{idx}] {name}: low={float(action_space.low[idx]):+.4f}, "
                f"high={float(action_space.high[idx]):+.4f}"
            )

        initial_obs, _ = env.reset(seed=args.seed)
        initial_pose = _pose_from_obs(np.asarray(initial_obs, dtype=float))
        print("\nInitial robot pose")
        print(_format_pose(initial_pose))
        print(
            "Head note: the observation gives the robot center and theta, not an explicit "
            "head point. This script infers a front tip as center + radius * [cos(theta), sin(theta)]."
        )

        rotate = _clip_action(_build_action(dtheta=turn_step), action_space)
        dx_action = _clip_action(_build_action(dx=move_step), action_space)
        dy_action = _clip_action(_build_action(dy=move_step), action_space)

        rotate_probe = _single_probe(
            env,
            seed=args.seed,
            name="rotate_only",
            action=rotate,
        )
        dx_probe = _single_probe(
            env,
            seed=args.seed,
            name="dx_only_from_reset",
            action=dx_action,
        )
        dx_after_turn_probe = _single_probe(
            env,
            seed=args.seed,
            name="dx_after_rotate",
            pre_action=rotate,
            action=dx_action,
        )
        dy_probe = _single_probe(
            env,
            seed=args.seed,
            name="dy_only_from_reset",
            action=dy_action,
        )
        dy_after_turn_probe = _single_probe(
            env,
            seed=args.seed,
            name="dy_after_rotate",
            pre_action=rotate,
            action=dy_action,
        )

        for probe in (
            rotate_probe,
            dx_probe,
            dx_after_turn_probe,
            dy_probe,
            dy_after_turn_probe,
        ):
            _print_probe(probe)

        print("\nInterpretation")
        print(_infer_frame(
            rotate_probe,
            dx_probe,
            dx_after_turn_probe,
            dy_probe,
            dy_after_turn_probe,
        ))
        print(
            "Theta note: for pure navigation, you usually should not need dtheta if dx/dy already "
            "move the base toward the target. Changing theta only matters if the environment or a "
            "policy chooses heading-dependent behavior."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
