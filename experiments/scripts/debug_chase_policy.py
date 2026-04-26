"""Debug and evaluate a handwritten CaP-style policy on Chase.

The script reports both episode success and step-level agreement with
the hand-coded Chase expert.  It also prints the policy's inferred phase
and internal state so mistakes are easier to localize.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import OmegaConf

PIL_IMAGE: Any | None = None
PIL_IMAGE_DRAW: Any | None = None
PIL_IMAGE_FONT: Any | None = None
IMAGEIO: Any | None = None

try:
    from PIL import Image as _PIL_IMAGE  # type: ignore
    from PIL import ImageDraw as _PIL_IMAGE_DRAW  # type: ignore
    from PIL import ImageFont as _PIL_IMAGE_FONT  # type: ignore

    PIL_IMAGE = _PIL_IMAGE
    PIL_IMAGE_DRAW = _PIL_IMAGE_DRAW
    PIL_IMAGE_FONT = _PIL_IMAGE_FONT
except Exception:  # pylint: disable=broad-exception-caught
    pass

try:
    import imageio.v2 as _IMAGEIO  # type: ignore

    IMAGEIO = _IMAGEIO
except Exception:  # pylint: disable=broad-exception-caught
    pass

from programmatic_policy_learning.approaches.experts.grid_experts import (
    get_grid_expert,
)
from programmatic_policy_learning.envs.registry import EnvRegistry

os.environ.setdefault(
    "MPLCONFIGDIR", str((Path(".pytest_cache") / "matplotlib").resolve())
)

POLICY_DEBUG = True
TOKEN_COLORS = {
    "empty": (245, 245, 245),
    "target": (255, 214, 102),
    "agent": (76, 154, 255),
    "wall": (55, 65, 81),
    "drawn": (239, 68, 68),
    "left_arrow": (34, 197, 94),
    "right_arrow": (34, 197, 94),
    "up_arrow": (34, 197, 94),
    "down_arrow": (34, 197, 94),
}


def _parse_env_nums(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        return []
    if "-" in spec:
        start_s, end_s = spec.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise ValueError("env range end must be >= start")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _token_name(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def _action_tuple(action: Any) -> tuple[int, int]:
    arr = np.asarray(action).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Expected 2D grid action, got {action!r}")
    return int(arr[0]), int(arr[1])


def _print_grid(obs: np.ndarray) -> None:
    for row in obs:
        print(" ".join(f"{_token_name(v):>11}" for v in row))


def _find_token(obs: np.ndarray, token_name: str) -> tuple[int, int] | None:
    for r in range(obs.shape[0]):
        for c in range(obs.shape[1]):
            if _token_name(obs[r, c]) == token_name:
                return (r, c)
    return None


def _derive_debug(obs: np.ndarray, action: tuple[int, int]) -> dict[str, Any]:
    clicked = None
    if 0 <= action[0] < obs.shape[0] and 0 <= action[1] < obs.shape[1]:
        clicked = obs[action[0], action[1]]

    marker_positions = []
    for r in range(obs.shape[0]):
        for c in range(obs.shape[1]):
            if _token_name(obs[r, c]) == "drawn":
                marker_positions.append((r, c))

    if marker_positions:
        phase = "navigate"
    elif action == (0, 0):
        phase = "wait_for_goal_corner"
    else:
        phase = "place_marker"

    return {
        "phase": phase,
        "clicked": _token_name(clicked),
        "agent_pos": _find_token(obs, "agent"),
        "goal_pos": _find_token(obs, "target"),
        "marker_vals": ["drawn"] if marker_positions else None,
        "desired": None,
        "chosen_control": (
            _token_name(clicked) if "arrow" in _token_name(clicked) else None
        ),
        "ctl_mv": getattr(policy, "_st", {}).get("ctl_mv", None),
    }


def _grid_to_frame(
    obs: np.ndarray,
    *,
    title: str,
    action: tuple[int, int] | None = None,
    expert_action: tuple[int, int] | None = None,
    cell_size: int = 48,
) -> np.ndarray:
    h, w = obs.shape
    top = 72
    frame = np.full((top + h * cell_size, w * cell_size, 3), 255, dtype=np.uint8)

    for r in range(h):
        for c in range(w):
            name = _token_name(obs[r, c])
            color = TOKEN_COLORS.get(name, (180, 180, 180))
            y0 = top + r * cell_size
            y1 = top + (r + 1) * cell_size
            x0 = c * cell_size
            x1 = (c + 1) * cell_size
            frame[y0:y1, x0:x1] = color
            frame[y0 : y0 + 2, x0:x1] = 20
            frame[y1 - 2 : y1, x0:x1] = 20
            frame[y0:y1, x0 : x0 + 2] = 20
            frame[y0:y1, x1 - 2 : x1] = 20

    def outline(cell: tuple[int, int], color: tuple[int, int, int], width: int) -> None:
        rr, cc = cell
        if rr < 0 or rr >= h or cc < 0 or cc >= w:
            return
        y0 = top + rr * cell_size
        y1 = top + (rr + 1) * cell_size
        x0 = cc * cell_size
        x1 = (cc + 1) * cell_size
        frame[y0 : y0 + width, x0:x1] = color
        frame[y1 - width : y1, x0:x1] = color
        frame[y0:y1, x0 : x0 + width] = color
        frame[y0:y1, x1 - width : x1] = color

    if expert_action is not None:
        outline(expert_action, (168, 85, 247), 4)
    if action is not None:
        outline(action, (0, 0, 0), 6)

    try:
        if PIL_IMAGE is None or PIL_IMAGE_DRAW is None or PIL_IMAGE_FONT is None:
            raise RuntimeError("Pillow is not available.")
        image_mod = cast(Any, PIL_IMAGE)
        image_draw_mod = cast(Any, PIL_IMAGE_DRAW)
        image_font_mod = cast(Any, PIL_IMAGE_FONT)
        image = image_mod.fromarray(frame)
        draw = image_draw_mod.Draw(image)
        font = image_font_mod.load_default()
        draw.rectangle((0, 0, w * cell_size, top), fill=(255, 255, 255))
        draw.text((10, 8), title, fill=(0, 0, 0), font=font)
        draw.text(
            (10, 30),
            "black=policy/chosen action, purple=expert action",
            fill=(50, 50, 50),
            font=font,
        )
        return np.asarray(image)
    except Exception:  # pylint: disable=broad-exception-caught
        return frame


def _save_mp4(frames: list[np.ndarray], path: Path, fps: int = 4) -> None:
    if not frames:
        raise ValueError(f"No frames to save for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if IMAGEIO is None:
        raise RuntimeError("Saving MP4 files requires imageio.")
    imageio_mod = cast(Any, IMAGEIO)
    imageio_mod.mimsave(path, cast(list[Any], frames), fps=fps, macro_block_size=1)


def reset_policy_state() -> None:
    """Clear state accumulated by ``policy`` between Chase instances."""
    if hasattr(policy, "_st"):
        delattr(policy, "_st")
    if hasattr(policy, "_last_debug"):
        delattr(policy, "_last_debug")


def policy(obs: np.ndarray) -> tuple[int, int]:
    """Return the handwritten Chase action for a grid observation.

    The policy first tries to advance the target, then marks a stopping
    cell, and finally routes the agent toward the target or marker via
    the directional arrow tokens.
    """
    # Token mapping
    # 0: empty, 1: wall, 2: agent, 3: target, 4: drawn,
    # 5: right_arrow, 6: left_arrow, 7: up_arrow, 8: down_arrow

    arr = obs
    # Find all positions
    target_pos = None
    agent_pos = None
    right_arrow = None
    left_arrow = None
    up_arrow = None
    down_arrow = None
    drawn_pos = None
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            v = arr[r, c]
            if v == 2:
                agent_pos = (r, c)
            elif v == 3:
                target_pos = (r, c)
            elif v == 4:
                drawn_pos = (r, c)
            elif v == 5:
                right_arrow = (r, c)
            elif v == 6:
                left_arrow = (r, c)
            elif v == 7:
                up_arrow = (r, c)
            elif v == 8:
                down_arrow = (r, c)
    # Step 1: Move target in increasing column direction if possible
    if target_pos is not None:
        r, c = target_pos
        if c + 1 < arr.shape[1] and arr[r, c + 1] == 0:
            return (0, 0)
        # If not, try decreasing column direction (for Trajectory 1)
        if c - 1 >= 0 and arr[r, c - 1] == 0:
            return (0, 0)
        # If not, try increasing row direction (for Trajectory 2)
        if r + 1 < arr.shape[0] and arr[r + 1, c] == 0:
            return (0, 0)
        # If not, try decreasing row direction
        if r - 1 >= 0 and arr[r - 1, c] == 0:
            return (0, 0)
        # Step 2: Draw a marker in the empty cell adjacent to the target,
        # preferring the direction opposite the movement attempt.
        # Try to find an empty cell adjacent to the target (prior direction)
        # Try left
        if c - 1 >= 0 and arr[r, c - 1] == 0:
            return (r, c - 1)
        # Try right
        if c + 1 < arr.shape[1] and arr[r, c + 1] == 0:
            return (r, c + 1)
        # Try up
        if r - 1 >= 0 and arr[r - 1, c] == 0:
            return (r - 1, c)
        # Try down
        if r + 1 < arr.shape[0] and arr[r + 1, c] == 0:
            return (r + 1, c)
    # Step 3: Move agent toward target using arrows
    if agent_pos is not None and target_pos is not None:
        ar, ac = agent_pos
        tr, tc = target_pos
        # If agent is not in same row as target, move along row
        if ar < tr and down_arrow is not None:
            return down_arrow
        if ar > tr and up_arrow is not None:
            return up_arrow
        # If agent is not in same column as target, move along column
        if ac < tc and right_arrow is not None:
            return right_arrow
        if ac > tc and left_arrow is not None:
            return left_arrow
        # If agent is at target, do nothing (but must return something)
        return (ar, ac)
    # If agent is present but no target, move toward drawn if present
    if agent_pos is not None and drawn_pos is not None:
        ar, ac = agent_pos
        dr, dc = drawn_pos
        if ar < dr and down_arrow is not None:
            return down_arrow
        if ar > dr and up_arrow is not None:
            return up_arrow
        if ac < dc and right_arrow is not None:
            return right_arrow
        if ac > dc and left_arrow is not None:
            return left_arrow
        return (ar, ac)
    # Fallback: click top-left
    return (0, 0)


def run_episode(
    env: Any,
    env_num: int,
    *,
    expert: Any,
    max_steps: int,
    print_grid: bool,
    stop_on_mismatch: bool,
    video_path: Path | None = None,
) -> dict[str, Any]:
    """Run one Chase episode with policy/expert comparison and logging."""
    reset_policy_state()
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    total_reward = 0.0
    matches = 0
    mismatches = 0
    invalid_actions = 0
    frames: list[np.ndarray] = []

    print(f"\n=== Chase env {env_num} ===")
    if print_grid:
        _print_grid(obs)

    for step in range(max_steps):
        expert_action = _action_tuple(expert(obs))
        action = _action_tuple(policy(obs))
        dbg = getattr(policy, "_last_debug", {})
        if not dbg:
            dbg = _derive_debug(obs, action)

        h, w = obs.shape
        valid = 0 <= action[0] < h and 0 <= action[1] < w
        exact_match = action == expert_action
        matches += int(exact_match)
        mismatches += int(not exact_match)
        invalid_actions += int(not valid)

        print(
            f"step={step:03d} phase={dbg.get('phase')} "
            f"action={action} clicked={dbg.get('clicked')} "
            f"expert={expert_action} match={exact_match} valid={valid}"
        )
        print(
            f"    agent={dbg.get('agent_pos')} goal={dbg.get('goal_pos')} "
            f"marker={dbg.get('marker_vals')} desired={dbg.get('desired')} "
            f"chosen_control={dbg.get('chosen_control')} "
            f"ctl_mv={dbg.get('ctl_mv')}"
        )
        if dbg.get("inferred_identity") is not None:
            print(f"    inferred_identity={dbg.get('inferred_identity')}")
        if dbg.get("learned_move") is not None:
            print(f"    learned_move={dbg.get('learned_move')}")

        if video_path is not None:
            frames.append(
                _grid_to_frame(
                    obs,
                    title=(
                        f"policy env={env_num} step={step} "
                        f"phase={dbg.get('phase')} match={exact_match}"
                    ),
                    action=action,
                    expert_action=expert_action,
                )
            )

        if stop_on_mismatch and not exact_match:
            print("    stopping early because action differs from expert")
            break

        obs_next, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        print(
            f"    reward={reward} terminated={terminated} "
            f"truncated={truncated} total_reward={total_reward}"
        )
        obs = obs_next
        if print_grid:
            _print_grid(obs)

        if terminated or truncated:
            break

    if video_path is not None:
        frames.append(
            _grid_to_frame(
                obs,
                title=f"policy env={env_num} final total_reward={total_reward}",
            )
        )
        _save_mp4(frames, video_path)
        print(f"saved policy video: {video_path}")

    success = total_reward > 0
    summary = {
        "env_num": env_num,
        "success": success,
        "total_reward": total_reward,
        "matches": matches,
        "mismatches": mismatches,
        "invalid_actions": invalid_actions,
    }
    print(
        f"summary env={env_num}: success={success} total_reward={total_reward} "
        f"matches={matches} mismatches={mismatches} invalid={invalid_actions}"
    )
    return summary


def record_expert_episode(
    env: Any,
    env_num: int,
    *,
    expert: Any,
    max_steps: int,
    video_path: Path,
) -> dict[str, Any]:
    """Record an expert-only Chase rollout as an MP4 and summary."""
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    frames: list[np.ndarray] = []
    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        action = _action_tuple(expert(obs))
        frames.append(
            _grid_to_frame(
                obs,
                title=f"expert env={env_num} step={step}",
                action=action,
            )
        )
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        steps = step + 1
        if terminated or truncated:
            break

    frames.append(
        _grid_to_frame(
            obs,
            title=f"expert env={env_num} final total_reward={total_reward}",
        )
    )
    _save_mp4(frames, video_path)
    print(f"saved expert video: {video_path}")
    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": total_reward > 0,
    }


def main() -> None:
    """Parse arguments and evaluate the handwritten Chase policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-nums",
        default="0-4",
        help="e.g. '0-4' or '0,3,7'",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-grid", action="store_true")
    parser.add_argument("--policy-debug", action="store_true")
    parser.add_argument("--stop-on-mismatch", action="store_true")
    parser.add_argument(
        "--video-dir",
        default="logs/chase_policy_debug_videos",
        help="Directory for expert and policy MP4 files.",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Disable MP4 recording and only print step-level debug output.",
    )
    args = parser.parse_args()

    global POLICY_DEBUG  # pylint: disable=global-statement
    POLICY_DEBUG = args.policy_debug

    random.seed(args.seed)
    np.random.seed(args.seed)

    env_cfg_path = Path("experiments/conf/env/ggg_chase.yaml")
    env_cfg = OmegaConf.load(env_cfg_path)
    registry = EnvRegistry()
    expert = get_grid_expert("Chase")
    env_nums = _parse_env_nums(args.env_nums)
    video_dir = Path(args.video_dir)

    summaries = []
    for env_num in env_nums:
        if not args.no_videos:
            expert_env = registry.load(env_cfg, instance_num=env_num)
            record_expert_episode(
                expert_env,
                env_num,
                expert=expert,
                max_steps=args.max_steps,
                video_path=video_dir / f"chase_{env_num:03d}_expert.mp4",
            )

        env = registry.load(env_cfg, instance_num=env_num)
        summaries.append(
            run_episode(
                env,
                env_num,
                expert=expert,
                max_steps=args.max_steps,
                print_grid=args.print_grid,
                stop_on_mismatch=args.stop_on_mismatch,
                video_path=(
                    None
                    if args.no_videos
                    else video_dir / f"chase_{env_num:03d}_policy.mp4"
                ),
            )
        )

    successes = sum(int(s["success"]) for s in summaries)
    total = len(summaries)
    total_matches = sum(int(s["matches"]) for s in summaries)
    total_mismatches = sum(int(s["mismatches"]) for s in summaries)
    total_invalid = sum(int(s["invalid_actions"]) for s in summaries)
    print("\n=== aggregate ===")
    print(f"successes={successes}/{total}")
    print(f"exact_expert_matches={total_matches}")
    print(f"expert_mismatches={total_mismatches}")
    print(f"invalid_actions={total_invalid}")


if __name__ == "__main__":
    main()
