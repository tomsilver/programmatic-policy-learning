"""Collect manual ARC-AGI-3 demonstrations in the repo-native demo format."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from arc_agi import OperationMode
from arcengine import GameAction

from programmatic_policy_learning.data.demo_io import DemoRecord, save_demo_record
from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.envs.arc_agi3 import (
    ArcAgi3GymEnv,
    Ls20InitialStateVariant,
    coerce_action,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect manual ARC-AGI-3 demonstrations."
    )
    parser.add_argument("--game", default="ls20", help="ARC game id")
    parser.add_argument(
        "--seeds",
        default="0",
        help="comma-separated seeds or inclusive range, e.g. 0,1,2 or 0..2",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="manual_demos")
    parser.add_argument(
        "--demo-name",
        default=None,
        help=(
            "optional filename stem for the saved demo; useful when collecting "
            "multiple demos for the same game and seed"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a seed when its target demo pickle already exists",
    )
    parser.add_argument(
        "--mode",
        default=OperationMode.NORMAL.value,
        choices=[OperationMode.NORMAL.value, OperationMode.OFFLINE.value],
    )
    parser.add_argument("--offline", action="store_true", help="shortcut for offline")
    parser.add_argument("--env-dir", default="environment_files")
    parser.add_argument("--recordings-dir", default="recordings")
    parser.add_argument(
        "--randomize-initial-state",
        action="store_true",
        help=(
            "sample a deterministic local ls20 level-0 player/switch variant "
            "from each seed; seed 0 keeps the official original start"
        ),
    )
    parser.add_argument(
        "--randomize-shape",
        action="store_true",
        help=(
            "Stage-B option: with --randomize-initial-state, also randomize "
            "the shared target/reference shape for seeds 1 and above"
        ),
    )
    parser.add_argument(
        "--player-position",
        type=parse_position,
        help="offline ls20 player top-left position as X,Y",
    )
    parser.add_argument(
        "--switch-position",
        type=parse_position,
        help="offline ls20 rotation-switch top-left position as X,Y",
    )
    parser.add_argument(
        "--shape-index",
        type=int,
        choices=range(6),
        help=(
            "offline ls20 shared shape identity for the rotating target and "
            "fixed reference"
        ),
    )
    parser.add_argument(
        "--render-mode",
        default="terminal",
        choices=["terminal", "terminal-fast", "human", "none"],
        help="official SDK render mode to use while collecting",
    )
    args = parser.parse_args(argv)
    if args.randomize_shape and not args.randomize_initial_state:
        parser.error("--randomize-shape requires --randomize-initial-state.")

    mode = OperationMode.OFFLINE if args.offline else OperationMode(args.mode)
    seeds = parse_seeds(args.seeds)
    for seed in seeds:
        collect_one_demo(
            game_id=args.game,
            seed=seed,
            operation_mode=mode,
            env_dir=args.env_dir,
            recordings_dir=args.recordings_dir,
            render_mode=None if args.render_mode == "none" else args.render_mode,
            output_dir=Path(args.output_dir),
            max_steps=args.max_steps,
            demo_name=args.demo_name,
            skip_existing=args.skip_existing,
            randomize_initial_state=args.randomize_initial_state,
            randomize_shape=args.randomize_shape,
            initial_state_variant=Ls20InitialStateVariant(
                player_position=args.player_position,
                rotation_switch_position=args.switch_position,
                shape_index=args.shape_index,
            ),
        )
    return 0


def parse_seeds(raw: str) -> list[int]:
    """Parse a comma-separated seed list or inclusive range."""
    text = raw.strip()
    if ".." in text:
        lo_text, hi_text = text.split("..", maxsplit=1)
        lo = int(lo_text)
        hi = int(hi_text)
        if hi < lo:
            raise ValueError(f"Invalid seed range: {raw!r}")
        return list(range(lo, hi + 1))
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def parse_position(raw: str) -> tuple[int, int]:
    """Parse an ARC sprite position written as ``X,Y``."""
    try:
        x_text, y_text = raw.split(",", maxsplit=1)
        return int(x_text), int(y_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an X,Y coordinate, got {raw!r}."
        ) from exc


def collect_one_demo(
    *,
    game_id: str,
    seed: int,
    operation_mode: OperationMode,
    env_dir: str,
    recordings_dir: str,
    render_mode: str | None,
    output_dir: Path,
    max_steps: int,
    demo_name: str | None,
    skip_existing: bool,
    randomize_initial_state: bool,
    randomize_shape: bool,
    initial_state_variant: Ls20InitialStateVariant,
) -> Path | None:
    """Collect one manual demonstration for a game and SDK seed."""
    expected_path = expected_demo_path(
        game_id=game_id,
        seed=seed,
        output_dir=output_dir,
        demo_name=demo_name,
    )
    if skip_existing and expected_path.exists():
        print(f"Skipping seed {seed}; demo already exists at {expected_path}")
        return None

    env = ArcAgi3GymEnv(
        game_id=game_id,
        observation_format="processed",
        operation_mode=operation_mode,
        environments_dir=env_dir,
        recordings_dir=recordings_dir,
        render_mode=render_mode,
        seed=seed,
        terminate_on_level_complete=True,
    )
    if randomize_initial_state:
        if initial_state_variant.to_dict():
            raise ValueError(
                "--randomize-initial-state cannot be combined with explicit "
                "initial-state options."
            )
        selected_variant = env.sample_initial_state_variant(
            seed, randomize_shape=randomize_shape
        )
    else:
        selected_variant = (
            initial_state_variant if initial_state_variant.to_dict() else None
        )
    reset_options = (
        {"initial_state_variant": selected_variant}
        if selected_variant is not None
        else None
    )
    obs, info = env.reset(seed=seed, options=reset_options)
    steps: list[tuple[Any, GameAction]] = []
    rewards: list[float] = []
    terminated = False
    truncated = False

    print("\nCollecting ARC-AGI-3 demo")
    print(f"game={game_id} seed={seed} mode={operation_mode.value}")
    if info.get("initial_state_variant"):
        print(f"initial_state_variant={info['initial_state_variant']}")
    print("Commands: 1-7/action1-action7 to step, status, reset, save, skip, quit")
    print_status(obs, env.available_actions, info)

    while len(steps) < max_steps:
        raw = input("arc-demo> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt("Manual ARC demo collection interrupted.")
        if raw in {"skip"}:
            print("Skipping this seed without saving.")
            return None
        if raw in {"status", "score"}:
            print_status(obs, env.available_actions, {"scorecard": env.get_scorecard()})
            continue
        if raw in {"reset", "r"}:
            obs, info = env.reset(seed=seed, options=reset_options)
            steps.clear()
            rewards.clear()
            terminated = truncated = False
            print("Reset current demo; cleared unsaved actions.")
            print_status(obs, env.available_actions, info)
            continue
        if raw in {"save", "s"}:
            return save_arc_demo(
                game_id=game_id,
                seed=seed,
                output_dir=output_dir,
                steps=steps,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                final_obs=obs,
                scorecard=env.get_scorecard(),
                demo_name=demo_name,
                initial_state_variant=info.get("initial_state_variant"),
            )

        try:
            action = coerce_action(raw)
        except (TypeError, ValueError) as exc:
            print(f"Invalid command/action: {exc}")
            continue
        if not action_is_available(action, env.available_actions):
            print(
                f"Unavailable action. Available: {format_actions(env.available_actions)}"
            )
            continue

        prev_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append((prev_obs, action))
        rewards.append(float(reward))
        print(f"Recorded step {len(steps)} with {format_action(action)}")
        print_status(obs, env.available_actions, info)
        if terminated or truncated:
            episode_boundary = info.get("episode_boundary")
            if episode_boundary == "level_completed":
                print("Level 0 completed; saving this single-level demo.")
            else:
                print("Episode ended; saving demo.")
            return save_arc_demo(
                game_id=game_id,
                seed=seed,
                output_dir=output_dir,
                steps=steps,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                final_obs=None if episode_boundary == "level_completed" else obs,
                scorecard=env.get_scorecard(),
                demo_name=demo_name,
                initial_state_variant=info.get("initial_state_variant"),
                episode_boundary=episode_boundary,
            )

    print(f"Reached max_steps={max_steps}; saving partial demo.")
    return save_arc_demo(
        game_id=game_id,
        seed=seed,
        output_dir=output_dir,
        steps=steps,
        rewards=rewards,
        terminated=terminated,
        truncated=True,
        final_obs=obs,
        scorecard=env.get_scorecard(),
        demo_name=demo_name,
        initial_state_variant=info.get("initial_state_variant"),
        episode_boundary="max_steps",
    )


def save_arc_demo(
    *,
    game_id: str,
    seed: int,
    output_dir: Path,
    steps: list[tuple[Any, GameAction]],
    rewards: list[float],
    terminated: bool,
    truncated: bool,
    final_obs: Any,
    scorecard: Any,
    demo_name: str | None,
    initial_state_variant: dict[str, Any] | None,
    episode_boundary: str | None = None,
) -> Path | None:
    """Persist a collected ARC demo as a DemoRecord."""
    if not steps:
        print("No actions recorded; not saving an empty demo.")
        return None

    serializable_steps = [
        (to_jsonable(obs), int(action.value)) for obs, action in steps
    ]
    env_id = f"arc_agi3/{game_id}"
    record = DemoRecord(
        env_id=env_id,
        seed=seed,
        trajectory=Trajectory(steps=serializable_steps),
        rewards=list(rewards),
        terminated=bool(terminated),
        truncated=bool(truncated),
        metadata={
            "source": "manual_arc_agi3",
            "game_id": game_id,
            "final_observation": to_jsonable(final_obs),
            "scorecard": to_jsonable(scorecard),
            "num_actions": len(steps),
            "saved_at_unix": int(time.time()),
            "initial_state_variant": initial_state_variant,
            "episode_boundary": episode_boundary,
        },
    )
    out_dir = output_dir / "arc_agi3" / game_id
    out_path = out_dir / f"{make_demo_stem(seed, demo_name, out_dir)}.pkl"
    save_demo_record(out_path, record)
    print(f"Saved ARC-AGI-3 demo to {out_path}")
    return out_path


def make_demo_stem(seed: int, demo_name: str | None, out_dir: Path) -> str:
    """Build a non-overwriting demo filename stem."""
    if demo_name:
        safe_name = sanitize_name(demo_name)
        return f"seed_{seed:04d}_{safe_name}"

    base = f"seed_{seed:04d}"
    if not (out_dir / f"{base}.pkl").exists():
        return base

    run_idx = 1
    while (out_dir / f"{base}_run_{run_idx:02d}.pkl").exists():
        run_idx += 1
    return f"{base}_run_{run_idx:02d}"


def expected_demo_path(
    *,
    game_id: str,
    seed: int,
    output_dir: Path,
    demo_name: str | None,
) -> Path:
    """Return the first target path a collection run would save to."""
    out_dir = output_dir / "arc_agi3" / game_id
    if demo_name:
        stem = f"seed_{seed:04d}_{sanitize_name(demo_name)}"
    else:
        stem = f"seed_{seed:04d}"
    return out_dir / f"{stem}.pkl"


def sanitize_name(name: str) -> str:
    """Sanitize a short user-provided demo name for filenames."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError("demo-name must contain at least one letter or number.")
    return cleaned


def to_jsonable(value: Any) -> Any:
    """Convert SDK/Pydantic values into pickle-stable plain data."""
    if (
        isinstance(value, dict)
        and "raw" in value
        and "grid" in value
        and "objects_by_color" in value
    ):
        return value
    if isinstance(value, GameAction):
        return int(value.value)
    if hasattr(value, "tolist"):
        return value.tolist()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        data = model_dump(mode="json")
        frame = getattr(value, "frame", None)
        if frame is not None and "frame" not in data:
            data["frame"] = to_jsonable(frame)
        return data
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        data = as_dict()
        frame = getattr(value, "frame", None)
        if frame is not None and "frame" not in data:
            data["frame"] = to_jsonable(frame)
        return data
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(val) for val in value]
    return value


def action_is_available(action: GameAction, actions: Sequence[Any]) -> bool:
    return any(
        getattr(candidate, "value", candidate) == action.value for candidate in actions
    )


def format_actions(actions: Sequence[Any]) -> str:
    return ", ".join(format_action(action) for action in actions)


def format_action(action: Any) -> str:
    if isinstance(action, GameAction):
        return f"{action.value}/{action.name.lower()}"
    for candidate in GameAction:
        if candidate.value == action:
            return f"{candidate.value}/{candidate.name.lower()}"
    return str(action)


def print_status(
    obs: Any, available_actions: Sequence[Any], info: dict[str, Any]
) -> None:
    if isinstance(obs, dict):
        state = obs.get("state", "?")
        levels = f"{obs.get('levels_completed', '?')}/{obs.get('win_levels', '?')}"
    else:
        state = getattr(getattr(obs, "state", None), "name", getattr(obs, "state", "?"))
        levels = (
            f"{getattr(obs, 'levels_completed', '?')}/"
            f"{getattr(obs, 'win_levels', '?')}"
        )
    print(
        f"state={state} levels={levels} "
        f"available=[{format_actions(available_actions)}]"
    )
    if "scorecard" in info:
        print("scorecard:")
        print(info["scorecard"])


if __name__ == "__main__":
    raise SystemExit(main())
