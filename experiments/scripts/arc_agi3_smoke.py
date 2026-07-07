"""Smoke-test the official ARC-AGI-3 SDK adapter."""

from __future__ import annotations

import argparse
import pdb
from collections.abc import Sequence
from typing import Any

from arc_agi import OperationMode
from arcengine import GameAction

from programmatic_policy_learning.envs.arc_agi3 import (
    ArcAgi3Env,
    coerce_action,
    list_environments,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test ARC-AGI-3 through the official arc-agi SDK."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list available/cached games")
    add_common_args(list_parser)

    smoke_parser = subparsers.add_parser(
        "smoke", help="launch a game, reset it, and take one action"
    )
    add_common_args(smoke_parser)
    smoke_parser.add_argument("--game", default="ls20", help="ARC game id")
    smoke_parser.add_argument("--action", default="1", help="action id or name")
    smoke_parser.add_argument(
        "--pdb",
        action="store_true",
        help="open pdb after reset with initial_obs and env available",
    )

    args = parser.parse_args(argv)
    if args.command == "list":
        return list_games(args)
    if args.command == "smoke":
        return smoke(args)
    return 2


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        default=OperationMode.NORMAL.value,
        choices=[OperationMode.NORMAL.value, OperationMode.OFFLINE.value],
        help="official SDK operation mode",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="shortcut for --mode offline",
    )
    parser.add_argument(
        "--env-dir",
        default="environment_files",
        help="directory used by the SDK for cached game files",
    )
    parser.add_argument(
        "--recordings-dir",
        default="recordings",
        help="directory used by the SDK for scorecards/recordings",
    )


def selected_mode(args: argparse.Namespace) -> OperationMode:
    if args.offline:
        return OperationMode.OFFLINE
    return OperationMode(args.mode)


def list_games(args: argparse.Namespace) -> int:
    try:
        environments = list_environments(
            operation_mode=selected_mode(args),
            environments_dir=args.env_dir,
            recordings_dir=args.recordings_dir,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print_graceful_failure(exc, selected_mode(args), args.env_dir)
        return 1

    if not environments:
        print("No ARC-AGI-3 games found.")
        if selected_mode(args) == OperationMode.OFFLINE:
            print(f"Offline mode only checks the local cache at {args.env_dir!r}.")
        return 1

    for env in environments:
        tags = ", ".join(getattr(env, "tags", None) or [])
        print(
            f"{env.game_id:16} {env.title:10} "
            f"actions={env.baseline_actions} tags={tags}"
        )
    return 0


def smoke(args: argparse.Namespace) -> int:
    mode = selected_mode(args)
    try:
        env = ArcAgi3Env(
            game_id=args.game,
            operation_mode=mode,
            environments_dir=args.env_dir,
            recordings_dir=args.recordings_dir,
        )
        initial_obs = env.reset()
        if args.pdb:
            print(
                "Entering pdb after reset. Useful names: initial_obs, env, args, mode."
            )
            pdb.set_trace()
        action = coerce_action(args.action)
        if not action_is_available(action, env.available_actions):
            print(
                f"{format_action(action)} is not available. "
                f"Available: {format_actions(env.available_actions)}"
            )
            return 1
        obs = env.step(action)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print_graceful_failure(exc, mode, args.env_dir, game_id=args.game)
        return 1

    print("Initial observation:")
    print_observation(initial_obs)
    print(f"Action taken: {format_action(action)}")
    print("Post-step observation:")
    print_observation(obs)
    print(f"Available actions: {format_actions(env.available_actions)}")
    print("Scorecard:")
    print(env.get_scorecard())
    return 0


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


def print_observation(obs: Any) -> None:
    if obs is None:
        print("  <no observation returned>")
        return
    state = getattr(obs, "state", None)
    state_name = getattr(state, "name", state)
    print(f"  state={state_name}")
    print(
        "  levels_completed="
        f"{getattr(obs, 'levels_completed', '?')}/{getattr(obs, 'win_levels', '?')}"
    )
    print(f"  game_id={getattr(obs, 'game_id', '?')}")
    print(
        "  available_actions="
        f"{format_actions(getattr(obs, 'available_actions', []) or [])}"
    )
    print(f"  raw={obs}")


def print_graceful_failure(
    exc: Exception,
    mode: OperationMode,
    env_dir: str,
    *,
    game_id: str | None = None,
) -> None:
    target = f" {game_id!r}" if game_id else ""
    print(f"Could not load ARC-AGI-3{target}: {exc}")
    if mode == OperationMode.OFFLINE:
        print(
            f"Offline mode requires the game to already be cached in {env_dir!r}. "
            "Run once in normal mode after setting ARC_API_KEY."
        )
    else:
        print(
            "Normal mode may need a registered ARC_API_KEY. Set it in the "
            "environment or in .env, then retry."
        )


if __name__ == "__main__":
    raise SystemExit(main())
