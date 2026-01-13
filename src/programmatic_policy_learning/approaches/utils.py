"""Utils for Testing LPP Approach."""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any


def run_single_episode(
    env: Any,
    policy: Callable[[Any], Any],
    record_video: bool = False,
    video_out_path: str | None = None,
    max_num_steps: int = 100,
) -> float:
    """Run a single episode in the environment using the given policy."""

    if record_video:
        env.start_recording_video(video_out_path=video_out_path)

    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_num_steps):
        action = policy(obs)
        new_obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        obs = new_obs

        if done:
            break
    env.close()

    return total_reward


def load_hint_text(env_name: str, encoding_method: str, hints_root: str | Path) -> str:
    """Load the latest hint text for an environment/encoding pair."""
    hint_dir = Path(hints_root) / env_name / encoding_method
    if not hint_dir.exists():
        raise FileNotFoundError(f"Missing hint directory: {hint_dir}")
    hint_files = sorted(hint_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not hint_files:
        raise FileNotFoundError(f"No hint files found in {hint_dir}")
    latest_file = hint_files[-1]
    raw_text = latest_file.read_text(encoding="utf-8").strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text

    if isinstance(data, list):
        return "\n".join(str(x) for x in data)
    if isinstance(data, dict):
        if "hints" in data:
            return "\n".join(str(x) for x in data["hints"])
        if "aggregated_hints" in data:
            return "\n".join(str(x) for x in data["aggregated_hints"])
    return raw_text


def convert_dir_lists_to_tuples(programs: list[str]) -> list[str]:
    """Convert direction lists like [-1, -1] to tuples (-1, -1) inside program
    strings.

    Args:
        programs (list[str]): list of program strings

    Returns:
        list[str]: rewritten program strings
    """
    _DIR_LIST_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")

    def repl(match: re.Match[str]) -> str:
        x, y = match.groups()
        return f"({x}, {y})"

    return [_DIR_LIST_RE.sub(repl, p) for p in programs]
