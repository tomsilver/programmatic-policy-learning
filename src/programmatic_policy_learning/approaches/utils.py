"""Utils for Testing LPP Approach."""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from programmatic_policy_learning.approaches.experts.grid_experts import (
    get_grid_expert,
)
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    grid_encoder,
    grid_hint_config,
)
from programmatic_policy_learning.dsl.llm_primitives.hint_generation.llm_based.hint_extractor import (
    collect_full_episode,
)

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



def sample_transition_example(
    env_factory: Callable[[int], Any],
    env_name: str,
    encoding_method: str,
    max_steps: int = 40,
) -> tuple[str, str, str]:
    """Sample a single (s_t, a_t, s_t1) example and format with encoding."""
    expert = get_grid_expert(env_name)
    env = env_factory(0)
    traj = collect_full_episode(env, expert, max_steps=max_steps, sample_count=None)
    env.close()
    if not traj:
        raise ValueError("No trajectory data collected.")
    obs_t, action, obs_t1 = traj[0]

    symbol_map = grid_hint_config.get_symbol_map(env_name)
    encoder = grid_encoder.GridStateEncoder(
        grid_encoder.GridStateEncoderConfig(
            symbol_map=symbol_map,
            empty_token="empty",
            coordinate_style="rc",
        )
    )
    salient_tokens = grid_hint_config.SALIENT_TOKENS[env_name]

    def list_literal(obs: Any) -> str:
        rows: list[str] = []
        for r in range(obs.shape[0]):
            entries: list[str] = []
            for c in range(obs.shape[1]):
                token = obs[r, c]
                char = symbol_map.get(token, "?")
                entries.append(f"'{char}'")
            rows.append(f"[{', '.join(entries)}]")
        return "[\n" + "\n".join(f"  {row}" for row in rows) + "\n]"

    def listing(obs: Any) -> str:
        tokens = list(dict.fromkeys([*salient_tokens, encoder.cfg.empty_token]))
        objs = encoder.extract_objects(obs, tokens)
        entries: list[tuple[tuple[int, int], str]] = []
        for token, coords in objs.items():
            if not coords:
                continue
            for coord in coords:
                entries.append((coord, token))
        entries.sort(key=lambda item: item[0])
        return "\n".join(f"{coord} - '{label}'" for coord, label in entries)

    if encoding_method == "1":
        state_t = list_literal(obs_t)
        state_t1 = list_literal(obs_t1)
        state_t1 +=  f"\nToken meanings: {grid_hint_config.SYMBOL_MAPS[env_name]}"
    else:
        state_t = listing(obs_t)
        state_t1 = listing(obs_t1)

    action_token = obs_t[action]
    action_text = f"{action}: '{action_token}'"
    return state_t, action_text, state_t1
