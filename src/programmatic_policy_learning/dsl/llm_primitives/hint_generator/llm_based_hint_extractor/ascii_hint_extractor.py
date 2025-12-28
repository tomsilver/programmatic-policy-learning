"""Script for extracting textual hints from grid trajectories."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel, PretrainedLargeModel
from prpl_llm_utils.reprompting import query_with_reprompts
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.approaches.experts.grid_experts import (
    get_grid_expert,
)

# pylint: disable=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.hint_generator.llm_based_hint_extractor import (
    grid_encoder,
    grid_hint_config,
    trajectory_serializer,
    transition_analyzer,
)
from programmatic_policy_learning.envs.registry import EnvRegistry


def collect_full_episode(
    env: Any,
    expert_fn: Callable[[Any], Any],
    max_steps: int = 200,
    sample_count: int | None = None,
) -> list[tuple[Any, Any, Any]]:
    """Roll out expert policy, optionally sampling a subset of (obs, action,
    obs_next)."""
    obs, _ = env.reset()
    trajectory = []

    for _ in range(max_steps):
        action = expert_fn(obs)
        obs_next, _, term, trunc, _ = env.step(action)
        trajectory.append((obs, action, obs_next))
        obs = obs_next
        if term or trunc:
            break

    if sample_count is None or sample_count >= len(trajectory):
        return trajectory
    if sample_count <= 0:
        return []

    sampled: list[tuple[Any, Any, Any]] = []
    if trajectory:
        sampled.append(trajectory[0])

    if len(sampled) >= sample_count or len(trajectory) <= 1:
        return sampled[:sample_count]

    remaining = trajectory[1:]
    needed = min(sample_count - len(sampled), len(remaining))
    if needed > 0:
        sampled.extend(random.sample(remaining, needed))

    return sampled


def build_joint_hint_prompt(
    all_trajectories_text: str, env_name: str, encoding: str
) -> str:
    """Return the LLM prompt for summarising expert trajectories."""
    token_meanings = ""
    action_mask = ""
    if encoding == "1":
        action_mask = "The ACTION MASK has a single 1 at the clicked cell."
        token_meanings = f"Token meanings: {grid_hint_config.SYMBOL_MAPS[env_name]}\n"

    return f"""
You are given a few expert demonstrations of the SAME task.
Each demonstration is a full trajectory from a different initial state.

IMPORTANT:
- Coordinates are 0-indexed (row, col) with (0,0) at top-left.
- An action is "Click cell (r,c)". {action_mask}
- Steps within a trajectory are temporally consecutive and highly correlated.
- Your job is to infer the underlying strategy/policy that generalizes across trajectories.
- Do NOT describe individual steps.
- Do NOT narrate what happens over time.
- Prefer rules that are consistent across trajectories; ignore trajectory-specific ones.

{token_meanings}

========================
DEMONSTRATIONS
========================
{all_trajectories_text}

Your task:
Infer the rule used to choose the action from the observation, then implement it.

OUTPUT FORMAT (STRICT):
- Return ONLY executable Python code.
- The code MUST be wrapped in a Markdown code block that starts with ```python and ends with ```.
- Do NOT include any text, explanation, headings, or comments outside the code block.

CODE REQUIREMENTS:
- Define a function with signature: def policy(obs):
- The function takes a NumPy array observation and returns an action (row, col).
- Use numeric coordinates only (NumPy convention: (0, 0) is top-left).
- Do NOT use words like left, right, up, or down.
- Do NOT import external libraries.
"""


def env_factory(
    instance_num: int | None = None,
    env_name: str | None = None,
) -> Any:
    """Env Factory."""
    if env_name is None:
        raise ValueError("env_name must be provided.")
    registry = EnvRegistry()
    return registry.load(
        OmegaConf.create(
            {
                "provider": "ggg",
                "make_kwargs": {
                    "base_name": env_name,
                    "id": f"{env_name}0-v0",
                },
                "instance_num": instance_num,
            }
        )
    )


def run(
    llm_client: PretrainedLargeModel,
    env_name: str,
    encoding_method: str,
    num_initial_states: int = 10,
    max_steps_per_traj: int = 40,
) -> str:
    """Collect multiple env trajectories and summarise hints via the LLM."""

    # ------------------------------------------------------------
    # 1) Setup encoder + analyzer
    # ------------------------------------------------------------
    symbol_map = grid_hint_config.get_symbol_map(env_name)

    enc_cfg = grid_encoder.GridStateEncoderConfig(
        symbol_map=symbol_map,
        empty_token="empty",
        coordinate_style="rc",
    )
    encoder = grid_encoder.GridStateEncoder(enc_cfg)
    analyzer = transition_analyzer.GenericTransitionAnalyzer()

    # ------------------------------------------------------------
    # 2) Collect expert trajectories
    # ------------------------------------------------------------
    expert = get_grid_expert(env_name)
    trajectories: list[list[tuple[Any, Any, Any]]] = []

    for init_idx in range(num_initial_states):
        env = env_factory(init_idx, env_name)
        traj = collect_full_episode(env, expert, sample_count=None)
        env.close()
        trajectories.append(traj)

    # ------------------------------------------------------------
    # 3) All trajectories hint extraction
    # ------------------------------------------------------------

    all_traj_texts = []

    for i, traj in enumerate(trajectories):
        text = trajectory_serializer.trajectory_to_text(
            traj,
            encoder=encoder,
            analyzer=analyzer,
            salient_tokens=grid_hint_config.SALIENT_TOKENS[env_name],
            encoding_method=encoding_method,
            max_steps=max_steps_per_traj,
        )
        all_traj_texts.append(f"\n---[TRAJECTORY {i}]---\n{text}\n\n")
        # all_traj_texts.append(f"\n{text}")

    combined_text = "\n\n".join(all_traj_texts)
    print(combined_text)

    prompt = build_joint_hint_prompt(combined_text, env_name, encoding_method)
    query = Query(prompt)

    response = query_with_reprompts(
        llm_client,
        query,
        reprompt_checks=[],
        max_attempts=5,
    )

    final_hint = response.text

    logging.info("\n=== FINAL AGGREGATED EXPERT BEHAVIOR HINTS ===\n")
    logging.info(final_hint)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("hints") / env_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{num_initial_states}_{timestamp}.txt"
    out_file.write_text(final_hint, encoding="utf-8")

    return final_hint


if __name__ == "__main__":
    _env_name = "Chase"
    encoding_mode = "1"  # 1=ascii+mask 2=coordinate-based 3=2+diff 4=3+relations
    cache_path = Path("hint_llm_cache.db")
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    client = OpenAIModel("gpt-4.1", cache)
    run(client, _env_name, encoding_mode, num_initial_states=4)
