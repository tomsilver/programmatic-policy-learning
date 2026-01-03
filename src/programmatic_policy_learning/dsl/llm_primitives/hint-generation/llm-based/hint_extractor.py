"""LLM-based hint extractor for LPP.

Given expert demonstrations, this module queries an LLM to extract
strategy-level hints that can later be used for DSL generation.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from prpl_llm_utils.models import OpenAIModel
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.CaP_baseline import *


def _configure_rng(seed: int) -> None:
    """Seed Python/NumPy RNGs for reproducible demos."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------


def build_hint_prompt(trajectories_text: str) -> str:
    """Build the LLM prompt for extracting strategy hints from demonstrations.

    trajectories_text:
        A human-readable representation of expert trajectories
        (ASCII, object-based, etc.)
    """
    return f"""
You are given expert demonstrations from a grid-based environment.

Each demonstration consists of:
- State s_t (before action)
- Action a_t (clicked cell: row, col)
- State s_t+1 (after deterministic transition)

DEMONSTRATIONS:
{trajectories_text}

----

Your task:

Extract **abstract, reusable decision-time features** that could inform the design of a DSL for selecting actions.

IMPORTANT:
- Use ONLY information observable in the current state (s_t).
- Ignore state transitions and future effects.
- Do NOT describe game rules or mechanics.
- Do NOT mention specific object names (e.g., arrow, wall, target).
- Treat cell contents as abstract VALUEs.
- Do NOT describe a policy or strategy.

Instead:
- Describe **generic spatial, relational, or comparison-based predicates**
- Each hint should correspond to something that could plausibly be implemented as: (Cell, Obs) -> Bool

OUTPUT FORMAT:
- Return ONLY a list of short hints
- One hint per line
- No introduction or explanation text
"""


# Your task:

# Extract hints about **what properties of the current state (s_t)** are being used to select the action.
# """

# Your task:

# Extract hints about **what properties of the current state (s_t)**
# are being used to select the action.

# IMPORTANT:
# - Only use information observable in s_t.
# - Ignore how the board changes after the action.
# - Do NOT explain why an action is good in terms of future outcomes.
# - Do NOT describe environment rules or transition dynamics.

# OUTPUT FORMAT:
# - Return ONLY a bullet list of concise hints.
# - Do not add any extra text in your answer.

# Focus on **state-based, spatial, or relational predicates** that could be implemented as reusable DSL functions.
# """


def extract_hints(llm_client, trajectories_text: str) -> dict[str, Any]:
    """Query the LLM and return parsed hint JSON."""
    prompt = build_hint_prompt(trajectories_text)

    query = Query(prompt)
    reprompt_checks: list[RepromptCheck] = []
    response = query_with_reprompts(
        llm_client,
        query,
        reprompt_checks=reprompt_checks,
        max_attempts=5,
    )

    # try:
    response_text = response.text if hasattr(response, "text") else str(response)
    # hints = json.loads(response_text)
    # except json.JSONDecodeError as e:
    #     raise ValueError(
    #         f"LLM did not return valid JSON.\nResponse:\n{response_text}"
    #     ) from e
    print(response_text)
    # Minimal sanity checks
    # assert "high_level_rules" in hints
    # assert "local_rules" in hints

    return response_text


# ---------------------------------------------------------------------
# Saving utilities
# ---------------------------------------------------------------------


def save_hints(
    hints: dict[str, Any] | str,
    env_name: str,
    seed: int,
    out_dir: Path | str = "hints",
    filename: str | None = None,
) -> Path:
    """
    Save extracted hints under:
        hints/{env_name}/{filename}
    """
    out_dir = Path(out_dir) / env_name
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = filename or f"hints_seed{seed}_{timestamp}.json"
    out_path = out_dir / filename
    if isinstance(hints, str):
        out_path.write_text(hints, encoding="utf-8")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(hints, f, indent=2)

    return out_path


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    num_initial_states = 4
    env_name = "StopTheFall"
    encoding_method = "4"
    max_steps_per_traj = 40
    seed = 0
    _configure_rng(seed)

    expert = get_grid_expert(env_name)
    trajectories: list[list[tuple[Any, Any, Any]]] = []
    for init_idx in range(num_initial_states):
        env = env_factory(init_idx, env_name)
        traj = collect_full_episode(env, expert, sample_count=None)
        env.close()
        trajectories.append(traj)

    symbol_map = grid_hint_config.get_symbol_map(env_name)

    enc_cfg = grid_encoder.GridStateEncoderConfig(
        symbol_map=symbol_map,
        empty_token="empty",
        coordinate_style="rc",
    )
    encoder = grid_encoder.GridStateEncoder(enc_cfg)
    analyzer = transition_analyzer.GenericTransitionAnalyzer()

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

    combined_text = "\n\n".join(all_traj_texts)

    cache_path = Path("cache.db")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_client = OpenAIModel("gpt-4.1", cache)
    reprompt_checks = []

    hints = extract_hints(llm_client, combined_text)

    path = save_hints(
        hints,
        env_name=env_name,
        seed=seed,
    )

    print(f"Hints saved to {path}")
