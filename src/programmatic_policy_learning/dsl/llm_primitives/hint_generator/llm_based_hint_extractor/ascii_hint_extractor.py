"""Script for extracting textual hints from grid trajectories."""

from __future__ import annotations

import logging
import tempfile
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
) -> list[tuple[Any, Any, Any]]:
    """Roll out expert policy and capture (obs, action, next_obs) tuples."""
    obs, _ = env.reset()
    trajectory = []

    for _ in range(max_steps):
        action = expert_fn(obs)
        obs_next, _, term, trunc, _ = env.step(action)
        trajectory.append((obs, action, obs_next))
        obs = obs_next
        if term or trunc:
            break

    return trajectory


nim_game_description = """
        Environment description:

        - The environment is a grid-based implementation of the game 'Nim'.
        - The observation `obs` is a 2D Python NumPy array (rows × columns).
        - Do not use boolean checks such as `if obs`, `if not obs`, or `obs == []`. 
        - Each cell contains one of the following values:
        - 'empty'
        - 'token'


        ## Game Dynamics and Rules (TwoPileNim)

        - The grid encodes a two-pile Nim position using **two token columns** (two vertical stacks of `'token'`).
        - **Each pile corresponds to one column**: the number of `'token'` cells in that column is the pile size.
        - The game is **turn-based**. On each agent turn, the policy selects exactly one action `(row, col)`.

        ### Action meaning
        - The action `(row, col)` indicates choosing **pile `col`** and removing tokens according to the selected **row**.
        - A move is legal only if the selected `(row, col)` corresponds to a location in a token-stack column that results in removing **at least one** token from that pile.
        - Intuitively: selecting a cell in a pile column removes tokens **from that cell and all tokens “below it” in the same column** (i.e., it reduces the pile to the tokens strictly above the selected cell).  
        - Selecting higher rows removes fewer tokens; selecting lower rows removes more tokens.

        ### Terminal condition and winner
        - The game ends when **no `'token'` cells remain in the grid** (both piles are empty).
        - The player who makes the move that leaves the grid with **no remaining tokens** wins (standard normal-play Nim).

        ### Optimal play objective
        - The optimal strategy is the standard Nim strategy: on your turn, make a move that leaves the opponent a **losing position** (for two piles, this corresponds to leaving the piles with **equal sizes**, when possible).
"""


def build_joint_hint_prompt(all_trajectories_text: str) -> str:
    """Build a prompt that includes all trajectories."""
    return f"""
        You are analyzing multiple expert demonstrations from the SAME environment.
        Each trajectory starts from a different initial state.
        Your task:
        - Infer the agent's STRATEGY that is consistent ACROSS trajectories.
        - Focus ONLY on behaviors that generalize across different initial states.
        - Do NOT describe trajectory-specific or one-off behaviors.
        - If a behavior appears in only one trajectory, treat it as incidental.

        Hard constraints:
        - Do NOT describe per-trajectory actions (e.g.“in trajectory 1 the agent did X”).
        - Do NOT overfit to a single column, cell index, or fixed action order.
        - Do NOT assume optimality unless it is supported across trajectories.
        - If multiple plausible strategies exist, describe the one which is most 
        supported by the evidence.

        Your output should:
        - Describe the agent’s strategy at the ENVIRONMENT LEVEL
        - Be invariant to initial state
        - Use relational, abstract language
        - Avoid referencing specific coordinates, columns, or indices

        Below are multiple expert trajectories:

        {all_trajectories_text}

        ---
        Output:
        A concise but precise description of the agent’s inferred strategy.
        Do not include headers, lists, or analysis scaffolding.
        Do not mention individual trajectories.
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
        traj = collect_full_episode(env, expert)
        print(traj)
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
            max_steps=max_steps_per_traj,
        )
        all_traj_texts.append(f"[TRAJECTORY {i}]\n{text}")

    combined_text = "\n\n".join(all_traj_texts)
    print(combined_text)

    prompt = build_joint_hint_prompt(combined_text)
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
    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    client = OpenAIModel("gpt-4.1", cache)
    run(client, _env_name, num_initial_states=5)
