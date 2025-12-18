"""Script for extracting textual hints from grid trajectories."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.reprompting import RepromptCheck, query_with_reprompts
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
from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel
import tempfile
from pathlib import Path
from omegaconf import OmegaConf
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

HINT_EXTRACTION_PROMPT_TEMPLATE = """

    You are analyzing expert demonstrations from a grid-based environment.

    You are given a sequence of states and actions expressed as:
    - ASCII grid representations
    - Coordinate-based descriptions of objects
    - Transitions showing how the grid changes after each action

    The agent is the ONLY controlled entity.

    Your task:
    Infer the decision-making strategy demonstrated by the agent.

    Focus on:
    - What the agent appears to optimize or prioritize
    - How the agent chooses actions based on relative positions, obstacles, and boundaries
    - What conditions cause the agent to prefer one direction or move over another
    - How the agent adapts its behavior when simple direct movement is blocked

    Constraints:
    - Do NOT propose DSL primitives, function names, or code.
    - Use clear, causal, spatial language grounded in the observations.
    - Describe *conditional logic* when relevant (e.g., “when X holds, the agent does Y”).
    - If behavior seems inconsistent or ambiguous, state that explicitly.

    ---
    Demonstrations:\n\n

    {TRAJECTORY_TEXT}

    ---
    Output format:
    Write a concise but structured explanation of the strategy, as if explaining the agent’s reasoning to a human.
    Do NOT invent mechanics not visible in the data.
    Output ONLY the strategy description text.
    No preamble. No conclusion. No formatting markers.

    """


def build_hint_prompt(trajectory_text: str) -> str:
    """Wrap the serialized trajectory text in the hint prompt."""
    return HINT_EXTRACTION_PROMPT_TEMPLATE.format(TRAJECTORY_TEXT=trajectory_text)

def env_factory(instance_num: int | None = None, env_name: str = None) -> Any:
    """Env Factory."""
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
        env.close()
        trajectories.append(traj)

    # ------------------------------------------------------------
    # 3) Per-trajectory hint extraction
    # ------------------------------------------------------------
    per_traj_hints: list[str] = []

    for i, traj in enumerate(trajectories):
        text = trajectory_serializer.trajectory_to_text(
            traj,
            encoder=encoder,
            analyzer=analyzer,
            salient_tokens=grid_hint_config.SALIENT_TOKENS[env_name],
            max_steps=max_steps_per_traj,
        )


        prompt = build_hint_prompt(text)
        query = Query(prompt)
        per_traj_checks: list[RepromptCheck] = []  # TODOO: add actual checks

        response = query_with_reprompts(
            llm_client,
            query,
            per_traj_checks,  # type: ignore[arg-type]
            max_attempts=5,
        )
        per_traj_hints.append(response.text)

    # ------------------------------------------------------------
    # 4) Aggregate hints (machine-side)
    # ------------------------------------------------------------
    blocks = []
    for i, hint in enumerate(per_traj_hints, start=1):
        blocks.append(f"[ANALYSIS {i}]\n{hint.strip()}")
    consolidation_input =  "\n\n".join(blocks)

    # ------------------------------------------------------------
    # 5) Final abstraction pass (LLM)
    # ------------------------------------------------------------

    final_prompt = f"""
    You are given multiple independent analyses of expert behavior,
    each derived from a different initial state of the same environment.

    Your task:
    - Identify the core strategy that is consistent across trajectories
    - Identify behaviors that appear context-dependent
    - Discard incidental or contradictory explanations
    - Produce a unified, environment-level description of the agent’s strategy

    Do NOT invent new behaviors.
    Do NOT refer to individual trajectories explicitly.
    Do NOT average language — reason conceptually.

    Below are the analyses:

    {consolidation_input}
    

    \nOutput ONLY the structured hint template used previously.
    """

    final_query = Query(final_prompt)
    aggregation_checks: list[RepromptCheck] = []  # TODOO: add actual checks

    final_response = query_with_reprompts(
        llm_client,
        final_query,
        aggregation_checks,  # type: ignore[arg-type]
        max_attempts=5,
    )

    print("\n=== FINAL AGGREGATED EXPERT BEHAVIOR HINTS ===\n")
    print(final_response.text)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("hints") / env_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{num_initial_states}_{timestamp}.txt"
    out_file.write_text(final_response.text, encoding="utf-8")

    return final_response.text


if __name__ == "__main__":
    _env_name = "TwoPileNim"
    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    client = OpenAIModel("gpt-4.1", cache)
    run(client, _env_name, num_initial_states=5)
