"""LLM-based hint extractor for LPP.

Given expert demonstrations, this module queries an LLM to extract
strategy-level hints that can later be used for DSL generation.
"""

from __future__ import annotations

import json
import logging
import random

# import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from omegaconf import OmegaConf
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel, PretrainedLargeModel
from prpl_llm_utils.reprompting import RepromptCheck, query_with_reprompts
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.approaches.experts.grid_experts import (
    get_grid_expert,
)
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    grid_encoder,
    grid_hint_config,
    trajectory_serializer,
    transition_analyzer,
)
from programmatic_policy_learning.envs.registry import EnvRegistry


def _configure_rng(seed: int) -> None:
    """Seed Python/NumPy RNGs for reproducible demos."""
    random.seed(seed)
    np.random.seed(seed)


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


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------


def build_hint_prompt_v1(
    trajectories_text: str, env_name: str, encoding_method: str
) -> str:
    """Build the LLM prompt for extracting strategy hints from demonstrations.

    trajectories_text:
        A human-readable representation of expert trajectories
        (ASCII, object-based, etc.)
    """
    place_holder = ""
    token_map_text = _format_token_map(_build_token_map(env_name))

    if encoding_method == "1":
        place_holder = f"""TOKEN MAP:
    At the very end, output a instruction named TOKEN_MAP that maps the raw grid characters to canonical names.
    Use EXACTLY this format (no extra keys, no commentary):
    
    {token_map_text}
    """
    template = _load_hint_prompt_template("hint_prompt_v1.txt")
    return template.replace("<<DEMONSTRATIONS>>", trajectories_text).replace(
        "<<PLACE_HOLDER>>", place_holder
    )


# Extract **abstract, reusable decision-time predicates/features**
# that could inform the design of a DSL.

# IMPORTANT:
# - Use ONLY information observable in the current state (s_t).
# - Use ONLY information available at decision time
# (do not use future states or transition outcomes).
# - Do NOT describe game rules or mechanics.
# - Do NOT mention specific object names (e.g., arrow, wall, target).
# - Treat cell contents as abstract VALUEs.
# - Do NOT describe a policy or strategy.

# Instead:
# - Describe **generic spatial, relational, or comparison-based predicates**
# - Each hint should correspond to something that could plausibly
# be implemented as a boolean predicate over
# +  (Cell, Obs) and optionally a distinguished reference from the decision-time inputs.

# OUTPUT FORMAT:
# - Return ONLY a list of short hints
# - One hint per line
# - No introduction or explanation text
# """


# Explain the decision making rule, how action a is selected in state s.
# """


def _build_token_map(env_name: str) -> dict[str, str]:
    token_to_canonical = {
        "StopTheFall": {
            "empty": "stf.EMPTY",
            "red_token": "stf.RED",
            "static_token": "stf.STATIC",
            "falling_token": "stf.FALLING",
            "drawn_token": "stf.DRAWN",
            "advance_token": "stf.ADVANCE",
        },
        "TwoPileNim": {
            "empty": "tpn.EMPTY",
            "token": "tpn.TOKEN",
        },
        "ReachForTheStar": {
            "empty": "rfts.EMPTY",
            "agent": "rfts.AGENT",
            "star": "rfts.STAR",
            "left_arrow": "rfts.LEFT_ARROW",
            "right_arrow": "rfts.RIGHT_ARROW",
            "drawn": "rfts.DRAWN",
        },
        "Chase": {
            "empty": "ec.EMPTY",
            "agent": "ec.AGENT",
            "target": "ec.TARGET",
            "wall": "ec.WALL",
            "drawn": "ec.DRAWN",
            "left_arrow": "ec.LEFT_ARROW",
            "right_arrow": "ec.RIGHT_ARROW",
            "up_arrow": "ec.UP_ARROW",
            "down_arrow": "ec.DOWN_ARROW",
        },
        "CheckmateTactic": {
            "empty": "ct.EMPTY",
            "black_king": "ct.BLACK_KING",
            "white_king": "ct.WHITE_KING",
            "white_queen": "ct.WHITE_QUEEN",
            "highlighted_white_king": "ct.HIGHLIGHTED_WHITE_KING",
            "highlighted_white_queen": "ct.HIGHLIGHTED_WHITE_QUEEN",
        },
    }
    try:
        mapping = token_to_canonical[env_name]
    except KeyError as exc:
        raise KeyError(f"No TOKEN_MAP configured for {env_name}") from exc

    symbol_map = grid_hint_config.get_symbol_map(env_name)
    token_map: dict[str, str] = {}
    for token_name, char in symbol_map.items():
        canonical = mapping.get(token_name)
        if canonical is None:
            raise KeyError(
                f"Missing canonical token mapping for {env_name}: {token_name}"
            )
        token_map[char] = canonical
    return token_map


def _format_token_map(token_map: dict[str, str]) -> str:
    items = sorted(token_map.items(), key=lambda item: item[0])
    lines = ["TOKEN_MAP = {"]
    for key, value in items:
        lines.append(f'  "{key}": "{value}",')
    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def _load_hint_prompt_template(filename: str) -> str:
    prompt_dir = Path(__file__).resolve().parent / "hint_gen_prompt"
    prompt_path = prompt_dir / filename
    return prompt_path.read_text(encoding="utf-8")


def build_hint_prompt_prev(trajectories_text: str) -> str:
    """Build the LLM prompt for extracting strategy hints from demonstrations.

    trajectories_text:
        A human-readable representation of expert trajectories
        (ASCII, object-based, etc.)
    """
    template = _load_hint_prompt_template("hint_prompt_prev.txt")
    return template.replace("<<DEMONSTRATIONS>>", trajectories_text)


def build_hint_structured(
    trajectories_text: str, env_name: str, encoding_method: str
) -> str:
    """Build structural hints."""
    place_holder = ""
    token_map_text = _format_token_map(_build_token_map(env_name))
    if encoding_method == "1":
        place_holder = f"""4) TOKEN MAP (required, last section):
At the very end, output an instruction named TOKEN_MAP that maps the raw grid characters to canonical names.
Use EXACTLY this format (no extra keys, no commentary):

{token_map_text}"""
    template = _load_hint_prompt_template("hint_structured.txt")
    return template.replace("<<DEMONSTRATIONS>>", trajectories_text).replace(
        "<<PLACE_HOLDER>>", place_holder
    )


def build_new_hint_structured(
    trajectories_text: str, env_name: str, _encoding_method: str
) -> str:
    """Build hints in a structured way."""
    template = _load_hint_prompt_template("new_hint_structured.txt")
    token_map_text = _format_token_map(_build_token_map(env_name))
    return template.replace("<<DEMONSTRATIONS>>", trajectories_text).replace(
        "<<TOKEN_MAP>>", token_map_text
    )


def extract_hints(
    llm_client: PretrainedLargeModel,
    trajectories_text: str,
    env_name: str,
    seed: int,
    encoding_method: str,
    structured: bool,
) -> str:
    """Query the LLM and return parsed hint JSON."""
    if structured:
        # prompt = build_hint_structured(trajectories_text, env_name, encoding_method)
        # # print(prompt)
        prompt = build_new_hint_structured(trajectories_text, env_name, encoding_method)
        # input(prompt)
    else:
        prompt = build_hint_prompt_v1(trajectories_text, env_name, encoding_method)
    prompt = f"{prompt}\n\nSEED: {seed}\n"
    # print(prompt)
    query = Query(prompt, hyperparameters={"temperature": 0.0, "seed": seed})
    reprompt_checks: list[RepromptCheck] = []
    response = query_with_reprompts(
        llm_client,
        query,
        reprompt_checks=reprompt_checks,
        max_attempts=5,
    )
    response_text = response.text if hasattr(response, "text") else str(response)
    logging.info(response_text)
    return response_text


def extract_dsl(llm_client: PretrainedLargeModel, prompt: str) -> str:
    """Query the LLM and return parsed hint JSON."""
    query = Query(prompt)
    reprompt_checks: list[RepromptCheck] = []
    response = query_with_reprompts(
        llm_client,
        query,
        reprompt_checks=reprompt_checks,
        max_attempts=5,
    )
    response_text = response.text if hasattr(response, "text") else str(response)
    logging.info(response_text)
    return response_text


# ---------------------------------------------------------------------
# Saving utilities
# ---------------------------------------------------------------------


def save_hints(
    hints: dict[str, Any] | str,
    env_name: str,
    seed: int,
    encoding_method: str,
    num_demos: int,
    flag: bool,
    out_dir: Path | str = "hints",
    filename: str | None = None,
) -> Path:
    """
    Save extracted hints under:
        hints/{env_name}/{filename}
    """
    out_dir = Path(out_dir) / env_name / f"enc_{encoding_method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    if flag:
        filename = filename or f"structured/hints_seed{seed}_{num_demos}.json"
    else:
        filename = filename or f"simple/hints_seed{seed}_{num_demos}.json"
    out_path = out_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(hints, str):
        out_path.write_text(hints, encoding="utf-8")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(hints, f, indent=2)

    return out_path


def main() -> None:
    """Entry point for running hint and DSL extraction."""
    max_steps_per_traj = 50
    seed = 0
    cache_path = Path("hint_cache.db")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_client = OpenAIModel("gpt-4.1", cache)

    env_names = [
        # "Chase",
        # "TwoPileNim",
        "ReachForTheStar",
        # "StopTheFall"
        # "CheckmateTactic"
    ]
    # encoding_methods = ["5"]  # "1",
    # num_initial_states = [0, 2, 6]  # 4 9 deleted
    # structured_modes = [True]
    encoding_methods = ["6"]  # "1", "6"
    num_initial_states = [0, 2, 4, 6, 9]  # 4 9 deleted
    structured_modes = [True]
    for env_name in env_names:
        for encoding_method in encoding_methods:
            for structured in structured_modes:
                _configure_rng(seed)
                expert = get_grid_expert(env_name)
                trajectories: list[list[tuple[Any, Any, Any]]] = []
                object_types: Sequence[str] | None = None
                for init_idx in num_initial_states:
                    print(
                        f"Collecting trajectory for {env_name}, init_idx={init_idx}..."
                    )
                    env = env_factory(init_idx, env_name)
                    traj = collect_full_episode(env, expert, sample_count=None)
                    env.close()
                    object_types = env.get_object_types()
                    trajectories.append(traj)

                if object_types is None:
                    raise RuntimeError("No object types collected from environments.")

                symbol_map = grid_hint_config.get_symbol_map(env_name)
                enc_cfg = grid_encoder.GridStateEncoderConfig(
                    symbol_map=symbol_map,
                    empty_token="empty",
                    coordinate_style="rc",
                )
                encoder = grid_encoder.GridStateEncoder(enc_cfg)
                analyzer = transition_analyzer.GenericTransitionAnalyzer()
                all_traj_texts = []
                for idx, traj in enumerate(trajectories):
                    # NEW VERSION OF ENC 1
                    # text = trajectory_serializer.trajectory_to_diff_text(
                    #     traj,
                    #     encoder=encoder,
                    #     max_steps=max_steps_per_traj,
                    # )
                    # ENC 1 - 6
                    text = trajectory_serializer.trajectory_to_text(
                        traj,
                        encoder=encoder,
                        analyzer=analyzer,
                        salient_tokens=grid_hint_config.SALIENT_TOKENS[env_name],
                        encoding_method=encoding_method,
                        max_steps=max_steps_per_traj,
                    )
                    all_traj_texts.append(f"\n---[TRAJECTORY {idx}]---\n{text}\n\n")

                combined_text = "\n\n".join(all_traj_texts)
                hints = extract_hints(
                    llm_client,
                    combined_text,
                    env_name,
                    seed,
                    encoding_method,
                    structured,
                )
                print(combined_text)
                input()
                # dsl_prompt = build_dsl_generation_prompt_final(hints, object_types)
                # output = extract_dsl(llm_client, dsl_prompt)

                path = save_hints(
                    hints,
                    env_name=env_name,
                    seed=seed,
                    encoding_method=encoding_method,
                    num_demos=len(num_initial_states),
                    flag=structured,
                    out_dir="new_hints",
                )
                logging.info(f"Hints saved to {path}")


if __name__ == "__main__":
    main()
