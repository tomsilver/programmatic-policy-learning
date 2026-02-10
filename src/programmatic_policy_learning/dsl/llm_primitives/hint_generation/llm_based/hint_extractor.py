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
Based on the demonstrations, describe the strategy the expert is using to take action a 
given state s. Explain it in one paragraph. No extra output.
In another paragraph, describe the dynamics of the game that you've observed.

----
{place_holder}
"""


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
    return f"""
You are given expert demonstrations from a grid-based environment.

Each demonstration consists of transitions (s_t, a_t, s_t+1) where a_t = (row, col).

DEMONSTRATIONS:
{trajectories_text}

TASK:
Infer the expert’s decision rule for choosing action a from state s.

OUTPUT (no extra text):

1) POLICY RULES (8–15 bullets):
Write rules as testable predicates over (s, a). Use relative positions to the action cell and directions (up/down/left/right/diagonals).
If a rule depends on patterns “along a direction”, describe it using phrases like “first hit”, “blocked by”, “before reaching”, or “until”.
After each bullet, add “(seen in ~N/M demos)”.

2) COUNTERFACTUAL DISTRACTORS (5 bullets):
Write patterns that often appear near the chosen action but where the expert does NOT click. Use the same predicate style.

3) GAME DYNAMICS (5–10 bullets):
Describe deterministic transition patterns you can infer from s_t → s_t+1 (movement, falling, barriers, redraw, etc.).

{place_holder}
"""


def build_hint_prompt_v2(trajectories_text: str) -> str:
    """Build the LLM prompt for extracting strategy hints from demonstrations.

    trajectories_text:
        A human-readable representation of expert trajectories
        (ASCII, object-based, etc.)
    """
    return f"""
## SYSTEM
You are given expert demonstrations from a grid-based environment.

Your task is to analyze the demonstrations and infer what kinds of
state-level queries or relational measurements would be necessary
to reproduce the expert’s behavior.

Do NOT assume any predefined object roles or game semantics.
Do NOT invent rules not supported by the demonstrations.

## INPUT
Each demonstration consists of:
- a grid observation (before action),
- a clicked cell (row, col),
- the resulting grid after a deterministic transition.

## TASK
Based on the demonstrations, describe at a high level:

- What kinds of information about the grid state must be measurable
  in order to explain the expert’s action choices.
- What kinds of spatial, relational, or temporal relationships
  appear to matter.
- What kinds of distinctions between action effects are required
  (e.g., actions that change state directionally vs. actions that do not).

IMPORTANT:
- Do NOT define persistent entity types or value-classes (no “the movable entity”, no “type A/B”, no placeholders like V1/V2).
- Do NOT track identities across time (avoid phrasing like “the same object moves”).
- Describe requirements only in terms of observable transition statistics and action-effect signatures.
- Each bullet must start with: “We need the ability to …”

Express your answer in terms of **needed functions, relations, or measurements**
over the grid (e.g., distance between cells satisfying some property,
relative position of selected cells, whether an action produces a directional change).

Do NOT name or infer object roles (e.g., agent, target).
Do NOT name specific grid symbols or controls.
Do NOT write code.
Do NOT propose a DSL or formal grammar.

Be concise and factual.
If something is uncertain, explicitly state the uncertainty.

## DEMONSTRATIONS
{trajectories_text}\n

AGAIN DOUBLE CHECK THE RESPONSE AND AVOID OBJECT TYPES (such as agent, wall, etc.)
"""


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
        prompt = build_hint_structured(trajectories_text, env_name, encoding_method)
    else:
        prompt = build_hint_prompt_v1(trajectories_text, env_name, encoding_method)
    prompt = f"{prompt}\n\nSEED: {seed}\n"
    print(prompt)
    input()
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


def build_dsl_generation_prompt_final(
    hint_text: str,
    grid_object_types: Sequence[str],
    *,
    min_primitives: int = 5,
    max_primitives: int = 10,
) -> str:
    """DSL generation prompt."""

    type_system = r"""
Type system:
- Obs := grid observation (2D grid of discrete values)
- Cell := (int, int)                # (row, col)
- Value := one of OBJECT_TYPES
- Direction := (int, int)           # (dr, dc)
- Bool := {True, False}

Fixed nonterminals and denotations:
- START: a Bool expression evaluated in environment {a: Cell, s: Obs}
- LOCAL_PROGRAM: a function (Cell, Obs) -> Bool
- CONDITION: a function (Cell, Obs) -> Bool
- DIRECTION: a Direction constant
- VALUE: a Value constant
""".strip()

    primitive_constraints = r"""
Primitive constraints (binding):
- Propose between MIN_PRIMITIVES and MAX_PRIMITIVES primitives.
- EVERY primitive must:
  (1) be executable Python,
  (2) return Bool,
  (3) take Cell and Obs as the LAST TWO arguments: (..., cell: Cell, obs: Obs) -> Bool
  (4) may take additional typed args (Value, Direction, Int, Bool, Cell, (Cell,Obs)->Bool, etc.).
- Names should be descriptive and generic; do NOT use domain-specific words (agent/target/wall/arrow/...).
- Avoid redundancy: primitives should cover distinct families of checks.
""".strip()

    grammar_rules = r"""
Grammar rules (binding):
- You must output a full grammar with productions for:
  START, LOCAL_PROGRAM, CONDITION, DIRECTION, VALUE.

- Cross-links are allowed if type-correct:
  e.g., START may expand to LOCAL_PROGRAM(a, s) or CONDITION(a, s),
        LOCAL_PROGRAM may expand to CONDITION,
        CONDITION may expand to LOCAL_PROGRAM, etc.
  (You decide which links are useful, but they must be type-safe.)

- Primitive uniqueness per nonterminal (very important):
  Each primitive name may appear as a direct terminal application in the production list
  of AT MOST ONE of {START, LOCAL_PROGRAM, CONDITION}.
  (You can still use nonterminal cross-links to share structure instead of repeating primitives.)
  Example: if "is_value" appears in CONDITION productions, it must not appear directly in START or LOCAL_PROGRAM productions.

- START environment variables:
  In START productions, use variables (a, s) (not (cell, obs)).
  If you call a predicate of type (Cell,Obs)->Bool, you must pass (a, s) into it.
""".strip()

    recursion_constraints = r"""
Recursion + productivity constraints (mandatory):
1) Reachability: every nonterminal must be reachable from START.
2) Productivity: the grammar must generate MANY distinct programs.
   Include at least one productive recursive cycle involving CONDITION and/or LOCAL_PROGRAM,
   where recursion strictly grows the expression (not CONDITION -> CONDITION).
3) Termination: include base cases so derivations can end.
4) Combinatorial growth: use boolean composition or other branching structure to yield many programs.
""".strip()

    output_schema = r"""
Output ONLY valid JSON with this schema:

{
  "object_types": ["..."],

  "types": {
    "Obs": "...",
    "Cell": "...",
    "Value": "...",
    "Direction": "...",
    "Bool": "..."
  },

  "primitives": [
    {
      "name": "primitive_name",
      "type": "(..., Cell, Obs) -> Bool",
      "args": [
        {"name": "...", "type": "Value|Direction|Int|Bool|Cell|(Cell,Obs)->Bool|..."},
        {"name": "cell", "type": "Cell"},
        {"name": "obs", "type": "Obs"}
      ],
      "description": "generic description; no domain-specific words",
      "python_impl": "def primitive_name(..., cell, obs):\n    ...\n"
    }
  ],

  "grammar": {
    "nonterminals": ["START","LOCAL_PROGRAM","CONDITION","DIRECTION","VALUE"],
    "start_symbol": "START",
    "productions": {
      "START": [
        "... Bool expression using a,s and/or calling LOCAL_PROGRAM/CONDITION with (a,s) ...",
        "..."
      ],
      "LOCAL_PROGRAM": [
        "... expression of type (Cell,Obs)->Bool ...",
        "..."
      ],
      "CONDITION": [
        "... expression of type (Cell,Obs)->Bool ...",
        "..."
      ],
      "DIRECTION": ["(1,0)", "(0,1)", "(-1,0)", "(0,-1)", "(1,1)", "(-1,1)", "(1,-1)", "(-1,-1)"],
      "VALUE": ["<OBJECT_TYPES>"]
    }
  },

  "sanity": {
    "type_safe": true,
    "all_nonterminals_reachable_from_start": true,
    "has_productive_recursion": true,
    "has_base_cases": true,
    "num_primitives": "between MIN_PRIMITIVES and MAX_PRIMITIVES",
    "all_primitives_return_bool": true,
    "all_primitives_take_cell_obs_last": true,
    "start_uses_a_s_only": true,
    "primitive_used_in_only_one_nonterminal": true,
    "no_redundant_primitives": true
  },

  "notes": ["optional"]
}

Hard constraints:
- No text outside JSON.
- Every primitive returns Bool and ends with (..., cell, obs).
- In START productions, only (a,s) may appear as the state variables.
- Each primitive appears in productions of at most one of START/LOCAL_PROGRAM/CONDITION.
- Grammar must be type-correct and productively recursive.
""".strip()

    return f"""
## SYSTEM
You are an expert language designer for program synthesis systems.

Your job: given a high-level hint extracted from expert demonstrations,
propose:
(1) 5–10 generic Boolean grid-game primitives (Python implementations),
(2) a type-safe grammar over the fixed nonterminals START/LOCAL_PROGRAM/CONDITION/DIRECTION/VALUE,
that is expressive and productively recursive (generates many programs).

## INPUT 1: Level-1 hint (capability requirements)
{hint_text}

## INPUT 2: OBJECT_TYPES
{json.dumps(list(grid_object_types))}

## TYPE SYSTEM (binding)
{type_system}

MIN_PRIMITIVES = {min_primitives}
MAX_PRIMITIVES = {max_primitives}

## PRIMITIVE CONSTRAINTS (binding)
{primitive_constraints}

## GRAMMAR RULES (binding)
{grammar_rules}

## RECURSION / PRODUCTIVITY (binding)
{recursion_constraints}

## OUTPUT FORMAT
Output ONLY valid JSON following this schema:

{output_schema}
""".strip()


def main() -> None:
    """Entry point for running hint and DSL extraction."""
    max_steps_per_traj = 40
    seed = 0
    cache_path = Path("cache.db")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_client = OpenAIModel("gpt-4o-mini", cache)

    env_names = [
        # "Chase",
        # "TwoPileNim",
        "ReachForTheStar",
        # "StopTheFall"
        # "CheckmateTactic"
    ]
    encoding_methods = ["1"]  # "1",
    num_initial_states = 10
    structured_modes = [True]

    for env_name in env_names:
        for encoding_method in encoding_methods:
            for structured in structured_modes:
                _configure_rng(seed)
                expert = get_grid_expert(env_name)
                trajectories: list[list[tuple[Any, Any, Any]]] = []
                object_types: Sequence[str] | None = None
                for init_idx in range(num_initial_states):
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
                    text = trajectory_serializer.trajectory_to_diff_text(
                        traj,
                        encoder=encoder,
                        max_steps=max_steps_per_traj,
                    )

                    # text = trajectory_serializer.trajectory_to_text(
                    #     traj,
                    #     encoder=encoder,
                    #     analyzer=analyzer,
                    #     salient_tokens=grid_hint_config.SALIENT_TOKENS[env_name],
                    #     encoding_method=encoding_method,
                    #     max_steps=max_steps_per_traj,
                    # )
                    all_traj_texts.append(f"\n---[TRAJECTORY {idx}]---\n{text}\n\n")
                print(all_traj_texts)
                input()
                combined_text = "\n\n".join(all_traj_texts)
                hints = extract_hints(
                    llm_client,
                    combined_text,
                    env_name,
                    seed,
                    encoding_method,
                    structured,
                )
                # dsl_prompt = build_dsl_generation_prompt_final(hints, object_types)
                # output = extract_dsl(llm_client, dsl_prompt)

                path = save_hints(
                    hints,
                    env_name=env_name,
                    seed=seed,
                    encoding_method=encoding_method,
                    num_demos=num_initial_states,
                    flag=structured,
                    out_dir="final_hints",
                )
                logging.info(f"Hints saved to {path}")


if __name__ == "__main__":
    main()
