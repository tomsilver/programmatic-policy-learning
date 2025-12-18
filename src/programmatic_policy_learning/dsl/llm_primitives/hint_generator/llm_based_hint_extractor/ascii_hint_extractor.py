"""Script for extracting textual hints from grid trajectories."""

from __future__ import annotations

from collections import Counter
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

We ONLY control the agent, not other entities.

Your job:
1) Infer the agent’s high-level objective.
2) Extract RECURRING spatial–relational patterns used for decision-making.
3) Identify directional asymmetries, alignment behavior, and distance-based cues.
4) Produce NON-CHEATY hints that could guide a symbolic policy or DSL design.

Hard constraints:
- Do NOT propose DSL primitives, function names, or code.
- Use descriptive relational language only.
- Focus on observable spatial relations, not abstract strategy names.
- If uncertain, say so.

You are given a sequence of expert state transitions below.

Each step contains:
- ASCII grid of the state
- Coordinate-based object listing
- Action taken
- Observed transition effects

---

{TRAJECTORY_TEXT}

---

Output ONLY the following template:

## DEMONSTRATION-INFERRED FEATURES (HINTS)

### High-frequency relational patterns:
- ...

### Useful directional / asymmetry relations:
- ...

### Example state–action correlations:
- ...

### Frequently observed local spatial configurations:
- ...

### Observed distance thresholds or step ranges:
- ...
"""


def build_hint_prompt(trajectory_text: str) -> str:
    """Wrap the serialized trajectory text in the hint prompt."""
    return HINT_EXTRACTION_PROMPT_TEMPLATE.format(TRAJECTORY_TEXT=trajectory_text)


def run_chase_example(
    llm_client: PretrainedLargeModel,
    env_factory: Callable[[int, str], Any],
    num_initial_states: int = 10,
    max_steps_per_traj: int = 40,
) -> str:
    """Collect multiple Chase trajectories and summarise hints via the LLM."""

    # ------------------------------------------------------------
    # 1) Setup encoder + analyzer
    # ------------------------------------------------------------
    symbol_map = {
        "empty": ".",
        "agent": "A",
        "target": "T",
        "wall": "#",
        "drawn": "*",
        "left_arrow": "<",
        "right_arrow": ">",
        "up_arrow": "^",
        "down_arrow": "v",
    }

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
    expert = get_grid_expert("Chase")
    trajectories: list[list[tuple[Any, Any, Any]]] = []

    for init_idx in range(num_initial_states):
        env = env_factory(init_idx, "Chase")
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
            salient_tokens=grid_hint_config.SALIENT_TOKENS["Chase"],
            max_steps=max_steps_per_traj,
        )

        print(f"\n=== TEXT TO FEED INTO LLM (trajectory {i}) ===\n")
        print(text)

        prompt = build_hint_prompt(text)
        query = Query(prompt)
        per_traj_checks: list[RepromptCheck] = []  # TODOO: add actual checks

        response = query_with_reprompts(
            llm_client,
            query,
            per_traj_checks,  # type: ignore[arg-type]
            max_attempts=5,
        )

        print(f"\n=== HINTS FROM TRAJECTORY {i} ===\n")
        print(response.text)

        per_traj_hints.append(response.text)

    # ------------------------------------------------------------
    # 4) Aggregate hints (machine-side)
    # ------------------------------------------------------------
    counter: Counter[str] = Counter()
    for hint_block in per_traj_hints:
        for line in hint_block.splitlines():
            line = line.strip()
            if line.startswith("-"):
                counter[line] += 1

    print("\n=== AGGREGATED RAW HINT COUNTS ===\n")
    for k, v in counter.most_common():
        print(f"{k}  [{v}/{num_initial_states}]")

    # ------------------------------------------------------------
    # 5) Final abstraction pass (LLM)
    # ------------------------------------------------------------
    summary_lines = [
        f"{k}  (observed in {v}/{num_initial_states} trajectories)"
        for k, v in counter.most_common()
    ]

    aggregation_text = "\n".join(summary_lines)

    final_prompt = f"""
You are analyzing expert behavior across multiple trajectories
in a grid-based Chase environment.

Below are relational patterns extracted independently from
{num_initial_states} expert demonstrations.

Your task:
- Identify invariant decision-making structures
- Downweight rare or inconsistent patterns
- Produce concise, reusable relational hints
- Do NOT invent new behavior
- Do NOT propose code, DSL primitives, or function names

Patterns:
{aggregation_text}


Output ONLY the structured hint template used previously.
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

    return final_response.text
