
from typing import Any
from programmatic_policy_learning.envs.registry import EnvRegistry
from omegaconf import OmegaConf
from programmatic_policy_learning.approaches.experts.grid_experts import get_grid_expert
import tempfile
from prpl_llm_utils.structs import Query
from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel
from pathlib import Path
from programmatic_policy_learning.approaches.experts.grid_experts import get_grid_expert
from grid_encoder import GridStateEncoder, GridStateEncoderConfig
from transition_analyzer import GenericTransitionAnalyzer
from trajectory_serializer import trajectory_to_text
from grid_hint_config import SALIENT_TOKENS
from prpl_llm_utils.reprompting import query_with_reprompts


def collect_full_episode(env, expert_fn, max_steps=200):
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
def run_chase_example(llm_client):
    # ------------------------------------------------------------
    # 1) Roll out a full expert episode
    # ------------------------------------------------------------
    env = env_factory(0, "Chase")
    expert = get_grid_expert("Chase")

    trajectory = collect_full_episode(env, expert)
    env.close()
    print("DPNE")
    # ------------------------------------------------------------
    # 2) Build encoder (ASCII + coordinates)
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

    enc_cfg = GridStateEncoderConfig(
        symbol_map=symbol_map,
        empty_token="empty",
        coordinate_style="rc",  # (row, col)
    )
    encoder = GridStateEncoder(enc_cfg)

    # ------------------------------------------------------------
    # 3) Transition analyzer (generic works for Chase)
    # ------------------------------------------------------------
    analyzer = GenericTransitionAnalyzer()

    # ------------------------------------------------------------
    # 4) Convert trajectory → structured text evidence
    # ------------------------------------------------------------
    trajectory_text = trajectory_to_text(
        trajectory,
        encoder=encoder,
        analyzer=analyzer,
        salient_tokens=SALIENT_TOKENS["Chase"],
        max_steps=40,  # keep prompt size sane
    )

    print("\n=== TEXT FED INTO LLM ===\n")
    print(trajectory_text)

    # ------------------------------------------------------------
    # 5) Wrap text in hint-extraction prompt and query LLM
    # ------------------------------------------------------------
    hint_prompt = build_hint_prompt(trajectory_text)

    reprompt_checks = [  # TODOO: Add for full version
    ]
    query = Query(hint_prompt)
    response = query_with_reprompts(
        llm_client,
        query,
        reprompt_checks,  # type: ignore[arg-type]
        max_attempts=5,
    )
    hints = response.text

    print("\n=== PROMPT-READY EXPERT BEHAVIOR HINTS ===\n")
    print(hints)

    return hints


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
    return HINT_EXTRACTION_PROMPT_TEMPLATE.format(
        TRAJECTORY_TEXT=trajectory_text
    )

def extract_hints_with_llm(
    trajectory_text: str,
    llm: PretrainedLargeModel,
) -> str:
    prompt = build_hint_prompt(trajectory_text)
    query = Query(prompt)

    response = llm(query)

    return response.text


def run_hint_extraction_pipeline(
    trajectory,
    encoder,
    analyzer,
    llm,
    salient_tokens,
    max_steps: int = 40,
) -> str:
    # (1) Convert trajectory → text
    trajectory_text = trajectory_to_text(
        trajectory,
        encoder=encoder,
        analyzer=analyzer,
        salient_tokens=salient_tokens,
        max_steps=max_steps,
    )

    # (2–3) Ask LLM for hints
    hints = extract_hints_with_llm(
        trajectory_text=trajectory_text,
        llm=llm,
    )

    return hints


if __name__ == "__main__":
    print("MAIN")

    
    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_client = OpenAIModel("gpt-4.1", cache)
    # Run the Chase expert rollout + hint extraction
    run_chase_example(llm_client)
