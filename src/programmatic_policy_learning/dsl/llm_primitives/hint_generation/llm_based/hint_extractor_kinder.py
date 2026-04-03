"""LLM-based hint extractor for KinDER Motion2D."""

from __future__ import annotations

import json
import logging
import random

# import time
from pathlib import Path
from typing import Any, Callable, Protocol, TypeGuard, cast

import numpy as np
from omegaconf import OmegaConf
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel, PretrainedLargeModel
from prpl_llm_utils.reprompting import RepromptCheck, query_with_reprompts
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.approaches.experts.kinder_experts import (
    create_kinder_expert,
)
from programmatic_policy_learning.dsl.llm_primitives.env_specs import (
    get_env_llm_spec,
)
from programmatic_policy_learning.envs.registry import EnvRegistry


class _StatefulExpert(Protocol):
    """Minimal protocol for stateful experts used in rollout collection."""

    def reset(self, obs: Any, info: Any) -> None:
        """Reset the expert to the current environment state."""

    def step(self) -> Any:
        """Produce the next expert action."""

    def update(self, obs: Any, reward: float, done: bool, info: Any) -> None:
        """Update the expert with the latest transition."""


ExpertPolicy = Callable[[Any], Any] | _StatefulExpert


def _is_stateful_expert(expert_fn: ExpertPolicy) -> TypeGuard[_StatefulExpert]:
    """Return True when the expert exposes reset/step/update methods."""
    return all(hasattr(expert_fn, attr) for attr in ("reset", "step", "update"))


def _configure_rng(seed: int) -> None:
    """Seed Python/NumPy RNGs for reproducible demos."""
    random.seed(seed)
    np.random.seed(seed)


def env_factory(
    instance_num: int | None = None,
    env_name: str | None = None,
    *,
    num_passages: int = 1,
) -> Any:
    """Env Factory."""
    if env_name is None:
        raise ValueError("env_name must be provided.")
    registry = EnvRegistry()
    return registry.load(
        OmegaConf.create(
            {
                "provider": "kinder",
                "num_passages": num_passages,
                "action_mode": "continuous",
                "make_kwargs": {
                    "base_name": f"{env_name}-p{num_passages}",
                    "id": f"kinder/{env_name}-p{num_passages}-v0",
                    "render_mode": "rgb_array",
                },
                "instance_num": instance_num,
            }
        )
    )


def collect_full_episode(
    env: Any,
    expert_fn: ExpertPolicy,
    max_steps: int = 200,
    sample_count: int | None = None,
) -> list[tuple[Any, Any, Any]]:
    """Roll out expert policy, optionally sampling a subset of (obs, action,
    obs_next)."""
    obs, info = env.reset()
    trajectory = []
    stateful_expert: _StatefulExpert | None = None
    callable_expert: Callable[[Any], Any] | None = None
    if _is_stateful_expert(expert_fn):
        stateful_expert = expert_fn
        stateful_expert.reset(obs, info)
    else:
        callable_expert = cast(Callable[[Any], Any], expert_fn)

    for _ in range(max_steps):
        if stateful_expert is not None:
            action = stateful_expert.step()
        else:
            assert callable_expert is not None
            action = callable_expert(obs)
        obs_next, reward, term, trunc, next_info = env.step(action)
        trajectory.append((obs, action, obs_next))
        if stateful_expert is not None:
            stateful_expert.update(
                obs_next,
                float(reward),
                bool(term or trunc),
                next_info,
            )
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


def _load_hint_prompt_template(filename: str) -> str:
    prompt_dir = Path(__file__).resolve().parent / "hint_gen_prompt"
    prompt_path = prompt_dir / filename
    return prompt_path.read_text(encoding="utf-8")


def build_hint_prompt(trajectories_text: str) -> str:
    """Build the structured KinDER hint prompt for enc_5 trajectories."""
    template = _load_hint_prompt_template("new_hint_structured_2.txt")
    return template.replace("<<DEMONSTRATIONS>>", trajectories_text)


def extract_hints(
    llm_client: PretrainedLargeModel,
    trajectories_text: str,
    seed: int,
) -> str:
    """Query the LLM and return structured hint text."""
    prompt = build_hint_prompt(trajectories_text)
    prompt = f"{prompt}\n\nSEED: {seed}\n"
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


def save_hints(
    hints: dict[str, Any] | str,
    env_name: str,
    seed: int,
    num_demos: int,
    out_dir: Path | str = "hints",
    filename: str | None = None,
) -> Path:
    """
    Save extracted hints under:
        hints/{env_name}/{filename}
    """
    out_dir = Path(out_dir) / env_name / "enc_5" / "structured"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = filename or f"hints_seed{seed}_{num_demos}.json"
    out_path = out_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(hints, str):
        out_path.write_text(hints, encoding="utf-8")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(hints, f, indent=2)

    return out_path


def main() -> None:
    """Collect Motion2D demos, serialize them with enc_5, and query hints."""
    max_steps_per_traj = 80
    seed = 0
    num_passages = 0
    llm_model = "gpt-4.1"
    model_slug = "".join(ch if ch.isalnum() else "_" for ch in llm_model).strip("_")
    cache_path = Path(f"hint_cache_{model_slug}.db")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_client = OpenAIModel(llm_model, cache)

    env_names = ["Motion2D"]
    num_initial_states = [0, 1]
    for env_name in env_names:
        _configure_rng(seed)
        trajectories: list[list[tuple[Any, Any, Any]]] = []
        for init_idx in num_initial_states:
            print(f"Collecting trajectory for {env_name}, init_idx={init_idx}...")
            env = env_factory(init_idx, env_name, num_passages=num_passages)
            expert = create_kinder_expert(
                env_name,
                env.action_space,
                seed=seed + init_idx,
                observation_space=env.observation_space,
                num_passages=num_passages,
                expert_kind="bilevel",
            )
            traj = collect_full_episode(env, expert, sample_count=None)
            trajectories.append(traj)
            env.close()

        llm_spec = get_env_llm_spec(
            env_name,
            env_specs={
                "action_mode": "continuous",
                "num_passages": num_passages,
            },
        )
        combined_text = llm_spec.serialize_demonstrations(
            trajectories,
            encoding_method="enc_5",
            max_steps=max_steps_per_traj,
        )
        hints = extract_hints(
            llm_client,
            combined_text,
            seed,
        )
        path = save_hints(
            hints,
            env_name=f"{env_name}-p{num_passages}",
            seed=seed,
            num_demos=len(num_initial_states),
            out_dir="new_hints_kinder",
        )
        logging.info(f"Hints saved to {path}")


if __name__ == "__main__":
    main()
