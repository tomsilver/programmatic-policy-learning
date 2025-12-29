"""Script for extracting textual hints from grid trajectories."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.code import (
    FunctionOutputRepromptCheck,
    SyntaxRepromptCheck,
)
from prpl_llm_utils.models import OpenAIModel, PretrainedLargeModel
from prpl_llm_utils.reprompting import RepromptCheck, query_with_reprompts
from prpl_llm_utils.structs import Query

from programmatic_policy_learning.approaches.experts.grid_experts import (
    get_grid_expert,
)
from programmatic_policy_learning.approaches.utils import run_single_episode

# pylint: disable=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
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

========================
Your task:
Infer the rule used to choose the action from the observation, then implement it.

OUTPUT FORMAT (STRICT):
- Return ONLY executable Python code.
- The function MUST return a valid (row, col) tuple on EVERY call (returning None is NOT allowed).
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
    reprompt_checks: list[RepromptCheck] = [
        SyntaxRepromptCheck(),
        # FunctionOutputRepromptCheck(
        #     function_name,
        #     [(self.example_observation,)],
        #     [self.action_space.contains],
        # ),
    ]
    response = query_with_reprompts(
        llm_client,
        query,
        reprompt_checks=reprompt_checks,
        max_attempts=5,
    )

    final_hint = response.text
    return final_hint


def _parse_cli_args() -> argparse.Namespace:
    """Return CLI args for batch CaP experiments."""

    parser = argparse.ArgumentParser(
        description="Generate code-as-policies hints for multiple encodings/seeds."
    )
    parser.add_argument("--env", required=True, help="Grid environment name.")
    parser.add_argument(
        "--encodings",
        nargs="+",
        default=["3"],
        help="Encoding modes to evaluate (matching `run`).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Random seeds for reproducible rollouts.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="LLM identifier passed to OpenAIModel (default: gpt-4.1).",
    )
    parser.add_argument(
        "--num-initial-states",
        type=int,
        default=4,
        help="Number of expert rollouts per run.",
    )
    parser.add_argument(
        "--max-steps-per-traj",
        type=int,
        default=40,
        help="Maximum trajectory length fed to the LLM.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/CaP"),
        help="Directory for storing per-run outputs and metadata.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("baseline_llm_cache.db"),
        help="SQLite cache used by the OpenAI model wrapper.",
    )
    parser.add_argument(
        "--sleep-between-runs",
        type=float,
        default=60.0,
        help="Seconds to sleep between sequential runs to throttle LLM queries.",
    )
    parser.add_argument(
        "--function-name",
        default="policy",
        help="Expected name of the generated policy function.",
    )
    parser.add_argument(
        "--eval-env-nums",
        nargs="*",
        type=int,
        default=[11,12,13,14,15,16,17,18,19],
        help="Optional list of environment indices for post-generation evaluation.",
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=100,
        help="Maximum number of steps per evaluation rollout.",
    )
    parser.add_argument(
        "--eval-run-expert",
        type=bool,
        default=True,
        help="Also evaluate the handcrafted expert policy during evaluation.",
    )
    parser.add_argument(
        "--expert-reference-seed",
        type=int,
        default=0,
        help="Only run the expert when the current seed equals this value.",
    )
    parser.add_argument(
        "--plot-results",
        action="store_true",
        help="Generate a plot summarizing averaged CaP vs expert success rates.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional destination for the plot. Defaults to <output_dir>/<env>.",
    )
    return parser.parse_args()


def _configure_rng(seed: int) -> None:
    """Seed Python/NumPy RNGs for deterministic rollouts."""

    random.seed(seed)
    np.random.seed(seed)


def _strip_code_block(text: str) -> str:
    """Remove Markdown fences from code."""

    text = text.strip()
    if "```" in text:
        if "```python" in text:
            text = text.split("```python", 1)[1]
        else:
            text = text.split("```", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    return text


def _compile_policy_function(code: str, function_name: str) -> Callable[[Any], Any]:
    """Compile Python code defining `function_name` into a callable."""

    globals_dict: dict[str, Any] = {}
    locals_dict: dict[str, Any] = {}
    exec(code, globals_dict, locals_dict)  # pylint: disable=exec-used

    if function_name in locals_dict:
        fn = locals_dict[function_name]
    elif function_name in globals_dict:
        fn = globals_dict[function_name]
    else:
        raise RuntimeError(f"Function '{function_name}' not found in generated code.")

    if not callable(fn):
        raise RuntimeError(f"'{function_name}' is not callable.")

    return fn


def _evaluate_policy_function(
    policy_fn: Callable[[Any], Any],
    env_builder: Callable[[int], Any],
    test_env_nums: Sequence[int],
    max_num_steps: int,
    *,
    env_name: str,
    run_expert: bool,
) -> tuple[list[bool], list[bool]]:
    """Roll out CaP (and optional expert) policies on the requested envs."""

    cap_results: list[bool] = []
    expert_results: list[bool] = []
    expert_fn = get_grid_expert(env_name) if run_expert else None
    print(run_expert)

    for env_idx in test_env_nums:
        env = env_builder(env_idx)
        cap_success = run_single_episode(
            env, policy_fn, max_num_steps=max_num_steps
        ) > 0
        cap_results.append(cap_success)
        env.close()

        if run_expert:
            if expert_fn is None:
                raise RuntimeError("Expert policy unavailable.")
            env_e = env_builder(env_idx)
            expert_success = (
                run_single_episode(env_e, expert_fn, max_num_steps=max_num_steps) > 0
            )
            print(expert_success)
            expert_results.append(expert_success)
            env_e.close()

    return cap_results, expert_results


def bootstrap_ci_success(
    successes: Sequence[bool],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Return mean success and half-width of a bootstrap CI."""
    rng = np.random.default_rng(seed)
    s = np.asarray(successes, dtype=float)
    N = len(s)
    if N == 0:
        raise ValueError("Cannot compute CI for an empty list of successes.")

    boots = rng.choice(s, size=(n_boot, N), replace=True).mean(axis=1)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return s.mean(), (hi - lo) / 2


def plot_expert_vs_cap(
    labels: Sequence[str],
    expert_means: Sequence[float],
    expert_cis: Sequence[float],
    cap_means: Sequence[float],
    cap_cis: Sequence[float],
    *,
    title: str,
    save_path: str | Path,
) -> None:
    """Bar plot comparing expert and CaP performance with error bars."""

    if not labels:
        raise ValueError("No labels supplied for plotting.")

    x = np.arange(len(labels))
    width = 0.22  # thinner bars

    _, ax = plt.subplots(figsize=(9, 4.2))

    # Expert bars
    ax.bar(
        x - width / 2,
        expert_means,
        width,
        yerr=expert_cis,
        label="Expert",
        capsize=4,
        facecolor="none",
        edgecolor="dimgray",
        linewidth=2,
    )

    # CaP bars
    ax.bar(
        x + width / 2,
        cap_means,
        width,
        yerr=cap_cis,
        label="CaP",
        capsize=4,
        facecolor="tab:blue",
        edgecolor="tab:blue",
        alpha=0.7,
    )

    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 105)

    ax.set_title(title)
    ax.legend(frameon=False)

    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    logging.info("Saved CaP vs Expert plot to %s", save_path.resolve())
    plt.close()


def _prepare_results_for_plot(
    encoding_eval_results: dict[str, dict[str, Any]],
    encoding_order: Sequence[str],
) -> tuple[list[str], list[float], list[float], list[float], list[float]]:
    """Aggregate evaluation results per encoding for plotting."""

    labels: list[str] = []
    expert_means: list[float] = []
    expert_cis: list[float] = []
    cap_means: list[float] = []
    cap_cis: list[float] = []

    for encoding in encoding_order:
        stats = encoding_eval_results.get(encoding)
        if not stats:
            continue
        cap_runs = stats.get("cap_runs", [])
        expert_run = stats.get("expert_run")
        if not cap_runs:
            logging.warning("No CaP evaluation results recorded for encoding %s", encoding)
            continue
        if expert_run is None:
            logging.warning(
                "Expert results missing for encoding %s; skipping it in the plot.", encoding
            )
            continue

        cap_flat = [bool(result) for run in cap_runs for result in run]
        cap_mean, cap_ci = bootstrap_ci_success(cap_flat)
        expert_mean, expert_ci = bootstrap_ci_success(expert_run)

        labels.append(f"enc_{encoding}")
        cap_means.append(100 * cap_mean)
        cap_cis.append(100 * cap_ci)
        expert_means.append(100 * expert_mean)
        expert_cis.append(100 * expert_ci)

    return labels, expert_means, expert_cis, cap_means, cap_cis


def main() -> None:
    """Batch entry-point mirroring run_code_as_policies."""

    args = _parse_cli_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = SQLite3PretrainedLargeModelCache(args.cache_path)
    client = OpenAIModel(args.model, cache)

    summary: list[dict[str, str | int]] = []
    encoding_eval_results: dict[str, dict[str, Any]] = {}
    for encoding in args.encodings:
        encoding_dir = args.output_dir / args.env / f"encoding_{encoding}"
        encoding_dir.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            _configure_rng(seed)
            final_hint = run(
                client,
                args.env,
                encoding,
                num_initial_states=args.num_initial_states,
                max_steps_per_traj=args.max_steps_per_traj,
            )

            policy_filename = f"policy_seed{seed}.txt"
            hint_path = encoding_dir / policy_filename
            hint_path.write_text(final_hint, encoding="utf-8")

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            metadata = {
                "env": args.env,
                "encoding": encoding,
                "seed": seed,
                "model": args.model,
                "num_initial_states": args.num_initial_states,
                "max_steps_per_traj": args.max_steps_per_traj,
                "timestamp": timestamp,
                "hint_path": str(hint_path.resolve()),
            }
            meta_path = encoding_dir / f"metadata_seed{seed}.json"
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            summary.append(
                {
                    "env": args.env,
                    "encoding": encoding,
                    "seed": seed,
                    "hint_file": str(hint_path),
                }
            )

            if args.eval_env_nums:
                code_str = _strip_code_block(final_hint)
                policy_fn = _compile_policy_function(code_str, args.function_name)
                env_builder = lambda idx: env_factory(idx, args.env)
                # print(args.eval_run_expert)
                # print(args.expert_reference_seed)
                # print(seed)

                run_expert = args.eval_run_expert and seed == args.expert_reference_seed
                # print(run_expert)
                # input()
                cap_results, expert_results = _evaluate_policy_function(
                    policy_fn,
                    env_builder,
                    args.eval_env_nums,
                    args.eval_max_steps,
                    env_name=args.env,
                    run_expert=run_expert,
                )
                enc_summary = encoding_eval_results.setdefault(
                    encoding,
                    {"cap_runs": [], "expert_run": None},
                )
                enc_summary["cap_runs"].append(cap_results)
                if run_expert:
                    enc_summary["expert_run"] = expert_results
                metadata.setdefault("evaluation", {})
                metadata["evaluation"] = {
                    "cap_results": cap_results,
                    "expert_results": expert_results,
                }
                meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            time.sleep(args.sleep_between_runs)
    print(encoding_eval_results)
    input()
    if args.plot_results:
        if not args.eval_env_nums:
            logging.warning(
                "Plotting requested but no evaluation environments were configured; skipping."
            )
        elif not encoding_eval_results:
            logging.warning(
                "Plotting requested but no evaluation results were recorded; skipping."
            )
        else:
            (
                labels,
                expert_means,
                expert_cis,
                cap_means,
                cap_cis,
            ) = _prepare_results_for_plot(encoding_eval_results, args.encodings)
            if labels:
                plot_path = args.plot_path
                if plot_path is None:
                    plot_path = args.output_dir / args.env / "cap_vs_expert.png"
                title = (
                    f"{args.env}: Expert vs CaP "
                    f"(averaged over {len(args.seeds)} seed{'s' if len(args.seeds) > 1 else ''})"
                )
                plot_expert_vs_cap(
                    labels,
                    expert_means,
                    expert_cis,
                    cap_means,
                    cap_cis,
                    title=title,
                    save_path=plot_path,
                )
            else:
                logging.warning(
                    "Plotting requested but no encodings had both CaP and expert results."
                )

    manifest_path = args.output_dir / args.env / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
