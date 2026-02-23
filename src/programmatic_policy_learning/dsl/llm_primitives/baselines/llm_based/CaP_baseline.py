"""Script for running Code-as-Policies baseline."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import gymnasium
import kinder
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from prpl_llm_utils import models as llm_models
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.code import (
    FunctionOutputRepromptCheck,
    SyntaxRepromptCheck,
)
from prpl_llm_utils.models import OpenAIModel, PretrainedLargeModel
from prpl_llm_utils.reprompting import RepromptCheck, query_with_reprompts
from prpl_llm_utils.structs import Query
from tqdm import tqdm

from programmatic_policy_learning.approaches.experts.grid_experts import (
    get_grid_expert,
)
from programmatic_policy_learning.approaches.experts.kinder_experts import (
    create_kinder_expert,
)
from programmatic_policy_learning.approaches.utils import run_single_episode

# pylint: disable=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    continuous_encoder,
    continuous_hint_config,
    continuous_trajectory_serializer,
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
    *,
    start_obs: Any = None,
) -> list[tuple[Any, Any, Any]]:
    """Roll out expert policy, optionally sampling a subset of (obs, action,
    obs_next) transitions.

    If *start_obs* is provided the environment is **not** reset, avoiding a
    double-reset when the caller already called ``env.reset(seed=...)``.

    Parameters
    ----------
    env : Any
        Gymnasium-compatible environment instance.
    expert_fn : Callable[[Any], Any]
        Expert policy mapping an observation to an action.
    max_steps : int, optional
        Maximum number of environment steps (default 200).
    sample_count : int | None, optional
        If set, randomly sub-sample the trajectory down to this many
        transitions (always keeping the first step).  ``None`` returns the
        full trajectory.
    start_obs : Any, optional
        If provided, skip ``env.reset()`` and use this observation as the
        initial state.

    Returns
    -------
    list[tuple[Any, Any, Any]]
        List of ``(obs_t, action_t, obs_{t+1})`` transitions.

    Examples
    --------
    Collect a full expert trajectory (no sub-sampling)::

        env = gym.make("Motion2D-p1-v0")
        traj = collect_full_episode(env, expert_fn, max_steps=200)
        # traj[0] == (obs_0, action_0, obs_1)

    Collect 5 randomly sampled transitions (first step always kept)::

        traj = collect_full_episode(env, expert_fn, sample_count=5)
    """
    if start_obs is None:
        obs, _ = env.reset()
    else:
        obs = start_obs
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
    """Return the LLM prompt for summarising expert trajectories.

    Parameters
    ----------
    all_trajectories_text : str
        Concatenated text of all serialized expert trajectories.
    env_name : str
        Grid environment name (used to look up symbol maps).
    encoding : str
        Encoding mode (``"1"`` adds token-meaning metadata).

    Returns
    -------
    str
        Complete prompt string ready to send to the LLM.
    """
    token_meanings = ""
    action_mask = ""
    if encoding == "1":
        action_mask = "The ACTION MASK has a single 1 at the clicked cell."
        token_meanings = f"Token meanings: {grid_hint_config.SYMBOL_MAPS[env_name]}\nIMPORTANT: Observations are represented as object-type identifiers (e.g., {list(grid_hint_config.SYMBOL_MAPS[env_name].keys())}), not ASCII characters. Any inferred rule or policy must compare against object types, not symbols like '.' or '*'."

    return f"""
You are given a few expert demonstrations of the SAME task.
Each demonstration is a full trajectory from a different initial state.

IMPORTANT:
- Coordinates are 0-indexed (row, col) with (0,0) at top-left.
- An action is "Click cell (r,c)". {action_mask}
- Steps within a trajectory are temporally consecutive.
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
- Write the inferred rule as a docstring of that function.
- The function MUST return a valid (row, col) tuple on EVERY call (returning None is NOT allowed).
- The code MUST be wrapped in a Markdown code block that starts with ```python and ends with ```.
- Do NOT include any text, explanation, headings, or comments outside the code block.

CODE REQUIREMENTS:
- Define a function with signature: def policy(obs):
- The function takes a 2D NumPy array observation and returns an action (row, col).
- Use numeric coordinates only (NumPy convention: (0, 0) is top-left).
- Do NOT use words like left, right, up, or down.
- Do NOT import external libraries.

"""


def build_continuous_hint_prompt(
    all_trajectories_text: str,
    env_description: str,
    action_field_names: list[str],
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> str:
    """Return the LLM prompt for continuous-action expert trajectories.

    Parameters
    ----------
    all_trajectories_text : str
        Concatenated text of all serialized expert trajectories.
    env_description : str
        Human-readable task description inserted under ``TASK DESCRIPTION``.
    action_field_names : list[str]
        Names for each action dimension (e.g. ``["dx", "dy", ...]``).
    action_low : np.ndarray
        Per-dimension lower bounds of the action space.
    action_high : np.ndarray
        Per-dimension upper bounds of the action space.

    Returns
    -------
    str
        Complete prompt string ready to send to the LLM.

    Examples
    --------
    ::

        prompt = build_continuous_hint_prompt(
            combined_text,
            "A 2-D navigation task ...",
            ["dx", "dy", "dtheta", "darm", "vac"],
            action_space.low,
            action_space.high,
        )
        assert "TASK DESCRIPTION" in prompt
        assert "dx in [" in prompt
    """
    bounds_lines = ", ".join(
        f"{name} in [{lo:.4f}, {hi:.4f}]"
        for name, lo, hi in zip(action_field_names, action_low, action_high)
    )
    return f"""
You are given a few expert demonstrations of the SAME task.
Each demonstration is a full trajectory from a different initial state.

TASK DESCRIPTION:
{env_description}

IMPORTANT:
- The observation is a 1-D NumPy float array.  Field names are given in each step.
- The action is a NumPy array of shape ({len(action_field_names)},) with bounds: {bounds_lines}.
- Steps within a trajectory are temporally consecutive.
- Your job is to infer the underlying strategy/policy that generalizes across trajectories.
- Do NOT describe individual steps.
- Do NOT narrate what happens over time.
- Prefer rules that are consistent across trajectories; ignore trajectory-specific ones.

========================
DEMONSTRATIONS
========================
{all_trajectories_text}

========================
Your task:
Infer the rule used to choose the action from the observation, then implement it.

OUTPUT FORMAT (STRICT):
- Return ONLY executable Python code.
- Write the inferred rule as a docstring of that function.
- The function MUST return a valid NumPy array on EVERY call (returning None is NOT allowed).
- The code MUST be wrapped in a Markdown code block that starts with ```python and ends with ```.
- Do NOT include any text, explanation, headings, or comments outside the code block.

CODE REQUIREMENTS:
- Define a function with signature: def policy(obs):
- The function takes a 1-D NumPy float array and returns a NumPy array of shape ({len(action_field_names)},).
- You may use `np` (NumPy) and `math`. Both are pre-imported.
- Do NOT import external libraries.
- Ensure the returned values are within the action bounds.
- Return exactly: np.array([{', '.join(action_field_names)}], dtype=np.float32)
- Do NOT return a Python list or tuple.
"""


def env_factory(
    instance_num: int | None = None,
    env_name: str | None = None,
) -> Any:
    """Create a grid (GGG) environment instance.

    Parameters
    ----------
    instance_num : int | None, optional
        Board-layout index to load.
    env_name : str | None, optional
        Environment name (e.g. ``"TicTacToe"``).

    Returns
    -------
    Any
        A Gymnasium-compatible grid environment.

    Raises
    ------
    ValueError
        If ``env_name`` is ``None``.

    Examples
    --------
    ::

        env = env_factory(instance_num=0, env_name="TicTacToe")
        obs, info = env.reset()
        env.close()
    """
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


def continuous_env_factory(
    env_name: str,
    num_passages: int = 1,
    seed: int | None = None,
) -> tuple[Any, np.ndarray]:
    """Create a KinDER environment, reset it with *seed*, and return (env,
    obs). Currently only supports Motion2D with different numbers of
    passages (explicity specified).

    Unlike GGG environments (which use ``instance_num`` to load different
    board layouts), KinDER environments vary initial states via
    ``env.reset(seed=...)``.

    Parameters
    ----------
    env_name : str
        Environment family name (e.g. ``"Motion2D"``).
    num_passages : int, optional
        Number of wall passages for Motion2D variants (default 1).
    seed : int | None, optional
        Random seed passed to ``env.reset()``.

    Returns
    -------
    tuple[Any, np.ndarray]
        ``(env, initial_observation)`` pair.
    """
    env_name = continuous_hint_config.canonicalize_env_name(env_name)
    kinder.register_all_environments()
    if env_name == "Motion2D" and num_passages:
        env_id = f"kinder/{env_name}-p{num_passages}-v0"
    else:
        raise ValueError(
            f"Unsupported env_name={env_name!r} or num_passages={num_passages}. "
            "Add a branch in continuous_env_factory()."
        )
    env = kinder.make(env_id)
    assert isinstance(env.action_space, gymnasium.spaces.Box)
    obs: np.ndarray
    obs, _ = env.reset(seed=seed)
    return env, obs


def run(
    llm_client: PretrainedLargeModel,
    env_name: str,
    encoding_method: str,
    seed: int,
    num_initial_states: int = 10,
    max_steps_per_traj: int = 40,
    function_name: str = "policy",
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
    logging.info(
        "Collecting %d expert trajectories for %s ...", num_initial_states, env_name
    )
    expert = get_grid_expert(env_name)
    trajectories: list[list[tuple[Any, Any, Any]]] = []

    for init_idx in range(num_initial_states):
        env = env_factory(init_idx, env_name)
        traj = collect_full_episode(env, expert, sample_count=None)
        env.close()
        trajectories.append(traj)
    logging.info("Collected %d trajectories.", len(trajectories))

    # ------------------------------------------------------------
    # 3) All trajectories hint extraction
    # ------------------------------------------------------------

    all_traj_texts = []

    for i, traj in enumerate(trajectories):
        # text = trajectory_serializer.trajectory_to_diff_text(
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
    logging.info("Building prompt (encoding=%s) ...", encoding_method)
    prompt = build_joint_hint_prompt(combined_text, env_name, encoding_method)
    prompt = f"{prompt}\n\nSEED: {seed}\n"
    query = Query(
        prompt, hyperparameters={"temperature": 0.0, "seed": seed}
    )  # "top_p": 1.0

    # example demo for reprompt check
    env0 = env_factory(0, env_name)
    obs, _ = env0.reset()
    action_space = env0.action_space
    env0.close()
    reprompt_checks: list[RepromptCheck] = [
        SyntaxRepromptCheck(),
        FunctionOutputRepromptCheck(
            function_name,
            [(obs,)],
            [action_space.contains],
        ),
    ]
    logging.info("Querying LLM (%s) — this may take a while ...", llm_client.get_id())
    start_time = time.time()
    response = query_with_reprompts(
        llm_client,
        query,
        reprompt_checks=reprompt_checks,
        max_attempts=5,
    )
    elapsed_time = time.time() - start_time
    logging.info("LLM response received in %.2f seconds.", elapsed_time)
    logging.info("LLM response: %s", response.text[:100] + "...")
    final_code = response.text
    return final_code


def _parse_cli_args() -> argparse.Namespace:
    """Return CLI args for batch CaP experiments."""

    parser = argparse.ArgumentParser(
        description="Generate code-as-policies hints for multiple encodings/seeds."
    )
    parser.add_argument("--env", required=True, help="Grid environment name.")
    parser.add_argument(
        "--encodings",
        nargs="+",
        default=["4"],
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
        "--use-response-model",
        action="store_true",
        help=(
            "Use OpenAIResponsesModel instead of OpenAIModel. "
            "This is required for response-style models like gpt-5.2-pro."
        ),
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
        default=Path("logs/CaP_baseline"),
        help="Directory for storing per-run outputs and metadata.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("baseline_llm_cache.db"),
        help=(
            "Base SQLite cache filename. We automatically append env/encoding/seed "
            "to create per-run caches unless you reuse the same path manually."
        ),
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
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
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

    globals_dict: dict[str, Any] = {"np": np}
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

    for env_idx in tqdm(test_env_nums, desc="Evaluating envs"):
        env = env_builder(env_idx)

        def safe_policy(obs: Any, *, _env: Any = env) -> Any:
            """Guard policy execution to avoid crashes from invalid
            assumptions."""

            try:
                action = policy_fn(obs)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Exception: {e}")
                action = None
            if action is None:
                return _env.action_space.sample()
            return action

        cap_success = (
            run_single_episode(env, safe_policy, max_num_steps=max_num_steps) > 0
        )
        cap_results.append(cap_success)
        env.close()

        if run_expert:
            if expert_fn is None:
                raise RuntimeError("Expert policy unavailable.")
            env_e = env_builder(env_idx)
            expert_success = (
                run_single_episode(env_e, expert_fn, max_num_steps=max_num_steps) > 0
            )
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
            logging.warning(
                "No CaP evaluation results recorded for encoding %s", encoding
            )
            continue
        if expert_run is None:
            logging.warning(
                "Expert results missing for encoding %s; skipping it in the plot.",
                encoding,
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


def _load_eval_results_from_disk(
    output_dir: Path,
    env_name: str,
    num_initial_states: int,
    encodings: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Read saved metadata JSONs and rebuild encoding_eval_results.

    If *encodings* is None, auto-discover all ``encoding_*`` directories.
    """
    base_dir = output_dir / env_name / str(num_initial_states)
    if encodings is not None:
        enc_dirs = [(enc, base_dir / f"encoding_{enc}") for enc in encodings]
    else:
        enc_dirs = sorted(
            (d.name.removeprefix("encoding_"), d)
            for d in base_dir.iterdir()
            if d.is_dir() and d.name.startswith("encoding_")
        )
    encoding_eval_results: dict[str, dict[str, Any]] = {}
    for encoding, enc_dir in enc_dirs:
        if not enc_dir.exists():
            continue
        enc_summary: dict[str, Any] = {"cap_runs": [], "expert_run": None}
        for meta_file in sorted(enc_dir.glob("metadata_seed*.json")):
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if "evaluation" not in meta:
                logging.warning("No evaluation in %s, skipping.", meta_file)
                continue
            cap = meta["evaluation"]["cap_results"]
            expert = meta["evaluation"]["expert_results"]
            if cap:
                enc_summary["cap_runs"].append(cap)
            if expert and enc_summary["expert_run"] is None:
                enc_summary["expert_run"] = expert
        if enc_summary["cap_runs"]:
            encoding_eval_results[encoding] = enc_summary
            logging.info(
                "Loaded encoding %s: %d seed run(s), expert=%s",
                encoding,
                len(enc_summary["cap_runs"]),
                enc_summary["expert_run"] is not None,
            )
    return encoding_eval_results


def manual_eval() -> None:
    """Evaluate a compiled policy function and print results."""
    final_code = """
def policy(obs):
    n_rows, n_cols = obs.shape
    TOKEN, EMPTY = "token", "empty"

    candidates = []
    any_tokens = []

    for r in range(n_rows):
        for c in range(n_cols):
            if obs[r, c] == TOKEN:
                any_tokens.append((r, c))
                row_has_empty_elsewhere = False
                for c2 in range(n_cols):
                    if c2 != c and obs[r, c2] == EMPTY:
                        row_has_empty_elsewhere = True
                        break
                if row_has_empty_elsewhere:
                    candidates.append((r, c))

    if candidates:
        r_max = max(rc[0] for rc in candidates)
        best = min((rc for rc in candidates if rc[0] == r_max), key=lambda x: x[1])
        return best

    if any_tokens:
        r_max = max(rc[0] for rc in any_tokens)
        best = min((rc for rc in any_tokens if rc[0] == r_max), key=lambda x: x[1])
        return best

    return (0, 0)
    """
    args = _parse_cli_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_str = _strip_code_block(final_code)
    policy_fn = _compile_policy_function(code_str, "policy")
    env_builder = lambda idx: env_factory(idx, args.env)
    run_expert = True
    cap_results, _expert_results = _evaluate_policy_function(
        policy_fn,
        env_builder,
        args.eval_env_nums,
        args.eval_max_steps,
        env_name=args.env,
        run_expert=run_expert,
    )
    print(cap_results)
    sys.exit()


def main() -> None:
    """Batch entry-point mirroring run_code_as_policies."""
    #  manual_eval() if we want to test a script manually (offline mode)

    args = _parse_cli_args()
    logging.info(
        "CLI args parsed: env=%s, model=%s, encodings=%s, seeds=%s",
        args.env,
        args.model,
        args.encodings,
        args.seeds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, str | int]] = []
    encoding_eval_results: dict[str, dict[str, Any]] = {}
    for encoding in args.encodings:
        encoding_dir = (
            args.output_dir
            / args.env
            / f"{str(args.num_initial_states)}"
            / f"encoding_{encoding}"
        )
        encoding_dir.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            meta_path = encoding_dir / f"metadata_seed{seed}.json"
            if meta_path.exists():
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
                if existing["evaluation"]["cap_results"]:
                    logging.info(
                        "Skipping env=%s encoding=%s seed=%d — results already exist at %s",
                        args.env,
                        encoding,
                        seed,
                        meta_path,
                    )
                    continue

            logging.info(
                "--- env=%s  encoding=%s  seed=%d ---", args.env, encoding, seed
            )
            _configure_rng(seed)
            cache_base = args.cache_path
            cache_dir = (
                cache_base.parent if cache_base.parent != Path("") else Path(".")
            )
            stem = cache_base.stem or "cache"
            suffix = cache_base.suffix or ".db"
            cache_path = (
                cache_dir
                / f"{stem}_{args.env}_enc{encoding}_initial_{args.num_initial_states}_seed{seed}{suffix}"
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache = SQLite3PretrainedLargeModelCache(cache_path)
            use_response_model = args.use_response_model or args.model == "gpt-5.2-pro"
            logging.info(
                "Creating LLM client (use_response_model=%s) ...", use_response_model
            )
            if use_response_model:
                response_cls = getattr(llm_models, "OpenAIResponsesModel", None)
                if response_cls is None:

                    raise ImportError(
                        "OpenAIResponsesModel is not available in prpl_llm_utils. "
                        "Install/upgrade the package or disable --use-response-model."
                    )
                client = response_cls(args.model, cache)
            else:
                client = OpenAIModel(args.model, cache)
            logging.info("LLM client created: %s", client.get_id())
            final_code = run(
                client,
                args.env,
                encoding,
                seed,
                num_initial_states=args.num_initial_states,
                max_steps_per_traj=args.max_steps_per_traj,
            )

            policy_filename = f"policy_seed{seed}.txt"
            policy_path = encoding_dir / policy_filename
            policy_path.write_text(final_code, encoding="utf-8")

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            metadata = {
                "env": args.env,
                "encoding": encoding,
                "seed": seed,
                "model": args.model,
                "num_initial_states": args.num_initial_states,
                "max_steps_per_traj": args.max_steps_per_traj,
                "timestamp": timestamp,
                "policy_path": str(policy_path.resolve()),
            }
            meta_path = encoding_dir / f"metadata_seed{seed}.json"
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            summary.append(
                {
                    "env": args.env,
                    "encoding": encoding,
                    "seed": seed,
                    "policy_file": str(policy_path),
                }
            )
            if args.eval_env_nums:
                code_str = _strip_code_block(final_code)
                policy_fn = _compile_policy_function(code_str, args.function_name)
                env_builder = lambda idx: env_factory(idx, args.env)
                run_expert = args.eval_run_expert and seed == args.expert_reference_seed
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
    if args.plot_results:
        if not args.eval_env_nums:
            logging.warning(
                "Plotting requested but no evaluation environments were configured; skipping."
            )
        else:
            disk_results = _load_eval_results_from_disk(
                args.output_dir,
                args.env,
                args.num_initial_states,
            )
            for enc, stats in disk_results.items():
                if enc not in encoding_eval_results:
                    encoding_eval_results[enc] = stats

            if not encoding_eval_results:
                logging.warning(
                    "Plotting requested but no evaluation results were recorded in memory or on disk; skipping."
                )
            else:
                all_encodings = sorted(encoding_eval_results.keys())
                (
                    labels,
                    expert_means,
                    expert_cis,
                    cap_means,
                    cap_cis,
                ) = _prepare_results_for_plot(encoding_eval_results, all_encodings)
                if labels:
                    plot_path = args.plot_path
                    if plot_path is None:
                        plot_path = (
                            args.output_dir
                            / args.env
                            / f"cap_vs_expert_{args.num_initial_states}.png"
                        )
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.info("CaP_baseline starting ...")
    main()
