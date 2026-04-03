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
import kinder  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from prpl_llm_utils import models as llm_models
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.code import FunctionOutputRepromptCheck, SyntaxRepromptCheck
from prpl_llm_utils.models import OpenAIModel, PretrainedLargeModel
from prpl_llm_utils.reprompting import RepromptCheck, query_with_reprompts
from prpl_llm_utils.structs import Query
from tqdm import tqdm

from programmatic_policy_learning.approaches.experts.grid_experts import get_grid_expert
from programmatic_policy_learning.approaches.experts.kinder_experts import (
    create_kinder_expert,
)
from programmatic_policy_learning.approaches.lpp_utils.utils import run_single_episode

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
    expert_fn: Any,
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
    expert_is_stateful = all(
        hasattr(expert_fn, attr) for attr in ("reset", "step", "update")
    )
    if start_obs is None:
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs, info = reset_out, {}
    else:
        obs = start_obs
        info = {}
    if expert_is_stateful:
        expert_fn.reset(obs, info)
    trajectory = []

    for _ in range(max_steps):
        action = expert_fn.step() if expert_is_stateful else expert_fn(obs)
        obs_next, reward, term, trunc, info = env.step(action)
        if expert_is_stateful:
            expert_fn.update(obs_next, reward, term or trunc, info)
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
    obs). Currently only supports Motion2D with different numbers of passages
    (explicity specified).

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
    if env_name == "Motion2D" and num_passages >= 0:
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
) -> tuple[str, str]:
    """Collect multiple env trajectories and summarise hints via the LLM.

    Parameters
    ----------
    llm_client : PretrainedLargeModel
        LLM client used for querying.
    env_name : str
        Grid environment name.
    encoding_method : str
        Trajectory encoding mode (``"1"``-``"4"``).
    seed : int
        Random seed for reproducibility.
    num_initial_states : int, optional
        Number of expert rollouts to collect (default 10).
    max_steps_per_traj : int, optional
        Maximum trajectory length fed to the LLM (default 40).
    function_name : str, optional
        Expected name of the generated policy function (default ``"policy"``).

    Returns
    -------
    tuple[str, str]
        A ``(prompt, llm_response)`` tuple.
    """
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
    return prompt, final_code


def run_continuous(
    llm_client: PretrainedLargeModel,
    env_name: str,
    encoding_method: str,
    seed: int,
    num_passages: int = 1,
    num_initial_states: int = 10,
    max_steps_per_traj: int = 40,
    function_name: str = "policy",
    skip_rate: int = 1,
    collect_max_steps: int = 500,
) -> tuple[str, str]:
    """Collect expert trajectories in a continuous env and query the LLM.

    Parameters
    ----------
    llm_client : PretrainedLargeModel
        LLM client used for querying.
    env_name : str
        Continuous environment family name (e.g. ``"Motion2D"``).
    encoding_method : str
        Trajectory encoding mode (``"1"``-``"4"``).
    seed : int
        Random seed for reproducibility.
    num_passages : int, optional
        Number of wall passages (default 1).
    num_initial_states : int, optional
        Number of expert rollouts to collect (default 10).
    max_steps_per_traj : int, optional
        Maximum trajectory length fed to the LLM (default 40).
    function_name : str, optional
        Expected name of the generated policy function (default ``"policy"``).
    skip_rate : int, optional
        Sub-sampling rate for trajectories (default 1, no skipping).
    collect_max_steps : int, optional
        Maximum env steps when rolling out the expert (default 500).

    Returns
    -------
    tuple[str, str]
        A ``(prompt, llm_response)`` tuple.
    """
    env_name = continuous_hint_config.canonicalize_env_name(env_name)
    # ------------------------------------------------------------
    # 1) Setup encoder
    # ------------------------------------------------------------
    obs_fields = continuous_hint_config.obs_field_names_for_motion2d(num_passages)
    action_fields = continuous_hint_config.ACTION_FIELD_NAMES[env_name]
    enc_cfg = continuous_encoder.ContinuousStateEncoderConfig(
        obs_field_names=obs_fields,
        action_field_names=action_fields,
        salient_indices=continuous_hint_config.salient_obs_indices_for_motion2d(
            num_passages
        ),
    )
    encoder = continuous_encoder.ContinuousStateEncoder(enc_cfg)

    # ------------------------------------------------------------
    # 2) Collect expert trajectories
    # ------------------------------------------------------------
    logging.info(
        "Collecting %d expert trajectories for %s (num_passages=%d) ...",
        num_initial_states,
        env_name,
        num_passages,
    )
    ref_env, ref_obs = continuous_env_factory(env_name, num_passages, seed=seed)
    assert isinstance(ref_env.action_space, gymnasium.spaces.Box)
    expert = create_kinder_expert(
        env_name,
        ref_env.action_space,
        seed=seed,
        observation_space=ref_env.observation_space,
        num_passages=num_passages,
        expert_kind="bilevel",
    )
    action_space = ref_env.action_space
    logging.info(
        "Action space: shape=%s dtype=%s low=%s high=%s",
        action_space.shape,
        action_space.dtype,
        action_space.low,
        action_space.high,
    )
    ref_env.close()

    trajectories: list[list[tuple[Any, Any, Any]]] = []
    for init_idx in range(num_initial_states):
        env, obs0 = continuous_env_factory(env_name, num_passages, seed=seed + init_idx)
        traj = collect_full_episode(
            env, expert, max_steps=collect_max_steps, start_obs=obs0
        )
        env.close()
        trajectories.append(traj)
    logging.info("Collected %d trajectories.", len(trajectories))

    # ------------------------------------------------------------
    # 3) Serialize trajectories
    # ------------------------------------------------------------
    all_traj_texts = []
    for i, traj in enumerate(trajectories):
        text = continuous_trajectory_serializer.trajectory_to_text(
            traj,
            encoder=encoder,
            num_passages=num_passages,
            encoding_method=encoding_method,
            max_steps=max_steps_per_traj,
            skip_rate=skip_rate,
        )
        all_traj_texts.append(f"\n---[TRAJECTORY {i}]---\n{text}\n\n")

    combined_text = "\n\n".join(all_traj_texts)
    logging.info("Building continuous prompt (encoding=%s) ...", encoding_method)

    env_desc = continuous_hint_config.get_env_description(env_name, num_passages)
    prompt = build_continuous_hint_prompt(
        combined_text,
        env_desc,
        action_fields,
        action_space.low,
        action_space.high,
    )
    prompt = f"{prompt}\n\nSEED: {seed}\n"
    query = Query(prompt, hyperparameters={"temperature": 0.0, "seed": seed})

    def _validate_box_action(action: Any) -> bool:
        # 1) check if the action is a ndarray
        if not isinstance(action, np.ndarray):
            raise AssertionError(
                f"Action must be a NumPy ndarray, got {type(action)}. "
                "Return e.g. np.array([dx, dy], dtype=np.float32)."
            )

        # 2) check if the action is convertible to the action space shape and dtype
        try:
            a = np.asarray(action, dtype=action_space.dtype).reshape(action_space.shape)
        except Exception as e:
            raise AssertionError(
                f"Action must be convertible to shape={action_space.shape} "
                f"and dtype={action_space.dtype}. Got {action}."
            ) from e

        # 3) check if the action is within the bounds
        low = action_space.low
        high = action_space.high
        if not (np.all(a >= low) and np.all(a <= high)):
            raise AssertionError(
                f"Action out of bounds. "
                f"Expected within [{low}, {high}] but got {a}."
            )

        return True

    reprompt_checks: list[RepromptCheck] = [
        SyntaxRepromptCheck(),
        FunctionOutputRepromptCheck(
            function_name,
            [(ref_obs,)],
            [_validate_box_action],
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
    return prompt, final_code


def _parse_cli_args() -> argparse.Namespace:
    """Return CLI args for batch CaP experiments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments including ``env``, ``encodings``,
        ``seeds``, ``model``, evaluation flags, and output paths.

    Examples
    --------
    Typical invocation from the command line::

        python CaP_baseline.py --env Motion2D --env-type continuous \\
            --num-passages 1 --encodings 4 --seeds 0 \\
            --model gpt-4.1 --eval-max-steps 500
    """

    parser = argparse.ArgumentParser(
        description="Generate code-as-policies hints for multiple encodings/seeds."
    )
    parser.add_argument("--env", required=True, help="Environment name.")
    parser.add_argument(
        "--env-type",
        choices=["grid", "continuous"],
        default="grid",
        help="Environment type: grid (discrete, GGG) or continuous (Box, KinDER).",
    )
    parser.add_argument(
        "--num-passages",
        type=int,
        default=1,
        help="Number of wall passages (only used when --env-type=continuous).",
    )
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
        "--skip-rate",
        type=int,
        default=5,
        help="Sub-sampling rate for continuous trajectories (1 = no skipping).",
    )
    parser.add_argument(
        "--collect-max-steps",
        type=int,
        default=500,
        help="Maximum env steps when collecting expert trajectories (default 500).",
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
        action=argparse.BooleanOptionalAction,
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
    parser.add_argument(
        "--manual-eval-code-file",
        type=str,
        default=None,
        help=(
            "Manual-eval code file basename under manual_eval_code/ (without .txt). "
            "Example: --manual-eval-code-file continuous_policy"
        ),
    )
    return parser.parse_args()


def _configure_rng(seed: int) -> None:
    """Seed Python/NumPy RNGs for deterministic rollouts.

    Parameters
    ----------
    seed : int
        Random seed applied to both ``random`` and ``numpy.random``.

    Examples
    --------
    >>> _configure_rng(42)
    >>> random.random()  # deterministic after seeding
    0.6394267984578837
    """

    random.seed(seed)
    np.random.seed(seed)


def _strip_code_block(text: str) -> str:
    """Remove Markdown fences from LLM-generated code.

    Parameters
    ----------
    text : str
        Raw LLM response that may be wrapped in ````` ```python ... ``` `````.

    Returns
    -------
    str
        Cleaned code string with fences removed.

    Examples
    --------
    >>> _strip_code_block('```python\\ndef f(): pass\\n```')
    'def f(): pass'
    >>> _strip_code_block('def f(): pass')
    'def f(): pass'
    """

    text = text.strip()
    if "```" in text:
        if "```python" in text:
            text = text.split("```python", 1)[1]
        else:
            text = text.split("```", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    return text


def _compile_policy_function(code: str, function_name: str) -> Callable[[Any], Any]:
    """Compile Python code defining `function_name` into a callable.

    The code is executed via ``exec()`` with ``np`` and ``math``
    pre-imported in the global namespace.

    Parameters
    ----------
    code : str
        Python source code that defines a function named *function_name*.
    function_name : str
        Name of the function to extract after execution.

    Returns
    -------
    Callable[[Any], Any]
        The compiled policy function.

    Raises
    ------
    RuntimeError
        If *function_name* is not found or is not callable.

    Examples
    --------
    >>> fn = _compile_policy_function('def policy(obs): return obs * 2', 'policy')
    >>> fn(np.array([1.0, 2.0]))
    array([2., 4.])
    """

    globals_dict: dict[str, Any] = {"np": np, "math": math}
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
    run_expert: bool,
    expert_fn: Any | None = None,
    env_type: str = "grid",
) -> tuple[list[bool], list[bool]]:
    """Roll out CaP (and optional expert) policies on the requested envs.

    Parameters
    ----------
    policy_fn : Callable[[Any], Any]
        LLM-generated policy function.
    env_builder : Callable[[int], Any]
        Factory that creates an environment from an index.
    test_env_nums : Sequence[int]
        Environment indices to evaluate on.
    max_num_steps : int
        Maximum steps per evaluation rollout.
    run_expert : bool
        Whether to also evaluate the expert policy.
    expert_fn : Any | None, optional
        Expert policy object. May be a simple ``(obs) -> action`` callable or
        a stateful agent exposing ``reset()``, ``step()``, and ``update()``.
    env_type : str, optional
        ``"grid"`` or ``"continuous"`` — selects the action guard
        (default ``"grid"``).

    Returns
    -------
    tuple[list[bool], list[bool]]
        ``(cap_results, expert_results)`` where each entry is ``True``
        if the episode succeeded (positive reward).

    Raises
    ------
    RuntimeError
        If ``run_expert`` is ``True`` but ``expert_fn`` is ``None``.

    Examples
    --------
    ::

        cap_res, expert_res = _evaluate_policy_function(
            policy_fn,
            env_builder=lambda idx: env_factory(idx, "TicTacToe"),
            test_env_nums=[0, 1, 2],
            max_num_steps=100,
            run_expert=True,
            expert_fn=get_grid_expert("TicTacToe"),
        )
        # cap_res == [True, False, True], expert_res == [True, True, True]
    """

    cap_results: list[bool] = []
    expert_results: list[bool] = []

    for env_idx in tqdm(test_env_nums, desc="Evaluating envs"):
        env = env_builder(env_idx)

        def safe_policy(obs: Any, *, _env: Any = env) -> Any:
            """Guard policy execution to avoid crashes from invalid assumptions
            in grid environments."""

            try:
                action = policy_fn(obs)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.exception(f"Exception: {e}")
                action = None
            if action is None:
                return _env.action_space.sample()
            return action

        def continuous_safe_policy(obs: Any, *, _env: Any = env) -> Any:
            """Guard for Box (continuous) action spaces.

            Converts to the correct dtype/shape and clips to bounds.
            Falls back to a random sample on any conversion failure.
            """
            try:
                action = policy_fn(obs)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.exception(f"Exception: {e}")
                return _env.action_space.sample()
            if action is None:
                return _env.action_space.sample()
            try:
                a = np.asarray(action, dtype=_env.action_space.dtype)
                a = a.reshape(_env.action_space.shape)
                a = np.clip(a, _env.action_space.low, _env.action_space.high)
                return a
            except (ValueError, TypeError):
                return _env.action_space.sample()

        guarded_policy = (
            continuous_safe_policy if env_type == "continuous" else safe_policy
        )
        # NOTE (double-reset for continuous envs):
        # continuous_env_factory already calls env.reset(seed=idx), but
        # run_single_episode calls env.reset() again without a seed.
        # The first state is discarded; the second is still deterministic
        # (Gymnasium preserves the seeded RNG), so results are reproducible.
        # Evaluation states differ from training demonstrations — this is
        # intentional: CaP evaluation tests generalisation to unseen initial
        # conditions.  To match training states exactly, pass the seed
        # through to run_single_episode.
        reward, terminated = run_single_episode(
            env, guarded_policy, max_num_steps=max_num_steps
        )
        logging.info(
            "env_idx: %s, reward: %s, terminated: %s", env_idx, reward, terminated
        )
        logging.info(
            "reward type: %s, terminated type: %s", type(reward), type(terminated)
        )
        cap_success = terminated if env_type == "continuous" else reward > 0
        cap_results.append(bool(cap_success))
        env.close()

        if run_expert:
            if expert_fn is None:
                raise RuntimeError("Expert policy unavailable.")
            env_e = env_builder(env_idx)
            reward_e, terminated_e = run_single_episode(
                env_e, expert_fn, max_num_steps=max_num_steps
            )
            logging.info("reward_e: %s, terminated_e: %s", reward_e, terminated_e)
            logging.info(
                "reward_e type: %s, terminated_e type: %s",
                type(reward_e),
                type(terminated_e),
            )
            expert_success = terminated_e if env_type == "continuous" else reward_e > 0
            expert_results.append(bool(expert_success))
            env_e.close()

    return cap_results, expert_results


def bootstrap_ci_success(
    successes: Sequence[bool],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Return mean success rate and half-width of a bootstrap CI (Confidence
    Interval).

    Parameters
    ----------
    successes : Sequence[bool]
        Per-episode success indicators.
    n_boot : int, optional
        Number of bootstrap resamples (default 10000).
    alpha : float, optional
        Significance level for the CI (default 0.05 for 95% CI).
    seed : int, optional
        RNG seed for reproducibility (default 0).

    Returns
    -------
    tuple[float, float]
        ``(mean_success_rate, ci_half_width)``.

    Raises
    ------
    ValueError
        If ``successes`` is empty.

    Examples
    --------
    >>> mean, ci = bootstrap_ci_success([True, True, False, True, False])
    >>> float(mean)  # 3 successes / 5 trials
    0.6
    >>> round(float(ci), 2)  # (hi - lo) / 2 = (1.00 - 0.20) / 2
    0.4
    """
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
    """Bar plot comparing expert and CaP performance with error bars.

    Parameters
    ----------
    labels : Sequence[str]
        X-axis labels (one per encoding).
    expert_means : Sequence[float]
        Expert success rate (%) per encoding.
    expert_cis : Sequence[float]
        Expert CI half-width (%) per encoding.
    cap_means : Sequence[float]
        CaP success rate (%) per encoding.
    cap_cis : Sequence[float]
        CaP CI half-width (%) per encoding.
    title : str
        Plot title.
    save_path : str | Path
        Destination file path for the saved figure.

    Raises
    ------
    ValueError
        If ``labels`` is empty.
    """

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
    """Aggregate evaluation results per encoding for plotting.

    Parameters
    ----------
    encoding_eval_results : dict[str, dict[str, Any]]
        Mapping from encoding name to ``{"cap_runs": [...], "expert_run": [...]}``.
    encoding_order : Sequence[str]
        Order in which encodings appear on the x-axis.

    Returns
    -------
    tuple[list[str], list[float], list[float], list[float], list[float]]
        ``(labels, expert_means, expert_cis, cap_means, cap_cis)``
        where means and CIs are in percent (0-100).

    Examples
    --------
    >>> results = {
    ...     "4": {
    ...         "cap_runs": [[True, False, True]],
    ...         "expert_run": [True, True, True],
    ...     },
    ... }
    >>> labels, em, ec, cm, cc = _prepare_results_for_plot(results, ["4"])
    >>> labels
    ['enc_4']
    >>> float(em[0])  # expert mean = 100%
    100.0
    """

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

    If *encodings* is ``None``, auto-discover all ``encoding_*``
    directories under ``output_dir / env_name / num_initial_states``.

    Parameters
    ----------
    output_dir : Path
        Root output directory (e.g. ``logs/CaP_baseline``).
    env_name : str
        Environment tag used as a subdirectory name.
    num_initial_states : int
        Number of initial states (used as a subdirectory name).
    encodings : Sequence[str] | None, optional
        Specific encodings to load.  ``None`` auto-discovers all.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from encoding name to
        ``{"cap_runs": [[bool, ...], ...], "expert_run": [bool, ...] | None}``.

    Examples
    --------
    ::

        results = _load_eval_results_from_disk(
            Path("logs/CaP_baseline"),
            env_name="Motion2D-p1/gpt-4.1",
            num_initial_states=4,
            encodings=["4"],
        )
        # results["4"]["cap_runs"] == [[True, False, ...], ...]
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
    """Evaluate a hard-coded policy function offline and print results.

    This is a developer-facing utility for quick manual testing; it
    loads policy code from ``manual_eval_code/`` and evaluates it on the
    environment specified via CLI ``--env``. The code file is loaded from
    ``manual_eval_code/<name>.txt`` where ``<name>`` comes from
    ``--manual-eval-code-file`` (or defaults by ``--env-type``).

    Raises
    ------
    FileNotFoundError
        If the requested manual-eval code file does not exist under
        ``manual_eval_code/``.
    RuntimeError
        If the loaded code cannot be compiled into the expected policy
        function.
    """
    args = _parse_cli_args()
    if args.env_type == "continuous":
        args.env = continuous_hint_config.canonicalize_env_name(args.env)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manual_code_dir = Path(__file__).resolve().parent / "manual_eval_code"
    code_base = args.manual_eval_code_file
    if code_base is None:
        raise ValueError("Manual eval code file not specified.")
    code_filename = f"{code_base}.txt"
    final_code = (manual_code_dir / code_filename).read_text(encoding="utf-8")
    code_str = _strip_code_block(final_code)
    policy_fn = _compile_policy_function(code_str, "policy")
    run_expert = True
    eval_expert: Any | None
    if args.env_type == "continuous":
        env_builder: Callable[[int], Any] = lambda idx: continuous_env_factory(
            args.env, args.num_passages, seed=idx
        )[0]
        ref_env, _ = continuous_env_factory(args.env, args.num_passages, seed=0)
        assert isinstance(ref_env.action_space, gymnasium.spaces.Box)
        eval_expert = create_kinder_expert(
            args.env,
            ref_env.action_space,
            seed=0,
            observation_space=ref_env.observation_space,
            num_passages=args.num_passages,
            expert_kind="bilevel",
        )
        ref_env.close()
    else:
        env_builder = lambda idx: env_factory(idx, args.env)
        eval_expert = get_grid_expert(args.env)
    cap_results, _expert_results = _evaluate_policy_function(
        policy_fn,
        env_builder,
        args.eval_env_nums,
        args.eval_max_steps,
        run_expert=run_expert,
        expert_fn=eval_expert,
        env_type=args.env_type,
    )
    logging.info(cap_results)
    sys.exit()


def main() -> None:
    """Batch entry-point mirroring Code-as-Policies experiments.

    Iterates over all ``(encoding, seed)`` combinations specified via
    CLI, collects expert trajectories, queries the LLM, evaluates the
    generated policy, saves metadata/logs, and optionally produces a
    summary plot.
    """
    #  manual_eval() if we want to test a script manually (offline mode)

    args = _parse_cli_args()
    if args.env_type == "continuous":
        args.env = continuous_hint_config.canonicalize_env_name(args.env)

    env_tag = (
        f"{args.env}-p{args.num_passages}"
        if args.env_type == "continuous"
        else args.env
    )
    run_tag = f"{env_tag}/{args.model}"

    logging.info(
        "CLI args parsed: env=%s, model=%s, encodings=%s, seeds=%s, run_tag=%s",
        args.env,
        args.model,
        args.encodings,
        args.seeds,
        run_tag,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, str | int]] = []
    encoding_eval_results: dict[str, dict[str, Any]] = {}
    for encoding in args.encodings:
        encoding_dir = (
            args.output_dir
            / run_tag
            / str(args.num_initial_states)
            / f"encoding_{encoding}"
        )
        encoding_dir.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            meta_path = encoding_dir / f"metadata_seed{seed}.json"
            if meta_path.exists():
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
                eval_section = existing.get("evaluation", {})
                if eval_section.get("cap_results"):
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
                / f"{stem}_{env_tag}_enc{encoding}_initial_{args.num_initial_states}_seed{seed}{suffix}"
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
            if args.env_type == "continuous":
                try:
                    prompt_text, final_code = run_continuous(
                        client,
                        args.env,
                        encoding,
                        seed,
                        num_passages=args.num_passages,
                        num_initial_states=args.num_initial_states,
                        max_steps_per_traj=args.max_steps_per_traj,
                        skip_rate=args.skip_rate,
                        collect_max_steps=args.collect_max_steps,
                    )
                except Exception as _main_exc:  # pylint: disable=broad-exception-caught
                    logging.error("run_continuous failed: %s", _main_exc, exc_info=True)
                    continue
            else:
                prompt_text, final_code = run(
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

            code_str_for_log = _strip_code_block(final_code)
            debug_log_path = encoding_dir / f"debug_seed{seed}.log"
            debug_log_path.write_text(
                f"{'=' * 80}\n"
                f"PROMPT SENT TO LLM\n"
                f"{'=' * 80}\n"
                f"{prompt_text}\n\n"
                f"{'=' * 80}\n"
                f"RAW LLM RESPONSE\n"
                f"{'=' * 80}\n"
                f"{final_code}\n\n"
                f"{'=' * 80}\n"
                f"CODE AFTER _strip_code_block (passed to exec)\n"
                f"{'=' * 80}\n"
                f"{code_str_for_log}\n",
                encoding="utf-8",
            )
            logging.info("Debug log saved to %s", debug_log_path)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            metadata = {
                "env": args.env,
                "env_type": args.env_type,
                "encoding": encoding,
                "seed": seed,
                "model": args.model,
                "num_initial_states": args.num_initial_states,
                "max_steps_per_traj": args.max_steps_per_traj,
                "timestamp": timestamp,
                "policy_path": str(policy_path.resolve()),
            }
            if args.env_type == "continuous":
                metadata["num_passages"] = args.num_passages
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
                if args.env_type == "continuous":
                    # See NOTE in _evaluate_policy_function on double-reset.
                    env_builder: Callable[[int], Any] = (
                        lambda idx: continuous_env_factory(
                            args.env, args.num_passages, seed=idx
                        )[0]
                    )
                else:
                    env_builder = lambda idx: env_factory(idx, args.env)
                run_expert = args.eval_run_expert and seed == args.expert_reference_seed
                eval_expert: Any | None = None
                if run_expert:
                    if args.env_type == "continuous":
                        ref_env, _ = continuous_env_factory(
                            args.env, args.num_passages, seed=0
                        )
                        assert isinstance(ref_env.action_space, gymnasium.spaces.Box)
                        eval_expert = create_kinder_expert(
                            args.env,
                            ref_env.action_space,
                            seed=seed,
                            observation_space=ref_env.observation_space,
                            num_passages=args.num_passages,
                            expert_kind="bilevel",
                        )
                        ref_env.close()
                    else:
                        eval_expert = get_grid_expert(args.env)
                cap_results, expert_results = _evaluate_policy_function(
                    policy_fn,
                    env_builder,
                    args.eval_env_nums,
                    args.eval_max_steps,
                    run_expert=run_expert,
                    expert_fn=eval_expert,
                    env_type=args.env_type,
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
                    "eval_max_steps": args.eval_max_steps,
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
                run_tag,
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
                            / run_tag
                            / f"cap_vs_expert_{args.num_initial_states}.png"
                        )
                    title = (
                        f"{env_tag}: Expert vs CaP "
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

    manifest_path = args.output_dir / run_tag / "manifest.json"
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
