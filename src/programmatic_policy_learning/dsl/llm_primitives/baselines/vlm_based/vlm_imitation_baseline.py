"""Generate a symbolic Python policy from demonstration videos with a VLM."""

from __future__ import annotations

import argparse
import json
import logging
import math
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import numpy as np
from openai import OpenAI
from omegaconf import OmegaConf

from programmatic_policy_learning.approaches.lpp_utils.utils import (
    infer_episode_success,
    run_single_episode,
)
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    grid_hint_config,
)
from programmatic_policy_learning.envs.registry import EnvRegistry

try:
    from .video_frames_hint_extractor import video_to_data_urls
except ImportError:  # pragma: no cover - script entrypoint fallback
    from programmatic_policy_learning.dsl.llm_primitives.baselines.vlm_based.video_frames_hint_extractor import (  # noqa: E501
        video_to_data_urls,
    )


def env_factory(instance_num: int | None = None, env_name: str | None = None) -> Any:
    """Create a GGG environment instance by name and index."""
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


def _strip_code_block(text: str) -> str:
    """Remove Markdown fences from generated code."""
    stripped = text.strip()
    if "```" in stripped:
        if "```python" in stripped:
            stripped = stripped.split("```python", 1)[1]
        else:
            stripped = stripped.split("```", 1)[1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    return stripped


def _compile_policy_function(code: str, function_name: str) -> Callable[[Any], Any]:
    """Compile generated code and return the named policy function."""
    namespace: dict[str, Any] = {"np": np, "math": math}
    exec(code, namespace, namespace)  # pylint: disable=exec-used
    if function_name not in namespace:
        raise RuntimeError(f"Function '{function_name}' not found in generated code.")
    fn = namespace[function_name]
    if not callable(fn):
        raise RuntimeError(f"'{function_name}' is not callable.")
    return cast(Callable[[Any], Any], fn)


def _reset_policy_function_state(policy_fn: Callable[[Any], Any]) -> None:
    """Clear common mutable attrs attached by generated functions."""
    for attr in ("_st", "_last_debug", "_memory", "_state"):
        if hasattr(policy_fn, attr):
            delattr(policy_fn, attr)


_ENV_TASK_DESCRIPTIONS: dict[str, str] = {
    "TwoPileNim": textwrap.dedent(
        """
        Environment-specific action semantics:
        - The grid has two columns representing two piles of Nim tokens.
        - A cell containing 'token' is a removable token in that pile.
        - Returning action `(row, col)` means: choose pile `col` and remove the
          clicked token and every token below it in the same column.
        - A legal move should click a cell that currently contains 'token'.
        - The winning strategy is standard two-pile Nim: if the pile sizes are
          unequal, act on the larger pile so the two pile sizes become equal;
          if they are already equal, any legal token click is acceptable.
        - There is no separate agent avatar in this environment.
        """
    ).strip(),
    "StopTheFall": textwrap.dedent(
        """
        Environment-specific action semantics:
        - The observation contains falling and static support tokens.
        - Returning `(row, col)` clicks a cell to place or advance support in the
          environment's own semantics.
        - The policy should prioritize preventing the falling token from reaching
          the bottom by supporting its path.
        """
    ).strip(),
    "ReachForTheStar": textwrap.dedent(
        """
        Environment-specific action semantics:
        - The grid contains an 'agent', a 'star', directional arrow cells, and
          drawn staircase/path cells.
        - Returning `(row, col)` clicks either an arrow control or a grid cell
          to extend a path, depending on the current situation.
        - The goal is to get the agent to the star by moving and creating the
          required staircase/path structure.
        """
    ).strip(),
    "Chase": textwrap.dedent(
        """
        Environment-specific action semantics:
        - The grid contains an 'agent', a 'target', walls, directional arrow
          cells, and drawable cells.
        - Returning `(row, col)` can either click an arrow control to move the
          agent or click an empty cell to draw/block as needed.
        - The goal is to trap or reach the target using movement and drawing.
        """
    ).strip(),
    "CheckmateTactic": textwrap.dedent(
        """
        Environment-specific action semantics:
        - The grid contains chess-piece tokens such as kings and a queen.
        - Returning `(row, col)` selects the move target cell according to the
          environment's chess-like move rules.
        - The goal is to choose the move that forces checkmate/tactical success.
        """
    ).strip(),
}


def _env_specific_task_description(env_name: str) -> str:
    """Return concise action semantics and objective for a given grid env."""
    return _ENV_TASK_DESCRIPTIONS.get(
        env_name,
        "Return a valid `(row, col)` action based on the symbolic grid state.",
    )


def _base_prompt_for_env(env_name: str) -> str:
    symbol_map = grid_hint_config.get_symbol_map(env_name)
    object_types = list(symbol_map.keys())
    symbol_desc = "\n".join(
        f"  - {token!r} is rendered like {symbol!r}" for token, symbol in symbol_map.items()
    )
    return textwrap.dedent(
        f"""
        You are given expert demonstration videos from a grid environment.

        IMPORTANT:
        - The videos are rendered demonstrations only; your final output must be a symbolic Python policy over `obs`.
        - `obs` is a 2D NumPy array of Python strings/object-type identifiers, not pixels.
        - The policy must reason over these symbolic values directly.
        - The action returned by `policy(obs)` must always be a valid `(row, col)` tuple.
        - Do not assume there is an agent token unless the observation actually contains one.
        - Valid object types visible in `obs` include:
          {", ".join(repr(obj) for obj in object_types)}

        Video rendering legend:
        {symbol_desc}

        Each attached video is one expert rollout from a possibly different initial state.
        Frames within each video are ordered temporally.

        Your task:
        1. Infer the expert's decision rule from the videos.
        2. Write a general Python function `policy(obs)` that maps the symbolic observation to an action.

        Task and action semantics:
        {_env_specific_task_description(env_name)}

        Output constraints:
        - Return ONLY executable Python code.
        - Wrap the code in a fenced Markdown block starting with ```python and ending with ```.
        - Define exactly one callable named `policy`.
        - Signature must be: `def policy(obs):`
        - The function must return a valid `(row, col)` tuple on every call.
        - If there are no good options, return some conservative legal fallback action rather than `None`.
        - Do not import external libraries.
        - You may use `np` and `math`; they are pre-imported.
        - Do not write image-processing code or mention videos in the returned code.
        - Do not include explanations outside the code block.
        """
    ).strip()


def query_policy_from_videos(
    video_paths: Sequence[str],
    *,
    env_name: str,
    model: str = "gpt-4.1",
    max_frames_per_video: int = 12,
    sample_every_n_frames: int = 15,
    client: OpenAI | None = None,
) -> tuple[str, str]:
    """Send demonstration videos to a VLM and ask for Python policy code."""
    active_client = client or OpenAI()
    prompt_text = _base_prompt_for_env(env_name)
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": prompt_text},
    ]

    for index, video_path in enumerate(video_paths, start=1):
        data_urls = video_to_data_urls(
            str(video_path),
            max_frames=max_frames_per_video,
            sample_every_n_frames=sample_every_n_frames,
        )
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"\n\n--- Demonstration video {index}: {Path(video_path).name} ---\n"
                    "The following images are temporally ordered frames from a single expert rollout."
                ),
            }
        )
        for data_url in data_urls:
            content.append(
                {
                    "type": "input_image",
                    "image_url": data_url,
                }
            )

    input_payload = [{"role": "user", "content": content}]
    response = active_client.responses.create(
        model=model,
        input=cast(Any, input_payload),
        max_output_tokens=2000,
        temperature=0.0,
    )
    return prompt_text, response.output_text


def evaluate_policy_function(
    policy_fn: Callable[[Any], Any],
    *,
    env_name: str,
    test_env_nums: Sequence[int],
    max_num_steps: int = 100,
) -> list[bool]:
    """Evaluate a generated policy on a set of held-out env ids."""
    results: list[bool] = []
    for env_num in test_env_nums:
        _reset_policy_function_state(policy_fn)
        env = env_factory(int(env_num), env_name)
        try:
            reward, terminated, final_info = run_single_episode(
                env,
                _make_safe_policy(policy_fn),
                max_num_steps=max_num_steps,
                reset_seed=int(env_num),
            )
            results.append(
                infer_episode_success(
                    reward=float(reward),
                    terminated=bool(terminated),
                    action_mode="discrete",
                    base_class_name=env_name,
                    final_info=final_info,
                )
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.warning(
                "Policy evaluation failed on %s env %s: %s",
                env_name,
                env_num,
                exc,
            )
            results.append(False)
    return results


def _fallback_action(obs: Any) -> tuple[int, int]:
    """Return a conservative in-bounds action for malformed generated policies."""
    obs_arr = np.asarray(obs)
    if obs_arr.ndim != 2 or obs_arr.size == 0:
        return (0, 0)
    non_empty = np.argwhere(obs_arr != "empty")
    if len(non_empty) > 0:
        r, c = non_empty[0]
        return (int(r), int(c))
    return (0, 0)


def _normalize_action(action: Any, obs: Any) -> tuple[int, int]:
    """Ensure a generated action is a usable in-bounds `(row, col)` tuple."""
    obs_arr = np.asarray(obs)
    rows, cols = obs_arr.shape[:2]
    if isinstance(action, (tuple, list)) and len(action) == 2:
        try:
            row = int(action[0])
            col = int(action[1])
            if 0 <= row < rows and 0 <= col < cols:
                return (row, col)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    return _fallback_action(obs)


def _make_safe_policy(policy_fn: Callable[[Any], Any]) -> Callable[[Any], tuple[int, int]]:
    """Wrap a generated policy so malformed outputs become conservative actions."""

    def _safe_policy(obs: Any) -> tuple[int, int]:
        try:
            action = policy_fn(obs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.warning("Generated policy raised %s; using fallback action.", exc)
            return _fallback_action(obs)
        return _normalize_action(action, obs)

    return _safe_policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="Grid environment name.")
    parser.add_argument(
        "--video-paths",
        nargs="+",
        required=True,
        help="One or more expert demonstration .mp4 files.",
    )
    parser.add_argument("--model", default="gpt-4.1", help="Vision model name.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/vlm_imitation"),
        help="Directory for generated code and metadata.",
    )
    parser.add_argument(
        "--function-name",
        default="policy",
        help="Expected generated policy function name.",
    )
    parser.add_argument(
        "--eval-env-nums",
        nargs="*",
        type=int,
        default=[],
        help="Optional held-out env ids for evaluation.",
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=100,
        help="Max rollout length during evaluation.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=12,
        help="Maximum sampled frames from each video.",
    )
    parser.add_argument(
        "--sample-every-n-frames",
        type=int,
        default=15,
        help="Frame sampling stride within each video.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate a policy from videos and optionally evaluate it."""
    args = _parse_args()
    output_dir = args.output_dir / args.env / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    client = OpenAI()
    prompt_text, raw_response = query_policy_from_videos(
        args.video_paths,
        env_name=args.env,
        model=args.model,
        max_frames_per_video=args.max_frames_per_video,
        sample_every_n_frames=args.sample_every_n_frames,
        client=client,
    )
    final_code = raw_response
    code_str = _strip_code_block(final_code)
    policy_fn = _compile_policy_function(code_str, args.function_name)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    policy_path = output_dir / f"policy_{timestamp}.py.txt"
    policy_path.write_text(final_code, encoding="utf-8")

    debug_log_path = output_dir / f"debug_{timestamp}.log"
    debug_log_path.write_text(
        f"{'=' * 80}\n"
        f"PROMPT SENT TO VLM\n"
        f"{'=' * 80}\n"
        f"{prompt_text}\n\n"
        f"{'=' * 80}\n"
        f"VIDEO PATHS\n"
        f"{'=' * 80}\n"
        + "\n".join(str(path) for path in args.video_paths)
        + "\n\n"
        f"{'=' * 80}\n"
        f"RAW VLM RESPONSE\n"
        f"{'=' * 80}\n"
        f"{final_code}\n\n"
        f"{'=' * 80}\n"
        f"CODE AFTER _strip_code_block\n"
        f"{'=' * 80}\n"
        f"{code_str}\n",
        encoding="utf-8",
    )

    metadata: dict[str, Any] = {
        "env": args.env,
        "model": args.model,
        "timestamp": timestamp,
        "video_paths": [str(path) for path in args.video_paths],
        "max_frames_per_video": int(args.max_frames_per_video),
        "sample_every_n_frames": int(args.sample_every_n_frames),
        "policy_path": str(policy_path.resolve()),
        "debug_log_path": str(debug_log_path.resolve()),
        "task_description": prompt_text,
    }
    if args.eval_env_nums:
        eval_results = evaluate_policy_function(
            policy_fn,
            env_name=args.env,
            test_env_nums=args.eval_env_nums,
            max_num_steps=int(args.eval_max_steps),
        )
        metadata["evaluation"] = {
            "eval_env_nums": [int(each) for each in args.eval_env_nums],
            "eval_max_steps": int(args.eval_max_steps),
            "results": eval_results,
            "num_successes": int(sum(eval_results)),
            "success_rate": (
                float(sum(eval_results) / len(eval_results)) if eval_results else None
            ),
        }

    metadata_path = output_dir / f"metadata_{timestamp}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logging.info("Generated VLM imitation policy saved to %s", policy_path)
    logging.info("Metadata written to %s", metadata_path)


if __name__ == "__main__":
    main()
