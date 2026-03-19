"""Visualize a saved CaP policy on Motion2D train/test seeds.

This helper is aimed at continuous Motion2D runs saved by
``CaP_baseline.py``. It can:

- load a generated policy from ``policy_seed*.txt``
- replay it on the training seeds used for prompt collection
- replay it on evaluation/test seeds
- optionally compare against the handcrafted expert
- optionally save rollout videos

Examples
--------
Visualize the training seeds used by a seed-0 CaP run::

    python scripts/visualize_cap_policy.py \
      --policy-path \
      logs/CaP_baseline/Motion2D-p1/gpt-5.2-pro/4/encoding_4/policy_seed0.txt \
      --metadata-path \
      logs/CaP_baseline/Motion2D-p1/gpt-5.2-pro/4/encoding_4/metadata_seed0.json \
      --mode train \
      --record-video

Visualize a few test seeds and compare to the expert::

    python scripts/visualize_cap_policy.py \
      --policy-path \
      logs/CaP_baseline/Motion2D-p1/gpt-5.2-pro/4/encoding_4/policy_seed0.txt \
      --metadata-path \
      logs/CaP_baseline/Motion2D-p1/gpt-5.2-pro/4/encoding_4/metadata_seed0.json \
      --mode test \
      --seeds 0 1 2 3 4 \
      --compare-expert \
      --record-video
"""

# pylint: disable=protected-access

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import imageio.v2 as imageio
import numpy as np

from programmatic_policy_learning.approaches.experts.kinder_experts import (
    create_kinder_expert,
)
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    CaP_baseline as cap_baseline,)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("train", "test"),
        default="test",
        help=(
            "`train` replays the prompt-collection seeds; " "`test` replays eval seeds."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit seeds to replay. Overrides mode defaults.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override rollout length. Defaults to metadata eval_max_steps or 500.",
    )
    parser.add_argument(
        "--compare-expert",
        action="store_true",
        help="Also replay the handcrafted expert on the same seeds.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Save rollout videos under --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("videos/cap_visualizations"),
        help="Directory for saved rollout videos.",
    )
    parser.add_argument(
        "--double-reset-like-eval",
        action="store_true",
        help=(
            "Mimic CaP_baseline evaluation exactly by resetting once with the seed "
            "and then resetting again without a seed before the rollout."
        ),
    )
    return parser.parse_args()


def _load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_policy(policy_path: Path) -> Callable[[Any], Any]:
    code = policy_path.read_text(encoding="utf-8")
    return cap_baseline._compile_policy_function(
        cap_baseline._strip_code_block(code), "policy"
    )


def _render_frame(env: Any) -> np.ndarray | None:
    """Render one frame from the environment when available."""

    if not hasattr(env, "render"):
        return None
    try:
        frame = env.render()
    except (AttributeError, RuntimeError, TypeError):
        return None
    if frame is None:
        return None
    return np.asarray(frame)


def _write_video(
    frames: Sequence[np.ndarray[Any, Any]], path: Path, fps: int = 10
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimsave(path, cast(Any, list(frames)), fps=fps)


def _rollout(
    env_name: str,
    num_passages: int,
    reset_seed: int,
    policy: Callable[[Any], Any],
    max_steps: int,
    *,
    record_video: bool,
    video_path: Path | None = None,
    double_reset_like_eval: bool = False,
) -> dict[str, Any]:
    stateful_policy = all(hasattr(policy, attr) for attr in ("reset", "step", "update"))
    env, obs = cap_baseline.continuous_env_factory(
        env_name, num_passages, seed=reset_seed
    )

    if double_reset_like_eval:
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out
        else:
            obs = reset_out
    if stateful_policy:
        policy.reset(obs, {})

    frames: list[np.ndarray] = []
    if record_video:
        frame0 = _render_frame(env)
        if frame0 is not None:
            frames.append(frame0)

    total_reward = 0.0
    terminated = False
    truncated = False
    no_change_steps = 0
    blocked_steps = 0
    prev_xy = np.asarray(obs[:2], dtype=np.float32).copy()
    start_xy = prev_xy.copy()
    final_obs = obs

    for step_idx in range(max_steps):
        raw_action = policy.step() if stateful_policy else policy(obs)
        action = np.asarray(raw_action, dtype=env.action_space.dtype)
        action = action.reshape(env.action_space.shape)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs_next, reward, terminated, truncated, _ = env.step(action)
        if stateful_policy:
            policy.update(obs_next, float(reward), terminated or truncated, {})
        total_reward += float(reward)

        next_xy = np.asarray(obs_next[:2], dtype=np.float32).copy()
        moved = not np.allclose(next_xy, prev_xy)
        if not moved:
            no_change_steps += 1
            if np.linalg.norm(action[:2]) > 1e-6:
                blocked_steps += 1

        obs = obs_next
        final_obs = obs_next
        prev_xy = next_xy

        if record_video:
            frame = _render_frame(env)
            if frame is not None:
                frames.append(frame)

        if terminated or truncated:
            break

    env.close()

    if record_video and video_path is not None:
        _write_video(frames, video_path)

    return {
        "seed": reset_seed,
        "start_xy": tuple(float(x) for x in start_xy),
        "final_xy": tuple(float(x) for x in final_obs[:2]),
        "target_xywh": tuple(float(x) for x in final_obs[[9, 10, 17, 18]]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "total_reward": float(total_reward),
        "steps": step_idx + 1 if max_steps > 0 else 0,
        "no_change_steps": no_change_steps,
        "blocked_steps": blocked_steps,
    }


def _resolve_seeds(
    metadata: dict[str, Any], mode: str, explicit: list[int] | None
) -> list[int]:
    if explicit:
        return explicit
    if mode == "train":
        base_seed = int(metadata["seed"])
        num_initial_states = int(metadata.get("num_initial_states", 4))
        return [base_seed + i for i in range(num_initial_states)]
    eval_section = metadata.get("evaluation", {})
    cap_results = eval_section.get("cap_results", [])
    if cap_results:
        return list(range(len(cap_results)))
    return list(range(20))


def _resolve_max_steps(metadata: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return override
    eval_section = metadata.get("evaluation", {})
    return int(eval_section.get("eval_max_steps", 500))


def _print_summary(label: str, result: dict[str, Any]) -> None:
    start_x, start_y = result["start_xy"]
    final_x, final_y = result["final_xy"]
    target_x, target_y, target_w, target_h = result["target_xywh"]
    print(
        f"{label}: seed={result['seed']} "
        f"success={result['terminated']} "
        f"steps={result['steps']} "
        f"start=({start_x:.3f}, {start_y:.3f}) "
        f"final=({final_x:.3f}, {final_y:.3f}) "
        f"target=({target_x:.3f}, {target_y:.3f}, {target_w:.3f}, {target_h:.3f}) "
        f"blocked_steps={result['blocked_steps']} "
        f"no_change_steps={result['no_change_steps']}"
    )


def main() -> None:
    """Load the requested policy and visualize it on the selected seeds."""

    args = _parse_args()
    metadata = _load_metadata(args.metadata_path)
    policy = _load_policy(args.policy_path)

    env_name = str(metadata["env"])
    num_passages = int(metadata.get("num_passages", 1))
    seeds = _resolve_seeds(metadata, args.mode, args.seeds)
    max_steps = _resolve_max_steps(metadata, args.max_steps)

    expert_policy: Callable[[Any], Any] | None = None
    if args.compare_expert:
        ref_env, _ = cap_baseline.continuous_env_factory(env_name, num_passages, seed=0)
        expert_policy = create_kinder_expert(
            env_name,
            ref_env.action_space,
            seed=0,
            observation_space=ref_env.observation_space,
            num_passages=num_passages,
            expert_kind="bilevel",
        )
        ref_env.close()

    print(
        f"Visualizing {args.policy_path} on {env_name}-p{num_passages} "
        f"mode={args.mode} seeds={seeds} max_steps={max_steps}"
    )
    if args.double_reset_like_eval:
        print("Using double-reset mode to mimic CaP_baseline evaluation.")

    for seed in seeds:
        cap_video_path = None
        expert_video_path = None
        if args.record_video:
            stem = args.policy_path.stem
            suffix = "_double_reset" if args.double_reset_like_eval else ""
            cap_video_path = (
                args.output_dir / f"{stem}_{args.mode}_seed{seed}{suffix}_cap.mp4"
            )
            expert_video_path = (
                args.output_dir / f"{stem}_{args.mode}_seed{seed}{suffix}_expert.mp4"
            )

        cap_result = _rollout(
            env_name,
            num_passages,
            seed,
            policy,
            max_steps,
            record_video=args.record_video,
            video_path=cap_video_path,
            double_reset_like_eval=args.double_reset_like_eval,
        )
        _print_summary("CaP", cap_result)

        if expert_policy is not None:
            expert_result = _rollout(
                env_name,
                num_passages,
                seed,
                expert_policy,
                max_steps,
                record_video=args.record_video,
                video_path=expert_video_path,
                double_reset_like_eval=args.double_reset_like_eval,
            )
            _print_summary("Expert", expert_result)


if __name__ == "__main__":
    main()
