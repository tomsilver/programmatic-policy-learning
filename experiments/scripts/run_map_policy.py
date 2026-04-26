"""Run a MAP program as a policy on test envs and record videos."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
from omegaconf import OmegaConf

from programmatic_policy_learning.approaches.lpp_utils.lpp_feature_source_utils import (
    _parse_py_feature_sources,
)
from programmatic_policy_learning.approaches.lpp_utils.utils import run_single_episode
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    get_dsl_functions_dict,
)
from programmatic_policy_learning.dsl.state_action_program import (
    StateActionProgram,
    set_dsl_functions,
)
from programmatic_policy_learning.envs.registry import EnvRegistry
from programmatic_policy_learning.policies.lpp_policy import LPPPolicy


def build_py_feature_functions(
    feature_programs: list[str],
    dsl_functions: dict[str, Any],
) -> dict[str, Any]:
    """Build a dict of feature function names to callables from source
    strings."""
    functions, _ = _parse_py_feature_sources(feature_programs, dsl_functions)
    return functions


def _parse_env_nums(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        return []
    if "-" in spec:
        start_s, end_s = spec.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise ValueError("env range end must be >= start")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _load_feature_sources(path: Path) -> list[str]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON in {path}")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"Missing 'features' list in {path}")
    sources: list[str] = []
    for feat in features:
        if not isinstance(feat, dict) or "source" not in feat:
            raise ValueError("Each feature must be a dict with a 'source' key.")
        src = feat["source"]
        if not isinstance(src, str):
            raise ValueError("Feature 'source' must be a string.")
        sources.append(src.replace("\\n", "\n"))
    return sources


def _load_map_program(path: Path | None, inline: str | None) -> str:
    if path is not None:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty MAP program file: {path}")
        # Allow pasting full log lines; strip leading log prefix if present.
        if "] - " in text:
            text = text.split("] - ", 1)[1].strip()
        return text
    if inline is None or not inline.strip():
        raise ValueError("Provide --map-program or --map-program-file.")
    text = inline.strip()
    if "] - " in text:
        text = text.split("] - ", 1)[1].strip()
    return text


def main() -> None:
    """Run a MAP program as a policy on test envs and record videos."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--env",
        required=True,
        help="Env config name (e.g., ggg_stf, ggg_chase).",
    )
    p.add_argument(
        "--features-json",
        required=True,
        help="Path to expanded feature JSON (offline payload).",
    )
    p.add_argument(
        "--map-program",
        default=None,
        help="MAP program string (use --map-program-file for long strings).",
    )
    p.add_argument(
        "--map-program-file",
        default=None,
        help="Path to file containing the MAP program string.",
    )
    p.add_argument(
        "--test-env-nums",
        default="0-19",
        help="Env instances to test (e.g., '11-19' or '11,12,15').",
    )
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--video-dir", default="logs/map_policy_videos")
    p.add_argument("--video-format", default="mp4")
    p.add_argument("--seed", type=int, default=0)
    # p.add_argument(
    #     "--normalize-plp-actions",
    #     action="store_false",
    #     help="Enable PLP action-mass normalization (penalize permissive PLPs).",
    # )
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env_cfg_path = Path("experiments/conf/env") / f"{args.env}.yaml"
    if not env_cfg_path.exists():
        raise FileNotFoundError(f"Env config not found: {env_cfg_path}")
    env_cfg = OmegaConf.load(env_cfg_path)

    registry = EnvRegistry()
    env_factory: Callable[[int], Any] = lambda instance_num: registry.load(
        env_cfg, instance_num=instance_num
    )

    feature_sources = _load_feature_sources(Path(args.features_json))
    dsl_fns = get_dsl_functions_dict()
    dsl_fns.update(build_py_feature_functions(feature_sources, dsl_fns))
    set_dsl_functions(dsl_fns)

    map_program = _load_map_program(
        Path(args.map_program_file) if args.map_program_file else None,
        args.map_program,
    )
    plp = StateActionProgram(map_program)
    policy: LPPPolicy = LPPPolicy(
        [plp],
        [1.0],
        # normalize_plp_actions=args.normalize_plp_actions,
    )

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    env_nums = _parse_env_nums(args.test_env_nums)
    if not env_nums:
        raise ValueError("No test env numbers provided.")

    results: list[bool] = []
    for env_num in env_nums:
        env = env_factory(env_num)
        video_path = video_dir / f"{args.env}_{env_num}.{args.video_format}"
        reward, _terminated, _final_info = run_single_episode(
            env,
            policy,
            record_video=True,
            video_out_path=str(video_path),
            max_num_steps=args.max_steps,
        )
        success = reward > 0
        results.append(bool(success))

    print(f"Results: {results}")
    if results:
        print(f"Success rate: {sum(results) / len(results):.3f}")


if __name__ == "__main__":
    main()
