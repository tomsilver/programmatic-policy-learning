"""Run a single LPP evaluation and emit a normalized paper-curves result."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from programmatic_policy_learning.envs.registry import EnvRegistry
from programmatic_policy_learning.paper_curves.common import (
    ensure_dir,
    jsonable,
    read_json,
    setup_logging,
    utc_timestamp,
    write_json,
)


def _load_module(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {file_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _append_repo_paths(repo_root: Path) -> None:
    for candidate in (repo_root, repo_root / "src"):
        candidate_str = str(candidate.resolve())
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _build_synced_prompt_demo_subsets(
    demo_ids: list[int],
    *,
    max_subset_size: int = 3,
    max_num_subsets: int = 4,
    max_num_size3_subsets: int = 2,
) -> list[list[int]]:
    """Split synced demo ids into contiguous prompt subsets.

    Preference order:
    1. All size-2 subsets, if they fit within ``max_num_subsets``.
    2. Then allow up to ``max_num_size3_subsets`` size-3 subsets, keeping the
       remaining subsets size 2 where possible.
    3. Fall back to a singleton subset only when exact 2/3 packing is impossible.
    """
    if not demo_ids:
        return [[0]]
    if max_subset_size <= 0:
        raise ValueError("max_subset_size must be positive.")
    if max_num_subsets <= 0:
        raise ValueError("max_num_subsets must be positive.")
    if max_num_size3_subsets < 0:
        raise ValueError("max_num_size3_subsets cannot be negative.")

    num_demos = len(demo_ids)
    feasible_sizes: list[int] | None = None
    max_size3 = min(int(max_num_size3_subsets), int(max_num_subsets))

    # Prefer all size-2 subsets whenever they fit inside the subset budget.
    if num_demos % 2 == 0:
        num_pairs = num_demos // 2
        if num_pairs <= max_num_subsets:
            feasible_sizes = [2] * num_pairs

    for num_subsets in range(1, int(max_num_subsets) + 1):
        if feasible_sizes is not None:
            break
        for num_size3 in range(0, min(max_size3, num_subsets) + 1):
            remaining = num_demos - 3 * num_size3
            num_size2 = num_subsets - num_size3
            if remaining != 2 * num_size2:
                continue
            feasible_sizes = [3] * num_size3 + [2] * num_size2
            break
        if feasible_sizes is not None:
            break

    if feasible_sizes is None:
        # Allow one singleton subset only when exact 2/3 packing is impossible.
        for num_subsets in range(1, int(max_num_subsets) + 1):
            for num_size1 in range(0, num_subsets + 1):
                for num_size3 in range(0, min(max_size3, num_subsets - num_size1) + 1):
                    num_size2 = num_subsets - num_size1 - num_size3
                    if num_size2 < 0:
                        continue
                    total = num_size3 * 3 + num_size2 * 2 + num_size1
                    if total != num_demos:
                        continue
                    feasible_sizes = [3] * num_size3 + [2] * num_size2 + [1] * num_size1
                    break
                if feasible_sizes is not None:
                    break
            if feasible_sizes is not None:
                break

    if feasible_sizes is None:
        raise ValueError(
            "Cannot split demo ids into prompt subsets without exceeding limits: "
            f"{num_demos=} {max_subset_size=} {max_num_subsets=} "
            f"{max_num_size3_subsets=}."
        )

    subsets: list[list[int]] = []
    start = 0
    for subset_size in feasible_sizes:
        end = start + subset_size
        subsets.append(list(demo_ids[start:end]))
        start = end
    return subsets


def _build_lpp_cfg(job: dict[str, Any], repo_root: Path) -> Any:
    env_name = str(job["environment"]["lpp_env"])
    env_cfg = cast(
        DictConfig,
        OmegaConf.load(repo_root / "experiments" / "conf" / "env" / f"{env_name}.yaml"),
    )
    approach_cfg = cast(
        DictConfig,
        OmegaConf.load(repo_root / "experiments" / "conf" / "approach" / "lpp.yaml"),
    )
    instance_num = int(env_cfg.get("instance_num", 0))
    base_name = str(env_cfg.make_kwargs.base_name)
    concrete_env_id = f"{base_name}{instance_num}-v0"
    env_cfg.make_kwargs.id = concrete_env_id
    if "env_id" in env_cfg:
        env_cfg.env_id = concrete_env_id

    cfg = cast(
        DictConfig,
        OmegaConf.create(
            {
                "seed": int(job["seed"]),
                "approach_name": "lpp",
                "env_name": env_name,
                "expert_policy": "env.expert",
                "approach": approach_cfg,
                "env": env_cfg,
                "eval": {
                    "record_videos": False,
                    "video_format": "mp4",
                    "vector_field": {"enabled": False, "grid_size": 21},
                },
            }
        ),
    )
    cfg.approach.demo_numbers = list(job["demo_ids"])

    method = dict(job["method"])
    overrides = []
    for raw_override in method.get("overrides", []):
        override = str(raw_override)
        key = override.split("=", maxsplit=1)[0].strip()
        if key in {"approach", "env", "expert"} and "." not in key:
            logging.info(
                "Ignoring Hydra choice-style override in wrapper: %s", override
            )
            continue
        overrides.append(override)
    if method.get("sync_prompt_demo_subsets", True):
        max_subset_size = int(method.get("sync_prompt_max_subset_size", 3))
        max_num_subsets = int(method.get("sync_prompt_max_subsets", 4))
        max_num_size3_subsets = int(method.get("sync_prompt_max_size3_subsets", 2))
        synced_subsets = _build_synced_prompt_demo_subsets(
            list(job["demo_ids"]),
            max_subset_size=max_subset_size,
            max_num_subsets=max_num_subsets,
            max_num_size3_subsets=max_num_size3_subsets,
        )
        sync_override = (
            "approach.program_generation.multi_prompt_ensemble.demo_subsets="
            f"{synced_subsets}"
        )
        print(
            "Synced prompt demo subsets "
            f"(demo_count={len(job['demo_ids'])}, "
            f"max_subset_size={max_subset_size}, "
            f"max_num_subsets={max_num_subsets}, "
            f"max_num_size3_subsets={max_num_size3_subsets}):"
        )
        for subset_idx, demo_subset in enumerate(synced_subsets, start=1):
            print(f"  subset {subset_idx}: {demo_subset}")
            logging.info("Prompt subset %d: %s", subset_idx, demo_subset)
        if not any(
            override.startswith(
                "approach.program_generation.multi_prompt_ensemble.demo_subsets="
            )
            for override in overrides
        ):
            overrides.append(sync_override)

    if overrides:
        cfg = cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides)))
    _absolutize_lpp_paths(cfg, repo_root)
    return cfg


def _absolutize_lpp_paths(cfg: Any, repo_root: Path) -> None:
    """Convert common repo-relative LPP config paths to absolute paths."""

    def absolutize(path_value: Any) -> Any:
        if path_value is None:
            return None
        path_str = str(path_value)
        if not path_str:
            return path_str
        path = Path(path_str)
        if path.is_absolute():
            return str(path)
        return str((repo_root / path).resolve())

    program_generation = cfg.approach.program_generation
    for field in (
        "py_feature_gen_prompt",
        "py_feature_gen_batch_prompt",
        "dsl_generator_prompt",
        "feature_generator_prompt",
    ):
        if field in program_generation and program_generation.get(field) is not None:
            program_generation[field] = absolutize(program_generation.get(field))

    loading_cfg = program_generation.get("loading")
    if loading_cfg is not None and "offline_json_path" in loading_cfg:
        offline_json_path = loading_cfg.get("offline_json_path")
        if offline_json_path:
            loading_cfg["offline_json_path"] = absolutize(offline_json_path)


def _important_lpp_config(cfg: Any, method: dict[str, Any]) -> dict[str, Any]:
    approach = cfg.approach
    program_generation = approach.program_generation
    cross_demo_filter = approach.cross_demo_feature_filter
    return {
        "backend": "lpp",
        "method_overrides": list(method.get("overrides", [])),
        "demo_numbers": list(approach.demo_numbers),
        "strategy": program_generation.strategy,
        "encoding_method": program_generation.get("encoding_method"),
        "llm_model": program_generation.get("llm_model"),
        "num_features": program_generation.get("num_features"),
        "num_programs": approach.num_programs,
        "num_dts": approach.num_dts,
        "prompt_demo_subsets": jsonable(
            program_generation.get("multi_prompt_ensemble", {}).get("demo_subsets", [])
        ),
        "collision_feedback_enabled": approach.collision_feedback_enabled,
        "collision_feedback_enc": approach.collision_feedback_enc,
        "cross_demo_feature_filter_enabled": cross_demo_filter.enabled,
        "cross_demo_feature_filter_apply": cross_demo_filter.apply_filter,
        "permissive_filter_enabled": approach.permissive_filter_enabled,
        "prior_version": approach.prior_version,
        "prior_beta": approach.prior_beta,
        "dt_max_depth": approach.dt_max_depth,
    }


def _run(job: dict[str, Any]) -> int:
    artifact_dir = ensure_dir(Path(job["artifact_dir"]).resolve())
    result_path = Path(job["result_path"]).resolve()
    setup_logging(artifact_dir / "wrapper.log")
    repo_root = Path(job["repo_root"]).resolve()
    os.chdir(artifact_dir)
    shared_run_cache_dir = job.get("shared_run_cache_dir")
    if shared_run_cache_dir:
        os.environ["PPL_CACHE_DIR_OVERRIDE"] = str(shared_run_cache_dir)
    _append_repo_paths(repo_root)

    logging.info("Starting LPP single-run wrapper for %s", job["run_id"])
    logging.info("Using repo root: %s", repo_root)
    logging.info("Artifact dir: %s", artifact_dir)

    try:
        run_experiment = _load_module(
            "paper_curves_run_experiment",
            repo_root / "experiments" / "run_experiment.py",
        )
        cfg = _build_lpp_cfg(job, repo_root)
        serialized_cfg = OmegaConf.to_container(cfg, resolve=False)
        write_json(artifact_dir / "resolved_config.json", serialized_cfg)

        registry = EnvRegistry()
        env = registry.load(cfg.env)
        approach = run_experiment.instantiate_approach(cfg, env, registry)

        reset_output = env.reset(seed=int(job["seed"]))
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            obs, info = reset_output
        else:
            obs, info = reset_output, {}
        approach.reset(obs, info)

        test_env_nums = [int(each) for each in job["test_env_nums"]]
        test_results = approach.test_policy_on_envs(
            base_class_name=cfg.env.make_kwargs.base_name,
            test_env_nums=test_env_nums,
            max_num_steps=int(job.get("eval_max_steps", 100)),
            record_videos=False,
        )
        num_test_total = len(test_results)
        num_test_solved = int(sum(bool(each) for each in test_results))
        success_rate = (
            float(num_test_solved / num_test_total) if num_test_total else 0.0
        )

        result = {
            "status": "success",
            "timestamp": utc_timestamp(),
            "backend": "lpp",
            "environment": job["environment"]["name"],
            "environment_key": job["environment"].get(
                "key", job["environment"]["name"]
            ),
            "backend_environment": job["environment"]["lpp_env"],
            "method_name": job["method"]["name"],
            "method_display_name": job["method"].get(
                "display_name", job["method"]["name"]
            ),
            "demo_count": int(job["demo_count"]),
            "demo_ids": [int(each) for each in job["demo_ids"]],
            "feature_count": (
                int(job["feature_count"]) if "feature_count" in job else None
            ),
            "heldout_env_num": (
                int(job["heldout_env_num"]) if "heldout_env_num" in job else None
            ),
            "seed": int(job["seed"]),
            "test_env_nums": test_env_nums,
            "num_test_solved": num_test_solved,
            "num_test_total": num_test_total,
            "success_rate": success_rate,
            "config_fields": jsonable(_important_lpp_config(cfg, dict(job["method"]))),
            "artifact_dir": str(artifact_dir),
            "raw_artifact_paths": {
                "artifact_dir": str(artifact_dir),
                "wrapper_log": str((artifact_dir / "wrapper.log").resolve()),
                "resolved_config": str(
                    (artifact_dir / "resolved_config.json").resolve()
                ),
                "feature_scores": str((artifact_dir / "feature_scores.json").resolve()),
            },
            "backend_command": [
                sys.executable,
                "-m",
                "programmatic_policy_learning.paper_curves.lpp_single_run",
                "--job",
                str(Path(job["job_path"]).resolve()),
            ],
            "run_id": job["run_id"],
        }
        write_json(result_path, result)
        logging.info(
            "Completed LPP run %s with %d/%d solved (%.3f).",
            job["run_id"],
            num_test_solved,
            num_test_total,
            success_rate,
        )
        return 0
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.exception("LPP single-run wrapper failed: %s", exc)
        error_result = {
            "status": "error",
            "timestamp": utc_timestamp(),
            "backend": "lpp",
            "environment": job["environment"]["name"],
            "environment_key": job["environment"].get(
                "key", job["environment"]["name"]
            ),
            "backend_environment": job["environment"]["lpp_env"],
            "method_name": job["method"]["name"],
            "method_display_name": job["method"].get(
                "display_name", job["method"]["name"]
            ),
            "demo_count": int(job["demo_count"]),
            "demo_ids": [int(each) for each in job["demo_ids"]],
            "feature_count": (
                int(job["feature_count"]) if "feature_count" in job else None
            ),
            "heldout_env_num": (
                int(job["heldout_env_num"]) if "heldout_env_num" in job else None
            ),
            "seed": int(job["seed"]),
            "test_env_nums": [int(each) for each in job["test_env_nums"]],
            "num_test_solved": None,
            "num_test_total": len(job["test_env_nums"]),
            "success_rate": None,
            "config_fields": {"backend": "lpp"},
            "artifact_dir": str(artifact_dir),
            "raw_artifact_paths": {
                "artifact_dir": str(artifact_dir),
                "wrapper_log": str((artifact_dir / "wrapper.log").resolve()),
            },
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_id": job["run_id"],
        }
        write_json(result_path, error_result)
        return 1


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path, help="Path to job JSON.")
    args = parser.parse_args()
    job = read_json(args.job)
    job["job_path"] = str(args.job)
    return _run(job)


if __name__ == "__main__":
    raise SystemExit(main())
