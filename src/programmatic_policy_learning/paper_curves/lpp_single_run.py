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
        sync_override = (
            "approach.program_generation.multi_prompt_ensemble.demo_subsets="
            f"[{list(job['demo_ids'])}]"
        )
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
