"""Run a single FCN baseline evaluation and emit a normalized result."""

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


def _build_fcn_cfg(job: dict[str, Any], repo_root: Path) -> DictConfig:
    env_name = str(job["environment"]["lpp_env"])
    env_cfg = cast(
        DictConfig,
        OmegaConf.load(repo_root / "experiments" / "conf" / "env" / f"{env_name}.yaml"),
    )
    approach_cfg = cast(
        DictConfig,
        OmegaConf.load(repo_root / "experiments" / "conf" / "approach" / "fcn.yaml"),
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
                "approach_name": "fcn",
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

    overrides = [str(each) for each in job["method"].get("overrides", [])]
    if overrides:
        cfg = cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides)))
    return cfg


def _important_fcn_config(cfg: DictConfig, method: dict[str, Any]) -> dict[str, Any]:
    approach = cfg.approach
    return {
        "backend": "fcn",
        "method_overrides": list(method.get("overrides", [])),
        "demo_numbers": list(approach.demo_numbers),
        "num_epochs": approach.num_epochs,
        "min_epochs": approach.min_epochs,
        "batch_size": approach.batch_size,
        "learning_rate": approach.learning_rate,
        "weight_decay": approach.weight_decay,
        "num_conv_layers": approach.num_conv_layers,
        "input_hidden_channels": approach.input_hidden_channels,
        "hidden_channels": approach.hidden_channels,
        "device": approach.device,
    }


def _run(job: dict[str, Any]) -> int:
    artifact_dir = ensure_dir(Path(job["artifact_dir"]).resolve())
    result_path = Path(job["result_path"]).resolve()
    setup_logging(artifact_dir / "wrapper.log")
    repo_root = Path(job["repo_root"]).resolve()
    os.chdir(artifact_dir)
    _append_repo_paths(repo_root)

    logging.info("Starting FCN single-run wrapper for %s", job["run_id"])
    logging.info("Using repo root: %s", repo_root)
    logging.info("Artifact dir: %s", artifact_dir)

    try:
        run_experiment = _load_module(
            "paper_curves_run_experiment",
            repo_root / "experiments" / "run_experiment.py",
        )
        cfg = _build_fcn_cfg(job, repo_root)
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

        training_summary = getattr(approach, "training_summary", None)
        if training_summary is not None:
            write_json(artifact_dir / "training_summary.json", jsonable(training_summary))
        if hasattr(approach, "save"):
            try:
                approach.save(artifact_dir / "model_state.pt")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logging.warning("Could not save FCN model state: %s", exc)

        result = {
            "status": "success",
            "timestamp": utc_timestamp(),
            "backend": "fcn",
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
            "config_fields": jsonable(_important_fcn_config(cfg, dict(job["method"]))),
            "artifact_dir": str(artifact_dir),
            "raw_artifact_paths": {
                "artifact_dir": str(artifact_dir),
                "wrapper_log": str((artifact_dir / "wrapper.log").resolve()),
                "resolved_config": str(
                    (artifact_dir / "resolved_config.json").resolve()
                ),
                "training_summary": str(
                    (artifact_dir / "training_summary.json").resolve()
                ),
                "model_state": str((artifact_dir / "model_state.pt").resolve()),
            },
            "backend_command": [
                sys.executable,
                "-m",
                "programmatic_policy_learning.paper_curves.fcn_single_run",
                "--job",
                str(Path(job["job_path"]).resolve()),
            ],
            "run_id": job["run_id"],
        }
        write_json(result_path, result)
        logging.info(
            "Completed FCN run %s with %d/%d solved (%.3f).",
            job["run_id"],
            num_test_solved,
            num_test_total,
            success_rate,
        )
        return 0
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.exception("FCN single-run wrapper failed: %s", exc)
        error_result = {
            "status": "error",
            "timestamp": utc_timestamp(),
            "backend": "fcn",
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
            "config_fields": {"backend": "fcn"},
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

