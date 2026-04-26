"""Run a single CaP evaluation and emit a normalized paper-curves result."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from programmatic_policy_learning.paper_curves.common import (
    ensure_dir,
    jsonable,
    read_json,
    setup_logging,
    utc_timestamp,
    write_json,
)


def _latest_file(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No matching files found.")
    return max(paths, key=lambda path: path.stat().st_mtime)


def _run(job: dict[str, Any]) -> int:
    artifact_dir = ensure_dir(Path(job["artifact_dir"]).resolve())
    result_path = Path(job["result_path"]).resolve()
    setup_logging(artifact_dir / "wrapper.log")

    repo_root = Path(job["repo_root"]).resolve()
    backend_python = str(job.get("backend_python") or sys.executable)
    raw_output_dir = ensure_dir(artifact_dir / "cap_raw")
    cap_script = (
        repo_root
        / "src"
        / "programmatic_policy_learning"
        / "dsl"
        / "llm_primitives"
        / "baselines"
        / "llm_based"
        / "CaP_baseline.py"
    )
    if not cap_script.exists():
        raise FileNotFoundError(f"Could not find CaP baseline script at {cap_script}.")

    env_cfg = dict(job["environment"])
    method_cfg = dict(job["method"])
    cli_args = [str(each) for each in method_cfg.get("cli_args", [])]
    command = [
        backend_python,
        str(cap_script),
        *cli_args,
        "--env",
        str(env_cfg["cap_env"]),
        "--seeds",
        str(int(job["seed"])),
        "--demo-env-nums",
        *[str(int(each)) for each in job["demo_ids"]],
        "--eval-env-nums",
        *[str(int(each)) for each in job["test_env_nums"]],
        "--output-dir",
        str(raw_output_dir),
        "--sleep-between-runs",
        "0",
        "--eval-max-steps",
        str(int(job.get("eval_max_steps", 100))),
        "--no-plot-results",
        "--no-eval-run-expert",
    ]
    if "cap_env_type" in env_cfg:
        command.extend(["--env-type", str(env_cfg["cap_env_type"])])
    if "cap_num_passages" in env_cfg:
        command.extend(["--num-passages", str(int(env_cfg["cap_num_passages"]))])

    logging.info("Starting CaP single-run wrapper for %s", job["run_id"])
    logging.info("Running command: %s", " ".join(command))

    try:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"CaP subprocess failed with exit code {completed.returncode}."
            )

        metadata_path = _latest_file(
            list(raw_output_dir.rglob(f"metadata_seed{int(job['seed'])}.json"))
        )
        metadata = read_json(metadata_path)
        evaluation = dict(metadata.get("evaluation") or {})
        cap_results = [bool(each) for each in evaluation.get("cap_results", [])]
        if not cap_results:
            raise RuntimeError(
                f"No CaP evaluation results found in metadata file {metadata_path}."
            )
        num_test_total = len(cap_results)
        num_test_solved = int(sum(cap_results))
        success_rate = float(num_test_solved / num_test_total)

        encoding = metadata.get("encoding")
        policy_path = Path(str(metadata.get("policy_path", "")))
        debug_candidates = list(
            metadata_path.parent.glob(f"debug_seed{int(job['seed'])}.log")
        )
        debug_log_path = _latest_file(debug_candidates) if debug_candidates else None

        result = {
            "status": "success",
            "timestamp": utc_timestamp(),
            "backend": "cap",
            "environment": env_cfg["name"],
            "environment_key": env_cfg.get("key", env_cfg["name"]),
            "backend_environment": env_cfg["cap_env"],
            "method_name": method_cfg["name"],
            "method_display_name": method_cfg.get("display_name", method_cfg["name"]),
            "demo_count": int(job["demo_count"]),
            "demo_ids": [int(each) for each in job["demo_ids"]],
            "seed": int(job["seed"]),
            "test_env_nums": [int(each) for each in job["test_env_nums"]],
            "num_test_solved": num_test_solved,
            "num_test_total": num_test_total,
            "success_rate": success_rate,
            "config_fields": jsonable(
                {
                    "backend": "cap",
                    "cli_args": cli_args,
                    "encoding": encoding,
                    "model": metadata.get("model"),
                    "env_type": metadata.get("env_type"),
                    "demo_env_nums": metadata.get("demo_env_nums"),
                    "num_demos": metadata.get("num_demos"),
                    "eval_max_steps": evaluation.get("eval_max_steps"),
                }
            ),
            "artifact_dir": str(artifact_dir),
            "raw_artifact_paths": jsonable(
                {
                    "artifact_dir": str(artifact_dir),
                    "wrapper_log": str((artifact_dir / "wrapper.log").resolve()),
                    "cap_metadata": str(metadata_path.resolve()),
                    "cap_policy": str(policy_path.resolve()) if policy_path else None,
                    "cap_debug_log": (
                        str(debug_log_path.resolve())
                        if debug_log_path is not None
                        else None
                    ),
                    "cap_output_dir": str(raw_output_dir.resolve()),
                }
            ),
            "backend_command": command,
            "run_id": job["run_id"],
        }
        write_json(result_path, result)
        logging.info(
            "Completed CaP run %s with %d/%d solved (%.3f).",
            job["run_id"],
            num_test_solved,
            num_test_total,
            success_rate,
        )
        return 0
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.exception("CaP single-run wrapper failed: %s", exc)
        error_result = {
            "status": "error",
            "timestamp": utc_timestamp(),
            "backend": "cap",
            "environment": env_cfg["name"],
            "environment_key": env_cfg.get("key", env_cfg["name"]),
            "backend_environment": env_cfg["cap_env"],
            "method_name": method_cfg["name"],
            "method_display_name": method_cfg.get("display_name", method_cfg["name"]),
            "demo_count": int(job["demo_count"]),
            "demo_ids": [int(each) for each in job["demo_ids"]],
            "seed": int(job["seed"]),
            "test_env_nums": [int(each) for each in job["test_env_nums"]],
            "num_test_solved": None,
            "num_test_total": len(job["test_env_nums"]),
            "success_rate": None,
            "config_fields": {"backend": "cap", "cli_args": cli_args},
            "artifact_dir": str(artifact_dir),
            "raw_artifact_paths": {
                "artifact_dir": str(artifact_dir),
                "wrapper_log": str((artifact_dir / "wrapper.log").resolve()),
                "cap_output_dir": str(raw_output_dir.resolve()),
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
