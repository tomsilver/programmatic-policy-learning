"""Run a single VLM-imitation evaluation and emit a normalized result."""

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


def _resolve_demo_video_paths(
    *,
    job: dict[str, Any],
    repo_root: Path,
) -> list[Path]:
    env_cfg = dict(job["environment"])
    method_cfg = dict(job["method"])
    demo_ids = [int(each) for each in job["demo_ids"]]

    explicit_paths = method_cfg.get("demo_video_paths")
    if explicit_paths is not None:
        resolved = [(repo_root / str(path)).resolve() for path in explicit_paths]
    else:
        pattern = str(
            method_cfg.get(
                "demo_video_pattern",
                "videos/expert_demonstration_{cap_env}_{demo_id}.mp4",
            )
        )
        resolved = []
        for demo_id in demo_ids:
            rel_path = pattern.format(
                demo_id=int(demo_id),
                env_name=str(env_cfg.get("name", "")),
                environment=str(env_cfg.get("name", "")),
                environment_key=str(env_cfg.get("key", env_cfg.get("name", ""))),
                cap_env=str(env_cfg.get("cap_env", "")),
                lpp_env=str(env_cfg.get("lpp_env", "")),
            )
            resolved.append((repo_root / rel_path).resolve())

    missing = [path for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing demo videos for VLM imitation: "
            + ", ".join(str(path) for path in missing)
        )
    return resolved


def _run(job: dict[str, Any]) -> int:
    artifact_dir = ensure_dir(Path(job["artifact_dir"]).resolve())
    result_path = Path(job["result_path"]).resolve()
    setup_logging(artifact_dir / "wrapper.log")

    repo_root = Path(job["repo_root"]).resolve()
    backend_python = str(job.get("backend_python") or sys.executable)
    raw_output_dir = ensure_dir(artifact_dir / "vlm_raw")
    vlm_script = (
        repo_root
        / "src"
        / "programmatic_policy_learning"
        / "dsl"
        / "llm_primitives"
        / "baselines"
        / "vlm_based"
        / "vlm_imitation_baseline.py"
    )
    if not vlm_script.exists():
        raise FileNotFoundError(
            f"Could not find VLM imitation script at {vlm_script}."
        )

    env_cfg = dict(job["environment"])
    method_cfg = dict(job["method"])
    model = str(method_cfg.get("model", "gpt-4.1"))
    max_frames_per_video = int(method_cfg.get("max_frames_per_video", 12))
    sample_every_n_frames = int(method_cfg.get("sample_every_n_frames", 15))
    function_name = str(method_cfg.get("function_name", "policy"))
    demo_video_paths = _resolve_demo_video_paths(job=job, repo_root=repo_root)

    command = [
        backend_python,
        str(vlm_script),
        "--env",
        str(env_cfg["cap_env"]),
        "--video-paths",
        *[str(path) for path in demo_video_paths],
        "--model",
        model,
        "--output-dir",
        str(raw_output_dir),
        "--function-name",
        function_name,
        "--eval-env-nums",
        *[str(int(each)) for each in job["test_env_nums"]],
        "--eval-max-steps",
        str(int(job.get("eval_max_steps", 100))),
        "--max-frames-per-video",
        str(max_frames_per_video),
        "--sample-every-n-frames",
        str(sample_every_n_frames),
    ]

    logging.info("Starting VLM imitation single-run wrapper for %s", job["run_id"])
    logging.info("Running command: %s", " ".join(command))

    try:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            logging.info("VLM imitation stdout:\n%s", completed.stdout)
        if completed.stderr:
            logging.warning("VLM imitation stderr:\n%s", completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                "VLM imitation subprocess failed with exit code "
                f"{completed.returncode}.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        metadata_path = _latest_file(list(raw_output_dir.rglob("metadata_*.json")))
        metadata = read_json(metadata_path)
        evaluation = dict(metadata.get("evaluation") or {})
        results = [bool(each) for each in evaluation.get("results", [])]
        if not results:
            raise RuntimeError(
                f"No VLM imitation evaluation results found in {metadata_path}."
            )
        num_test_total = len(results)
        num_test_solved = int(sum(results))
        success_rate = float(num_test_solved / num_test_total)
        policy_path = Path(str(metadata.get("policy_path", "")))
        debug_log_path = Path(str(metadata.get("debug_log_path", "")))

        result = {
            "status": "success",
            "timestamp": utc_timestamp(),
            "backend": "vlm",
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
                    "backend": "vlm",
                    "model": model,
                    "max_frames_per_video": max_frames_per_video,
                    "sample_every_n_frames": sample_every_n_frames,
                    "function_name": function_name,
                    "demo_video_paths": [str(path) for path in demo_video_paths],
                    "task_description": metadata.get("task_description"),
                }
            ),
            "artifact_dir": str(artifact_dir),
            "raw_artifact_paths": jsonable(
                {
                    "artifact_dir": str(artifact_dir),
                    "wrapper_log": str((artifact_dir / "wrapper.log").resolve()),
                    "vlm_metadata": str(metadata_path.resolve()),
                    "vlm_policy": str(policy_path.resolve()) if policy_path else None,
                    "vlm_debug_log": (
                        str(debug_log_path.resolve()) if debug_log_path else None
                    ),
                    "vlm_output_dir": str(raw_output_dir.resolve()),
                }
            ),
            "backend_command": command,
            "run_id": job["run_id"],
        }
        write_json(result_path, result)
        logging.info(
            "Completed VLM imitation run %s with %d/%d solved (%.3f).",
            job["run_id"],
            num_test_solved,
            num_test_total,
            success_rate,
        )
        return 0
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.exception("VLM imitation single-run wrapper failed: %s", exc)
        error_result = {
            "status": "error",
            "timestamp": utc_timestamp(),
            "backend": "vlm",
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
            "config_fields": {
                "backend": "vlm",
                "model": model,
            },
            "artifact_dir": str(artifact_dir),
            "raw_artifact_paths": {
                "artifact_dir": str(artifact_dir),
                "wrapper_log": str((artifact_dir / "wrapper.log").resolve()),
                "vlm_output_dir": str(raw_output_dir.resolve()),
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
