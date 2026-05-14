"""Top-level driver for running paper-curve experiments and plots."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from programmatic_policy_learning.paper_curves.aggregate import (
    compute_summary,
    load_results_dataframe,
)
from programmatic_policy_learning.paper_curves.common import (
    demo_ids_for_count,
    ensure_dir,
    load_yaml_config,
    read_json,
    shared_sqlite_cache_dir,
    setup_logging,
    slugify,
    utc_timestamp,
    write_json,
)
from programmatic_policy_learning.paper_curves.plotting import save_environment_plots


def _select_entries(
    entries: list[dict[str, Any]],
    names: list[str] | None,
    *,
    field_names: tuple[str, ...] = ("name", "key"),
) -> list[dict[str, Any]]:
    if not names:
        return entries
    selected = []
    name_set = set(names)
    for entry in entries:
        tokens = {str(entry.get(field, "")) for field in field_names}
        if tokens & name_set:
            selected.append(entry)
    return selected


def _default_output_dir(config: dict[str, Any]) -> Path:
    output_root = Path(str(config.get("output_root", "results/paper_curves")))
    experiment_name = config.get("experiment_name")
    suffix = slugify(str(experiment_name)) if experiment_name else utc_timestamp()
    return output_root / suffix


def _shared_run_cache_dir(
    *,
    results_dir: Path,
    env_key: str,
    method_cfg: dict[str, Any],
    train_demo_ids: list[int],
    seed: int,
) -> str | None:
    """Return a shared run-matrix cache dir for repeatable LPP jobs."""
    if not bool(method_cfg.get("shared_run_cache", False)):
        return None

    ignore_prefixes = [
        str(prefix)
        for prefix in method_cfg.get("shared_cache_ignore_override_prefixes", [])
    ]
    signature_overrides: list[str] = []
    for raw_override in method_cfg.get("overrides", []):
        override = str(raw_override)
        if any(override.startswith(prefix) for prefix in ignore_prefixes):
            continue
        signature_overrides.append(override)

    signature_payload = {
        "method_name": str(
            method_cfg.get("shared_cache_base_name", method_cfg.get("name", ""))
        ),
        "train_demo_ids": [int(each) for each in train_demo_ids],
        "signature_overrides": sorted(signature_overrides),
        "seed": int(seed),
        "kind": "run_cache",
    }
    signature = hashlib.sha1(
        __import__("json").dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    cache_dir = ensure_dir(
        results_dir / "shared_caches" / slugify(env_key) / "run_cache"
    )
    return str(
        (
            cache_dir
            / (
                f"{slugify(str(method_cfg.get('shared_cache_base_name', method_cfg.get('name', 'method'))))}"
                f"__seed{int(seed)}__{signature}"
            )
        ).resolve()
    )


def _build_jobs(
    config: dict[str, Any],
    *,
    results_dir: Path,
    environments: list[dict[str, Any]],
    methods: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    demo_counts = [int(each) for each in config["demo_counts"]]
    global_seeds = [int(each) for each in config["seeds"]]
    demo_id_pool = [int(each) for each in config.get("demo_id_pool", list(range(11)))]
    test_env_nums = [
        int(each) for each in config.get("test_env_nums", list(range(10, 20)))
    ]
    codebases = dict(config.get("codebases", {}))
    jobs: list[dict[str, Any]] = []

    for env_cfg in environments:
        env_key = str(env_cfg.get("key", env_cfg["name"]))
        for method_cfg in methods:
            backend = str(method_cfg["backend"]).lower()
            if backend not in {"lpp", "cap", "fcn", "vlm"}:
                raise ValueError(
                    f"Unsupported backend '{backend}' for method {method_cfg['name']}."
                )
            backend_cfg = dict(codebases.get(backend, {}))
            repo_root = Path(str(backend_cfg.get("root_dir", "."))).resolve()
            backend_python = str(backend_cfg.get("python_executable", sys.executable))
            method_seeds = [int(each) for each in method_cfg.get("seeds", global_seeds)]
            for demo_count in demo_counts:
                demo_ids = demo_ids_for_count(demo_id_pool, demo_count)
                for seed in method_seeds:
                    shared_run_cache_dir = _shared_run_cache_dir(
                        results_dir=results_dir,
                        env_key=env_key,
                        method_cfg=method_cfg,
                        train_demo_ids=demo_ids,
                        seed=seed,
                    )
                    run_id = (
                        f"{slugify(env_key)}__{slugify(str(method_cfg['name']))}"
                        f"__d{demo_count}__s{seed}"
                    )
                    artifact_dir = results_dir / "runs" / run_id
                    result_path = artifact_dir / "result.json"
                    job = {
                        "run_id": run_id,
                        "repo_root": str(repo_root),
                        "backend_python": backend_python,
                        "environment": env_cfg,
                        "method": method_cfg,
                        "seed": int(seed),
                        "demo_count": int(demo_count),
                        "demo_ids": demo_ids,
                        "train_env_nums": list(demo_ids),
                        "test_env_nums": test_env_nums,
                        "shared_sqlite_cache_dir": str(
                            shared_sqlite_cache_dir(results_dir, backend).resolve()
                        ),
                        "shared_run_cache_dir": shared_run_cache_dir,
                        "eval_max_steps": int(config.get("eval_max_steps", 100)),
                        "artifact_dir": str(artifact_dir.resolve()),
                        "result_path": str(result_path.resolve()),
                    }
                    jobs.append(job)
    return jobs


def _wrapper_module_for_backend(backend: str) -> str:
    if backend == "lpp":
        return "programmatic_policy_learning.paper_curves.lpp_single_run"
    if backend == "cap":
        return "programmatic_policy_learning.paper_curves.cap_single_run"
    if backend == "fcn":
        return "programmatic_policy_learning.paper_curves.fcn_single_run"
    if backend == "vlm":
        return "programmatic_policy_learning.paper_curves.vlm_single_run"
    raise ValueError(f"Unsupported backend '{backend}'.")


def _should_skip_existing_result(result_path: Path) -> bool:
    """Return True only for existing successful result files."""
    if not result_path.exists():
        return False
    try:
        payload = read_json(result_path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.warning(
            "Existing result %s is unreadable (%s); rerunning.",
            result_path,
            exc,
        )
        return False
    status = str(payload.get("status", "")).lower()
    if status == "success":
        return True
    logging.info(
        "Existing result %s has status=%r; rerunning.",
        result_path,
        payload.get("status"),
    )
    return False


def _run_jobs(
    jobs: list[dict[str, Any]],
    *,
    results_dir: Path,
    skip_existing: bool,
    continue_on_error: bool,
    dry_run: bool,
) -> None:
    del results_dir
    orchestrator_root = Path(__file__).resolve().parents[3]
    orchestrator_src = orchestrator_root / "src"
    for index, job in enumerate(jobs, start=1):
        artifact_dir = ensure_dir(Path(job["artifact_dir"]))
        result_path = Path(job["result_path"])
        job_path = artifact_dir / "job.json"
        write_json(job_path, job)
        backend = str(job["method"]["backend"]).lower()
        module_name = _wrapper_module_for_backend(backend)
        command = [sys.executable, "-m", module_name, "--job", str(job_path)]
        env = os.environ.copy()
        pythonpath_entries = [str(orchestrator_src)]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        logging.info(
            "[%d/%d] %s | env=%s method=%s demos=%d seed=%d",
            index,
            len(jobs),
            backend.upper(),
            job["environment"]["name"],
            job["method"]["name"],
            job["demo_count"],
            job["seed"],
        )
        if skip_existing and _should_skip_existing_result(result_path):
            logging.info("Skipping existing result: %s", result_path)
            continue
        if dry_run:
            logging.info("Dry run command: %s", " ".join(command))
            continue

        completed = subprocess.run(command, check=False, env=env)
        if completed.returncode != 0:
            message = (
                f"Job {job['run_id']} failed with exit code {completed.returncode}. "
                f"See {artifact_dir / 'wrapper.log'}."
            )
            if continue_on_error:
                logging.error(message)
                continue
            raise RuntimeError(message)


def _write_outputs(
    results_dir: Path,
    config: dict[str, Any],
    environments: list[dict[str, Any]],
    methods: list[dict[str, Any]],
) -> None:
    results_df = load_results_dataframe(results_dir)
    results_csv = results_dir / "all_results.csv"
    results_jsonl = results_dir / "all_results.jsonl"
    summary_csv = results_dir / "summary.csv"

    if results_df.empty:
        logging.warning("No normalized result files were found under %s.", results_dir)
        return

    results_df.to_csv(results_csv, index=False)
    with results_jsonl.open("w", encoding="utf-8") as handle:
        for record in results_df.to_dict(orient="records"):
            handle.write(
                __import__("json").dumps(record, sort_keys=True, default=str) + "\n"
            )

    summary_df = compute_summary(results_df)
    if not summary_df.empty:
        summary_df.to_csv(summary_csv, index=False)
        save_environment_plots(
            summary_df,
            plots_dir=results_dir / "plots",
            environments=environments,
            methods=methods,
            error_band=str(config.get("error_band", "sem")).lower(),
        )
    else:
        logging.warning("No successful runs available for aggregation/plotting.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="YAML config path.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Existing or target results directory. Defaults to "
            "results/paper_curves/<name-or-timestamp>/."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Skip launching experiments and plot from normalized results "
            "already on disk."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs without launching them.",
    )
    parser.add_argument(
        "--environments",
        nargs="*",
        default=None,
        help="Optional subset of environment names/keys to run.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional subset of method names to run.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rerun jobs even if normalized result files already exist.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = _parse_args()
    config = load_yaml_config(args.config.resolve())
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else _default_output_dir(config).resolve()
    )
    ensure_dir(results_dir)
    setup_logging(results_dir / "driver.log")

    environments_cfg = _select_entries(
        [dict(each) for each in config["environments"]],
        args.environments,
        field_names=("name", "key", "lpp_env", "cap_env"),
    )
    methods_cfg = _select_entries(
        [dict(each) for each in config["methods"]],
        args.methods,
        field_names=("name", "display_name"),
    )
    if not environments_cfg:
        raise ValueError("No environments selected after filtering.")
    if not methods_cfg:
        raise ValueError("No methods selected after filtering.")

    write_json(results_dir / "resolved_driver_config.json", config)

    jobs = _build_jobs(
        config,
        results_dir=results_dir,
        environments=environments_cfg,
        methods=methods_cfg,
    )
    logging.info("Prepared %d jobs.", len(jobs))

    if not args.plot_only:
        _run_jobs(
            jobs,
            results_dir=results_dir,
            skip_existing=not args.no_skip_existing,
            continue_on_error=bool(config.get("continue_on_error", True)),
            dry_run=args.dry_run,
        )
    if not args.dry_run:
        _write_outputs(results_dir, config, environments_cfg, methods_cfg)
        logging.info("Results written to %s", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
