"""Top-level driver for running feature-budget curve experiments and plots."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

from programmatic_policy_learning.paper_curves.aggregate import (
    compute_summary,
    load_results_dataframe,
)
from programmatic_policy_learning.paper_curves.common import (
    ensure_dir,
    load_yaml_config,
    setup_logging,
    slugify,
    write_json,
)
from programmatic_policy_learning.paper_curves.driver import (
    _default_output_dir,
    _run_jobs,
    _select_entries,
)
from programmatic_policy_learning.paper_curves.plotting import save_environment_plots


def _apply_feature_budget_override(
    method_cfg: dict[str, Any],
    feature_count: int,
) -> dict[str, Any]:
    method = dict(method_cfg)
    override_key = method.get("feature_budget_override")
    overrides = [str(override) for override in method.get("overrides", [])]
    if override_key:
        prefix = f"{override_key}="
        overrides = [
            override for override in overrides if not str(override).startswith(prefix)
        ]
        overrides.append(f"{override_key}={int(feature_count)}")
    method["overrides"] = overrides
    return method


def _shared_py_feature_cache_override(
    *,
    results_dir: Path,
    env_key: str,
    method_cfg: dict[str, Any],
    train_demo_ids: list[int],
    seed: int,
) -> str | None:
    if not bool(method_cfg.get("shared_py_feature_cache", False)):
        return None

    override_key = str(
        method_cfg.get(
            "feature_budget_override",
            "approach.cross_demo_feature_filter.top_k_features",
        )
    )
    signature_overrides = [
        str(override)
        for override in method_cfg.get("overrides", [])
        if not str(override).startswith(f"{override_key}=")
    ]
    signature_payload = {
        "method_name": str(method_cfg.get("name", "")),
        "train_demo_ids": [int(each) for each in train_demo_ids],
        "signature_overrides": sorted(signature_overrides),
        "seed": int(seed),
    }
    signature = hashlib.sha1(
        __import__("json").dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    cache_dir = ensure_dir(results_dir / "shared_caches" / slugify(env_key))
    cache_path = cache_dir / (
        f"{slugify(str(method_cfg.get('name', 'method')))}"
        f"__seed{int(seed)}__{signature}.db"
    )
    return f"approach.program_generation.py_feature_cache_path={cache_path.resolve()}"


def _shared_run_cache_dir(
    *,
    results_dir: Path,
    env_key: str,
    method_cfg: dict[str, Any],
    train_demo_ids: list[int],
    seed: int,
) -> str | None:
    if not bool(method_cfg.get("shared_run_cache", False)):
        return None

    override_key = str(
        method_cfg.get(
            "feature_budget_override",
            "approach.cross_demo_feature_filter.top_k_features",
        )
    )
    signature_overrides = [
        str(override)
        for override in method_cfg.get("overrides", [])
        if not str(override).startswith(f"{override_key}=")
    ]
    signature_payload = {
        "method_name": str(method_cfg.get("name", "")),
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
                f"{slugify(str(method_cfg.get('name', 'method')))}"
                f"__seed{int(seed)}__{signature}"
            )
        ).resolve()
    )


def _compute_feature_summary(results_df: Any) -> Any:
    """Aggregate per-seed feature-budget results into mean/std/sem curves."""
    if "feature_count" not in results_df.columns:
        return results_df.iloc[0:0].copy()

    success_df = results_df[results_df["status"] == "success"].copy()
    success_df = success_df[success_df["feature_count"].notna()].copy()
    if success_df.empty:
        return success_df
    summary_df = compute_summary(success_df, x_key="feature_count")
    if not summary_df.empty:
        summary_df = summary_df.rename(columns={"num_seeds": "num_seed_runs"})
    return summary_df


def _build_jobs(
    config: dict[str, Any],
    *,
    results_dir: Path,
    environments: list[dict[str, Any]],
    methods: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    feature_counts = [int(each) for each in config["feature_counts"]]
    global_seeds = [int(each) for each in config["seeds"]]
    train_demo_ids = [int(each) for each in config["train_demo_ids"]]
    test_env_nums = [int(each) for each in config["test_env_nums"]]
    codebases = dict(config.get("codebases", {}))
    jobs: list[dict[str, Any]] = []

    for env_cfg in environments:
        env_key = str(env_cfg.get("key", env_cfg["name"]))
        for method_cfg in methods:
            backend = str(method_cfg["backend"]).lower()
            if backend not in {"lpp", "cap"}:
                raise ValueError(
                    f"Unsupported backend '{backend}' for method {method_cfg['name']}."
                )
            backend_cfg = dict(codebases.get(backend, {}))
            repo_root = Path(str(backend_cfg.get("root_dir", "."))).resolve()
            backend_python = str(backend_cfg.get("python_executable", "python"))
            method_seeds = [int(each) for each in method_cfg.get("seeds", global_seeds)]
            for feature_count in feature_counts:
                feature_method_cfg = _apply_feature_budget_override(
                    method_cfg,
                    feature_count,
                )
                for seed in method_seeds:
                    shared_cache_override = _shared_py_feature_cache_override(
                        results_dir=results_dir,
                        env_key=env_key,
                        method_cfg=feature_method_cfg,
                        train_demo_ids=train_demo_ids,
                        seed=seed,
                    )
                    shared_run_cache_dir = _shared_run_cache_dir(
                        results_dir=results_dir,
                        env_key=env_key,
                        method_cfg=feature_method_cfg,
                        train_demo_ids=train_demo_ids,
                        seed=seed,
                    )
                    if shared_cache_override is not None:
                        feature_method_cfg = dict(feature_method_cfg)
                        feature_method_cfg["overrides"] = list(
                            feature_method_cfg.get("overrides", [])
                        ) + [shared_cache_override]
                    run_id = (
                        f"{env_key.lower()}__"
                        f"{str(feature_method_cfg['name']).replace('_', '-').lower()}__"
                        f"__f{feature_count}__s{seed}"
                    )
                    artifact_dir = results_dir / "runs" / run_id
                    result_path = artifact_dir / "result.json"
                    jobs.append(
                        {
                            "run_id": run_id,
                            "repo_root": str(repo_root),
                            "backend_python": backend_python,
                            "environment": env_cfg,
                            "method": feature_method_cfg,
                            "seed": int(seed),
                            "demo_count": len(train_demo_ids),
                            "demo_ids": list(train_demo_ids),
                            "feature_count": int(feature_count),
                            "test_env_nums": list(test_env_nums),
                            "shared_run_cache_dir": shared_run_cache_dir,
                            "eval_max_steps": int(config.get("eval_max_steps", 100)),
                            "artifact_dir": str(artifact_dir.resolve()),
                            "result_path": str(result_path.resolve()),
                        }
                    )
    return jobs


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

    if "feature_count" in results_df.columns:
        results_df = results_df[results_df["feature_count"].notna()].copy()
    else:
        logging.warning(
            "No feature_count column found in normalized results under %s; "
            "skipping feature-curve aggregation.",
            results_dir,
        )
        return

    if results_df.empty:
        logging.warning(
            "No feature-curve result files found under %s after filtering.", results_dir
        )
        return

    results_df.to_csv(results_csv, index=False)
    with results_jsonl.open("w", encoding="utf-8") as handle:
        for record in results_df.to_dict(orient="records"):
            handle.write(
                __import__("json").dumps(record, sort_keys=True, default=str) + "\n"
            )

    summary_df = _compute_feature_summary(results_df)
    if not summary_df.empty:
        summary_df.to_csv(summary_csv, index=False)
        save_environment_plots(
            summary_df,
            plots_dir=results_dir / "plots",
            environments=environments,
            methods=methods,
            error_band=str(config.get("error_band", "sem")).lower(),
            x_key="feature_count",
            x_label="Number of features",
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

    write_json(results_dir / "resolved_feature_driver_config.json", config)

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
