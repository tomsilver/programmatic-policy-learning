"""Plotting utilities for paper-curve experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from programmatic_policy_learning.paper_curves.common import ensure_dir, slugify


def _apply_plot_style() -> None:
    """Configure a clean matplotlib style for paper plots."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def save_environment_plots(
    summary_df: pd.DataFrame,
    *,
    plots_dir: Path,
    environments: list[dict[str, Any]],
    methods: list[dict[str, Any]],
    error_band: str = "sem",
    x_key: str = "demo_count",
    x_label: str = "Number of demonstrations",
) -> list[Path]:
    """Save train/test split plots and a combined plot per environment."""
    if summary_df.empty:
        return []

    _apply_plot_style()
    ensure_dir(plots_dir)
    saved_paths: list[Path] = []
    method_order = [str(method["name"]) for method in methods]
    method_labels = {
        str(method["name"]): str(method.get("display_name", method["name"]))
        for method in methods
    }
    method_styles = {
        str(method["name"]): dict(method.get("plot_style", {})) for method in methods
    }

    for env_cfg in environments:
        env_name = str(env_cfg["name"])
        env_key = str(env_cfg.get("key", env_name))
        caption = str(env_cfg.get("plot_caption", "")).strip()
        env_df = summary_df[summary_df["environment_key"] == env_key].copy()
        if env_df.empty:
            continue

        split_order = ["train", "test"]
        available_splits = (
            [split for split in split_order if split in set(env_df["eval_split"])]
            if "eval_split" in env_df.columns
            else ["test"]
        )

        for split_name in available_splits:
            split_df = (
                env_df[env_df["eval_split"] == split_name].copy()
                if "eval_split" in env_df.columns
                else env_df.copy()
            )
            if split_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(6.4, 4.1))
            color_cycle = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])

            for method_name in method_order:
                method_df = split_df[
                    split_df["method_name"] == method_name
                ].sort_values(x_key)
                if method_df.empty:
                    continue
                style = dict(method_styles.get(method_name, {}))
                color = style.pop("color", next(color_cycle, None))
                linestyle = style.pop("linestyle", "-")
                marker = style.pop("marker", "o")
                x_values = method_df[x_key].to_numpy()
                y_values = method_df["mean_success_rate"].to_numpy()
                band_values = None
                if error_band == "std":
                    band_values = method_df["std_success_rate"].to_numpy()
                elif error_band == "sem":
                    band_values = method_df["sem"].to_numpy()

                ax.plot(
                    x_values,
                    y_values,
                    label=method_labels.get(method_name, method_name),
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2.0,
                    markersize=5.5,
                    **style,
                )
                if band_values is not None:
                    lower = (y_values - band_values).clip(0.0, 1.0)
                    upper = (y_values + band_values).clip(0.0, 1.0)
                    ax.fill_between(x_values, lower, upper, color=color, alpha=0.18)

            metric_label = (
                str(split_df["metric_label"].iloc[0])
                if "metric_label" in split_df.columns and not split_df.empty
                else (
                    "Train success rate"
                    if split_name == "train"
                    else "Test success rate"
                )
            )
            split_title = "Train" if split_name == "train" else "Test"
            ax.set_xlabel(x_label)
            ax.set_ylabel(metric_label)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(left=0)
            ax.set_title(f"{env_cfg.get('plot_title', env_name)} ({split_title})")
            ax.legend(frameon=False)
            ax.set_axisbelow(True)
            if caption:
                fig.text(
                    0.5,
                    0.01,
                    caption,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    wrap=True,
                )
                fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
            else:
                fig.tight_layout()

            stem = slugify(env_key)
            suffix = slugify(split_name)
            png_path = plots_dir / f"{stem}_{suffix}.png"
            pdf_path = plots_dir / f"{stem}_{suffix}.pdf"
            caption_path = plots_dir / f"{stem}_{suffix}.caption.txt"
            fig.savefig(png_path, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
            saved_paths.extend([png_path, pdf_path])
            if caption:
                caption_path.write_text(f"{caption}\n", encoding="utf-8")
                saved_paths.append(caption_path)

        if "eval_split" in env_df.columns and set(env_df["eval_split"]) & {
            "train",
            "test",
        }:
            fig, ax = plt.subplots(figsize=(6.8, 4.3))
            base_colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
            method_color_map: dict[str, Any] = {}

            for method_name in method_order:
                base_method_df = env_df[env_df["method_name"] == method_name]
                if base_method_df.empty:
                    continue
                style = dict(method_styles.get(method_name, {}))
                color = style.get("color", next(base_colors, None))
                method_color_map[method_name] = color

            split_styles = {
                "train": {"linestyle": "-", "marker": "o", "alpha": 0.18},
                "test": {"linestyle": "--", "marker": "s", "alpha": 0.12},
            }

            for method_name in method_order:
                for split_name in ("train", "test"):
                    method_df = env_df[
                        (env_df["method_name"] == method_name)
                        & (env_df["eval_split"] == split_name)
                    ].sort_values(x_key)
                    if method_df.empty:
                        continue
                    style = dict(method_styles.get(method_name, {}))
                    color = method_color_map.get(method_name)
                    split_style = split_styles[split_name]
                    x_values = method_df[x_key].to_numpy()
                    y_values = method_df["mean_success_rate"].to_numpy()
                    band_values = None
                    if error_band == "std":
                        band_values = method_df["std_success_rate"].to_numpy()
                    elif error_band == "sem":
                        band_values = method_df["sem"].to_numpy()

                    ax.plot(
                        x_values,
                        y_values,
                        label=f"{method_labels.get(method_name, method_name)} ({split_name})",
                        color=color,
                        linestyle=split_style["linestyle"],
                        marker=split_style["marker"],
                        linewidth=2.0,
                        markersize=5.2,
                        **{
                            k: v
                            for k, v in style.items()
                            if k not in {"color", "linestyle", "marker"}
                        },
                    )
                    if band_values is not None:
                        lower = (y_values - band_values).clip(0.0, 1.0)
                        upper = (y_values + band_values).clip(0.0, 1.0)
                        ax.fill_between(
                            x_values,
                            lower,
                            upper,
                            color=color,
                            alpha=float(split_style["alpha"]),
                        )

            ax.set_xlabel(x_label)
            ax.set_ylabel("Success rate")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(left=0)
            ax.set_title(f"{env_cfg.get('plot_title', env_name)} (Train + Test)")
            ax.legend(frameon=False, ncol=2)
            ax.set_axisbelow(True)
            if caption:
                fig.text(
                    0.5,
                    0.01,
                    caption,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    wrap=True,
                )
                fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
            else:
                fig.tight_layout()

            stem = slugify(env_key)
            png_path = plots_dir / f"{stem}_combined.png"
            pdf_path = plots_dir / f"{stem}_combined.pdf"
            caption_path = plots_dir / f"{stem}_combined.caption.txt"
            fig.savefig(png_path, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
            saved_paths.extend([png_path, pdf_path])
            if caption:
                caption_path.write_text(f"{caption}\n", encoding="utf-8")
                saved_paths.append(caption_path)

    return saved_paths
