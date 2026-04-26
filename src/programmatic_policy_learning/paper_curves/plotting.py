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
) -> list[Path]:
    """Save one line plot per environment."""
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
        env_df = summary_df[summary_df["environment_key"] == env_key].copy()
        if env_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.4, 4.1))
        color_cycle = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        for method_name in method_order:
            method_df = env_df[env_df["method_name"] == method_name].sort_values(
                "demo_count"
            )
            if method_df.empty:
                continue
            style = dict(method_styles.get(method_name, {}))
            color = style.pop("color", next(color_cycle, None))
            linestyle = style.pop("linestyle", "-")
            marker = style.pop("marker", "o")
            x_values = method_df["demo_count"].to_numpy()
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

        ax.set_xlabel("Number of demonstrations")
        ax.set_ylabel("Test success rate")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(left=0)
        ax.set_title(str(env_cfg.get("plot_title", env_name)))
        ax.legend(frameon=False)
        ax.set_axisbelow(True)
        fig.tight_layout()

        stem = slugify(env_key)
        png_path = plots_dir / f"{stem}.png"
        pdf_path = plots_dir / f"{stem}.pdf"
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        saved_paths.extend([png_path, pdf_path])

    return saved_paths
