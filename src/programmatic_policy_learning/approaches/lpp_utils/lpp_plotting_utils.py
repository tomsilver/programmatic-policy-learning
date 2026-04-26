"""Plotting helpers for LPP diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from hydra.core.hydra_config import HydraConfig

plt: Any | None = None
Rectangle: Any | None = None

try:
    import matplotlib.pyplot as _plt
    from matplotlib.patches import Rectangle as _Rectangle

    plt = _plt
    Rectangle = _Rectangle
except ImportError:
    pass


def plot_policy_vector_fields(
    *,
    base_class_name: str,
    approach_base_class_name: str,
    policy: Callable[[Any], Any],
    env_factory: Callable[[int | None], Any],
    env_specs: Mapping[str, Any],
    env_nums: Sequence[int],
    grid_size: int = 21,
    split_name: str = "eval",
) -> list[str]:
    """Save Motion2D policy vector-field plots for selected envs."""
    if "motion2d" not in approach_base_class_name.lower():
        logging.info(
            "Vector field plotting skipped: base_class_name=%s is not Motion2D.",
            approach_base_class_name,
        )
        return []

    if plt is None or Rectangle is None:
        logging.warning("Vector field plotting skipped: matplotlib is unavailable.")
        return []

    try:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:  # pylint: disable=broad-exception-caught
        output_dir = Path.cwd()
    vector_dir = output_dir / "vector_fields"
    vector_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    action_mode = str(env_specs.get("action_mode", "discrete"))
    if action_mode != "continuous":
        logging.info(
            "Vector field plotting skipped: action_mode=%s is not continuous.",
            action_mode,
        )
        return []

    for env_num in env_nums:
        env = env_factory(int(env_num))
        try:
            try:
                reset_out = env.reset(seed=int(env_num))
            except TypeError:
                reset_out = env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                obs, _info = reset_out
            else:
                obs, _info = reset_out, {}

            base_obs = np.asarray(obs, dtype=float).reshape(-1).copy()
            if base_obs.size < 19:
                logging.warning(
                    "Vector field plotting skipped for env %d: unexpected obs shape.",
                    env_num,
                )
                continue

            x_low = 0.2
            x_high = 2.5
            y_low = 0.2
            y_high = 2.5

            xs = np.linspace(x_low, x_high, max(2, int(grid_size)))
            ys = np.linspace(y_low, y_high, max(2, int(grid_size)))
            grid_x, grid_y = np.meshgrid(xs, ys)
            u = np.full_like(grid_x, np.nan, dtype=float)
            v = np.full_like(grid_y, np.nan, dtype=float)

            obstacles: list[tuple[float, float, float, float]] = []
            num_obstacles = max(0, (base_obs.size - 19) // 10)
            for idx in range(num_obstacles):
                base = 19 + 10 * idx
                obstacles.append(
                    (
                        float(base_obs[base]),
                        float(base_obs[base + 1]),
                        float(base_obs[base + 8]),
                        float(base_obs[base + 9]),
                    )
                )
            obstacle_boxes = tuple(obstacles)

            def _inside_any_obstacle(
                x: float,
                y: float,
                boxes: tuple[tuple[float, float, float, float], ...] = obstacle_boxes,
            ) -> bool:
                for ox, oy, ow, oh in boxes:
                    if ox <= x <= ox + ow and oy <= y <= oy + oh:
                        return True
                return False

            for row_idx in range(grid_y.shape[0]):
                for col_idx in range(grid_x.shape[1]):
                    x = float(grid_x[row_idx, col_idx])
                    y = float(grid_y[row_idx, col_idx])
                    if _inside_any_obstacle(x, y):
                        continue
                    probe_obs = base_obs.copy()
                    probe_obs[0] = x
                    probe_obs[1] = y
                    action = np.asarray(policy(probe_obs), dtype=float).reshape(-1)
                    if action.size < 2:
                        continue
                    u[row_idx, col_idx] = float(action[0])
                    v[row_idx, col_idx] = float(action[1])

            plt_mod = cast(Any, plt)
            rectangle_cls = cast(Any, Rectangle)
            fig, ax = plt_mod.subplots(figsize=(7, 7))
            ax.quiver(
                grid_x,
                grid_y,
                u,
                v,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
                color="tab:blue",
            )

            target_x = float(base_obs[9])
            target_y = float(base_obs[10])
            target_w = float(base_obs[17])
            target_h = float(base_obs[18])
            target_cx = target_x + target_w / 2.0
            target_cy = target_y + target_h / 2.0
            ax.add_patch(
                rectangle_cls(
                    (target_x, target_y),
                    target_w,
                    target_h,
                    facecolor="tab:green",
                    edgecolor="tab:green",
                    alpha=0.25,
                    linewidth=2,
                )
            )
            ax.scatter(
                [target_cx],
                [target_cy],
                c="tab:green",
                s=70,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                label="target center",
                zorder=5,
            )
            ax.annotate(
                "target",
                (target_cx, target_cy),
                xytext=(6, 6),
                textcoords="offset points",
                color="tab:green",
                fontsize=9,
                weight="bold",
            )

            for ox, oy, ow, oh in obstacles:
                ax.add_patch(
                    rectangle_cls(
                        (ox, oy),
                        ow,
                        oh,
                        facecolor="tab:red",
                        edgecolor="tab:red",
                        alpha=0.25,
                        linewidth=1.5,
                    )
                )

            ax.scatter(
                [float(base_obs[0])],
                [float(base_obs[1])],
                c="black",
                s=40,
                label="reset state",
            )
            ax.set_xlim(x_low, x_high)
            ax.set_ylim(y_low, y_high)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(
                f"{base_class_name} {split_name} env {env_num} policy vector field"
            )
            ax.set_xlabel("robot x")
            ax.set_ylabel("robot y")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()

            out_path = (
                vector_dir
                / f"lfd_{base_class_name}_{split_name}_env{env_num}_vector_field.png"
            )
            fig.savefig(out_path, dpi=200)
            plt_mod.close(fig)
            saved_paths.append(str(out_path))
            logging.info("Saved vector field for env %d to %s", env_num, out_path)
        finally:
            if hasattr(env, "close"):
                env.close()
    return saved_paths
