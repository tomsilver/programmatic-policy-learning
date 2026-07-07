"""HTML visualization for ARC-AGI-3 LPP decisions."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

_ACTION_NAMES = {
    1: "Up",
    2: "Down",
    3: "Left",
    4: "Right",
}

_ARC_COLORS = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
    "#FFFFFF",
    "#39CCCC",
    "#B10DC9",
    "#01FF70",
    "#85144B",
]


def generate_arc_lpp_decision_trace(
    *,
    env: Any,
    policy: Any,
    output_path: str | Path,
    max_steps: int = 30,
    reset_seed: int = 0,
    feature_json_path: str | Path | None = None,
    stop_after_levels: int | None = None,
) -> Path:
    """Run the learned policy and save an explained ARC rollout as HTML."""
    explain = getattr(policy, "explain_finite_discrete_decision", None)
    if not callable(explain):
        raise TypeError("The learned policy does not support decision explanations.")

    feature_names = _load_feature_names(feature_json_path)
    obs, info = env.reset(seed=reset_seed)
    steps: list[dict[str, Any]] = []
    total_reward = 0.0

    for step_index in range(max_steps):
        explanation = explain(obs)
        action = explanation["chosen_action"]
        before_levels = _levels_completed(obs, info)
        frame_uri = _frame_data_uri(obs)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        total_reward += float(reward)
        steps.append(
            {
                "index": step_index,
                "frame_uri": frame_uri,
                "levels_completed": before_levels,
                "explanation": explanation,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        obs, info = next_obs, next_info
        if (
            terminated
            or truncated
            or (
                stop_after_levels is not None
                and _levels_completed(obs, info) >= stop_after_levels
            )
        ):
            break

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _render_html(
            steps=steps,
            feature_names=feature_names,
            total_reward=total_reward,
            final_levels=_levels_completed(obs, info),
            map_program=str(getattr(policy, "map_program", "")),
            map_posterior=float(getattr(policy, "map_posterior", 0.0)),
        ),
        encoding="utf-8",
    )
    env.close()
    return out_path


def _load_feature_names(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    feature_path = Path(path)
    if not feature_path.exists():
        return {}
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    return {
        str(feature.get("id")): str(feature.get("name", feature.get("id")))
        for feature in payload.get("features", [])
        if isinstance(feature, dict) and feature.get("id")
    }


def _levels_completed(obs: Any, info: Any) -> int:
    if isinstance(obs, dict):
        value = obs.get("levels_completed", 0)
    else:
        value = getattr(obs, "levels_completed", 0)
    if value is None and isinstance(info, dict):
        value = info.get("levels_completed", 0)
    return int(value or 0)


def _extract_grid(obs: Any) -> np.ndarray | None:
    if isinstance(obs, dict) and "grid" in obs:
        try:
            grid = np.asarray(obs["grid"], dtype=int)
        except (TypeError, ValueError):
            return None
        return grid if grid.ndim == 2 else None
    frame = obs.get("frame") if isinstance(obs, dict) else getattr(obs, "frame", None)
    if frame is None:
        return None
    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    while (
        isinstance(frame, (list, tuple))
        and frame
        and isinstance(frame[0], (list, tuple))
        and frame[0]
        and isinstance(frame[0][0], (list, tuple))
    ):
        frame = frame[0]
    try:
        grid = np.asarray(frame, dtype=int)
    except (TypeError, ValueError):
        return None
    if grid.ndim != 2:
        return None
    return grid


def _frame_data_uri(obs: Any) -> str | None:
    grid = _extract_grid(obs)
    if grid is None:
        return None
    buffer = io.BytesIO()
    plt.imsave(
        buffer,
        grid,
        cmap=ListedColormap(_ARC_COLORS),
        vmin=0,
        vmax=len(_ARC_COLORS) - 1,
        format="png",
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _action_label(action: Any) -> str:
    value = int(getattr(action, "value", action))
    return f"{value}: {_ACTION_NAMES.get(value, 'Action')}"


def _feature_labels(
    feature_ids: list[str],
    feature_names: dict[str, str],
) -> str:
    if not feature_ids:
        return '<span class="muted">none</span>'
    labels = [
        f'<span class="feature">{html.escape(feature_names.get(fid, fid))}</span>'
        for fid in feature_ids
    ]
    return " ".join(labels)


def _render_html(
    *,
    steps: list[dict[str, Any]],
    feature_names: dict[str, str],
    total_reward: float,
    final_levels: int,
    map_program: str,
    map_posterior: float,
) -> str:
    step_sections: list[str] = []
    for step in steps:
        explanation = step["explanation"]
        chosen = explanation["chosen_action"]
        rows: list[str] = []
        for action_row in explanation["actions"]:
            action = action_row["action"]
            chosen_class = (
                " chosen"
                if int(getattr(action, "value", action))
                == int(getattr(chosen, "value", chosen))
                else ""
            )
            rows.append(
                "<tr class='" + chosen_class.strip() + "'>"
                f"<td>{html.escape(_action_label(action))}</td>"
                f"<td>{float(action_row['probability']):.3f}</td>"
                f"<td>{'yes' if action_row['map_accepts'] else 'no'}</td>"
                "<td>"
                + _feature_labels(action_row["active_features"], feature_names)
                + "</td></tr>"
            )
        image = (
            f'<img src="{step["frame_uri"]}" alt="ARC frame">'
            if step["frame_uri"]
            else '<div class="no-frame">No frame data</div>'
        )
        status = []
        if step["reward"]:
            status.append(f"reward {step['reward']:.1f}")
        if step["terminated"]:
            status.append("terminated")
        if step["truncated"]:
            status.append("truncated")
        status_text = ", ".join(status) or "continuing"
        step_sections.append(f"""
            <section class="step">
              <header>
                <div>
                  <span class="step-number">Step {step['index']}</span>
                  <strong>Chose {_action_label(chosen)}</strong>
                </div>
                <div class="status">
                  level {step['levels_completed']} · {html.escape(status_text)}
                </div>
              </header>
              <div class="step-body">
                <div class="frame">{image}</div>
                <table>
                  <thead>
                    <tr>
                      <th>Action</th>
                      <th>Mixture probability</th>
                      <th>MAP accepts</th>
                      <th>Active MAP features</th>
                    </tr>
                  </thead>
                  <tbody>{''.join(rows)}</tbody>
                </table>
              </div>
            </section>
            """)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ARC-AGI-3 LPP Decision Trace</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      background: #f4f5f7;
      color: #17191c;
    }}
    body {{ margin: 0; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    .summary {{ margin-bottom: 24px; color: #4d535b; }}
    details {{
      background: #fff;
      border: 1px solid #d9dde3;
      border-radius: 6px;
      padding: 12px 14px;
      margin-bottom: 20px;
    }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }}
    .step {{
      background: #fff;
      border: 1px solid #d9dde3;
      border-radius: 6px;
      margin-bottom: 16px;
      overflow: hidden;
    }}
    .step header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid #e5e8ec;
      background: #fafbfc;
    }}
    .step-number {{ color: #666e78; margin-right: 14px; }}
    .status {{ color: #666e78; }}
    .step-body {{
      display: grid;
      grid-template-columns: minmax(260px, 420px) 1fr;
      gap: 18px;
      padding: 16px;
      align-items: start;
    }}
    .frame img {{
      width: 100%;
      aspect-ratio: 1;
      image-rendering: pixelated;
      border: 1px solid #cfd4da;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{
      text-align: left;
      padding: 9px;
      border-bottom: 1px solid #e5e8ec;
      vertical-align: top;
    }}
    th {{ color: #555d67; font-weight: 600; }}
    tr.chosen {{ background: #e9f6ec; }}
    .feature {{
      display: inline-block;
      margin: 0 4px 4px 0;
      padding: 2px 6px;
      border: 1px solid #cbd2da;
      border-radius: 4px;
      background: #f4f6f8;
    }}
    .muted, .no-frame {{ color: #7a828c; }}
    @media (max-width: 820px) {{
      main {{ padding: 16px; }}
      .step-body {{ grid-template-columns: 1fr; }}
      .step header {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>ARC-AGI-3 LPP Decision Trace</h1>
    <div class="summary">
      {len(steps)} decisions · total reward {total_reward:.1f} ·
      final levels completed {final_levels} · MAP posterior {map_posterior:.4f}
    </div>
    <details>
      <summary>Learned MAP PLP</summary>
      <pre>{html.escape(map_program)}</pre>
    </details>
    {''.join(step_sections)}
  </main>
</body>
</html>
"""
