"""Serialize expert trajectories into textual hint blocks."""

from typing import Sequence

import numpy as np

# pylint: disable=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.grid_encoder import (
    GridStateEncoder,
)
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.transition_analyzer import (
    GenericTransitionAnalyzer,
    extract_relational_facts,
)


def trajectory_to_text(
    trajectory: Sequence[tuple[np.ndarray, tuple[int, int], np.ndarray]],
    *,
    encoder: GridStateEncoder,
    analyzer: GenericTransitionAnalyzer,
    salient_tokens: list[str],
    encoding_method: str,
    max_steps: int | None = None,
) -> str:
    """Convert (obs, action, next_obs) tuples to structured text."""
    blocks: list[str] = []
    steps = trajectory[:max_steps] if max_steps else trajectory

    for i, (obs_t, action, obs_t1) in enumerate(steps):

        if encoding_method == "1":
            ascii_t = encoder.to_ascii_list_literal(obs_t, action, i)
            block = ascii_t
            blocks.append(block)
        else:
            tokens = list(dict.fromkeys([*salient_tokens, encoder.cfg.empty_token]))
            objs = encoder.extract_objects(obs_t, tokens)
            # objs_next = encoder.extract_objects(obs_t1, tokens)
            listing = encoder.format_cell_value_listing(
                objs,
                i,
                action=action,
            )

            if encoding_method == "2":
                block = listing
                blocks.append(block)

            elif encoding_method == "3":
                events = analyzer.analyze(
                    obs_t,
                    action,
                    obs_t1,
                    encoder=encoder,
                )
                change_summary = "\n".join(f"- {event}" for event in events)

                block = listing
                blocks.append(block)
                blocks.append(f"\nCHANGES SUMMARY:\n{change_summary}")
            elif encoding_method == "4":
                events = analyzer.analyze(
                    obs_t,
                    action,
                    obs_t1,
                    encoder=encoder,
                )
                change_summary = "\n".join(f"- {event}" for event in events)

                relational_facts = extract_relational_facts(
                    obs_t,
                    objs,
                    symbol_map=encoder.cfg.symbol_map,
                    reference_tokens=salient_tokens,
                    empty_tokens=(encoder.cfg.empty_token,),
                )

                block = listing
                blocks.append(block)
                blocks.append(f"\nCHANGES SUMMARY:\n{change_summary}\n\n")
                blocks.append(
                    f"Relational Facts:\n{', '.join(str(each) for each in relational_facts)}"
                )
    if steps:
        final_obs = steps[-1][2]
        final_step_idx = len(steps)
        if encoding_method == "1":
            blocks.append(
                encoder.to_ascii_list_literal(
                    final_obs, action=None, step_index=final_step_idx
                )
            )
        else:
            tokens = list(dict.fromkeys([*salient_tokens, encoder.cfg.empty_token]))
            objs = encoder.extract_objects(final_obs, tokens)
            blocks.append(
                encoder.format_cell_value_listing(
                    objs,
                    final_step_idx,
                    action=None,
                )
            )
    return "\n\n".join(blocks)


def trajectory_to_diff_text(
    trajectory: Sequence[tuple[np.ndarray, tuple[int, int], np.ndarray]],
    *,
    encoder: GridStateEncoder,
    max_steps: int | None = None,
) -> str:
    """Convert (obs, action, next_obs) tuples to diff-style text."""
    steps = trajectory[:max_steps] if max_steps else trajectory
    if not steps:
        return ""

    symbol_map = encoder.cfg.symbol_map

    def token_to_symbol(token: str) -> str:
        return symbol_map.get(token, "?")

    def grid_to_literal(obs: np.ndarray) -> str:
        row_blocks: list[str] = []
        for r in range(obs.shape[0]):
            entries: list[str] = []
            for c in range(obs.shape[1]):
                entries.append(f"'{token_to_symbol(obs[r, c])}'")
            row_blocks.append(f"[{', '.join(entries)}];")
        indented_rows = "\n".join(f"  {row}" for row in row_blocks)
        return f"[\n{indented_rows}\n]"

    blocks: list[str] = []

    first_obs = steps[0][0]
    blocks.append(f"Observation (s_0):\n{grid_to_literal(first_obs)}")

    for i, (obs_t, action, obs_t1) in enumerate(steps):
        r, c = action
        token_name = obs_t[r, c]
        token_symbol = token_to_symbol(token_name)
        token_label = str(token_name)

        step_lines: list[str] = []
        step_lines.append(f"### Step {i}")
        step_lines.append(
            f"Action: Click `({r}, {c})`, cell contained `{token_symbol}` ({token_label}).  "
        )

        changes: list[str] = []
        for rr in range(obs_t.shape[0]):
            for cc in range(obs_t.shape[1]):
                if obs_t[rr, cc] != obs_t1[rr, cc]:
                    before = token_to_symbol(obs_t[rr, cc])
                    after = token_to_symbol(obs_t1[rr, cc])
                    changes.append(f"- `({rr}, {cc}): {before} -> {after}`")

        step_lines.append(f"Diff (`s_{i} → s_{i + 1}`):")
        if changes:
            step_lines.extend(changes)
        else:
            step_lines.append("- (no changes)")
        step_lines.append(f"Unchanged: all other cells same as `s_{i}`.")
        blocks.append("\n".join(step_lines))

    final_obs = steps[-1][2]
    final_step_idx = len(steps)
    final_lines = [
        f"### Step {final_step_idx}",
        f"Observation (s_{final_step_idx}):\n{grid_to_literal(final_obs)}",
        "Action: None (terminal state).",
    ]
    blocks.append("\n".join(final_lines))

    return "\n\n".join(blocks)
