"""Helpers for encoding grid observations into text features."""

from dataclasses import dataclass

import numpy as np


@dataclass
class GridStateEncoderConfig:
    """Simple configuration for mapping tokens to ASCII symbols."""

    symbol_map: dict[str, str]
    empty_token: str = "empty"
    coordinate_style: str = "rc"  # row, col


class GridStateEncoder:
    """Encode observations to ASCII and coordinate listings."""

    def __init__(self, cfg: GridStateEncoderConfig):
        """Store the encoder configuration."""
        self.cfg = cfg

    def to_ascii(
        self, obs_t: np.ndarray, action: tuple[int, int], step_index: int
    ) -> str:
        """Render the current/next grid plus a binary action mask."""

        def render(obs: np.ndarray) -> str:
            rows: list[str] = []
            for r in range(obs.shape[0]):
                row_chars = []
                for c in range(obs.shape[1]):
                    token = obs[r, c]
                    char = self.cfg.symbol_map.get(token, "?")
                    row_chars.append(char)
                rows.append(",".join(row_chars))
            return "\n".join(rows)

        def render_action_mask(shape: tuple[int, ...]) -> str:
            mask = np.zeros(shape, dtype=int)
            mask[action] = 1
            rows = [",".join(str(cell) for cell in mask[r]) for r in range(shape[0])]
            return "\n".join(rows)

        token_name = obs_t[action]
        token_symbol = self.cfg.symbol_map.get(token_name, "?")

        ascii_grid = f"*** Step {step_index} ***\nObservation (s_{step_index}):\n"
        ascii_grid += render(obs_t)

        ascii_grid += (
            f"\n\nAction: Click cell {action}, containing "
            f"a '{token_name}' token represented by '{token_symbol}'."
        )

        ascii_grid += "\n\n=== ACTION MASK ===\n"

        ascii_grid += render_action_mask(obs_t.shape)
        return ascii_grid

    def to_ascii_list_literal(
        self,
        obs_t: np.ndarray,
        action: tuple[int, int],
        step_index: int,
    ) -> str:
        """Render the grid using list-of-lists literals, mirroring
        `to_ascii`."""

        def build_literal(obs: np.ndarray, mark_action: bool) -> str:
            """Return a bracketed literal, optionally highlighting the
            action."""
            row_blocks: list[str] = []
            for r in range(obs.shape[0]):
                entries: list[str] = []
                for c in range(obs.shape[1]):
                    token = obs[r, c]
                    if mark_action and (r, c) == action:
                        char = "*"
                        assert char not in self.cfg.symbol_map.values()
                    else:
                        char = self.cfg.symbol_map.get(token, "?")
                    entries.append(f"'{char}'")
                row_blocks.append(f"[{', '.join(entries)}]")
            indented_rows = "\n".join(f"  {row}" for row in row_blocks)
            return f"[\n{indented_rows}\n]"

        def build_mask_literal(shape: tuple[int, ...]) -> str:
            mask = np.zeros(shape, dtype=int)
            mask[action] = 1
            row_blocks = []
            for r in range(shape[0]):
                entries = ", ".join(str(cell) for cell in mask[r])
                row_blocks.append(f"[{entries}]")
            indented_rows = "\n".join(f"  {row}" for row in row_blocks)
            return f"[\n{indented_rows}\n]"

        token_name = obs_t[action]
        token_symbol = self.cfg.symbol_map.get(token_name, "?")

        ascii_grid = f"*** Step {step_index} ***\nObservation (s_{step_index}):\n"
        ascii_grid += build_literal(obs_t, mark_action=False)

        ascii_grid += (
            f"\n\nAction: Click cell {action}, containing "
            f"a '{token_name}' token represented by '{token_symbol}'."
        )

        ascii_grid += "\n\n=== ACTION MASK ===\n"
        ascii_grid += build_mask_literal(obs_t.shape)

        return ascii_grid

    def extract_objects(
        self,
        obs: np.ndarray,
        salient_tokens: list[str],
    ) -> dict[str, list[tuple[int, int]]]:
        """Collect coordinates for the requested tokens."""
        objects: dict[str, list[tuple[int, int]]] = {t: [] for t in salient_tokens}
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                token = obs[r, c]
                if token in objects:
                    objects[token].append((r, c))
        return objects

    def format_cell_value_listing(
        self,
        objects: dict[str, list[tuple[int, int]]],
        step_index: int,
        action: tuple[int, int] | None = None,
    ) -> str:
        """Return per-cell lines like '(r,c) - Token Name', mirroring the ASCII
        layout."""

        def render(obj_dict: dict[str, list[tuple[int, int]]]) -> str:
            entries: list[tuple[tuple[int, int], str]] = []
            for token, coords in obj_dict.items():
                if not coords:
                    continue
                label = token.replace("_", " ").title()
                for coord in coords:
                    entries.append((coord, label))
            entries.sort(key=lambda item: item[0])
            return "\n".join(f"{coord} - {label}" for coord, label in entries)

        sections: list[str] = []
        sections.append(
            # pylint: disable=line-too-long
            f"*** Step {step_index} ***\nObservation (s_{step_index}):\n{render(objects)}"
        )

        if action is not None:
            sections.append(f"Action Taken: {action}")

        return "\n\n".join(sections)
