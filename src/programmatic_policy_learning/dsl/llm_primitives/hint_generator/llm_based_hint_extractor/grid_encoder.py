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

    def to_ascii(self, obs: np.ndarray) -> str:
        """Render the grid as ASCII art."""
        rows: list[str] = []
        for r in range(obs.shape[0]):
            row_chars = []
            for c in range(obs.shape[1]):
                token = obs[r, c]
                row_chars.append(self.cfg.symbol_map.get(token, "?"))
            rows.append("".join(row_chars))
        return "\n".join(rows)

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

    def format_coordinate_listing(
        self,
        objects: dict[str, list[tuple[int, int]]],
        max_per_token: int = 8,
    ) -> str:
        """Return a human-readable summary of token coordinates."""
        lines = []
        for token, coords in objects.items():
            if not coords:
                continue
            shown = coords[:max_per_token]
            suffix = " ..." if len(coords) > max_per_token else ""
            lines.append(f"- {token}: {shown}{suffix}")
        return "\n".join(lines)
