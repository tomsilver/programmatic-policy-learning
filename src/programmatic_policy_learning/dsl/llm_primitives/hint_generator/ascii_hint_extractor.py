from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
from programmatic_policy_learning.envs.registry import EnvRegistry
from omegaconf import OmegaConf
import numpy as np


@dataclass(frozen=True)
class GridStateEncoderConfig:
    """
    Configuration for encoding a grid observation.

    symbol_map:
        Maps cell values (strings in obs) -> a single ASCII character.
        Example: {"empty": " ", "wall": "#", "agent": "@", "token": "T"}

    empty_token:
        The canonical value used in obs for empty cells. Used to decide
        which cells to skip in extract_objects by default.

    unknown_char:
        Character used when a cell value is not present in symbol_map.

    coordinate_style:
        - "rc": use (row, col)
        - "xy": use (x, y) where x=col, y=row (sometimes nicer for prompts)

    list_all_non_empty:
        If True, extract_objects returns positions for every non-empty token
        encountered in the grid (except empty_token). If False, you should pass
        `salient_tokens` to extract_objects() (recommended).
    """

    symbol_map: dict[str, str]
    empty_token: str = "empty"
    unknown_char: str = "?"
    coordinate_style: str = "rc"  # "rc" or "xy"
    list_all_non_empty: bool = True


class GridStateEncoder:
    """
    Encodes grid observations into:
      (1) ASCII representation
      (2) Coordinate-based object listing

    Assumptions:
      - obs is a 2D grid of tokens (strings), typically np.ndarray dtype=object/str
      - Each cell contains a single token string (e.g., "empty", "agent", "wall")
    """

    def __init__(self, cfg: GridStateEncoderConfig):
        self.cfg = cfg
        if self.cfg.coordinate_style not in ("rc", "xy"):
            raise ValueError(
                f"coordinate_style must be 'rc' or 'xy', got {self.cfg.coordinate_style}"
            )

        # Optional sanity check: encourage 1-char mappings for ASCII
        for k, v in self.cfg.symbol_map.items():
            if not isinstance(v, str) or len(v) != 1:
                raise ValueError(
                    f"symbol_map values must be 1-char strings. For key '{k}', got '{v}'."
                )

    # ----------------------------
    # Public API
    # ----------------------------

    def to_ascii(self, obs: Any) -> str:
        """
        Convert an observation to a multi-line ASCII string.

        Unknown tokens are rendered as cfg.unknown_char.
        """
        grid = self._ensure_2d_array(obs)

        rows: list[str] = []
        H, W = grid.shape
        for r in range(H):
            chars = []
            for c in range(W):
                token = self._cell_to_token(grid[r, c])
                chars.append(self.cfg.symbol_map.get(token, self.cfg.unknown_char))
            rows.append("".join(chars))
        return "\n".join(rows)

    def extract_objects(
        self,
        obs: Any,
        *,
        salient_tokens: Iterable[str] | None = None,
        include_empty: bool = False,
        sort_positions: bool = True,
    ) -> dict[str, list[tuple[int, int]]]:
        """
        Extract object positions from the grid.

        Returns:
            dict[token] -> list[(row, col)] or list[(x, y)] depending on cfg.coordinate_style

        Parameters:
            salient_tokens:
                If provided, only these tokens are returned (even if list_all_non_empty=True).
                Recommended for prompt compactness.

            include_empty:
                If True, includes empty_token in results. Usually False.

            sort_positions:
                If True, positions are sorted for deterministic outputs.
        """
        grid = self._ensure_2d_array(obs)
        H, W = grid.shape

        if salient_tokens is not None:
            salient = set(salient_tokens)
        else:
            salient = None

        out: dict[str, list[tuple[int, int]]] = {}

        for r in range(H):
            for c in range(W):
                token = self._cell_to_token(grid[r, c])

                if not include_empty and token == self.cfg.empty_token:
                    continue

                if salient is not None:
                    if token not in salient:
                        continue
                else:
                    # If no salient_tokens provided, follow config behavior
                    if not self.cfg.list_all_non_empty and token != self.cfg.empty_token:
                        # list_all_non_empty=False means caller should pass salient_tokens
                        # but we still behave reasonably: include all non-empty
                        pass

                coord = self._coord(r, c)
                out.setdefault(token, []).append(coord)

        if sort_positions:
            for token in out:
                out[token].sort()

        return out

    def format_coordinate_listing(
        self,
        objects: dict[str, list[tuple[int, int]]],
        *,
        token_name_map: dict[str, str] | None = None,
        max_lines: int | None = None,
    ) -> str:
        """
        Convert extracted objects into a prompt-friendly coordinate listing.

        Example output lines:
          - agent at (row=3, col=4)
          - wall at (row=0, col=0)
          - token at (row=6, col=1), (row=7, col=1), ...

        token_name_map:
            Optional mapping to rename tokens for readability in prompts
            (e.g., {"left_arrow": "LeftArrow"}).

        max_lines:
            Optionally truncate to the first N lines (useful to control prompt length).
        """
        token_name_map = token_name_map or {}

        lines: list[str] = []
        for token in sorted(objects.keys()):
            nice = token_name_map.get(token, token)
            coords = objects[token]

            if len(coords) == 1:
                lines.append(f"- {nice} at {self._format_coord(coords[0])}")
            else:
                coords_str = ", ".join(self._format_coord(p) for p in coords)
                lines.append(f"- {nice} at {coords_str}")

        if max_lines is not None:
            lines = lines[:max_lines]

        return "\n".join(lines)

    # ----------------------------
    # Helpers
    # ----------------------------

    def _ensure_2d_array(self, obs: Any) -> np.ndarray:
        """
        Convert obs to a 2D numpy array of dtype=object.

        Supports:
          - np.ndarray (2D)
          - list[list[...]] (2D)
        """
        if isinstance(obs, np.ndarray):
            arr = obs
        else:
            arr = np.asarray(obs, dtype=object)

        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D grid observation, got shape {arr.shape}")

        return arr

    def _cell_to_token(self, cell: Any) -> str:
        """
        Convert a cell value to a string token.
        Handles np.str_, Python str, etc.
        """
        if isinstance(cell, str):
            return cell
        # numpy scalar string types, or other objects
        try:
            return str(cell)
        except Exception:
            return self.cfg.unknown_char

    def _coord(self, r: int, c: int) -> tuple[int, int]:
        if self.cfg.coordinate_style == "rc":
            return (r, c)
        # xy: x=col, y=row
        return (c, r)

    def _format_coord(self, coord: tuple[int, int]) -> str:
        if self.cfg.coordinate_style == "rc":
            r, c = coord
            return f"(row={r}, col={c})"
        x, y = coord
        return f"(x={x}, y={y})"

if __name__ == "__main__":
    # symbol_map = {
    # "empty": " ",
    # "token": "T",
    # }
        
    symbol_map = {
        "empty": " ",
        "wall": "#",
        "agent": "A",
        "target": "G",
        "drawn": ".",
        "left_arrow": "<",
        "right_arrow": ">",
        "up_arrow": "^",
        "down_arrow": "v",
    }


    registry = EnvRegistry()
    def env_factory(
        instance_num: int | None = None, _env_name: str = None,
    ) -> Any:
        """Env Factory."""
        return registry.load(
            OmegaConf.create(
                {
                    "provider": "ggg",
                    "make_kwargs": {
                        "base_name": _env_name,
                        "id": f"{_env_name}0-v0",
                    },
                    "instance_num": instance_num,
                }
            )
        )

    # ------------------------------------------------------------------
    # Get example observation + action space
    # ------------------------------------------------------------------
    env0 = env_factory(0, "Chase")
    obs, _ = env0.reset()
    print(obs)
    # action_space = env0.action_space
    # env0.close()
    cfg = GridStateEncoderConfig(
        symbol_map=symbol_map,
        empty_token="empty",
        coordinate_style="rc",
    )
    enc = GridStateEncoder(cfg)

    ascii_grid = enc.to_ascii(obs)
    objs = enc.extract_objects(obs, salient_tokens=["token"])
    listing = enc.format_coordinate_listing(objs)

    print(ascii_grid)
    print(listing)

    