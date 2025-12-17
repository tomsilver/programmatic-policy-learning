# grid_encoder.py

from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, List, Tuple

@dataclass
class GridStateEncoderConfig:
    symbol_map: Dict[str, str]
    empty_token: str = "empty"
    coordinate_style: str = "rc"  # row, col


class GridStateEncoder:
    def __init__(self, cfg: GridStateEncoderConfig):
        self.cfg = cfg

    def to_ascii(self, obs: np.ndarray) -> str:
        rows = []
        for r in range(obs.shape[0]):
            row = []
            for c in range(obs.shape[1]):
                token = obs[r, c]
                row.append(self.cfg.symbol_map.get(token, "?"))
            rows.append("".join(row))
        return "\n".join(rows)

    def extract_objects(
        self,
        obs: np.ndarray,
        salient_tokens: List[str],
    ) -> Dict[str, List[Tuple[int, int]]]:
        objects: Dict[str, List[Tuple[int, int]]] = {t: [] for t in salient_tokens}
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                token = obs[r, c]
                if token in objects:
                    objects[token].append((r, c))
        return objects

    def format_coordinate_listing(
        self,
        objects: Dict[str, List[Tuple[int, int]]],
        max_per_token: int = 8,
    ) -> str:
        lines = []
        for token, coords in objects.items():
            if not coords:
                continue
            shown = coords[:max_per_token]
            suffix = " ..." if len(coords) > max_per_token else ""
            lines.append(f"- {token}: {shown}{suffix}")
        return "\n".join(lines)
