"""Utils for Testing LPP Approach."""

from __future__ import annotations

import json
import logging
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from programmatic_policy_learning.approaches.experts.grid_experts import (
    get_grid_expert,
)
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    grid_encoder,
    grid_hint_config,
)
from programmatic_policy_learning.dsl.llm_primitives.hint_generation.llm_based import (
    hint_extractor,
)


def run_single_episode(
    env: Any,
    policy: Callable[[Any], Any],
    record_video: bool = False,
    video_out_path: str | None = None,
    max_num_steps: int = 100,
) -> float:
    """Run a single episode in the environment using the given policy."""

    if record_video:
        env.start_recording_video(video_out_path=video_out_path)

    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_num_steps):
        action = policy(obs)
        new_obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        obs = new_obs

        if done:
            break
    env.close()

    return total_reward


def load_hint_text(
    env_name: str, encoding_method: str, structured_hint: bool, hints_root: str | Path
) -> str:
    """Load the latest hint text for an environment/encoding pair."""
    hint_type = "structured" if structured_hint else "simple"
    hint_dir = Path(hints_root) / env_name / encoding_method / hint_type
    if not hint_dir.exists():
        raise FileNotFoundError(f"Missing hint directory: {hint_dir}")
    hint_files = sorted(hint_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not hint_files:
        raise FileNotFoundError(f"No hint files found in {hint_dir}")
    latest_file = hint_files[-1]

    raw_text = latest_file.read_text(encoding="utf-8").strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text

    if isinstance(data, list):
        return "\n".join(str(x) for x in data)
    if isinstance(data, dict):
        if "hints" in data:
            return "\n".join(str(x) for x in data["hints"])
        if "aggregated_hints" in data:
            return "\n".join(str(x) for x in data["aggregated_hints"])
    return raw_text


def convert_dir_lists_to_tuples(programs: list[str]) -> list[str]:
    """Convert direction lists like [-1, -1] to tuples (-1, -1) inside program
    strings.

    Args:
        programs (list[str]): list of program strings

    Returns:
        list[str]: rewritten program strings
    """
    _DIR_LIST_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")

    def repl(match: re.Match[str]) -> str:
        x, y = match.groups()
        return f"({x}, {y})"

    return [_DIR_LIST_RE.sub(repl, p) for p in programs]


def sample_transition_example(
    env_factory: Callable[[int], Any],
    env_name: str,
    encoding_method: str,
    max_steps: int = 40,
) -> tuple[str, str, str]:
    """Sample a single (s_t, a_t, s_t1) example and format with encoding."""
    expert = get_grid_expert(env_name)
    env = env_factory(0)
    traj = hint_extractor.collect_full_episode(
        env, expert, max_steps=max_steps, sample_count=None
    )
    env.close()
    if not traj:
        raise ValueError("No trajectory data collected.")
    obs_t, action, obs_t1 = traj[0]

    symbol_map = grid_hint_config.get_symbol_map(env_name)
    encoder = grid_encoder.GridStateEncoder(
        grid_encoder.GridStateEncoderConfig(
            symbol_map=symbol_map,
            empty_token="empty",
            coordinate_style="rc",
        )
    )
    salient_tokens = grid_hint_config.SALIENT_TOKENS[env_name]

    def list_literal(obs: Any) -> str:
        rows: list[str] = []
        for r in range(obs.shape[0]):
            entries: list[str] = []
            for c in range(obs.shape[1]):
                token = obs[r, c]
                char = symbol_map.get(token, "?")
                entries.append(f"'{char}'")
            rows.append(f"[{', '.join(entries)}]")
        return "[\n" + "\n".join(f"  {row}" for row in rows) + "\n]"

    def listing(obs: Any) -> str:
        tokens = list(dict.fromkeys([*salient_tokens, encoder.cfg.empty_token]))
        objs = encoder.extract_objects(obs, tokens)
        entries: list[tuple[tuple[int, int], str]] = []
        for token, coords in objs.items():
            if not coords:
                continue
            for coord in coords:
                entries.append((coord, token))
        entries.sort(key=lambda item: item[0])
        return "\n".join(f"{coord} - '{label}'" for coord, label in entries)

    if encoding_method == "1":
        state_t = list_literal(obs_t)
        state_t1 = list_literal(obs_t1)
        state_t1 += f"\nToken meanings: {grid_hint_config.SYMBOL_MAPS[env_name]}"
    else:
        state_t = listing(obs_t)
        state_t1 = listing(obs_t1)

    action_token = obs_t[action]
    action_text = f"{action}: '{action_token}'"
    return state_t, action_text, state_t1


# @dataclass(frozen=True)
# class FeatureKeyBucket:
#     """Rows that share the exact same feature vector (same key)."""

#     key: bytes
#     indices: Tuple[int, ...]

#     @property
#     def size(self) -> int:
#         return len(self.indices)


# @dataclass(frozen=True)
# class CollisionBucket:
#     """A feature-key bucket that contains mixed labels (i.e., a true
#     collision)."""

#     key: bytes
#     indices: Tuple[int, ...]
#     num_pos: int
#     num_neg: int

#     @property
#     def size(self) -> int:
#         return len(self.indices)


# def _row_key_from_sparse(X: Any, i: int) -> bytes:
#     """Stable key for row i (feature vector)."""
#     row = X.getrow(i)
#     dense = row.toarray().ravel().astype(np.uint8)
#     return dense.tobytes()


# # ---------------------------
# # 1) Group ONLY by feature similarity (key)
# # ---------------------------


# def bucket_by_feature_key(X: Any) -> List[FeatureKeyBucket]:
#     """Group rows by identical feature vectors (feature similarity).

#     Returns all buckets (including pure buckets).
#     """
#     if X is None or X.shape[0] == 0:
#         return []

#     buckets: Dict[bytes, List[int]] = {}
#     for i in range(X.shape[0]):
#         key = _row_key_from_sparse(X, i)
#         buckets.setdefault(key, []).append(i)

#     out = [FeatureKeyBucket(key=k, indices=tuple(v)) for k, v in buckets.items()]
#     out.sort(key=lambda b: b.size, reverse=True)

#     return out


# def collision_buckets_from_key_buckets(
#     key_buckets: List[FeatureKeyBucket],
#     y: np.ndarray,
# ) -> List[CollisionBucket]:
#     """From feature-key buckets, keep ONLY those that are true collisions
#     (contain both labels)."""
#     labels = y.astype(int).flatten()

#     collisions: List[CollisionBucket] = []
#     for b in key_buckets:
#         labs = labels[list(b.indices)]
#         num_pos = int(np.sum(labs == 1))
#         num_neg = int(np.sum(labs == 0))
#         if num_pos > 0 and num_neg > 0:
#             collisions.append(
#                 CollisionBucket(
#                     key=b.key,
#                     indices=b.indices,
#                     num_pos=num_pos,
#                     num_neg=num_neg,
#                 )
#             )

#     collisions.sort(key=lambda b: b.size, reverse=True)
#     return collisions


# def select_largest_collision_bucket(
#     X: Any,
#     y: np.ndarray,
# ) -> Optional[CollisionBucket]:
#     """Choose the largest COLLIDING feature-key bucket (mixed labels)."""
#     key_buckets = bucket_by_feature_key(X)
#     print(key_buckets)
#     collisions = collision_buckets_from_key_buckets(key_buckets, y)
#     print(collisions)
#     input("HERE")
#     return collisions[0] if collisions else None


# # ---------------------------
# # 2) Sampling for prompt (still within ONE bucket)
# # ---------------------------


# def sample_examples_from_bucket(
#     bucket: CollisionBucket,
#     y: np.ndarray,
#     max_per_label: int = 5,
#     seed: int = 0,
# ) -> Tuple[List[int], List[int]]:
#     """Sample up to max_per_label positives and negatives from THIS ONE bucket.

#     Returns (pos_indices, neg_indices).
#     """
#     rng = np.random.default_rng(seed)
#     labels = y.astype(int).flatten()

#     pos = [i for i in bucket.indices if int(labels[i]) == 1]
#     neg = [i for i in bucket.indices if int(labels[i]) == 0]

#     # Shuffle deterministically and take first k
#     if len(pos) > max_per_label:
#         pos = [pos[i] for i in rng.permutation(len(pos))[:max_per_label]]
#     if len(neg) > max_per_label:
#         neg = [neg[i] for i in rng.permutation(len(neg))[:max_per_label]]

#     return pos, neg


# # ---------------------------
# # 3) Prompt building
# # ---------------------------


# def _to_py_token(x: Any) -> Any:
#     try:
#         return x.item()
#     except Exception:
#         return x


# def _format_grid_for_llm(grid: np.ndarray) -> str:
#     rows = []
#     for r in grid:
#         rows.append("  " + str([_to_py_token(x) for x in r]))
#     return "[\n" + "\n".join(rows) + "\n]"


# def _format_one_example(
#     s: np.ndarray,
#     a: Tuple[int, int],
#     label: int,
#     idx: int,
# ) -> str:
#     r, c = int(a[0]), int(a[1])
#     clicked = _to_py_token(s[r][c])
#     return textwrap.dedent(f"""
#     Example idx={idx} (label={label}):
#     Observation (s):
#     {_format_grid_for_llm(s)}

#     Action (a): click cell (row={r}, col={c}), clicked_value={clicked}
#     """).strip()


# def build_collision_repair_prompt(
#     examples: List[Tuple[np.ndarray, Tuple[int, int]]],
#     y: np.ndarray,
#     bucket: CollisionBucket,
#     pos_indices: List[int],
#     neg_indices: List[int],
#     env_name: Optional[str] = None,
#     existing_feature_summary: Optional[str] = None,
# ) -> str:
#     if not pos_indices or not neg_indices:
#         raise ValueError(
#             "Need at least 1 positive and 1 negative from the SAME feature-key bucket."
#         )

#     pos_blocks = []
#     for idx in pos_indices:
#         s, a = examples[idx]
#         pos_blocks.append(_format_one_example(s, a, label=1, idx=idx))

#     neg_blocks = []
#     for idx in neg_indices:
#         s, a = examples[idx]
#         neg_blocks.append(_format_one_example(s, a, label=0, idx=idx))

#     env_line = f"ENV: {env_name}\n" if env_name else ""
#     feat_line = ""
#     if existing_feature_summary:
#         feat_line = f"EXISTING FEATURES (summary):\n{existing_feature_summary}\n\n"

#     prompt = f"""
# SYSTEM:
# You are an expert feature-library designer for Logical Programmatic Policies (LPP) in grid-based games.
# Your task is to REPAIR representational failures in an existing feature set.

# {env_line}CONTEXT:
# - Observation s is a 2D grid (list of lists of tokens).
# - Action a is a clicked cell coordinate (row, col).
# - Each feature is a Python function f(s, a) -> bool.
# - Features must generalize across board sizes and positions.
# - Features depend ONLY on (s, a). No history.

# {feat_line}COLLISION EVIDENCE:
# All examples below produce IDENTICAL feature vectors under the current feature set (same feature-key),
# yet the expert labels differ. Therefore, the current features are provably insufficient.

# Collision bucket stats:
# - bucket_size = {bucket.size}
# - num_pos_in_bucket = {bucket.num_pos}
# - num_neg_in_bucket = {bucket.num_neg}

# POSITIVE EXAMPLES (label = 1):
# {chr(10).join(pos_blocks)}

# NEGATIVE EXAMPLES (label = 0):
# {chr(10).join(neg_blocks)}

# TASK:
# Propose new boolean feature functions that distinguish positives from negatives WITHIN THIS BUCKET.

# Each proposed feature must:
# 1) Be True for most positives and False for most negatives (or vice versa) within this bucket
# 2) Capture a meaningful semantic distinction (safety, reachability, blocking, threat, progress, etc.)
# 3) Be board-size invariant (no hard-coded coordinates)
# 4) Use ONLY (s, a)
# 5) Return a boolean

# DO NOT:
# - Hard-code exact positions
# - Memorize these examples
# - Use random logic
# - Output anything other than JSON

# OUTPUT FORMAT (STRICT JSON ONLY):

# {{
#   "features": [
#     {{
#       "id": "f_new_1",
#       "name": "short_descriptive_name",
#       "source": "def f_new_1(s, a):\\n    <python code>\\n"
#     }}
#   ]
# }}
# """.strip()

#     return prompt


# # ---------------------------
# # 4) Convenience: go from (X,y,examples) -> prompt (largest colliding bucket)
# # ---------------------------


# def build_prompt_for_largest_collision_bucket(
#     X: Any,
#     y: np.ndarray,
#     examples: List[Tuple[np.ndarray, Tuple[int, int]]],
#     max_per_label: int = 5,
#     seed: int = 0,
#     env_name: Optional[str] = None,
#     existing_feature_summary: Optional[str] = None,
# ) -> Optional[str]:
#     bucket = select_largest_collision_bucket(X, y)
#     if bucket is None:
#         return None

#     pos_idx, neg_idx = sample_examples_from_bucket(
#         bucket, y, max_per_label=max_per_label, seed=seed
#     )
#     return build_collision_repair_prompt(
#         examples=examples,
#         y=y,
#         bucket=bucket,
#         pos_indices=pos_idx,
#         neg_indices=neg_idx,
#         env_name=env_name,
#         existing_feature_summary=existing_feature_summary,
#     )


# # ---------------------------
# # 5) Optional logging helper
# # ---------------------------


# def log_collision_summary_and_prompt_info(
#     X: Any,
#     y: np.ndarray,
#     max_buckets_to_log: int = 5,
# ) -> None:
#     key_buckets = bucket_by_feature_key(X)
#     collisions = collision_buckets_from_key_buckets(key_buckets, y)

#     if not collisions:
#         logging.info("No feature collisions found.")
#         return

#     logging.info("Collision buckets found: %d", len(collisions))
#     for i, b in enumerate(collisions[:max_buckets_to_log]):
#         logging.info(
#             "CollisionBucket[%d]: size=%d pos=%d neg=%d",
#             i,
#             b.size,
#             b.num_pos,
#             b.num_neg,
#         )
