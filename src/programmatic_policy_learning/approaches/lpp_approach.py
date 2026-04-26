"""An approach that learns a logical programmatic policy from data."""

import json
import logging
import random
from collections import deque
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar, cast

import numpy as np
from gymnasium.spaces import Space
from hydra.core.hydra_config import HydraConfig

# from omegaconf import DictConfig
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from scipy.sparse import vstack
from scipy.special import logsumexp

from programmatic_policy_learning.approaches.base_approach import BaseApproach

# pylint: disable-next=line-too-long
from programmatic_policy_learning.approaches.lpp_utils.lpp_collision_feedback_utils import (
    run_collision_feedback_loop,
)
from programmatic_policy_learning.approaches.lpp_utils.lpp_feature_scoring_utils import (
    feature_score_keep_mask,
    score_cross_demo_features,
    write_feature_scores,
)
from programmatic_policy_learning.approaches.lpp_utils.lpp_feature_source_utils import (
    _extract_feature_names,
)

# pylint: disable-next=line-too-long
from programmatic_policy_learning.approaches.lpp_utils.lpp_program_generation_utils import (
    get_program_set,
    make_llm_client_for_model,
)
from programmatic_policy_learning.approaches.lpp_utils.lpp_program_setup_utils import (
    prepare_programs_and_dsl,
)
from programmatic_policy_learning.approaches.lpp_utils.lpp_split_matrix_utils import (
    filter_redundant_features,
    split_and_collect_demonstrations,
)

# pylint: disable-next=line-too-long
from programmatic_policy_learning.approaches.lpp_utils.lpp_structural_complexity_utils import (
    compute_program_structural_log_prior,
)
from programmatic_policy_learning.approaches.lpp_utils.utils import (
    assert_features_fire,
    build_collision_repair_prompt,
    deduplicate_negative_examples,
    drop_negative_exact_contradictions,
    infer_episode_success,
    is_kinder_env,
    log_exact_example_label_contradictions,
    log_feature_collisions,
    log_plp_violation_counts,
    run_single_episode,
)
from programmatic_policy_learning.data.dataset import (
    extract_examples_from_demonstration,
    extract_examples_from_demonstration_item,
    run_all_programs_on_demonstrations,
)
from programmatic_policy_learning.data.demo_types import Trajectory

# pylint: disable-next=line-too-long
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.continuous_trajectory_serializer import (
    extract_continuous_relational_facts,
)
from programmatic_policy_learning.dsl.llm_primitives.py_feature_generator import (
    PyFeatureGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.utils import (
    JSONStructureRepromptCheck,
)
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram
from programmatic_policy_learning.learning.decision_tree_learner import learn_plps
from programmatic_policy_learning.learning.particles_utils import select_particles
from programmatic_policy_learning.learning.plp_likelihood import compute_likelihood_plps
from programmatic_policy_learning.policies.lpp_policy import LPPPolicy
from programmatic_policy_learning.utils.action_canonicalization import (
    active_action_bounds,
    canonicalize_continuous_action,
    embed_active_action,
    get_active_action_dims,
    get_inactive_action_fill_value,
)
from programmatic_policy_learning.utils.action_quantization import (
    Motion2DActionQuantizer,
)
from programmatic_policy_learning.utils.grid_validation import require_grid_state_action

_filter_redundant_features = filter_redundant_features

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")
EnvFactory = Callable[[int | None], Any]


class LogicProgrammaticPolicyApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that learns a logical programmatic policy from data."""

    def _assert_continuous_demo_alignment(
        self,
        demonstrations: Trajectory[_ObsType, _ActType],
        negative_sampling_cfg: dict[str, Any] | None,
        *,
        max_checks: int = 5,
    ) -> None:
        """Sanity-check that demo centering matches dataset expansion."""
        if str(self.env_specs.get("action_mode", "discrete")) != "continuous":
            return
        for obs, action in demonstrations.steps[:max_checks]:
            pos_examples, _, _ = extract_examples_from_demonstration_item(
                (obs, action),
                negative_sampling=negative_sampling_cfg,
                action_mode="continuous",
                compute_sample_weights=False,
            )
            centered = np.asarray(
                self._center_continuous_action_for_scoring(action), dtype=float
            )
            expected = np.asarray(pos_examples[0][1], dtype=float)
            if not np.allclose(centered, expected, atol=1e-8):
                raise AssertionError(
                    "Continuous demo alignment mismatch between "
                    "_center_continuous_action_for_scoring() and "
                    "extract_examples_from_demonstration_item()."
                )

    def _log_sample_weight_stats(
        self,
        sample_weights: np.ndarray | None,
        *,
        label: str,
        preview_limit: int = 25,
        histogram_bins: int = 10,
    ) -> None:
        """Log sample-weight shape, preview values, and histogram."""
        if sample_weights is None:
            logging.info("%s sample weights: None", label)
            return

        weights = np.asarray(sample_weights, dtype=float).reshape(-1)
        logging.info(
            "%s sample weights: shape=%s dtype=%s min=%.6f max=%.6f "
            "mean=%.6f std=%.6f",
            label,
            tuple(sample_weights.shape),
            sample_weights.dtype,
            float(weights.min()),
            float(weights.max()),
            float(weights.mean()),
            float(weights.std()),
        )
        logging.info(
            "%s sample weights preview (first %d/%d): %s",
            label,
            min(preview_limit, weights.size),
            weights.size,
            weights[:preview_limit].tolist(),
        )

        unique_vals, unique_counts = np.unique(weights, return_counts=True)
        if unique_vals.size <= histogram_bins:
            exact_histogram = [
                (float(value), int(count))
                for value, count in zip(unique_vals, unique_counts)
            ]
            logging.info(
                "%s sample weights exact histogram: %s",
                label,
                exact_histogram,
            )
            return

        bin_edges = np.histogram_bin_edges(weights, bins=histogram_bins)
        counts, edges = np.histogram(weights, bins=bin_edges)
        range_histogram = [
            {
                "left": float(edges[idx]),
                "right": float(edges[idx + 1]),
                "count": int(counts[idx]),
            }
            for idx in range(len(counts))
        ]
        logging.info("%s sample weights histogram: %s", label, range_histogram)

    def _log_motion2d_demonstration_snapshots(
        self,
        demo_dict: Mapping[int, Trajectory[_ObsType, _ActType]],
        *,
        split_name: str,
    ) -> None:
        """Log per-state Motion2D demonstration snapshots for debugging."""
        if "motion2d" not in str(self.base_class_name).lower():
            return

        logging.info("Motion2D %s demonstration snapshots:", split_name)
        num_passages = int(self.env_specs.get("num_passages", 0))
        for demo_id, trajectory in sorted(demo_dict.items(), key=lambda item: item[0]):
            logging.info("Demo %s has %d states", demo_id, len(trajectory.steps))
            for state_idx, (obs, action) in enumerate(trajectory.steps):
                robot_pos = self._motion2d_robot_position(obs)
                if robot_pos is not None:
                    logging.info(
                        "Demo %s state %d robot_position=(%.3f, %.3f) action=%s",
                        demo_id,
                        state_idx,
                        robot_pos[0],
                        robot_pos[1],
                        action,
                    )
                else:
                    logging.info(
                        "Demo %s state %d robot_position=unknown action=%s",
                        demo_id,
                        state_idx,
                        action,
                    )
                try:
                    facts = extract_continuous_relational_facts(
                        np.asarray(obs, dtype=float).reshape(-1),
                        num_passages=num_passages,
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logging.info(
                        "Demo %s state %d snapshot unavailable: %s",
                        demo_id,
                        state_idx,
                        exc,
                    )
                    continue
                logging.info(
                    "Demo %s state %d snapshot: %s",
                    demo_id,
                    state_idx,
                    " | ".join(facts),
                )

    def _log_initial_action_space_ranges(self) -> None:
        """Print/log continuous action ranges for debugging."""
        action_space = self._action_space
        if not (hasattr(action_space, "low") and hasattr(action_space, "high")):
            return

        low = np.asarray(action_space.low, dtype=float).reshape(-1)
        high = np.asarray(action_space.high, dtype=float).reshape(-1)
        if low.size == 0 or high.size == 0:
            return

        msg = f"Action space bounds: low={low.tolist()} high={high.tolist()}"
        logging.info(msg)
        action_types = self.env_specs.get("action_types")
        if action_types:
            logging.info("Action space types: %s", list(action_types))

        if "motion2d" not in str(self.base_class_name).lower():
            return

        sampling_cfg = None
        if isinstance(self.program_generation, dict):
            sampling_cfg = self.program_generation.get("negative_sampling")
        active_dims = get_active_action_dims(
            sampling_cfg,
            total_dims=low.size,
            default_active_dims=(0, 1),
        )
        active_low, active_high = active_action_bounds(
            low,
            high,
            active_dims=active_dims,
        )
        active_names = ["dx", "dy"]
        for idx, dim in enumerate(active_dims[:2]):
            name = active_names[idx] if idx < len(active_names) else f"a[{int(dim)}]"
            dim_msg = (
                f"{name} valid range: [{float(active_low[idx]):.6f}, "
                f"{float(active_high[idx]):.6f}] (dim {int(dim)})"
            )
            logging.info(dim_msg)

    def __init__(
        self,
        environment_description: str,  # env_id
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
        expert: BaseApproach,
        env_factory: Any,
        base_class_name: str,
        demo_numbers: tuple[int, ...] = (1, 2),
        program_generation_step_size: int = 10,
        num_programs: int = 100,
        num_dts: int = 5,
        max_num_particles: int = 10,
        max_demo_length: int | float = np.inf,
        env_specs: dict[str, Any] | None = None,
        start_symbol: int = 0,
        program_generation: dict[str, Any] | None = None,
        normalize_plp_actions: bool = False,
        permissive_filter_enabled: bool = False,
        permissive_filter_max_avg_frac: float | None = None,
        permissive_filter_max_avg_count: int | None = None,
        dt_splitter: str = "random",
        cc_alpha: float = 0.0,
        collision_feedback_enc: str = "enc_1",
        collision_template_feedback: bool = True,
        collision_feedback_num_buckets: int = 2,
        collision_feedback_bucket_mode: str = "positive_anchor",
        collision_feedback_enabled: bool = False,
        collision_feedback_max_new_features: int = 10,
        collision_feedback_max_rounds: int = 3,
        collision_feedback_target_collisions: int = 0,
        prior_version: str = "v1",
        prior_beta: float = 1.0,
        alpha: float = 0.0,
        w_clauses: float = 0.1,
        w_literals: float = 0.05,
        w_max_clause: float = 0.1,
        w_depth: float = 0.05,
        w_ops: float = 0.03,
        val_frac: float | None = 0.2,
        val_size: int | None = None,
        split_seed: int = 0,
        split_strategy: str = "random",
        preserve_ordering: bool = False,
        dt_max_depth: int | None = 5,
        hyperparam_grid: Mapping[str, Sequence[Any] | Any] | None = None,
        recovery_augmentation: Mapping[str, Any] | None = None,
        cross_demo_feature_filter: Mapping[str, Any] | None = None,
    ) -> None:
        """LPP APProach."""
        super().__init__(environment_description, observation_space, action_space, seed)
        self.seed_num = seed
        self.configure_rng()
        self._policy: LPPPolicy | None = None
        self.env_factory = env_factory
        self.base_class_name = base_class_name
        self.expert = expert
        self.demo_numbers = demo_numbers
        self.program_generation_step_size = program_generation_step_size
        self.num_programs = num_programs
        self.num_dts = num_dts
        self.max_num_particles = max_num_particles
        self.max_demo_length = max_demo_length
        self.env_specs = env_specs if env_specs is not None else {}
        self.start_symbol = start_symbol
        self.program_generation = program_generation
        self.normalize_plp_actions = normalize_plp_actions
        self.permissive_filter_enabled = permissive_filter_enabled
        self.permissive_filter_max_avg_frac = permissive_filter_max_avg_frac
        self.permissive_filter_max_avg_count = permissive_filter_max_avg_count
        self.dt_splitter = dt_splitter
        self.cc_alpha = cc_alpha
        self.collision_feedback_enc = collision_feedback_enc
        self.collision_template_feedback = collision_template_feedback
        self.collision_feedback_num_buckets = max(
            1, min(2, int(collision_feedback_num_buckets))
        )
        if collision_feedback_bucket_mode not in {
            "positive_anchor",
            "global_mixed_label",
        }:
            raise ValueError(
                "collision_feedback_bucket_mode must be one of "
                "{'positive_anchor', 'global_mixed_label'}."
            )
        self.collision_feedback_bucket_mode = str(collision_feedback_bucket_mode)
        self.collision_feedback_enabled = collision_feedback_enabled
        self.collision_feedback_max_new_features = collision_feedback_max_new_features
        self.collision_feedback_max_rounds = collision_feedback_max_rounds
        self.collision_feedback_target_collisions = collision_feedback_target_collisions
        self.collision_feedback_reprompt_max_attempts = 5
        self._collision_llm_model: str | None = None
        self._collision_py_generator: PyFeatureGenerator | None = None
        if prior_version not in {"v1", "v2", "uniform"}:
            raise ValueError("prior_version must be one of {'v1', 'v2', 'uniform'}.")
        self.prior_version = prior_version
        self.prior_beta = float(prior_beta)
        self.alpha = float(alpha)
        self.w_clauses = float(w_clauses)
        self.w_literals = float(w_literals)
        self.w_max_clause = float(w_max_clause)
        self.w_depth = float(w_depth)
        self.w_ops = float(w_ops)
        self.val_frac = val_frac
        self.val_size = val_size
        self.split_seed = split_seed
        self.split_strategy = split_strategy
        self.preserve_ordering = preserve_ordering
        self.dt_max_depth = dt_max_depth
        self.hyperparam_grid = (
            dict(hyperparam_grid) if isinstance(hyperparam_grid, Mapping) else None
        )
        self.recovery_augmentation = (
            dict(recovery_augmentation)
            if isinstance(recovery_augmentation, Mapping)
            else {"enabled": False}
        )
        self.cross_demo_feature_filter = (
            dict(cross_demo_feature_filter)
            if isinstance(cross_demo_feature_filter, Mapping)
            else {"enabled": True, "apply_filter": False}
        )
        self._negative_sampling_cfg: dict[str, Any] | None = None
        self._log_initial_action_space_ranges()

    def _default_train_hyperparams(self) -> dict[str, Any]:
        return {
            "num_dts": int(self.num_dts),
            "program_generation_step_size": int(self.program_generation_step_size),
            "dt_splitter": self.dt_splitter,
            "cc_alpha": float(self.cc_alpha),
            "dt_max_depth": self.dt_max_depth,
            "max_num_particles": int(self.max_num_particles),
            "prior_version": self.prior_version,
            "alpha": float(self.alpha),
            "w_clauses": float(self.w_clauses),
            "w_literals": float(self.w_literals),
            "w_max_clause": float(self.w_max_clause),
            "w_depth": float(self.w_depth),
            "w_ops": float(self.w_ops),
        }

    @staticmethod
    def _realign_row_demo_ids(
        old_examples: list[tuple[_ObsType, _ActType]] | None,
        old_row_demo_ids: np.ndarray,
        new_examples: list[tuple[_ObsType, _ActType]] | None,
    ) -> np.ndarray:
        if old_examples is None or new_examples is None:
            return old_row_demo_ids
        row_id_by_example_identity = {
            id(example): old_row_demo_ids[idx]
            for idx, example in enumerate(old_examples)
        }
        return np.asarray(
            [row_id_by_example_identity[id(example)] for example in new_examples],
            dtype=int,
        )

    def _prompt_demo_ids_tag(self) -> Sequence[int] | None:
        """Return configured prompt demo ids for cache tagging."""
        program_generation = dict(self.program_generation or {})
        ensemble_cfg = dict(program_generation.get("multi_prompt_ensemble") or {})
        demo_subsets = ensemble_cfg.get("demo_subsets")
        if not isinstance(demo_subsets, Sequence) or isinstance(
            demo_subsets, (str, bytes)
        ):
            return None

        normalized_subsets = [
            list(dict.fromkeys(int(demo_id) for demo_id in demo_subset))
            for demo_subset in demo_subsets
            if isinstance(demo_subset, Sequence)
            and not isinstance(demo_subset, (str, bytes))
            and len(demo_subset) > 0
        ]
        if not normalized_subsets:
            return None
        if len(normalized_subsets) == 1:
            return normalized_subsets[0]

        union: list[int] = []
        seen: set[int] = set()
        for demo_subset in normalized_subsets:
            for demo_id in demo_subset:
                if demo_id not in seen:
                    seen.add(demo_id)
                    union.append(demo_id)
        return union

    def _feature_scores_output_path(self) -> Path:
        try:
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        except Exception:  # pylint: disable=broad-exception-caught
            output_dir = Path.cwd()
        return output_dir / "feature_scores.json"

    def _score_and_optionally_filter_features(
        self,
        X: Any,
        y: np.ndarray,
        row_demo_ids: np.ndarray,
        programs_sa: list[StateActionProgram],
        program_prior_log_probs: list[float] | None,
    ) -> tuple[Any, list[StateActionProgram], list[float] | None]:
        cfg = dict(self.cross_demo_feature_filter or {})
        if not bool(cfg.get("enabled", True)):
            logging.info("Cross-demo feature scoring disabled.")
            return X, programs_sa, program_prior_log_probs

        consistency_tau = float(cfg.get("consistency_tau", 0.05))
        scores = score_cross_demo_features(
            X,
            y,
            row_demo_ids,
            programs_sa,
            consistency_tau=consistency_tau,
        )
        ranked_scores = sorted(
            scores,
            key=lambda score: float(score["cross_demo_score"]),
            reverse=True,
        )
        logging.info(
            "Cross-demo feature scores computed for %d features.",
            len(scores),
        )
        for rank, score in enumerate(ranked_scores[:10], start=1):
            logging.info(
                "Feature score #%d score=%.4f pos_demo=%d "
                "neg_demo=%d mean_abs_demo_contrast=%.4f "
                "consistency=%.2f program=%s",
                rank,
                float(score["cross_demo_score"]),
                int(score["pos_demo_support_count"]),
                int(score["neg_demo_support_count"]),
                float(score["mean_demo_abs_contrast"]),
                float(score["contrast_consistency_frac"]),
                score["program"],
            )

        write_feature_scores(
            self._feature_scores_output_path(),
            scores,
            metadata={
                "kind": "cross_demo_feature_scores",
                "base_class_name": self.base_class_name,
                "seed": self.seed_num,
                "consistency_tau": consistency_tau,
                "num_features": len(scores),
                "num_rows": int(X.shape[0]),
                "demo_ids": sorted(int(x) for x in np.unique(row_demo_ids)),
            },
        )

        apply_filter = bool(cfg.get("apply_filter", cfg.get("filter_enabled", False)))
        if not apply_filter:
            return X, programs_sa, program_prior_log_probs

        num_demos = max(1, len(np.unique(row_demo_ids)))
        min_pos_demo_support = min(
            num_demos,
            int(cfg.get("min_pos_demo_support", 2 if num_demos > 1 else 1)),
        )
        keep_mask = feature_score_keep_mask(
            scores,
            min_pos_demo_support=min_pos_demo_support,
            min_neg_demo_support=int(
                cfg.get("min_neg_demo_support", min_pos_demo_support)
            ),
            allow_negative_support=bool(cfg.get("allow_negative_support", True)),
            min_abs_mean_demo_contrast=float(
                cfg.get("min_abs_mean_demo_contrast", 0.02)
            ),
            min_consistency_frac=float(cfg.get("min_consistency_frac", 0.4)),
            min_total_fire_rate=float(cfg.get("min_total_fire_rate", 0.005)),
            max_total_fire_rate=float(cfg.get("max_total_fire_rate", 0.95)),
        )
        if keep_mask.all():
            logging.info("Cross-demo feature filter kept all %d features.", len(scores))
            return X, programs_sa, program_prior_log_probs
        if not keep_mask.any():
            logging.warning(
                "Cross-demo feature filter would remove all features; "
                "keeping original set."
            )
            return X, programs_sa, program_prior_log_probs

        removed = int((~keep_mask).sum())
        logging.info(
            "Cross-demo feature filter removed %d/%d features.",
            removed,
            len(scores),
        )
        for score in [s for s in scores if not keep_mask[int(s["feature_index"])]][:10]:
            logging.info(
                "Cross-demo drop score=%.4f pos_demo=%d "
                "neg_demo=%d mean_abs_demo_contrast=%.4f "
                "consistency=%.2f program=%s",
                float(score["cross_demo_score"]),
                int(score["pos_demo_support_count"]),
                int(score["neg_demo_support_count"]),
                float(score["mean_demo_abs_contrast"]),
                float(score["contrast_consistency_frac"]),
                score["program"],
            )

        X = X[:, keep_mask]
        programs_sa = [p for i, p in enumerate(programs_sa) if keep_mask[i]]
        if program_prior_log_probs is not None:
            program_prior_log_probs = [
                lp for i, lp in enumerate(program_prior_log_probs) if keep_mask[i]
            ]
        return X, programs_sa, program_prior_log_probs

    @staticmethod
    def _normalize_grid_values(raw_values: Sequence[Any] | Any) -> list[Any]:
        if isinstance(raw_values, Sequence) and not isinstance(
            raw_values, (str, bytes)
        ):
            values = list(raw_values)
        else:
            values = [raw_values]
        if len(values) == 0:
            raise ValueError("Hyperparameter candidate list cannot be empty.")
        return values

    def _build_hyperparam_candidates(self) -> list[dict[str, Any]]:
        if self.hyperparam_grid is None:
            return [{}]

        supported = {
            "dt_max_depth",
            "num_dts",
            "program_generation_step_size",
            "dt_splitter",
            "cc_alpha",
            "max_num_particles",
            "prior_version",
            "alpha",
            "w_clauses",
            "w_literals",
            "w_max_clause",
            "w_depth",
            "w_ops",
            "normalize_plp_actions",
        }
        unknown = set(self.hyperparam_grid.keys()) - supported
        if unknown:
            unknown_str = ", ".join(sorted(unknown))
            raise ValueError(
                f"Unsupported hyperparameter(s) in hyperparam_grid: {unknown_str}"
            )

        keys = list(self.hyperparam_grid.keys())
        value_lists = [
            self._normalize_grid_values(self.hyperparam_grid[key]) for key in keys
        ]
        candidates: list[dict[str, Any]] = []
        for values in product(*value_lists):
            candidates.append(dict(zip(keys, values)))
        if len(candidates) == 0:
            return [{}]
        return candidates

    def _resolve_train_hyperparams(
        self, train_hyperparams: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        resolved = self._default_train_hyperparams()
        if train_hyperparams is None:
            return resolved
        for key, value in train_hyperparams.items():
            resolved[key] = value
        return resolved

    def configure_rng(self) -> None:
        """Seed Python/NumPy RNGs for deterministic rollouts."""
        random.seed(self.seed_num)
        np.random.seed(self.seed_num)

    def _get_collision_py_generator(self) -> PyFeatureGenerator:
        """Create and cache the PyFeatureGenerator used by collision
        feedback."""
        llm_model = (self.program_generation or {}).get("llm_model", "gpt-4.1")
        if (
            self._collision_py_generator is None
            or self._collision_llm_model != llm_model
        ):
            model_slug = "".join(ch if ch.isalnum() else "_" for ch in llm_model).strip(
                "_"
            )
            cache_path = Path(f"py_feature_cache_{model_slug}.db")
            cache = SQLite3PretrainedLargeModelCache(cache_path)
            llm_client = make_llm_client_for_model(
                llm_model,
                cache,
                use_response_model=(self.program_generation or {}).get(
                    "use_response_model"
                ),
            )
            self._collision_py_generator = PyFeatureGenerator(llm_client)
            self._collision_llm_model = llm_model
        return self._collision_py_generator

    def _generate_collision_features(
        self,
        prompt: str,
        *,
        start_index: int,
        collision_idx: int,
    ) -> tuple[list[str], dict[str, Any], Path]:
        if self.program_generation is None:
            logging.info("Collision feedback skipped: program_generation missing.")
            return [], {}, Path()
        if self.program_generation.get("strategy") != "py_feature_gen":
            logging.info(
                "Collision feedback skipped: strategy %s does not support LLM features.",
                self.program_generation.get("strategy"),
            )
            return [], {}, Path()

        py_generator = self._get_collision_py_generator()

        template_payload = py_generator.query_llm(
            prompt,
            max_attempts=self.collision_feedback_reprompt_max_attempts,
            reprompt_checks=[JSONStructureRepromptCheck(required_fields=["features"])],
            seed=self.seed_num,
        )
        logging.info(template_payload)
        is_kinder = is_kinder_env(self.base_class_name)
        if not is_kinder:
            expanded_payload = py_generator.expand_template_payload(
                template_payload, env_name=self.base_class_name, start_index=start_index
            )
        else:
            expanded_payload = py_generator.renumber_payload_features(
                template_payload, start_index=start_index
            )
        py_generator.write_json(
            f"py_feature_payload_collision_{collision_idx}.json", expanded_payload
        )
        feature_programs = py_generator.parse_feature_programs(expanded_payload)
        if self.collision_feedback_max_new_features > 0:
            feature_programs = feature_programs[
                : self.collision_feedback_max_new_features
            ]
        logging.info(
            "Collision feedback generated %d new feature(s).",
            len(feature_programs),
        )
        return feature_programs, expanded_payload, py_generator.output_path

    def _handle_collision_feedback(
        self,
        collision_groups: list[dict[str, Any]],
        examples: list[tuple[_ObsType, _ActType]] | None,
        *,
        existing_feature_summary: str | None = None,
        failed_attempt_summaries: str | None = None,
    ) -> str | None:
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        is_kinder = (
            "motion2d" in str(self.base_class_name).lower()
            or "kinder" in str(self.base_class_name).lower()
        )

        if action_mode != "discrete" and not is_kinder:
            logging.info(
                "Collision repair prompt generation unsupported "
                "for action_mode=%s env=%s.",
                action_mode,
                self.base_class_name,
            )
            return None

        if not collision_groups or examples is None:
            return None
        ranked_groups = sorted(
            collision_groups,
            key=lambda g: (
                int(g.get("total_count", g["max_occur"])),
                float(g.get("label_balance", 0.0)),
                int(g["max_occur"]),
            ),
            reverse=True,
        )
        logging.info(
            "Collision groups (top 5 by size/balance, mode=%s):",
            self.collision_feedback_bucket_mode,
        )
        for rank, group in enumerate(ranked_groups[:5], start=1):
            pos_list = group["pos"]
            neg_list = group["neg"]
            logging.info(
                "  %d) total=%d balance=%.3f max_occur=%d "
                "pos_count=%d neg_count=%d pos=%s neg=%s",
                rank,
                int(group.get("total_count", len(pos_list) + len(neg_list))),
                float(group.get("label_balance", 0.0)),
                group["max_occur"],
                len(pos_list),
                len(neg_list),
                pos_list[:5],
                neg_list[:10],
            )
        best_group = ranked_groups[0]
        second_group = (
            ranked_groups[1]
            if self.collision_feedback_num_buckets >= 2 and len(ranked_groups) > 1
            else None
        )

        prompt = build_collision_repair_prompt(
            pos_indices=best_group["pos"],
            neg_indices=best_group["neg"],
            # examples=cast(list[tuple[np.ndarray, tuple[int, int]]], examples),
            examples=examples,
            env_name=self.base_class_name,
            existing_feature_summary=existing_feature_summary,
            max_per_label=5,
            collision_feedback_enc=self.collision_feedback_enc,
            bucket_mode=self.collision_feedback_bucket_mode,
            pos_indices_2=second_group["pos"] if second_group else None,
            neg_indices_2=second_group["neg"] if second_group else None,
            seed=self.seed_num,
            collision_template_feedback=self.collision_template_feedback,
            failed_attempt_summaries=failed_attempt_summaries,
        )
        try:
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        except Exception:  # pylint: disable=broad-exception-caught
            output_dir = Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = "collision_prompt_"
        existing = sorted(output_dir.glob(f"{prefix}*.txt"))
        next_idx = len(existing) + 1
        out_path = output_dir / f"{prefix}{next_idx}.txt"
        out_path.write_text(prompt, encoding="utf-8")
        mode = "template" if self.collision_template_feedback else "feature_only"
        logging.info(
            "Collision prompt stats: mode=%s enc=%s buckets=%d chars=%d",
            mode,
            self.collision_feedback_enc,
            (2 if second_group is not None else 1),
            len(prompt),
        )
        logging.info("Collision repair prompt written to %s", out_path)
        return prompt

    def _train_policy_from_matrix(
        self,
        X: Any,
        y_bool: list[bool],
        sample_weight: np.ndarray | None,
        programs_sa: list[StateActionProgram],
        program_prior_log_probs_opt: list[float] | None,
        demonstrations: Trajectory[_ObsType, _ActType],
        dsl_functions: dict[str, Any],
        *,
        negative_sampling_cfg: dict[str, Any] | None = None,
        train_hyperparams: Mapping[str, Any] | None = None,
        tie_break_demonstrations: Trajectory[_ObsType, _ActType] | None = None,
    ) -> LPPPolicy:
        hp = self._resolve_train_hyperparams(train_hyperparams)
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        candidate_actions = (
            self._build_continuous_candidate_actions(negative_sampling_cfg)
            if action_mode == "continuous"
            else None
        )
        prior_version = str(hp["prior_version"])
        alpha = float(hp["alpha"])
        w_clauses = float(hp["w_clauses"])
        w_literals = float(hp["w_literals"])
        w_max_clause = float(hp["w_max_clause"])
        w_depth = float(hp["w_depth"])
        w_ops = float(hp["w_ops"])
        # gain = gini_gain_per_feature(X, y_bool)
        # ranked_cols = np.argsort(-gain)
        # X_ranked = X[:, ranked_cols]
        # programs_ranked = [programs_sa[i] for i in ranked_cols]
        X_ranked = X
        programs_ranked = programs_sa
        ranked_cols = list(range(len(programs_sa)))
        if program_prior_log_probs_opt is None:
            priors_ranked = [0.0 for _ in ranked_cols]
        else:
            priors_ranked = [program_prior_log_probs_opt[i] for i in ranked_cols]

        plps, plp_priors = learn_plps(
            X_ranked,
            y_bool,
            programs_ranked,
            priors_ranked,
            sample_weight=sample_weight,
            num_dts=int(hp["num_dts"]),
            program_generation_step_size=int(hp["program_generation_step_size"]),
            dt_splitter=str(hp["dt_splitter"]),
            cc_alpha=float(hp["cc_alpha"]),
            dt_max_depth=cast(int | None, hp["dt_max_depth"]),
            dsl_functions=dsl_functions,
        )
        if prior_version == "uniform":
            plp_priors = [-4.0] * len(plps)
        if alpha != 0.0:
            structural_log_priors = [
                compute_program_structural_log_prior(
                    plp,
                    alpha=alpha,
                    w_clauses=w_clauses,
                    w_literals=w_literals,
                    w_max_clause=w_max_clause,
                    w_depth=w_depth,
                    w_ops=w_ops,
                )
                for plp in plps
            ]
            plp_priors = [
                base_prior + struct_prior
                for base_prior, struct_prior in zip(plp_priors, structural_log_priors)
            ]
            logging.info(
                "Applied structural prior to %d PLPs (alpha=%.4f).",
                len(plps),
                alpha,
            )

        logging.info("LEN BEFORE FILTERING FALSE=%d", len(plps))
        filtered: list[tuple[StateActionProgram, float]] = []
        num_false = 0
        for plp, prior in zip(plps, plp_priors):
            if str(plp).strip() == "False":
                num_false += 1
                continue
            filtered.append((plp, prior))
        if filtered:
            filtered_plps_tuple, filtered_priors_tuple = zip(*filtered)
            plps = list(filtered_plps_tuple)
            plp_priors = list(filtered_priors_tuple)
        else:
            plps, plp_priors = [], []

        logging.info(
            "LEN AFTER FILTERING FALSE=%d (removed %d false PLPs)",
            len(plps),
            num_false,
        )

        aligned_demonstrations = self._align_demonstrations_for_continuous_scoring(
            demonstrations
        )
        # plps, plp_priors = self._filter_overly_permissive_plps(
        #     plps,
        #     plp_priors,
        #     aligned_demonstrations,
        #     candidate_actions=candidate_actions,
        # )

        log_plp_violation_counts(
            plps,
            aligned_demonstrations,
            dsl_functions,
            candidate_actions=candidate_actions,
            detailed_debug=False,
        )
        # logging.info("LEN AFTER filtering violations=%d", len(plps))
        likelihoods = compute_likelihood_plps(
            plps,
            aligned_demonstrations,
            dsl_functions,
            candidate_actions=candidate_actions,
        )
        # logging.info("LIKELIHOODS: %s", likelihoods)
        # logging.info("PRIORS: %s", plp_priors)
        particles = []
        particle_log_probs = []
        for plp, prior, likelihood in zip(plps, plp_priors, likelihoods):
            particles.append(plp)
            particle_log_probs.append(prior + likelihood)
            # print(f"Posterior: {prior + likelihood:.4f}")

        top_particles, top_particle_log_probs = select_particles(
            particles, particle_log_probs, int(hp["max_num_particles"])
        )
        # print([str(plp) + "**\n" for plp in top_particles])
        # input(top_particle_log_probs)
        if len(top_particle_log_probs) > 0:
            probs_arr = np.asarray(particle_log_probs)
            max_val = probs_arr.max()
            max_indices = np.flatnonzero(probs_arr == max_val)
            if len(max_indices) == 1:
                map_idx = int(max_indices[0])
            elif (
                tie_break_demonstrations is not None
                and len(tie_break_demonstrations.steps) > 0
            ):
                tied_idx = [int(i) for i in max_indices.tolist()]
                best_idx = tied_idx[0]
                best_risk = float("inf")
                for idx in tied_idx:
                    tie_policy: LPPPolicy = LPPPolicy(
                        [particles[idx]],
                        [1.0],
                        map_choices=True,
                        normalize_plp_actions=self.normalize_plp_actions,
                        action_mode=str(self.env_specs.get("action_mode", "discrete")),
                        action_space=cast(Any, self._action_space),
                        candidate_actions=candidate_actions,
                    )
                    risk = self._compute_policy_risk_on_demos(
                        tie_policy, tie_break_demonstrations
                    )
                    if risk < best_risk:
                        best_risk = risk
                        best_idx = idx
                    elif risk == best_risk and str(particles[idx]) < str(
                        particles[best_idx]
                    ):
                        best_idx = idx
                map_idx = best_idx
                logging.info(
                    "Resolved MAP posterior tie using tie-break risk on %d demos "
                    "(best risk=%.6f).",
                    len(tie_break_demonstrations.steps),
                    best_risk,
                )
            else:
                # Deterministic fallback for reproducibility when ties remain.
                map_idx = min(
                    (int(i) for i in max_indices.tolist()),
                    key=lambda i: str(particles[i]),
                )
            logging.info("MAP program index=%d", map_idx)
            logging.info("MAP program (%s):", particle_log_probs[map_idx])
            logging.info(particles[map_idx])
            logging.info("Detailed violation debugger for final MAP policy:")
            log_plp_violation_counts(
                [particles[map_idx]],
                aligned_demonstrations,
                dsl_functions,
                candidate_actions=candidate_actions,
                max_logged_plps=1,
                detailed_debug=True,
            )

            if not any(str(p) == str(particles[map_idx]) for p in top_particles):
                map_particle = particles[map_idx]
                map_log_prob = particle_log_probs[map_idx]
                max_particles = int(hp["max_num_particles"])
                if len(top_particles) < max_particles:
                    top_particles.append(map_particle)
                    top_particle_log_probs.append(map_log_prob)
                elif top_particles:
                    top_particles[-1] = map_particle
                    top_particle_log_probs[-1] = map_log_prob
                logging.info(
                    "Inserted resolved MAP program into truncated mixture policy."
                )

            top_particle_log_probs = np.array(top_particle_log_probs) - logsumexp(
                top_particle_log_probs
            )
            top_particle_probs = np.exp(top_particle_log_probs)
            logging.info("top_particle_probs: %s", top_particle_probs)
            policy: LPPPolicy = LPPPolicy(
                top_particles,
                top_particle_probs,
                map_choices=True,
                normalize_plp_actions=self.normalize_plp_actions,
                action_mode=str(self.env_specs.get("action_mode", "discrete")),
                action_space=cast(Any, self._action_space),
                candidate_actions=candidate_actions,
            )
            if not any(str(p) == str(particles[map_idx]) for p in top_particles):
                raise AssertionError(
                    "MAP representative is not included in the returned mixture "
                    "policy. Ensure particle truncation preserves the MAP PLP."
                )
            policy.map_program = str(particles[map_idx])
            policy.map_posterior = particle_log_probs[map_idx]
            return policy

        logging.info("no nontrivial particles found")
        return LPPPolicy(
            [StateActionProgram("False")],
            [1.0],
            map_choices=True,
            normalize_plp_actions=self.normalize_plp_actions,
            action_mode=str(self.env_specs.get("action_mode", "discrete")),
            action_space=cast(Any, self._action_space),
            candidate_actions=candidate_actions,
        )

    def _compute_policy_risk_on_demos(
        self,
        policy: LPPPolicy,
        demonstrations: Trajectory[_ObsType, _ActType],
        *,
        eps: float = 1e-12,
    ) -> float:
        """Compute empirical validation risk: mean NLL of expert actions."""
        if len(demonstrations.steps) == 0:
            return float("inf")
        aligned_demonstrations = self._align_demonstrations_for_continuous_scoring(
            demonstrations
        )
        losses: list[float] = []
        for obs, action in aligned_demonstrations.steps:
            prob = float(policy.get_action_prob(obs, action))
            losses.append(-float(np.log(max(prob, eps))))
        return float(np.mean(losses))

    def _compute_plp_avg_allowed_metrics(
        self,
        plp: StateActionProgram,
        demonstrations: Trajectory[_ObsType, _ActType],
        *,
        candidate_actions: list[_ActType] | None,
    ) -> tuple[float, float]:
        """Return (avg_allowed_frac, avg_allowed_count) over demo states."""
        if len(demonstrations.steps) == 0:
            return 0.0, 0.0

        total_allowed = 0.0
        total_possible = 0.0
        total_steps = 0

        for obs, action in demonstrations.steps:
            try:
                obs_grid, _ = require_grid_state_action(
                    obs,
                    action,
                    context="_compute_plp_avg_allowed_metrics",
                )
                rows, cols = obs_grid.shape[:2]
                possible = rows * cols
                allowed = 0
                for r in range(rows):
                    for c in range(cols):
                        if plp(obs_grid, (r, c)):
                            allowed += 1
            except TypeError:
                probe_actions = (
                    list(candidate_actions)
                    if candidate_actions is not None and len(candidate_actions) > 0
                    else [action]
                )
                possible = len(probe_actions)
                allowed = 0
                for probe_action in probe_actions:
                    if plp(obs, probe_action):
                        allowed += 1

            total_allowed += float(allowed)
            total_possible += float(max(1, possible))
            total_steps += 1

        if total_steps == 0:
            return 0.0, 0.0
        avg_allowed_frac = total_allowed / max(1.0, total_possible)
        avg_allowed_count = total_allowed / float(total_steps)
        return float(avg_allowed_frac), float(avg_allowed_count)

    def _filter_overly_permissive_plps(
        self,
        plps: list[StateActionProgram],
        plp_priors: list[float],
        demonstrations: Trajectory[_ObsType, _ActType],
        *,
        candidate_actions: list[_ActType] | None,
    ) -> tuple[list[StateActionProgram], list[float]]:
        """Drop PLPs that allow too many actions on average over demos."""
        if not self.permissive_filter_enabled or len(plps) == 0:
            return plps, plp_priors

        max_avg_frac = self.permissive_filter_max_avg_frac
        max_avg_count = self.permissive_filter_max_avg_count
        if max_avg_frac is None and max_avg_count is None:
            return plps, plp_priors

        kept: list[tuple[StateActionProgram, float]] = []
        removed: list[tuple[float, float, str]] = []
        for plp, prior in zip(plps, plp_priors):
            avg_frac, avg_count = self._compute_plp_avg_allowed_metrics(
                plp,
                demonstrations,
                candidate_actions=candidate_actions,
            )
            too_permissive_frac = (max_avg_frac is not None) and (
                avg_frac > float(max_avg_frac)
            )
            too_permissive_count = (max_avg_count is not None) and (
                avg_count > float(max_avg_count)
            )
            if too_permissive_frac or too_permissive_count:
                removed.append((avg_frac, avg_count, str(plp)))
                continue
            kept.append((plp, prior))

        if not kept:
            logging.warning(
                "Permissive filter removed all PLPs; keeping original set. "
                "Try relaxing permissive_filter_max_avg_frac/count."
            )
            return plps, plp_priors

        if removed:
            logging.info(
                "Permissive filter removed %d/%d PLPs "
                "(max_avg_frac=%s max_avg_count=%s).",
                len(removed),
                len(plps),
                str(max_avg_frac),
                str(max_avg_count),
            )
            removed_sorted = sorted(removed, key=lambda t: (t[0], t[1]), reverse=True)
            for idx, (avg_frac, avg_count, plp_str) in enumerate(
                removed_sorted[:5], start=1
            ):
                logging.info(
                    "Permissive drop #%d avg_allowed_frac=%.4f "
                    "avg_allowed_count=%.4f plp=%s",
                    idx,
                    avg_frac,
                    avg_count,
                    plp_str,
                )

        filtered_plps, filtered_priors = zip(*kept)
        return list(filtered_plps), list(filtered_priors)

    def _get_data_loading_config(
        self,
    ) -> tuple[str | None, dict[str, Any] | None]:
        if self.program_generation is None:
            raise ValueError("program_generation config is required.")
        offline_path_name = None
        loading_cfg = (self.program_generation or {}).get("loading")
        if isinstance(loading_cfg, Mapping) and loading_cfg.get("offline"):
            offline_path_name = loading_cfg.get("offline_json_path")
        raw_sampling_cfg = (self.program_generation or {}).get("negative_sampling")
        negative_sampling_cfg = (
            dict(raw_sampling_cfg) if isinstance(raw_sampling_cfg, Mapping) else None
        )
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        if (
            action_mode == "continuous"
            and isinstance(negative_sampling_cfg, dict)
            and hasattr(self._action_space, "low")
            and hasattr(self._action_space, "high")
        ):
            negative_sampling_cfg.setdefault(
                "action_low",
                np.asarray(getattr(self._action_space, "low"), dtype=float).tolist(),
            )
            negative_sampling_cfg.setdefault(
                "action_high",
                np.asarray(getattr(self._action_space, "high"), dtype=float).tolist(),
            )
            cont_cfg = negative_sampling_cfg.setdefault("continuous", {})
            if isinstance(cont_cfg, dict):
                cont_cfg.setdefault("inactive_action_fill_value", 0.0)
        return offline_path_name, negative_sampling_cfg

    def _build_continuous_candidate_actions(
        self,
        negative_sampling_cfg: dict[str, Any] | None,
    ) -> list[_ActType]:
        """Build a fixed continuous action catalog from quantized bucket
        centers."""
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        if action_mode != "continuous":
            return []

        if not hasattr(self._action_space, "low") or not hasattr(
            self._action_space, "high"
        ):
            raise ValueError(
                "Continuous inference requires an action space with low/high bounds."
            )

        sampling_cfg = dict(negative_sampling_cfg or {})
        cont_cfg = dict(sampling_cfg.get("continuous", {}))
        bucket_counts_cfg = cont_cfg.get("bucket_counts")
        bucket_edges_cfg = cont_cfg.get("bucket_edges")

        low_arr = np.asarray(getattr(self._action_space, "low"), dtype=float).reshape(
            -1
        )
        high_arr = np.asarray(getattr(self._action_space, "high"), dtype=float).reshape(
            -1
        )
        if low_arr.size == 0 or high_arr.size == 0:
            raise ValueError(
                "Continuous quantized inference requires at least 1 action dimension."
            )

        active_dims = get_active_action_dims(
            sampling_cfg,
            total_dims=low_arr.size,
            default_active_dims=None,
        )
        inactive_fill_value = get_inactive_action_fill_value(sampling_cfg)
        active_low_arr, active_high_arr = active_action_bounds(
            low_arr,
            high_arr,
            active_dims=active_dims,
        )

        quantizer = Motion2DActionQuantizer.from_bounds(
            active_low_arr,
            active_high_arr,
            bucket_counts=bucket_counts_cfg,
            bucket_edges=bucket_edges_cfg,
        )

        action_dtype = getattr(self._action_space, "dtype", None)
        candidate_actions: list[_ActType] = []
        for center in quantizer.all_bucket_centers():
            candidate = embed_active_action(
                center,
                template=np.zeros_like(low_arr, dtype=float),
                active_dims=active_dims,
                fill_value=inactive_fill_value,
            )
            candidate = np.clip(candidate, low_arr, high_arr)
            if action_dtype is not None:
                candidate = candidate.astype(action_dtype, copy=False)
            candidate_actions.append(cast(_ActType, candidate))
        return candidate_actions

    def _center_continuous_action_for_scoring(self, action: _ActType) -> _ActType:
        """Map a continuous action to the quantized bucket-center
        representation."""
        if not hasattr(self._action_space, "low") or not hasattr(
            self._action_space, "high"
        ):
            raise ValueError(
                "Continuous action alignment requires an action "
                "space with low/high bounds."
            )

        sampling_cfg = dict(self._negative_sampling_cfg or {})
        cont_cfg = dict(sampling_cfg.get("continuous", {}))
        bucket_counts_cfg = cont_cfg.get("bucket_counts")
        bucket_edges_cfg = cont_cfg.get("bucket_edges")

        base = np.asarray(action, dtype=float)
        if base.ndim == 0:
            base = base.reshape(1)

        low_arr = np.asarray(getattr(self._action_space, "low"), dtype=float).reshape(
            -1
        )
        high_arr = np.asarray(getattr(self._action_space, "high"), dtype=float).reshape(
            -1
        )
        if low_arr.shape != base.shape or high_arr.shape != base.shape:
            raise ValueError(
                "continuous action bounds shape mismatch during scoring alignment: "
                f"base={base.shape}, low={low_arr.shape}, high={high_arr.shape}"
            )

        active_dims = get_active_action_dims(
            sampling_cfg,
            total_dims=base.shape[0],
            default_active_dims=None,
        )
        inactive_fill_value = get_inactive_action_fill_value(sampling_cfg)
        canonical_base = canonicalize_continuous_action(
            base,
            active_dims=active_dims,
            fill_value=inactive_fill_value,
        )
        active_low_arr, active_high_arr = active_action_bounds(
            low_arr,
            high_arr,
            active_dims=active_dims,
        )

        quantizer = Motion2DActionQuantizer.from_bounds(
            active_low_arr,
            active_high_arr,
            bucket_counts=bucket_counts_cfg,
            bucket_edges=bucket_edges_cfg,
        )
        centered = embed_active_action(
            quantizer.dequantize(quantizer.quantize(canonical_base[active_dims])),
            template=canonical_base,
            active_dims=active_dims,
            fill_value=inactive_fill_value,
        )
        centered = np.clip(centered, low_arr, high_arr)

        if isinstance(action, np.ndarray):
            return cast(_ActType, centered.astype(action.dtype, copy=False))
        if isinstance(action, tuple):
            return cast(_ActType, tuple(float(x) for x in centered.tolist()))
        if isinstance(action, list):
            return cast(_ActType, [float(x) for x in centered.tolist()])
        return cast(_ActType, centered)

    def _align_demonstrations_for_continuous_scoring(
        self,
        demonstrations: Trajectory[_ObsType, _ActType],
    ) -> Trajectory[_ObsType, _ActType]:
        """Use bucket-center demo actions so scoring matches
        training/inference."""
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        if action_mode != "continuous":
            return demonstrations

        aligned_steps = [
            (obs, self._center_continuous_action_for_scoring(action))
            for obs, action in demonstrations.steps
        ]
        return Trajectory(steps=aligned_steps)

    def _recovery_obs_hash(self, obs: Any) -> Any:
        if isinstance(obs, np.ndarray):
            return ("np", str(obs.dtype), tuple(obs.shape), obs.tobytes())
        return ("repr", repr(obs))

    def _motion2d_target_distance(self, obs: Any) -> float | None:
        if "motion2d" not in str(self.base_class_name).lower():
            return None
        arr = np.asarray(obs, dtype=float).reshape(-1)
        if arr.size < 19:
            return None
        robot_x = float(arr[0])
        robot_y = float(arr[1])
        target_cx = float(arr[9] + arr[17] / 2.0)
        target_cy = float(arr[10] + arr[18] / 2.0)
        return float(np.hypot(target_cx - robot_x, target_cy - robot_y))

    def _motion2d_robot_position(self, obs: Any) -> tuple[float, float] | None:
        if "motion2d" not in str(self.base_class_name).lower():
            return None
        arr = np.asarray(obs, dtype=float).reshape(-1)
        if arr.size < 2:
            return None
        return float(arr[0]), float(arr[1])

    def _motion2d_reference_passage(self, obs: Any) -> dict[str, float | int] | None:
        if "motion2d" not in str(self.base_class_name).lower():
            return None
        arr = np.asarray(obs, dtype=float).reshape(-1)
        if arr.size < 39:
            return None
        robot_x = float(arr[0])
        robot_r = float(arr[3])
        num_obstacles = max(0, (arr.size - 19) // 10)
        passages: list[dict[str, float | int]] = []
        for passage_idx in range(num_obstacles // 2):
            bot_base = 19 + 20 * passage_idx
            top_base = bot_base + 10
            wall_x = float(arr[bot_base])
            wall_w = float(arr[bot_base + 8])
            gap_lower = float(arr[bot_base + 1] + arr[bot_base + 9] + robot_r)
            gap_upper = float(arr[top_base + 1] - robot_r)
            passages.append(
                {
                    "passage_idx": passage_idx,
                    "wall_left": wall_x,
                    "wall_right": wall_x + wall_w,
                    "gap_lower": gap_lower,
                    "gap_upper": gap_upper,
                    "gap_center": (gap_lower + gap_upper) / 2.0,
                }
            )
        if not passages:
            return None
        for passage in sorted(passages, key=lambda p: float(p["wall_left"])):
            if robot_x + robot_r < float(passage["wall_right"]):
                return passage
        return min(passages, key=lambda p: abs(robot_x - float(p["gap_center"])))

    def _format_recovery_action(self, action: Any) -> str:
        arr = np.asarray(action, dtype=float).reshape(-1)
        return np.array2string(arr, precision=3, separator=", ")

    def _log_recovery_decision_debug(
        self,
        *,
        policy: LPPPolicy,
        obs: _ObsType,
        policy_action: _ActType,
        expert_action: _ActType,
        env_num: int,
        trigger: str,
    ) -> None:
        """Log a compact snapshot for a recovery-triggered state."""
        try:
            robot_pos = self._motion2d_robot_position(obs)
            target_dist = self._motion2d_target_distance(obs)
            ref_passage = self._motion2d_reference_passage(obs)
            centered_expert = self._center_continuous_action_for_scoring(expert_action)
            debug_parts = [
                f"env={env_num}",
                f"trigger={trigger}",
                f"policy={self._format_recovery_action(policy_action)}",
                f"expert_centered={self._format_recovery_action(centered_expert)}",
            ]
            if robot_pos is not None:
                debug_parts.append(f"robot=({robot_pos[0]:.3f}, {robot_pos[1]:.3f})")
            if target_dist is not None:
                debug_parts.append(f"target_dist={target_dist:.3f}")
            if ref_passage is not None:
                debug_parts.append(
                    "passage="
                    f"{int(ref_passage['passage_idx'])}"
                    f"(wall=[{float(ref_passage['wall_left']):.3f},"
                    f"{float(ref_passage['wall_right']):.3f}],"
                    f"gap_center={float(ref_passage['gap_center']):.3f},"
                    f"gap=[{float(ref_passage['gap_lower']):.3f},"
                    f"{float(ref_passage['gap_upper']):.3f}])"
                )

            if (
                str(self.env_specs.get("action_mode", "discrete")) == "continuous"
                and policy.candidate_actions
            ):
                # Accessing the policy's scored recovery candidates is
                # intentional here for debugging.
                # pylint: disable-next=protected-access
                scores = policy._get_continuous_candidate_scores(obs)
                if scores.size > 0:
                    top_indices = np.argsort(scores)[::-1][:2]
                    top_parts = []
                    for rank, idx in enumerate(top_indices.tolist(), start=1):
                        action_text = self._format_recovery_action(
                            policy.candidate_actions[idx]
                        )
                        top_parts.append(
                            f"top{rank}={action_text}@{float(scores[idx]):.4f}"
                        )
                    # pylint: disable-next=protected-access
                    expert_idx = policy._find_continuous_candidate_index(
                        cast(_ActType, centered_expert)
                    )
                    if expert_idx is not None:
                        top_parts.append(
                            f"expert_score={float(scores[expert_idx]):.4f}"
                        )
                    debug_parts.extend(top_parts)

            logging.info("Recovery debug: %s", " | ".join(debug_parts))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.info("Recovery debug unavailable for env %d: %s", env_num, exc)

    def _actions_match_for_recovery(self, a: _ActType, b: _ActType) -> bool:
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        if action_mode != "continuous":
            return cast(bool, a == b)
        aa = np.asarray(self._center_continuous_action_for_scoring(a), dtype=float)
        bb = np.asarray(self._center_continuous_action_for_scoring(b), dtype=float)
        return bool(np.allclose(aa, bb, atol=1e-8))

    def _continuous_recovery_action_signature(
        self,
        action: _ActType,
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """Return quantized active-dim buckets and sign pattern for
        recovery."""
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        if action_mode != "continuous":
            return None
        if not hasattr(self._action_space, "low") or not hasattr(
            self._action_space, "high"
        ):
            return None

        base = np.asarray(action, dtype=float).reshape(-1)
        low_arr = np.asarray(getattr(self._action_space, "low"), dtype=float).reshape(
            -1
        )
        high_arr = np.asarray(getattr(self._action_space, "high"), dtype=float).reshape(
            -1
        )
        if base.shape != low_arr.shape or base.shape != high_arr.shape:
            return None

        sampling_cfg = dict(self._negative_sampling_cfg or {})
        cont_cfg = dict(sampling_cfg.get("continuous", {}))
        active_dims = get_active_action_dims(
            sampling_cfg,
            total_dims=base.shape[0],
            default_active_dims=None,
        )
        inactive_fill_value = get_inactive_action_fill_value(sampling_cfg)
        canonical_base = canonicalize_continuous_action(
            base,
            active_dims=active_dims,
            fill_value=inactive_fill_value,
        )
        active_low_arr, active_high_arr = active_action_bounds(
            low_arr,
            high_arr,
            active_dims=active_dims,
        )
        quantizer = Motion2DActionQuantizer.from_bounds(
            active_low_arr,
            active_high_arr,
            bucket_counts=cont_cfg.get("bucket_counts"),
            bucket_edges=cont_cfg.get("bucket_edges"),
        )
        active_action = canonical_base[active_dims]
        bucket = tuple(int(idx) for idx in quantizer.quantize(active_action))
        sign_pattern = tuple(
            int(np.sign(value)) if not np.isclose(value, 0.0, atol=1e-8) else 0
            for value in active_action.tolist()
        )
        return bucket, sign_pattern

    def _recovery_action_signature(self, action: _ActType) -> Any:
        """Return a hashable action signature for recovery-loop checks."""
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        if action_mode == "continuous":
            signature = self._continuous_recovery_action_signature(action)
            if signature is not None:
                return ("continuous", signature)
        if isinstance(action, np.ndarray):
            return ("np", str(action.dtype), tuple(action.shape), action.tobytes())
        return ("repr", repr(action))

    def _has_recovery_loop(self, recent_step_signatures: Sequence[Any]) -> bool:
        """Return True for simple repeated state-action loops."""
        window = list(recent_step_signatures)
        if len(window) < 2:
            return False
        if all(sig == window[0] for sig in window[1:]):
            return True
        if len(window) % 2 != 0 or window[0] == window[1]:
            return False
        return all(sig == window[idx % 2] for idx, sig in enumerate(window))

    def _get_recovery_stuck_indices(
        self, recent_step_signatures: Sequence[Any]
    ) -> list[int]:
        """Return representative indices for a simple stuck pattern."""
        window = list(recent_step_signatures)
        if len(window) < 2:
            return []
        if all(sig == window[0] for sig in window[1:]):
            return [len(window) - 1]
        if len(window) % 2 != 0 or window[0] == window[1]:
            return []
        if all(sig == window[idx % 2] for idx, sig in enumerate(window)):
            return [len(window) - 2, len(window) - 1]
        return []

    def _is_meaningful_recovery_deviation(
        self,
        learner_action: _ActType,
        expert_action: _ActType,
        *,
        bucket_tolerance: int,
        require_sign_flip: bool,
    ) -> bool:
        """Return True when learner action meaningfully diverges from
        expert."""
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        if action_mode != "continuous":
            return not self._actions_match_for_recovery(learner_action, expert_action)

        learner_sig = self._continuous_recovery_action_signature(learner_action)
        expert_sig = self._continuous_recovery_action_signature(expert_action)
        if learner_sig is None or expert_sig is None:
            return not self._actions_match_for_recovery(learner_action, expert_action)

        learner_bucket, learner_sign = learner_sig
        expert_bucket, expert_sign = expert_sig
        bucket_distance = sum(
            abs(int(a_idx) - int(b_idx))
            for a_idx, b_idx in zip(learner_bucket, expert_bucket)
        )
        sign_flip = any(
            ls != es and ls != 0 and es != 0
            for ls, es in zip(learner_sign, expert_sign)
        )
        if require_sign_flip:
            return sign_flip
        return bucket_distance > max(0, int(bucket_tolerance)) or sign_flip

    def _query_expert_action_for_recovery(
        self,
        obs: _ObsType,
        info: dict[str, Any],
        env: Any,
    ) -> _ActType | None:
        try:
            if hasattr(self.expert, "set_env"):
                self.expert.set_env(env)
            self.expert.reset(obs, info)
            return cast(_ActType, self.expert.step())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.warning("Recovery expert query failed: %s", exc)
            return None

    # TODOOO: Recovery augmentation is disabled for manual data collection
    # environments such as PushPullHook2D.
    def _augment_recovery_states(
        self,
        *,
        policy: LPPPolicy,
        train_demo_ids: tuple[int, ...],
        demonstrations_train: Trajectory[_ObsType, _ActType],
        demo_dict_train: dict[int, Trajectory[_ObsType, _ActType]],
        negative_sampling_cfg: dict[str, Any] | None = None,
    ) -> tuple[
        tuple[int, ...],
        Trajectory[_ObsType, _ActType],
        dict[int, Trajectory[_ObsType, _ActType]],
        bool,
    ]:
        """Roll out current policy on train envs and add recovery states."""
        cfg = dict(self.recovery_augmentation or {})
        if not bool(cfg.get("enabled", False)):
            return train_demo_ids, demonstrations_train, demo_dict_train, False

        max_steps = int(cfg.get("max_steps", 50))
        stuck_window = max(2, int(cfg.get("stuck_window", 6)))
        capture_deviation = bool(cfg.get("capture_deviation", True))
        capture_stuck = bool(cfg.get("capture_stuck", True))
        deviation_bucket_tolerance = int(cfg.get("deviation_bucket_tolerance", 2))
        deviation_require_sign_flip = bool(
            cfg.get("deviation_require_sign_flip", False)
        )
        queried_steps: list[tuple[_ObsType, _ActType]] = []
        seen_hashes: set[Any] = {
            self._recovery_obs_hash(obs) for obs, _ in demonstrations_train.steps
        }
        event_counts = {"deviation": 0, "stuck": 0}
        all_train_demo_ids = tuple(int(i) for i in train_demo_ids)
        rollout_env_ids = tuple(i for i in all_train_demo_ids if i >= 0)

        for env_num in rollout_env_ids:
            env = self.env_factory(int(env_num))
            try:
                try:
                    reset_out = env.reset(seed=int(env_num))
                except TypeError:
                    reset_out = env.reset()
                if isinstance(reset_out, tuple) and len(reset_out) == 2:
                    obs, info = reset_out
                else:
                    obs, info = reset_out, {}

                recent_history: deque[
                    tuple[_ObsType, dict[str, Any], _ActType | None]
                ] = deque(maxlen=stuck_window)
                recent_step_signatures: deque[Any] = deque(maxlen=stuck_window)
                captured_deviation = False
                captured_stuck = False
                env_added = 0
                env_duplicate_skips = 0

                for _ in range(max_steps):
                    expert_action = self._query_expert_action_for_recovery(
                        obs, info, env
                    )
                    policy_action = cast(_ActType, policy(obs))
                    recent_history.append((obs, dict(info), policy_action))
                    recent_step_signatures.append(
                        (
                            self._recovery_obs_hash(obs),
                            self._recovery_action_signature(policy_action),
                        )
                    )

                    if (
                        expert_action is not None
                        and capture_deviation
                        and not captured_deviation
                        and self._is_meaningful_recovery_deviation(
                            policy_action,
                            expert_action,
                            bucket_tolerance=deviation_bucket_tolerance,
                            require_sign_flip=deviation_require_sign_flip,
                        )
                    ):
                        obs_hash = self._recovery_obs_hash(obs)
                        if obs_hash not in seen_hashes:
                            queried_steps.append((obs, expert_action))
                            seen_hashes.add(obs_hash)
                            event_counts["deviation"] += 1
                            captured_deviation = True
                            env_added += 1
                            logging.info(
                                "Recovery env %d: added deviation state #%d.",
                                env_num,
                                env_added,
                            )
                            self._log_recovery_decision_debug(
                                policy=policy,
                                obs=obs,
                                policy_action=policy_action,
                                expert_action=expert_action,
                                env_num=env_num,
                                trigger="deviation",
                            )
                        else:
                            env_duplicate_skips += 1
                            logging.info(
                                "Recovery env %d: deviation trigger hit existing state; "
                                "skipping duplicate.",
                                env_num,
                            )
                    if (
                        capture_stuck
                        and not captured_stuck
                        and len(recent_step_signatures) == stuck_window
                    ):
                        stuck_indices = self._get_recovery_stuck_indices(
                            recent_step_signatures
                        )
                        added_stuck = 0
                        for idx in stuck_indices:
                            stuck_obs, stuck_info, stuck_policy_action = recent_history[
                                idx
                            ]
                            stuck_hash = self._recovery_obs_hash(stuck_obs)
                            if stuck_hash in seen_hashes:
                                continue
                            if idx == len(recent_history) - 1:
                                stuck_expert_action = expert_action
                            else:
                                stuck_expert_action = (
                                    self._query_expert_action_for_recovery(
                                        stuck_obs, stuck_info, env
                                    )
                                )
                            if stuck_expert_action is None:
                                continue
                            queried_steps.append((stuck_obs, stuck_expert_action))
                            seen_hashes.add(stuck_hash)
                            added_stuck += 1
                            env_added += 1
                            if stuck_policy_action is not None:
                                self._log_recovery_decision_debug(
                                    policy=policy,
                                    obs=stuck_obs,
                                    policy_action=stuck_policy_action,
                                    expert_action=stuck_expert_action,
                                    env_num=env_num,
                                    trigger="stuck",
                                )
                        if added_stuck > 0:
                            event_counts["stuck"] += added_stuck
                            captured_stuck = True
                            logging.info(
                                "Recovery env %d: added %d stuck-state correction(s).",
                                env_num,
                                added_stuck,
                            )
                        elif stuck_indices:
                            env_duplicate_skips += len(stuck_indices)
                            logging.info(
                                "Recovery env %d: detected stuck pattern but all "
                                "candidate states were duplicates or missing expert "
                                "actions.",
                                env_num,
                            )

                    step_out = env.step(policy_action)
                    if len(step_out) == 4:
                        obs, reward, done, info = step_out
                        terminated, truncated = done, False
                    else:
                        obs, reward, terminated, truncated, info = step_out
                    del reward
                    if terminated or truncated:
                        break
                logging.info(
                    "Recovery env %d summary: added=%d deviation=%s stuck=%s "
                    "duplicate_skips=%d",
                    env_num,
                    env_added,
                    captured_deviation,
                    captured_stuck,
                    env_duplicate_skips,
                )
            finally:
                if hasattr(env, "close"):
                    env.close()

        if not queried_steps:
            logging.info("Recovery augmentation found no new recovery states.")
            return train_demo_ids, demonstrations_train, demo_dict_train, False

        next_demo_id = min(all_train_demo_ids, default=0) - 1
        aug_traj = Trajectory[_ObsType, _ActType](steps=queried_steps)
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        recovery_pos, recovery_neg, _ = extract_examples_from_demonstration(
            aug_traj,
            negative_sampling=negative_sampling_cfg,
            action_mode=action_mode,
        )
        augmented_demo_dict = dict(demo_dict_train)
        augmented_demo_dict[next_demo_id] = aug_traj
        augmented_demo_ids = tuple(list(all_train_demo_ids) + [next_demo_id])
        augmented_demonstrations = Trajectory[_ObsType, _ActType](
            steps=list(demonstrations_train.steps) + queried_steps
        )
        logging.info(
            "Recovery augmentation added %d recovery states as demo %d "
            "(deviation=%d stuck=%d).",
            len(queried_steps),
            next_demo_id,
            event_counts["deviation"],
            event_counts["stuck"],
        )
        logging.info(
            "Recovery augmentation expansion: positives=%d negatives=%d "
            "total_examples=%d",
            len(recovery_pos),
            len(recovery_neg),
            len(recovery_pos) + len(recovery_neg),
        )
        return augmented_demo_ids, augmented_demonstrations, augmented_demo_dict, True

    def _build_and_process_train_matrix(
        self,
        *,
        train_demo_ids: tuple[int, ...],
        val_demo_ids: tuple[int, ...],
        demo_dict_train: dict[int, Trajectory[_ObsType, _ActType]],
        programs_sa: list[StateActionProgram],
        program_prior_log_probs_opt: list[float] | None,
        dsl_functions: dict[str, Any],
        negative_sampling_cfg: dict[str, Any] | None,
        offline_path_name: str | None,
        start_index: int,
        feature_display_names: list[str] | None = None,
    ) -> tuple[
        Any,
        np.ndarray,
        list[bool],
        np.ndarray | None,
        list[tuple[_ObsType, _ActType]] | None,
        list[StateActionProgram],
        list[float] | None,
    ]:
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        val_split_tag = (
            "-".join(str(x) for x in val_demo_ids) if val_demo_ids else "none"
        )
        self._assert_continuous_demo_alignment(
            (
                Trajectory(steps=list(demo_dict_train.values())[0].steps[:5])
                if demo_dict_train
                else Trajectory(steps=[])
            ),
            negative_sampling_cfg,
        )
        result = cast(
            tuple[
                Any | None,
                np.ndarray | None,
                list[tuple[_ObsType, _ActType]] | None,
                np.ndarray | None,
                np.ndarray,
            ],
            run_all_programs_on_demonstrations(
                self.base_class_name,
                train_demo_ids,
                programs_sa,
                demo_dict_train,
                dsl_functions,
                negative_sampling=negative_sampling_cfg,
                env_factory=self.env_factory,
                return_examples=True,
                offline_path_name=offline_path_name,
                prompt_demo_ids=self._prompt_demo_ids_tag(),
                split_tag=(
                    f"seed{self.split_seed}"
                    f"_train_{'-'.join(str(x) for x in train_demo_ids)}"
                    f"__val_{val_split_tag}"
                    "__role_train_core"
                ),
                seed=self.seed_num,
                action_mode=action_mode,
                return_demo_ids=True,
            ),
        )
        X, y, examples, sample_weights, row_demo_ids = result
        self._log_sample_weight_stats(
            sample_weights,
            label="train_core_raw",
        )

        if X is None or y is None:
            raise ValueError(
                "Train matrix is invalid. Ensure program execution results are valid."
            )

        logging.info("n_examples=%d n_features=%d", X.shape[0], X.shape[1])
        log_exact_example_label_contradictions(examples, y)
        old_examples = examples
        old_row_demo_ids = row_demo_ids
        X, y, examples, sample_weights = drop_negative_exact_contradictions(
            X, y, examples, sample_weights
        )
        row_demo_ids = self._realign_row_demo_ids(
            old_examples,
            old_row_demo_ids,
            examples,
        )
        old_examples = examples
        old_row_demo_ids = row_demo_ids
        X, y, examples, sample_weights = deduplicate_negative_examples(
            X, y, examples, sample_weights
        )
        row_demo_ids = self._realign_row_demo_ids(
            old_examples,
            old_row_demo_ids,
            examples,
        )
        discrete_cfg = dict((negative_sampling_cfg or {}).get("discrete", {}))
        use_discrete_sample_weights = bool(
            discrete_cfg.get("use_sample_weights", False)
        )
        if action_mode == "discrete" and not use_discrete_sample_weights:
            sample_weights = None
        X, programs_sa, program_prior_log_probs_opt, col_nnz = (
            _filter_redundant_features(X, programs_sa, program_prior_log_probs_opt)
        )
        col_nnz = np.asarray(X.getnnz(axis=0)).ravel()
        all_zero = np.where(col_nnz == 0)[0]
        all_one = np.where(col_nnz == X.shape[0])[0]
        logging.info("#all-zero features=%d indices=%s", len(all_zero), all_zero[:30])
        logging.info("#all-one features=%d indices=%s", len(all_one), all_one[:30])
        logging.info("n_examples=%d n_features=%d", X.shape[0], X.shape[1])

        X, programs_sa, program_prior_log_probs_opt = (
            self._score_and_optionally_filter_features(
                X,
                y,
                row_demo_ids,
                programs_sa,
                program_prior_log_probs_opt,
            )
        )

        collision_groups = log_feature_collisions(
            X,
            y,
            examples,
            bucket_mode=self.collision_feedback_bucket_mode,
        )

        if self.collision_feedback_enabled and examples is not None:
            max_rounds = max(1, self.collision_feedback_max_rounds)
            collision_attempt_memory: list[str] = []
            current_feature_display_names = feature_display_names

            def _summarize_existing_features() -> str:
                nonlocal current_feature_display_names
                if current_feature_display_names is not None and len(
                    current_feature_display_names
                ) != len(programs_sa):
                    logging.info(
                        "Feature display names are out of sync with the live "
                        "program list (%d names vs %d programs); falling back "
                        "to current programs for collision feedback prompts.",
                        len(current_feature_display_names),
                        len(programs_sa),
                    )
                    current_feature_display_names = None

                if current_feature_display_names:
                    display_names = current_feature_display_names[:40]
                    lines = [f"- {name}" for name in display_names]
                    remaining = len(current_feature_display_names) - len(display_names)
                    if remaining > 0:
                        lines.append(f"- ... plus {remaining} more existing features")
                    return "\n".join(lines)
                feature_names = [str(p) for p in programs_sa[:40]]
                if not feature_names:
                    return "- None"
                lines = [f"- {name}" for name in feature_names]
                remaining = len(programs_sa) - len(feature_names)
                if remaining > 0:
                    lines.append(f"- ... plus {remaining} more existing features")
                return "\n".join(lines)

            def _record_attempt_summary(
                round_idx: int,
                collision_payload: dict[str, Any],
                before_count: int,
                after_count: int,
            ) -> None:
                features = collision_payload.get("features", [])
                descriptions: list[str] = []
                if isinstance(features, list):
                    for feature in features[:3]:
                        if not isinstance(feature, dict):
                            continue
                        desc = str(feature.get("description", "")).strip()
                        if desc:
                            descriptions.append(desc.rstrip("."))
                lines = [f"Previous repair attempt {round_idx}:"]
                if descriptions:
                    lines.append(
                        "- Added features targeting: " + "; ".join(descriptions)
                    )
                else:
                    added_count = len(features) if isinstance(features, list) else 0
                    lines.append(f"- Added {added_count} new features.")
                if after_count < before_count:
                    lines.append(
                        "- Collision count improved from "
                        f"{before_count} to {after_count}, but collisions remain."
                    )
                elif after_count > before_count:
                    lines.append(
                        "- Collision count worsened from "
                        f"{before_count} to {after_count}; avoid feature families "
                        "that increase overlap across positive and negative rows."
                    )
                else:
                    lines.append(
                        "- Collision count stayed at "
                        f"{after_count}; avoid near-duplicate "
                        "rephrasings of those feature families."
                    )
                collision_attempt_memory.append("\n".join(lines))

            def _generate_collision_features(
                prompt: str, start_idx: int, collision_idx: int
            ) -> tuple[list[str], dict[str, Any], Path]:
                return self._generate_collision_features(
                    prompt, start_index=start_idx, collision_idx=collision_idx
                )

            def _make_collision_prompt(
                collision_groups: list[dict[str, Any]],
                examples: list[tuple[_ObsType, _ActType]],
            ) -> str | None:
                failed_attempts = (
                    "\n\n".join(collision_attempt_memory)
                    if collision_attempt_memory
                    else "- None yet."
                )
                return self._handle_collision_feedback(
                    collision_groups,
                    examples,
                    existing_feature_summary=_summarize_existing_features(),
                    failed_attempt_summaries=failed_attempts,
                )

            (
                X,
                programs_sa,
                program_prior_log_probs_opt,
                collision_payloads,
                collision_output_path,
                col_nnz,
            ) = run_collision_feedback_loop(
                collision_groups=collision_groups,
                examples=examples,
                max_rounds=max_rounds,
                target_collisions=self.collision_feedback_target_collisions,
                start_index=start_index,
                program_prior_log_probs=program_prior_log_probs_opt,
                X=X,
                y=y,
                programs_sa=programs_sa,
                dsl_functions=dsl_functions,
                generate_features=_generate_collision_features,
                make_prompt=_make_collision_prompt,
                record_attempt_summary=_record_attempt_summary,
                prior_version=self.prior_version,
                prior_beta=self.prior_beta,
                collision_bucket_mode=self.collision_feedback_bucket_mode,
            )
            if collision_payloads and collision_output_path is not None:
                out_path = (
                    collision_output_path / "py_feature_payload_collision_all.json"
                )
                out_path.write_text(
                    json.dumps({"collision_payloads": collision_payloads}, indent=4),
                    encoding="utf-8",
                )
            X, programs_sa, program_prior_log_probs_opt = (
                self._score_and_optionally_filter_features(
                    X,
                    y,
                    row_demo_ids,
                    programs_sa,
                    program_prior_log_probs_opt,
                )
            )
        logging.info("Data after collision feedback loop: X shape %s", X.shape)

        n = X.shape[0]
        logging.info("N=%d", n)
        col_nnz = np.asarray(X.getnnz(axis=0)).ravel()
        freq = col_nnz / n
        rare = np.where(freq <= 0.05)[0]
        common = np.where(freq >= 0.95)[0]
        logging.info("Almost-always-0=%d", len(rare))
        logging.info("Almost-always-1=%d", len(common))
        assert_features_fire(X, programs_sa)

        y_bool: list[bool] = list(y.astype(bool).flatten())
        pos = sum(y_bool)
        neg = len(y_bool) - pos
        logging.info(
            "y: n=%d pos=%d (%.2f%%) neg=%d",
            len(y_bool),
            pos,
            100 * pos / len(y_bool),
            neg,
        )
        if sample_weights is None and action_mode != "discrete":
            sample_weights = np.ones(len(y_bool), dtype=float)
        return (
            X,
            y,
            y_bool,
            sample_weights,
            examples,
            programs_sa,
            program_prior_log_probs_opt,
        )

    def _select_hyperparams(
        self,
        *,
        X: Any,
        y_bool: list[bool],
        sample_weight: np.ndarray | None,
        programs_sa: list[StateActionProgram],
        program_prior_log_probs_opt: list[float] | None,
        demonstrations_train: Trajectory[_ObsType, _ActType],
        demonstrations_val: Trajectory[_ObsType, _ActType],
        dsl_functions: dict[str, Any],
        negative_sampling_cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Select hyperparameters via validation risk.

        Returns a dict so future tunable knobs can be added without
        changing training/evaluation orchestration code.
        """
        candidate_hyperparams = self._build_hyperparam_candidates()
        has_validation = len(demonstrations_val.steps) > 0
        best_hyperparams = dict(candidate_hyperparams[0])
        if not has_validation:
            logging.info(
                "No validation data available; skipping eval-risk minimization and "
                "falling back to the first candidate hyperparameter set."
            )
            return best_hyperparams
        if len(candidate_hyperparams) == 1:
            logging.info(
                (
                    "Single candidate hyperparameter set=%s; "
                    "skipping validation model selection."
                ),
                best_hyperparams,
            )
            return best_hyperparams

        best_risk = float("inf")
        for hp in candidate_hyperparams:
            logging.info(
                "Training candidate on train_core with hyperparameters=%s.",
                hp,
            )
            candidate_policy = self._train_policy_from_matrix(
                X,
                y_bool,
                sample_weight,
                programs_sa,
                program_prior_log_probs_opt,
                demonstrations_train,
                dsl_functions,
                negative_sampling_cfg=negative_sampling_cfg,
                train_hyperparams=hp,
                tie_break_demonstrations=demonstrations_val,
            )
            val_risk = self._compute_policy_risk_on_demos(
                candidate_policy, demonstrations_val
            )
            logging.info(
                "Validation risk (mean NLL) for hyperparameters=%s: %.6f",
                hp,
                val_risk,
            )
            if val_risk < best_risk:
                best_risk = val_risk
                best_hyperparams = dict(hp)
        logging.info(
            "Selected hyperparameters=%s with best validation risk %.6f.",
            best_hyperparams,
            best_risk,
        )
        return best_hyperparams

    def _build_final_training_data(
        self,
        *,
        train_demo_ids: tuple[int, ...],
        val_demo_ids: tuple[int, ...],
        demo_dict_val: dict[int, Trajectory[_ObsType, _ActType]],
        programs_sa: list[StateActionProgram],
        dsl_functions: dict[str, Any],
        negative_sampling_cfg: dict[str, Any] | None,
        offline_path_name: str | None,
        X_train: Any,
        y_train: np.ndarray,
        sample_weight_train: np.ndarray | None,
        program_prior_log_probs_opt: list[float] | None,
        demonstrations_train: Trajectory[_ObsType, _ActType],
        demonstrations_val: Trajectory[_ObsType, _ActType],
    ) -> tuple[
        Any,
        list[bool],
        np.ndarray | None,
        list[StateActionProgram],
        list[float] | None,
        Trajectory[_ObsType, _ActType],
    ]:
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        if len(train_demo_ids) == 0:
            raise ValueError("No demonstrations available for final retraining.")
        demonstrations_final = Trajectory[_ObsType, _ActType](
            steps=list(demonstrations_train.steps) + list(demonstrations_val.steps)
        )

        X_final = X_train
        y_final = y_train
        sample_weight_final = sample_weight_train
        if val_demo_ids:
            val_result = run_all_programs_on_demonstrations(
                self.base_class_name,
                val_demo_ids,
                list(programs_sa),
                demo_dict_val,
                dsl_functions,
                negative_sampling=negative_sampling_cfg,
                env_factory=self.env_factory,
                return_examples=False,
                offline_path_name=offline_path_name,
                prompt_demo_ids=self._prompt_demo_ids_tag(),
                split_tag=(
                    f"seed{self.split_seed}"
                    f"_train_{'-'.join(str(x) for x in train_demo_ids)}"
                    f"__val_{'-'.join(str(x) for x in val_demo_ids)}"
                    "__role_val"
                ),
                seed=self.seed_num,
                action_mode=action_mode,
            )
            X_val, y_val, _, sample_weight_val = val_result[:4]
            if X_val is None or y_val is None:
                raise ValueError(
                    "X_val or y_val is None. Ensure the validation dataset is valid."
                )
            X_final = vstack([X_final, X_val]).tocsr()
            y_final = np.concatenate([y_final, y_val])
            if sample_weight_final is None:
                sample_weight_final = sample_weight_val
            elif sample_weight_val is not None:
                sample_weight_final = np.concatenate(
                    [sample_weight_final, sample_weight_val]
                )
        if action_mode == "discrete":
            sample_weight_final = None

        final_programs_sa = list(programs_sa)
        final_program_priors = (
            list(program_prior_log_probs_opt)
            if program_prior_log_probs_opt is not None
            else None
        )
        X_final, final_programs_sa, final_program_priors, _ = (
            _filter_redundant_features(X_final, final_programs_sa, final_program_priors)
        )
        y_final_bool: list[bool] = list(y_final.astype(bool).flatten())
        return (
            X_final,
            y_final_bool,
            sample_weight_final,
            final_programs_sa,
            final_program_priors,
            demonstrations_final,
        )

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        self._policy = self._train_policy()
        self._timestep = 0

    def _train_policy(self) -> LPPPolicy:
        """Train the logical programmatic policy using demonstrations."""
        outer_feedback = None
        offline_path_name, negative_sampling_cfg = self._get_data_loading_config()
        self._negative_sampling_cfg = negative_sampling_cfg

        split_result: tuple[
            tuple[int, ...],
            tuple[int, ...],
            Trajectory[_ObsType, _ActType],
            Trajectory[_ObsType, _ActType],
            dict[int, Trajectory[_ObsType, _ActType]],
            dict[int, Trajectory[_ObsType, _ActType]],
        ] = split_and_collect_demonstrations(
            env_factory=self.env_factory,
            expert=self.expert,
            demo_numbers=self.demo_numbers,
            val_frac=self.val_frac,
            val_size=self.val_size,
            split_seed=self.split_seed,
            split_strategy=self.split_strategy,
            preserve_ordering=self.preserve_ordering,
            negative_sampling_cfg=negative_sampling_cfg,
            action_mode=str(self.env_specs.get("action_mode", "discrete")),
        )
        (
            train_demo_ids,
            val_demo_ids,
            demonstrations_train,
            demonstrations_val,
            demo_dict_train,
            demo_dict_val,
        ) = split_result

        ###### Logging and validation checks on demonstrations ######
        self._log_motion2d_demonstration_snapshots(
            demo_dict_train,
            split_name="train",
        )
        if demo_dict_val:
            self._log_motion2d_demonstration_snapshots(
                demo_dict_val,
                split_name="validation",
            )
        ##############################################################
        (
            programs_sa,
            program_prior_log_probs_opt,
            dsl_functions,
            start_index,
            feature_display_names,
        ) = prepare_programs_and_dsl(
            num_programs=self.num_programs,
            base_class_name=self.base_class_name,
            env_factory=self.env_factory,
            expert=self.expert,
            env_specs=self.env_specs,
            start_symbol=self.start_symbol,
            program_generation=self.program_generation,
            train_demo_ids=train_demo_ids,
            outer_feedback=outer_feedback,
            seed_num=self.seed_num,
            prior_version=self.prior_version,
            prior_beta=self.prior_beta,
            get_program_set_fn=get_program_set,
            extract_feature_names_fn=_extract_feature_names,
        )
        (
            X_train,
            y_train,
            y_train_bool,
            sample_weight_train,
            _examples,
            programs_sa,
            program_prior_log_probs_opt,
        ) = self._build_and_process_train_matrix(
            train_demo_ids=train_demo_ids,
            val_demo_ids=val_demo_ids,
            demo_dict_train=demo_dict_train,
            programs_sa=programs_sa,
            program_prior_log_probs_opt=program_prior_log_probs_opt,
            dsl_functions=dsl_functions,
            negative_sampling_cfg=negative_sampling_cfg,
            offline_path_name=offline_path_name,
            start_index=start_index,
            feature_display_names=feature_display_names,
        )
        recovery_cfg = dict(self.recovery_augmentation or {})
        if bool(recovery_cfg.get("enabled", False)):
            recovery_rounds = max(1, int(recovery_cfg.get("rounds", 1)))
            for round_idx in range(recovery_rounds):
                logging.info(
                    "Recovery augmentation round %d/%d: training bootstrap policy.",
                    round_idx + 1,
                    recovery_rounds,
                )
                bootstrap_policy = self._train_policy_from_matrix(
                    X_train,
                    y_train_bool,
                    sample_weight_train,
                    programs_sa,
                    program_prior_log_probs_opt,
                    demonstrations_train,
                    dsl_functions,
                    negative_sampling_cfg=negative_sampling_cfg,
                    train_hyperparams=self._default_train_hyperparams(),
                    tie_break_demonstrations=None,
                )
                (
                    train_demo_ids,
                    demonstrations_train,
                    demo_dict_train,
                    changed,
                ) = self._augment_recovery_states(
                    policy=bootstrap_policy,
                    train_demo_ids=train_demo_ids,
                    demonstrations_train=demonstrations_train,
                    demo_dict_train=demo_dict_train,
                    negative_sampling_cfg=negative_sampling_cfg,
                )
                if not changed:
                    break
                (
                    X_train,
                    y_train,
                    y_train_bool,
                    sample_weight_train,
                    _examples,
                    programs_sa,
                    program_prior_log_probs_opt,
                ) = self._build_and_process_train_matrix(
                    train_demo_ids=train_demo_ids,
                    val_demo_ids=val_demo_ids,
                    demo_dict_train=demo_dict_train,
                    programs_sa=programs_sa,
                    program_prior_log_probs_opt=program_prior_log_probs_opt,
                    dsl_functions=dsl_functions,
                    negative_sampling_cfg=negative_sampling_cfg,
                    offline_path_name=offline_path_name,
                    start_index=start_index,
                    feature_display_names=feature_display_names,
                )
                logging.info(
                    "Recovery augmentation round %d rebuilt train matrix: "
                    "X_train.shape=%s rows=%d",
                    round_idx + 1,
                    X_train.shape,
                    X_train.shape[0],
                )
        logging.info("Final programs:")
        for i, prog in enumerate(programs_sa):
            logging.info(f"  Program {i}: {prog}")
        selected_hyperparams = self._select_hyperparams(
            X=X_train,
            y_bool=y_train_bool,
            sample_weight=sample_weight_train,
            programs_sa=programs_sa,
            program_prior_log_probs_opt=program_prior_log_probs_opt,
            demonstrations_train=demonstrations_train,
            demonstrations_val=demonstrations_val,
            dsl_functions=dsl_functions,
            negative_sampling_cfg=negative_sampling_cfg,
        )
        (
            X_final,
            y_final_bool,
            sample_weight_final,
            final_programs_sa,
            final_program_priors,
            demonstrations_final,
        ) = self._build_final_training_data(
            train_demo_ids=train_demo_ids,
            val_demo_ids=val_demo_ids,
            demo_dict_val=demo_dict_val,
            programs_sa=programs_sa,
            dsl_functions=dsl_functions,
            negative_sampling_cfg=negative_sampling_cfg,
            offline_path_name=offline_path_name,
            X_train=X_train,
            y_train=y_train,
            sample_weight_train=sample_weight_train,
            program_prior_log_probs_opt=program_prior_log_probs_opt,
            demonstrations_train=demonstrations_train,
            demonstrations_val=demonstrations_val,
        )
        logging.info(
            "Retraining final policy on train+val with hyperparameters=%s.",
            selected_hyperparams,
        )
        return self._train_policy_from_matrix(
            X_final,
            y_final_bool,
            sample_weight_final,
            final_programs_sa,
            final_program_priors,
            demonstrations_final,
            dsl_functions,
            negative_sampling_cfg=negative_sampling_cfg,
            train_hyperparams=selected_hyperparams,
            tie_break_demonstrations=demonstrations_val,
        )

    def test_policy_on_envs(
        self,
        base_class_name: str,
        test_env_nums: Sequence[int] = range(11, 20),
        max_num_steps: int = 50,
        record_videos: bool = False,
        video_format: str | None = "mp4",
    ) -> list[bool]:
        """Train the logical programmatic policy using demonstrations."""
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        accuracies = []
        for i in test_env_nums:
            env = self.env_factory(i)
            video_out_path = None
            if record_videos and video_format is not None:
                try:
                    output_dir = Path(HydraConfig.get().runtime.output_dir)
                except Exception:  # pylint: disable=broad-exception-caught
                    output_dir = Path.cwd()
                video_dir = output_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                video_out_path = str(
                    video_dir / f"lfd_{base_class_name}_env{i}.{video_format}"
                )
                logging.info(
                    "Recording test rollout for env %s to %s", i, video_out_path
                )
            assert self._policy is not None, "Policy must be trained before testing."
            reward, terminated, final_info = run_single_episode(
                env,
                self._policy,
                max_num_steps=max_num_steps,
                record_video=record_videos,
                video_out_path=video_out_path,
                reset_seed=i,
            )
            result = infer_episode_success(
                reward=float(reward),
                terminated=bool(terminated),
                action_mode=action_mode,
                base_class_name=base_class_name,
                final_info=final_info,
            )
            accuracies.append(result)
        return accuracies

    def _get_action(self) -> _ActType:
        assert self._policy is not None, "Call reset() first."
        assert self._last_observation is not None
        # Use the logical policy to select an action
        return self._policy(self._last_observation)
