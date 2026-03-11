"""An approach that learns a logical programmatic policy from data."""

import json
import logging
import random
from itertools import product
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar, cast

import numpy as np
from gymnasium.spaces import Space
from hydra.core.hydra_config import HydraConfig

# from omegaconf import DictConfig
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel
from scipy.sparse import vstack
from scipy.special import logsumexp

from programmatic_policy_learning.approaches.base_approach import BaseApproach

# pylint: disable-next=line-too-long
from programmatic_policy_learning.approaches.lpp_utils.lpp_collision_feedback_utils import (
    run_collision_feedback_loop,
)
from programmatic_policy_learning.approaches.lpp_utils.lpp_feature_source_utils import (
    _extract_feature_names,
)

# pylint: disable-next=line-too-long
from programmatic_policy_learning.approaches.lpp_utils.lpp_program_generation_utils import (
    get_program_set,
)
from programmatic_policy_learning.approaches.lpp_utils.lpp_program_setup_utils import (
    prepare_programs_and_dsl,
)
from programmatic_policy_learning.approaches.lpp_utils.lpp_split_matrix_utils import (
    filter_constant_features,
    split_and_collect_demonstrations,
)
from programmatic_policy_learning.approaches.lpp_utils.utils import (
    assert_features_fire,
    build_collision_repair_prompt,
    gini_gain_per_feature,
    log_feature_collisions,
    log_plp_violation_counts,
    run_single_episode,
)
from programmatic_policy_learning.data.dataset import (
    run_all_programs_on_demonstrations,
)
from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.dsl.llm_primitives.py_feature_generator import (
    PyFeatureGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.utils import (
    JSONStructureRepromptCheck,
)
from programmatic_policy_learning.dsl.state_action_program import (
    StateActionProgram,
)
from programmatic_policy_learning.learning.decision_tree_learner import learn_plps
from programmatic_policy_learning.learning.particles_utils import select_particles
from programmatic_policy_learning.learning.plp_likelihood import compute_likelihood_plps
from programmatic_policy_learning.policies.lpp_policy import LPPPolicy

_filter_constant_features = filter_constant_features

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")
EnvFactory = Callable[[int | None], Any]


class LogicProgrammaticPolicyApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that learns a logical programmatic policy from data."""

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

    def _default_train_hyperparams(self) -> dict[str, Any]:
        return {
            "num_dts": int(self.num_dts),
            "program_generation_step_size": int(self.program_generation_step_size),
            "dt_splitter": self.dt_splitter,
            "cc_alpha": float(self.cc_alpha),
            "dt_max_depth": self.dt_max_depth,
            "max_num_particles": int(self.max_num_particles),
        }

    @staticmethod
    def _normalize_grid_values(raw_values: Sequence[Any] | Any) -> list[Any]:
        if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
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
        }
        unknown = set(self.hyperparam_grid.keys()) - supported
        if unknown:
            unknown_str = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported hyperparameter(s) in hyperparam_grid: {unknown_str}")

        keys = list(self.hyperparam_grid.keys())
        value_lists = [
            self._normalize_grid_values(self.hyperparam_grid[key]) for key in keys
        ]
        candidates: list[dict[str, Any]] = []
        for values in product(*value_lists):
            candidates.append({k: v for k, v in zip(keys, values)})
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
            cache_path = Path("py_feature_cache.db")
            cache = SQLite3PretrainedLargeModelCache(cache_path)
            llm_client = OpenAIModel(llm_model, cache)
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
        print(template_payload)
        expanded_payload = py_generator.expand_template_payload(
            template_payload, env_name=self.base_class_name, start_index=start_index
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
    ) -> str | None:
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        if action_mode != "discrete":
            logging.info(
                "Collision repair prompt generation is grid-specific; "
                "skipping for action_mode=%s.",
                action_mode,
            )
            return None
        if not collision_groups or examples is None:
            return None
        ranked_groups = sorted(
            collision_groups, key=lambda g: g["max_occur"], reverse=True
        )
        logging.info("Collision groups (top 5 by max_occur):")
        for rank, group in enumerate(ranked_groups[:5], start=1):
            pos_list = group["pos"]
            neg_list = group["neg"]
            logging.info(
                "  %d) max_occur=%d pos_count=%d neg_count=%d pos=%s neg=%s",
                rank,
                group["max_occur"],
                len(pos_list),
                len(neg_list),
                pos_list[:5],
                neg_list[:10],
            )
        best_group = ranked_groups[0]
        second_group = ranked_groups[1] if len(ranked_groups) > 1 else None

        prompt = build_collision_repair_prompt(
            pos_indices=best_group["pos"],
            neg_indices=best_group["neg"],
            # examples=cast(list[tuple[np.ndarray, tuple[int, int]]], examples),
            examples=examples,
            env_name=self.base_class_name,
            existing_feature_summary=None,
            max_per_label=5,
            collision_feedback_enc=self.collision_feedback_enc,
            pos_indices_2=second_group["pos"] if second_group else None,
            neg_indices_2=second_group["neg"] if second_group else None,
            seed=self.seed_num,
            collision_template_feedback=self.collision_template_feedback,
        )
        try:
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        except Exception:  # pylint: disable=broad-exception-caught
            output_dir = Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = "collision_prompr_"
        existing = sorted(output_dir.glob(f"{prefix}*.txt"))
        next_idx = len(existing) + 1
        out_path = output_dir / f"{prefix}{next_idx}.txt"
        out_path.write_text(prompt, encoding="utf-8")
        logging.info("Collision repair prompt written to %s", out_path)
        return prompt

    def _train_policy_from_matrix(
        self,
        X: Any,
        y_bool: list[bool],
        programs_sa: list[StateActionProgram],
        program_prior_log_probs_opt: list[float] | None,
        demonstrations: Trajectory[_ObsType, _ActType],
        dsl_functions: dict[str, Any],
        *,
        train_hyperparams: Mapping[str, Any] | None = None,
    ) -> LPPPolicy:
        hp = self._resolve_train_hyperparams(train_hyperparams)
        gain = gini_gain_per_feature(X, y_bool)
        ranked_cols = np.argsort(-gain)
        X_ranked = X[:, ranked_cols]
        programs_ranked = [programs_sa[i] for i in ranked_cols]
        if program_prior_log_probs_opt is None:
            priors_ranked = [0.0 for _ in ranked_cols]
        else:
            priors_ranked = [program_prior_log_probs_opt[i] for i in ranked_cols]

        plps, plp_priors = learn_plps(
            X_ranked,
            y_bool,
            programs_ranked,
            priors_ranked,
            num_dts=int(hp["num_dts"]),
            program_generation_step_size=int(hp["program_generation_step_size"]),
            dt_splitter=str(hp["dt_splitter"]),
            cc_alpha=float(hp["cc_alpha"]),
            dt_max_depth=cast(int | None, hp["dt_max_depth"]),
            dsl_functions=dsl_functions,
        )
        if self.prior_version == "uniform":
            plp_priors = [-4.0] * len(plps)

        logging.info("LEN BEFORE FILTERING FALSE=%d", len(plps))
        filtered: list[tuple[StateActionProgram, float]] = []
        for plp, prior in zip(plps, plp_priors):
            if str(plp).strip() == "False":
                continue
            filtered.append((plp, prior))
        if filtered:
            filtered_plps_tuple, filtered_priors_tuple = zip(*filtered)
            plps = list(filtered_plps_tuple)
            plp_priors = list(filtered_priors_tuple)
        else:
            plps, plp_priors = [], []

        logging.info("LEN AFTER FILTERING FALSE=%d", len(plps))

        valid_plps = log_plp_violation_counts(plps, demonstrations, dsl_functions)
        logging.info("LEN AFTER filtering violations=%d", len(valid_plps))
        plps = valid_plps

        likelihoods = compute_likelihood_plps(
            plps,
            demonstrations,
            dsl_functions,
        )
        logging.info("LIKELIHOODS: %s", likelihoods)
        logging.info("PRIORS: %s", plp_priors)
        particles = []
        particle_log_probs = []
        for plp, prior, likelihood in zip(plps, plp_priors, likelihoods):
            particles.append(plp)
            particle_log_probs.append(prior + likelihood)

        top_particles, top_particle_log_probs = select_particles(
            particles, particle_log_probs, int(hp["max_num_particles"])
        )
        if len(top_particle_log_probs) > 0:
            probs_arr = np.asarray(particle_log_probs)
            max_val = probs_arr.max()
            max_indices = np.flatnonzero(probs_arr == max_val)
            map_idx = int(np.random.choice(max_indices))
            logging.info("MAP program index=%d", map_idx)
            logging.info("MAP program (%s):", particle_log_probs[map_idx])
            logging.info(particles[map_idx])

            top_particle_log_probs = np.array(top_particle_log_probs) - logsumexp(
                top_particle_log_probs
            )
            top_particle_probs = np.exp(top_particle_log_probs)
            logging.info("top_particle_probs: %s", top_particle_probs)
            policy: LPPPolicy = LPPPolicy(
                top_particles,
                top_particle_probs,
                normalize_plp_actions=self.normalize_plp_actions,
                action_mode=str(self.env_specs.get("action_mode", "discrete")),
                action_space=cast(Any, self._action_space),
            )
            policy.map_program = str(particles[map_idx])
            policy.map_posterior = particle_log_probs[map_idx]
            return policy

        logging.info("no nontrivial particles found")
        return LPPPolicy(
            [StateActionProgram("False")],
            [1.0],
            normalize_plp_actions=self.normalize_plp_actions,
            action_mode=str(self.env_specs.get("action_mode", "discrete")),
            action_space=cast(Any, self._action_space),
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
        losses: list[float] = []
        for obs, action in demonstrations.steps:
            prob = float(policy.get_action_prob(obs, action))
            losses.append(-float(np.log(max(prob, eps))))
        return float(np.mean(losses))

    def _get_data_loading_config(
        self,
    ) -> tuple[str | None, dict[str, Any] | None]:
        if self.program_generation is None:
            raise ValueError("program_generation config is required.")
        offline_path_name = None
        loading_cfg = (self.program_generation or {}).get("loading")
        if isinstance(loading_cfg, Mapping) and loading_cfg.get("offline"):
            offline_path_name = loading_cfg.get("offline_json_path")
        raw_data_imbalance_cfg = (self.program_generation or {}).get("data_imbalance")
        data_imbalance_cfg = (
            dict(raw_data_imbalance_cfg)
            if isinstance(raw_data_imbalance_cfg, Mapping)
            else raw_data_imbalance_cfg
        )
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        if (
            action_mode == "continuous"
            and isinstance(data_imbalance_cfg, dict)
            and hasattr(self._action_space, "low")
            and hasattr(self._action_space, "high")
        ):
            data_imbalance_cfg.setdefault(
                "continuous_action_low",
                np.asarray(getattr(self._action_space, "low"), dtype=float).tolist(),
            )
            data_imbalance_cfg.setdefault(
                "continuous_action_high",
                np.asarray(getattr(self._action_space, "high"), dtype=float).tolist(),
            )
        return offline_path_name, data_imbalance_cfg

    def _build_and_process_train_matrix(
        self,
        *,
        train_demo_ids: tuple[int, ...],
        val_demo_ids: tuple[int, ...],
        demo_dict_train: dict[int, Trajectory[_ObsType, _ActType]],
        programs_sa: list[StateActionProgram],
        program_prior_log_probs_opt: list[float] | None,
        dsl_functions: dict[str, Any],
        data_imbalance_cfg: dict[str, Any] | None,
        offline_path_name: str | None,
        start_index: int,
    ) -> tuple[
        Any,
        np.ndarray,
        list[bool],
        list[tuple[_ObsType, _ActType]] | None,
        list[StateActionProgram],
        list[float] | None,
    ]:
        action_mode = str(self.env_specs.get("action_mode", "discrete"))
        val_split_tag = (
            "-".join(str(x) for x in val_demo_ids) if val_demo_ids else "none"
        )
        X, y, examples = run_all_programs_on_demonstrations(
            self.base_class_name,
            train_demo_ids,
            programs_sa,
            demo_dict_train,
            dsl_functions,
            data_imbalance=data_imbalance_cfg,
            return_examples=True,
            offline_path_name=offline_path_name,
            demos_included=(self.program_generation or {}).get("demos_included"),
            split_tag=(
                f"seed{self.split_seed}"
                f"_train_{'-'.join(str(x) for x in train_demo_ids)}"
                f"__val_{val_split_tag}"
                "__role_train_core"
            ),
            seed=self.seed_num,
            action_mode=action_mode,
        )
        if X is None or y is None:
            raise ValueError(
                "Train matrix is invalid. Ensure program execution results are valid."
            )

        logging.info("n_examples=%d n_features=%d", X.shape[0], X.shape[1])
        X, programs_sa, program_prior_log_probs_opt, col_nnz = (
            _filter_constant_features(X, programs_sa, program_prior_log_probs_opt)
        )
        all_zero = np.where(col_nnz == 0)[0]
        all_one = np.where(col_nnz == X.shape[0])[0]
        logging.info("#all-zero features=%d indices=%s", len(all_zero), all_zero[:30])
        logging.info("#all-one features=%d indices=%s", len(all_one), all_one[:30])

        collision_groups = log_feature_collisions(X, y, examples)
        if self.collision_feedback_enabled and examples is not None:
            max_rounds = max(1, self.collision_feedback_max_rounds)

            def _generate_collision_features(
                prompt: str, start_idx: int, collision_idx: int
            ) -> tuple[list[str], dict[str, Any], Path]:
                return self._generate_collision_features(
                    prompt, start_index=start_idx, collision_idx=collision_idx
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
                make_prompt=self._handle_collision_feedback,
                prior_version=self.prior_version,
                prior_beta=self.prior_beta,
            )
            if collision_payloads and collision_output_path is not None:
                out_path = (
                    collision_output_path / "py_feature_payload_collision_all.json"
                )
                out_path.write_text(
                    json.dumps({"collision_payloads": collision_payloads}, indent=4),
                    encoding="utf-8",
                )
        logging.info("Data after collision feedback loop: X shape %s", X.shape)

        n = X.shape[0]
        logging.info("N=%d", n)
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
        return X, y, y_bool, examples, programs_sa, program_prior_log_probs_opt

    def _select_hyperparams(
        self,
        *,
        X: Any,
        y_bool: list[bool],
        programs_sa: list[StateActionProgram],
        program_prior_log_probs_opt: list[float] | None,
        demonstrations_train: Trajectory[_ObsType, _ActType],
        demonstrations_val: Trajectory[_ObsType, _ActType],
        dsl_functions: dict[str, Any],
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
                "Single candidate hyperparameter set=%s; skipping validation model selection.",
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
                programs_sa,
                program_prior_log_probs_opt,
                demonstrations_train,
                dsl_functions,
                train_hyperparams=hp,
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
        data_imbalance_cfg: dict[str, Any] | None,
        offline_path_name: str | None,
        X_train: Any,
        y_train: np.ndarray,
        program_prior_log_probs_opt: list[float] | None,
        demonstrations_train: Trajectory[_ObsType, _ActType],
        demonstrations_val: Trajectory[_ObsType, _ActType],
    ) -> tuple[
        Any,
        list[bool],
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
        if val_demo_ids:
            X_val, y_val, _ = run_all_programs_on_demonstrations(
                self.base_class_name,
                val_demo_ids,
                list(programs_sa),
                demo_dict_val,
                dsl_functions,
                data_imbalance=data_imbalance_cfg,
                return_examples=False,
                offline_path_name=offline_path_name,
                demos_included=(self.program_generation or {}).get("demos_included"),
                split_tag=(
                    f"seed{self.split_seed}"
                    f"_train_{'-'.join(str(x) for x in train_demo_ids)}"
                    f"__val_{'-'.join(str(x) for x in val_demo_ids)}"
                    "__role_val"
                ),
                seed=self.seed_num,
                action_mode=action_mode,
            )
            if X_val is None or y_val is None:
                raise ValueError(
                    "X_val or y_val is None. Ensure the validation dataset is valid."
                )
            X_final = vstack([X_final, X_val]).tocsr()
            y_final = np.concatenate([y_final, y_val])

        final_programs_sa = list(programs_sa)
        final_program_priors = (
            list(program_prior_log_probs_opt)
            if program_prior_log_probs_opt is not None
            else None
        )
        X_final, final_programs_sa, final_program_priors, _ = _filter_constant_features(
            X_final, final_programs_sa, final_program_priors
        )
        y_final_bool: list[bool] = list(y_final.astype(bool).flatten())
        return (
            X_final,
            y_final_bool,
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
        offline_path_name, data_imbalance_cfg = self._get_data_loading_config()

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
            data_imbalance_cfg=data_imbalance_cfg,
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

        (
            programs_sa,
            program_prior_log_probs_opt,
            dsl_functions,
            start_index,
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
        # print(programs_sa)
        # input("PROGRAM GENERATION COMPLETE. Press Enter to continue...")
        (
            X_train,
            y_train,
            y_train_bool,
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
            data_imbalance_cfg=data_imbalance_cfg,
            offline_path_name=offline_path_name,
            start_index=start_index,
        )
        selected_hyperparams = self._select_hyperparams(
            X=X_train,
            y_bool=y_train_bool,
            programs_sa=programs_sa,
            program_prior_log_probs_opt=program_prior_log_probs_opt,
            demonstrations_train=demonstrations_train,
            demonstrations_val=demonstrations_val,
            dsl_functions=dsl_functions,
        )
        (
            X_final,
            y_final_bool,
            final_programs_sa,
            final_program_priors,
            demonstrations_final,
        ) = self._build_final_training_data(
            train_demo_ids=train_demo_ids,
            val_demo_ids=val_demo_ids,
            demo_dict_val=demo_dict_val,
            programs_sa=programs_sa,
            dsl_functions=dsl_functions,
            data_imbalance_cfg=data_imbalance_cfg,
            offline_path_name=offline_path_name,
            X_train=X_train,
            y_train=y_train,
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
            final_programs_sa,
            final_program_priors,
            demonstrations_final,
            dsl_functions,
            train_hyperparams=selected_hyperparams,
        )

    def test_policy_on_envs(
        self,
        base_class_name: str,
        test_env_nums: Sequence[int] = range(11, 20),
        max_num_steps: int = 50,
        record_videos: bool = True,
        video_format: str | None = "mp4",
    ) -> list[bool]:
        """Train the logical programmatic policy using demonstrations."""
        accuracies = []
        for i in test_env_nums:
            env = self.env_factory(i)
            video_out_path = f"/tmp/lfd_{base_class_name}.{video_format}"
            assert self._policy is not None, "Policy must be trained before testing."
            reward, _terminated = run_single_episode(
                env,
                self._policy,
                max_num_steps=max_num_steps,
                record_video=record_videos,
                video_out_path=video_out_path,
            )
            result = reward > 0
            accuracies.append(result)
            env.close()
        return accuracies

    def _get_action(self) -> _ActType:
        assert self._policy is not None, "Call reset() first."
        assert self._last_observation is not None
        # Use the logical policy to select an action
        return self._policy(self._last_observation)
