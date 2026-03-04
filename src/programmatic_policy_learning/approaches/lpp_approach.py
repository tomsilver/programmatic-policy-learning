"""An approach that learns a logical programmatic policy from data."""

import ast
import hashlib
import json
import logging
import os
import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar, cast

import numpy as np
from gymnasium.spaces import Space
from hydra.core.hydra_config import HydraConfig

# from omegaconf import DictConfig
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel
from scipy.sparse import hstack
from scipy.special import logsumexp

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.approaches.experts.grid_experts import get_grid_expert
from programmatic_policy_learning.approaches.utils import (
    assert_features_fire,
    build_collision_repair_prompt,
    build_dnf_failure_payload,
    build_dnf_failure_prompt,
    convert_dir_lists_to_tuples,
    gini_gain_per_feature,
    load_hint_text,
    load_unique_hint,
    log_feature_collisions,
    log_plp_violation_counts,
    run_single_episode,
)
from programmatic_policy_learning.data.collect import get_demonstrations
from programmatic_policy_learning.data.dataset import (
    extract_examples_from_demonstration,
    run_all_programs_on_demonstrations,
    run_programs_on_examples,
)
from programmatic_policy_learning.dsl.generators.grammar_based_generator import (
    Grammar,
    GrammarBasedProgramGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based import (
    grid_encoder,
    grid_hint_config,
    trajectory_serializer,
    transition_analyzer,
)
from programmatic_policy_learning.dsl.llm_primitives.feature_generator import (
    LLMFeatureGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.feature_priors import (
    compute_feature_log_probs,
)
from programmatic_policy_learning.dsl.llm_primitives.hint_generation.llm_based import (
    hint_extractor,
)
from programmatic_policy_learning.dsl.llm_primitives.llm_generator import (
    LLMPrimitivesGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.py_feature_generator import (
    PyFeatureGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.utils import (
    JSONStructureRepromptCheck,
)
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    GridInput,
    LocalProgram,
    create_grammar,
    get_dsl_functions_dict,
    make_ablated_dsl,
    make_dsl,
)
from programmatic_policy_learning.dsl.state_action_program import (
    StateActionProgram,
    set_dsl_functions,
)
from programmatic_policy_learning.learning.decision_tree_learner import learn_plps
from programmatic_policy_learning.learning.particles_utils import select_particles
from programmatic_policy_learning.learning.plp_likelihood import compute_likelihood_plps
from programmatic_policy_learning.learning.prior_calculation import (
    priors_from_features,
    priors_from_features_v2,
)
from programmatic_policy_learning.policies.lpp_policy import LPPPolicy

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")
EnvFactory = Callable[[int | None], Any]


REPO_ROOT = Path(__file__).resolve().parents[1]
HINTS_ROOT = (
    REPO_ROOT / "dsl" / "llm_primitives" / "hint_generation" / "llm_based" / "new_hints"
)


def build_py_feature_functions(
    feature_programs: list[str],
    dsl_functions: dict[str, Any],
) -> dict[str, Any]:
    """Build a dict of feature function names to callables from source
    strings."""
    functions: dict[str, Any] = {}
    for source in feature_programs:

        tree = ast.parse(source)
        func_names = [
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        ]
        if not func_names:
            raise ValueError("Expected at least one function definition in feature.")

        module_globals = dict(dsl_functions)
        exec(source, module_globals)  # pylint: disable=exec-used
        for name in func_names:
            fn = module_globals.get(name)
            if not callable(fn):
                raise ValueError(f"Feature function '{name}' is not callable.")
            functions[name] = fn
    return functions


def _extract_feature_names(feature_programs: list[str]) -> list[str]:
    """Extract function names from feature source strings."""
    names: list[str] = []
    for source in feature_programs:
        tree = ast.parse(source)
        func_names = [
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        ]
        if not func_names:
            raise ValueError("Expected at least one function definition in feature.")
        names.extend(func_names)
    return names


def _flatten_boolop(node: ast.AST, op_type: type[ast.boolop]) -> list[ast.AST]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, op_type):
        out: list[ast.AST] = []
        for v in node.values:
            out.extend(_flatten_boolop(v, op_type))
        return out
    return [node]


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(child) for child in children)


def compute_program_structural_complexity(program: Any) -> dict[str, int]:
    """Compute structural complexity metrics from program syntax only."""
    expr = str(program).strip()
    try:
        tree = ast.parse(expr, mode="eval")
        root: ast.AST = tree.body
    except SyntaxError:
        return {
            "num_clauses": 1,
            "total_literals": 1,
            "max_clause_len": 1,
            "depth": 1,
            "ops": 0,
        }

    clauses = _flatten_boolop(root, ast.Or)
    num_clauses = max(1, len(clauses))
    clause_lit_counts: list[int] = []
    for clause in clauses:
        lits = _flatten_boolop(clause, ast.And)
        clause_lit_counts.append(max(1, len(lits)))

    total_literals = int(sum(clause_lit_counts)) if clause_lit_counts else 1
    max_clause_len = int(max(clause_lit_counts)) if clause_lit_counts else 1
    depth = _ast_depth(root)

    ops = 0
    for node in ast.walk(root):
        if isinstance(node, ast.BoolOp):
            ops += max(0, len(node.values) - 1)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            ops += 1

    return {
        "num_clauses": num_clauses,
        "total_literals": total_literals,
        "max_clause_len": max_clause_len,
        "depth": depth,
        "ops": int(ops),
    }


def compute_program_structural_log_prior(
    program: Any,
    *,
    alpha: float,
    w_clauses: float,
    w_literals: float,
    w_max_clause: float,
    w_depth: float,
    w_ops: float,
) -> float:
    """Structural log-prior term: -alpha * C_struct(program)."""
    c = compute_program_structural_complexity(program)
    cost = (
        w_clauses * c["num_clauses"]
        + w_literals * c["total_literals"]
        + w_max_clause * c["max_clause_len"]
        + w_depth * c["depth"]
        + w_ops * c["ops"]
    )
    return -float(alpha) * float(cost)


def _run_structural_prior_sanity_checks() -> None:
    """Unit-test-like sanity checks for structural prior monotonicity."""
    cfg = {
        "alpha": 1.0,
        "w_clauses": 1.0,
        "w_literals": 1.0,
        "w_max_clause": 1.0,
        "w_depth": 1.0,
        "w_ops": 1.0,
    }
    p_short = "f1(s, a)"
    p_long = "(f1(s, a) and f2(s, a)) or (f3(s, a) and f4(s, a))"
    lp_short = compute_program_structural_log_prior(p_short, **cfg)
    lp_long = compute_program_structural_log_prior(p_long, **cfg)
    assert lp_short > lp_long

    p_two = "f1(s, a) or f2(s, a)"
    p_three = "f1(s, a) or f2(s, a) or f3(s, a)"
    lp_two = compute_program_structural_log_prior(p_two, **cfg)
    lp_three = compute_program_structural_log_prior(p_three, **cfg)
    assert lp_two > lp_three


def split_dataset(
    demo_numbers: Sequence[int],
    *,
    val_frac: float | None = None,
    val_size: int | None = None,
    split_seed: int = 0,
    split_strategy: str = "random",
    preserve_ordering: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split demo ids into train_core/val deterministically."""
    if split_strategy != "random":
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")

    demo_ids: list[int] = []
    seen: set[int] = set()
    for d in demo_numbers:
        dd = int(d)
        if dd not in seen:
            seen.add(dd)
            demo_ids.append(dd)
    n = len(demo_ids)
    if n == 0:
        return tuple(), tuple()

    if val_size is not None:
        if val_size < 0 or val_size >= n:
            raise ValueError("val_size must be in [0, len(demo_numbers)-1].")
        n_val = int(val_size)
    else:
        frac = 0.0 if val_frac is None else float(val_frac)
        if frac < 0.0 or frac >= 1.0:
            raise ValueError("val_frac must be in [0.0, 1.0).")
        n_val = int(round(n * frac))

    if n_val <= 0:
        return tuple(demo_ids), tuple()

    work = list(demo_ids)
    if not preserve_ordering:
        rng = np.random.default_rng(split_seed)
        rng.shuffle(work)
    val_ids = tuple(work[:n_val])
    train_ids = tuple(work[n_val:])
    if len(train_ids) == 0:
        raise ValueError("Split produced empty train_core set.")
    return train_ids, val_ids


def _hash_state(obs: np.ndarray) -> str:
    arr = np.asarray(obs)
    hasher = hashlib.sha1()
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _count_states_and_expanded_examples(
    demo_ids: Sequence[int],
    demo_dict: dict[int, Any],
    *,
    data_imbalance: dict[str, Any] | None = None,
) -> tuple[int, int]:
    num_states = 0
    expanded = 0
    for demo_id in demo_ids:
        traj = demo_dict[int(demo_id)]
        num_states += len(traj.steps)
        pos, neg = extract_examples_from_demonstration(
            traj, data_imbalance=data_imbalance
        )
        expanded += len(pos) + len(neg)
    return num_states, expanded


def _assert_state_disjointness(
    demo_dict: dict[int, Any],
    train_ids: Sequence[int],
    val_ids: Sequence[int],
) -> None:
    train_hashes: set[str] = set()
    val_hashes: set[str] = set()
    for demo_id in train_ids:
        for obs, _ in demo_dict[int(demo_id)].steps:
            train_hashes.add(_hash_state(obs))
    for demo_id in val_ids:
        for obs, _ in demo_dict[int(demo_id)].steps:
            val_hashes.add(_hash_state(obs))
    overlap = train_hashes.intersection(val_hashes)
    if overlap:
        raise AssertionError(
            "train_core/val state leakage detected via state hash overlap."
        )


def _constant_feature_cols(
    X_csr: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = X_csr.shape[0]  # num examples
    # nnz per column = how many rows have a nonzero entry in that feature
    col_nnz = np.asarray(X_csr.getnnz(axis=0)).ravel()
    all_zero = np.where(col_nnz == 0)[0]
    all_one = np.where(col_nnz == n)[0]  # only valid if X is truly binary 0/1
    return all_zero, all_one, col_nnz


def _filter_constant_features(
    X: Any,
    programs_sa: list[StateActionProgram],
    program_prior_log_probs: list[float] | None,
    *,
    round_idx: int | None = None,
) -> tuple[Any, list[StateActionProgram], list[float] | None, np.ndarray]:
    all_zero, all_one, col_nnz = _constant_feature_cols(X)
    remove = np.unique(np.concatenate([all_zero, all_one]))
    if remove.size > 0:
        keep_mask = np.ones(X.shape[1], dtype=bool)
        keep_mask[remove] = False
        X = X[:, keep_mask]
        programs_sa = [p for i, p in enumerate(programs_sa) if keep_mask[i]]
        if program_prior_log_probs is not None:
            program_prior_log_probs = [
                lp for i, lp in enumerate(program_prior_log_probs) if keep_mask[i]
            ]
        if round_idx is None:
            logging.info("Filtered constant features. New X shape: %s", X.shape)
        else:
            logging.info(
                "Filtered constant features after feedback round %d. "
                "New X shape: %s",
                round_idx,
                X.shape,
            )
    return X, programs_sa, program_prior_log_probs, col_nnz


def _append_new_features_from_sources(
    X: Any,
    programs_sa: list[StateActionProgram],
    program_prior_log_probs: list[float] | None,
    dsl_functions: dict[str, Any],
    new_feature_sources: list[str],
    examples: list[tuple[np.ndarray, tuple[int, int]]],
    *,
    start_index: int,
    collision_loop_idx: int,
    prior_version: str = "v1",
    prior_beta: float = 1.0,
) -> tuple[Any, int]:
    dsl_functions.update(build_py_feature_functions(new_feature_sources, dsl_functions))
    new_feature_names = _extract_feature_names(new_feature_sources)
    new_programs = [f"{name}(s, a)" for name in new_feature_names]
    new_programs_sa = [StateActionProgram(p) for p in new_programs]
    X_new = run_programs_on_examples(
        new_programs_sa,
        examples,
        dsl_functions,
        feature_sources=new_feature_sources,
        collision_loop_idx=collision_loop_idx,
    )
    X = hstack([X, X_new]).tocsr()
    programs_sa.extend(new_programs_sa)
    if program_prior_log_probs is not None:
        if prior_version == "v2":
            new_priors = priors_from_features_v2(new_feature_sources, beta=prior_beta)[
                "beta_log_scores"
            ]
        elif prior_version in {"v1", "uniform"}:
            new_priors = priors_from_features(new_feature_sources)["logprobs"]
        else:
            raise ValueError(f"Unsupported prior_version: {prior_version}")
        program_prior_log_probs.extend(new_priors)
    return X, start_index + len(new_feature_names)


def _run_collision_feedback_loop(
    *,
    collision_groups: list[dict[str, Any]],
    examples: list[tuple[np.ndarray, tuple[int, int]]],
    max_rounds: int,
    target_collisions: int,
    start_index: int,
    program_prior_log_probs: list[float] | None,
    X: Any,
    y: np.ndarray | None,
    programs_sa: list[StateActionProgram],
    dsl_functions: dict[str, Any],
    generate_features: Callable[
        [str, int, int], tuple[list[str], dict[str, Any], Path]
    ],
    make_prompt: Callable[
        [list[dict[str, Any]], list[tuple[np.ndarray, tuple[int, int]]]], str | None
    ],
    prior_version: str = "v1",
    prior_beta: float = 1.0,
) -> tuple[
    Any,
    list[StateActionProgram],
    list[float] | None,
    list[dict[str, Any]],
    Path | None,
    np.ndarray,
]:
    collision_payloads: list[dict[str, Any]] = []
    collision_output_path: Path | None = None
    col_nnz = np.asarray(X.getnnz(axis=0)).ravel()
    for round_idx in range(max_rounds):
        num_collisions = len(collision_groups) if collision_groups else 0
        if num_collisions <= target_collisions:
            logging.info(
                "Collision feedback stopping: %d <= target %d.",
                num_collisions,
                target_collisions,
            )
            break
        prompt = make_prompt(collision_groups, examples)
        if prompt is None:
            break
        prompt = f"{prompt}\n\nCOLLISION_FEEDBACK_ROUND: {round_idx + 1}\n"
        new_feature_sources, collision_payload, output_path = generate_features(
            prompt, start_index, round_idx + 1
        )
        # print(f"Generated new features: {new_feature_sources}")
        # input()
        collision_payloads.append(collision_payload)
        collision_output_path = output_path

        if not new_feature_sources:
            logging.info("No new features generated; stopping feedback loop.")
            break
        X, start_index = _append_new_features_from_sources(
            X,
            programs_sa,
            program_prior_log_probs,
            dsl_functions,
            new_feature_sources,
            examples,
            start_index=start_index,
            collision_loop_idx=round_idx + 1,
            prior_version=prior_version,
            prior_beta=prior_beta,
        )
        X, programs_sa, program_prior_log_probs, col_nnz = _filter_constant_features(
            X, programs_sa, program_prior_log_probs, round_idx=round_idx + 1
        )
        collision_groups = log_feature_collisions(X, y, examples)
        logging.info(
            "Collision groups after feedback round %d: %d",
            round_idx + 1,
            len(collision_groups) if collision_groups else 0,
        )
    return (
        X,
        programs_sa,
        program_prior_log_probs,
        collision_payloads,
        collision_output_path,
        col_nnz,
    )


def get_program_set(
    num_programs: int,
    base_class_name: str,  # pylint: disable=unused-argument
    env_factory: EnvFactory,
    env_specs: dict[str, Any] | None = None,
    start_symbol: int = 0,
    program_generation: dict[str, Any] | None = None,
    demo_numbers: Sequence[int] | None = None,
    outer_feedback: str | None = None,
    seed: int = 0,
    prior_version: str = "v1",
    prior_beta: float = 1.0,
) -> tuple[list[Any], list[float], dict[str, Any]]:
    """Enumerate programs from the grammar and return programs + prior log-
    probs.

    This helper creates the DSL and the grammar-based generator, then
    samples `num_programs` programs from the generator. It returns a tuple of
    (programs, program_prior_log_probs).
    """
    if program_generation is None:
        raise ValueError(
            "program_generation configuration is required for LPP approach."
        )
    strategy = program_generation["strategy"]

    # Define strategies as a dictionary
    strategies = {
        "fixed_grid_v1": lambda: _generate_with_fixed_grid_v1(env_specs, start_symbol),
        "dsl_generator": lambda: _generate_with_dsl_generator(
            program_generation, env_specs, start_symbol, env_factory, outer_feedback
        ),
        "grid_v1_ablated": lambda: _generate_with_ablated_grid_v1(
            program_generation, env_specs, start_symbol
        ),
        "offline_loader": lambda: _generate_with_offline_loader(
            program_generation, env_specs, start_symbol
        ),
    }
    llm_model = (program_generation or {}).get("llm_model", "gpt-4.1")

    if strategy == "feature_generator":
        cache_path = Path("feature_cache.db")
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm_client = OpenAIModel(llm_model, cache)
        prompt_path = program_generation["feature_generator_prompt"]
        num_features = program_generation["num_features"]
        env = env_factory(0)
        object_types = env.get_object_types()
        generator = LLMFeatureGenerator(llm_client)
        hint_text = load_hint_text(
            base_class_name,
            program_generation["encoding_method"],
            program_generation["hint_structured"],
            HINTS_ROOT,
        )

        features, payload = generator.generate(
            prompt_path=prompt_path,
            object_types=object_types,
            hint_text=hint_text,
            num_features=num_features,
        )
        features = convert_dir_lists_to_tuples(features)

        program_prior_log_probs = compute_feature_log_probs(payload, object_types)
        dsl_fns = get_dsl_functions_dict()
        return features, program_prior_log_probs, dsl_fns
    if strategy == "py_feature_gen":
        cache_path = Path("py_feature_cache.db")
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm_client = OpenAIModel(llm_model, cache)
        prompt_path = program_generation["py_feature_gen_prompt"]
        batch_prompt_path = program_generation.get("py_feature_gen_batch_prompt")
        enc_method = str(program_generation["encoding_method"])
        enc_id = enc_method.replace("enc_", "")
        num_features = program_generation["num_features"]
        num_batches = program_generation.get("num_batches")
        py_generator = PyFeatureGenerator(llm_client)
        hint_text = load_unique_hint(base_class_name, HINTS_ROOT)

        demo_text: str | None = None
        try:
            expert = get_grid_expert(base_class_name)
            trajectories: list[list[tuple[Any, Any, Any]]] = []
            demos_included = program_generation.get("demos_included")
            if demos_included is None:
                demo_ids = list(demo_numbers) if demo_numbers is not None else [0]
            else:
                demo_ids = list(demos_included)
            for init_idx in demo_ids:
                env_demo = env_factory(init_idx)
                traj = hint_extractor.collect_full_episode(
                    env_demo, expert, max_steps=40, sample_count=None
                )
                env_demo.close()
                trajectories.append(traj)
            symbol_map = grid_hint_config.get_symbol_map(base_class_name)
            enc_cfg = grid_encoder.GridStateEncoderConfig(
                symbol_map=symbol_map,
                empty_token="empty",
                coordinate_style="rc",
            )
            encoder = grid_encoder.GridStateEncoder(enc_cfg)
            analyzer = transition_analyzer.GenericTransitionAnalyzer()
            salient_tokens = grid_hint_config.SALIENT_TOKENS[base_class_name]
            all_traj_texts: list[str] = []
            for i, traj in enumerate(trajectories):
                if enc_method == "enc_1":
                    text = trajectory_serializer.trajectory_to_diff_text(
                        traj,
                        encoder=encoder,
                        max_steps=50,
                    )
                else:
                    # ENCODING 2-6
                    text = trajectory_serializer.trajectory_to_text(
                        traj,
                        encoder=encoder,
                        analyzer=analyzer,
                        salient_tokens=salient_tokens,
                        encoding_method=enc_id,
                        max_steps=50,
                    )
                all_traj_texts.append(f"\n---[TRAJECTORY {i}]---\n{text}\n\n")
            demo_text = "\n\n".join(all_traj_texts)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.info("Failed to build demonstration text: %s", exc)

        features, payload = py_generator.generate(
            prompt_path=prompt_path,
            batch_prompt_path=batch_prompt_path,
            hint_text=hint_text,
            num_features=num_features,
            num_batches=num_batches,
            env_name=base_class_name,
            demonstration_data=demo_text,
            encoding_method=enc_method,
            _seed=seed,
            reprompt_checks=[JSONStructureRepromptCheck(required_fields=["features"])],
            loading=program_generation.get("loading"),
        )
        dsl_fns = get_dsl_functions_dict()
        dsl_fns.update(
            build_py_feature_functions(features, dsl_fns)
        )  # pylint: disable=exec-used
        if prior_version == "v2":
            out_dict = priors_from_features_v2(features, beta=prior_beta)
            program_prior_log_probs = out_dict["beta_log_scores"]
        elif prior_version in {"v1", "uniform"}:
            out_dict = priors_from_features(features)
            program_prior_log_probs = out_dict["logprobs"]
        else:
            raise ValueError(f"Unsupported prior_version: {prior_version}")
        return features, program_prior_log_probs, dsl_fns

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Call the appropriate strategy
    program_generator, dsl_fns = strategies[strategy]()
    # Generate programs using the shared helper function
    programs, program_prior_log_probs = _generate_programs(
        program_generator, num_programs
    )
    return programs, program_prior_log_probs, dsl_fns


def _generate_with_fixed_grid_v1(
    env_specs: dict[str, Any] | None, start_symbol: int
) -> tuple[GrammarBasedProgramGenerator, dict[str, Any]]:
    """Generate programs using the fixed grid_v1 DSL."""

    dsl = make_dsl()
    dsl_dict = get_dsl_functions_dict()

    program_generator = GrammarBasedProgramGenerator(
        cast(
            Callable[[dict[str, Any]], Grammar[LocalProgram, GridInput, Any]],
            create_grammar,
        ),
        dsl,
        env_spec=env_specs if env_specs is not None else {},
        start_symbol=start_symbol,
    )
    return program_generator, dsl_dict


def _generate_with_ablated_grid_v1(
    program_generation: dict[str, Any],
    env_specs: dict[str, Any] | None,
    start_symbol: int,
) -> tuple[GrammarBasedProgramGenerator, dict[str, Any]]:
    """Generate programs using the ablated grid_v1 DSL."""
    removed_primitive = program_generation["removed_primitive"]
    dsl = make_ablated_dsl(removed_primitive)
    dsl_dict = get_dsl_functions_dict(removed_primitive)
    program_generator = GrammarBasedProgramGenerator(
        cast(
            Callable[[dict[str, Any]], Grammar[LocalProgram, GridInput, Any]],
            create_grammar,
        ),
        dsl,
        env_spec=env_specs if env_specs is not None else {},
        start_symbol=start_symbol,
        removed_primitive=removed_primitive,
    )
    return program_generator, dsl_dict


def _generate_with_dsl_generator(
    program_generation: dict[str, Any],
    env_specs: dict[str, Any] | None,
    start_symbol: int,
    env_factory: EnvFactory,
    outer_feedback: str | None,
) -> tuple[GrammarBasedProgramGenerator, dict[str, Any]]:
    """Generate programs using the DSL generator."""
    cache_path = Path("llm_cache.db")
    # cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_model = program_generation.get("llm_model", "gpt-4.1")
    llm_client = OpenAIModel(llm_model, cache)
    prompt_path = program_generation["dsl_generator_prompt"]
    with open(
        prompt_path,
        "r",
        encoding="utf-8",
    ) as file:
        prompt = file.read()
    removed_primitive = program_generation["removed_primitive"]
    generator = LLMPrimitivesGenerator(llm_client, removed_primitive)
    if env_specs is None:
        raise ValueError("env_specs cannot be None when indexing.")
    _, new_dsl_dict, dsl = generator.generate_and_process_grammar(
        prompt,
        env_specs["object_types"],
        env_factory,  # type: ignore
        "full",
        outer_feedback,
    )

    program_generator = GrammarBasedProgramGenerator(
        generator.create_grammar,
        dsl,
        env_spec=env_specs if env_specs is not None else {},
        start_symbol=start_symbol,
    )

    return program_generator, new_dsl_dict


def _generate_with_offline_loader(
    program_generation: dict[str, Any],
    env_specs: dict[str, Any] | None,
    start_symbol: int,
) -> tuple[GrammarBasedProgramGenerator, dict[str, Any]]:
    """Generate programs using the offline loader."""
    run_id = program_generation["offline_loader_run_id"]
    removed_primitive = program_generation["removed_primitive"]
    generator = LLMPrimitivesGenerator(None, removed_primitive)
    _, new_dsl_dict, dsl = generator.offline_loader(run_id)
    program_generator = GrammarBasedProgramGenerator(
        generator.create_grammar,
        dsl,
        env_spec=env_specs if env_specs is not None else {},
        start_symbol=start_symbol,
    )
    return program_generator, new_dsl_dict


def _generate_programs(
    program_generator: GrammarBasedProgramGenerator, num_programs: int
) -> tuple[list[Any], list[float]]:
    """Shared logic for generating programs."""
    logging.info(f"Generating {num_programs} programs")

    programs: list[Any] = []
    program_prior_log_probs = []
    gen = program_generator.generate_programs()
    for _ in range(num_programs):
        try:
            program, prior = next(gen)
        except StopIteration:
            logging.info(
                f"Generator exhausted early — only produced {len(programs)} programs."
            )
            break
        programs.append(program)
        program_prior_log_probs.append(prior)
    return programs, program_prior_log_probs


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
        _run_structural_prior_sanity_checks()
        self.val_frac = val_frac
        self.val_size = val_size
        self.split_seed = split_seed
        self.split_strategy = split_strategy
        self.preserve_ordering = preserve_ordering

    def configure_rng(self) -> None:
        """Seed Python/NumPy RNGs for deterministic rollouts."""
        random.seed(self.seed_num)
        np.random.seed(self.seed_num)

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

        llm_model = (self.program_generation or {}).get("llm_model", "gpt-4.1")
        cache_path = Path("py_feature_cache.db")
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm_client = OpenAIModel(llm_model, cache)
        py_generator = PyFeatureGenerator(llm_client)

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
        examples: list[tuple[np.ndarray, tuple[int, int]]] | None,
    ) -> str | None:
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

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        self._policy = self._train_policy()
        self._timestep = 0

    def _train_policy(self) -> LPPPolicy:
        """Train the logical programmatic policy using demonstrations."""

        outer_feedback = None
        train_core_demos, val_demos = split_dataset(
            self.demo_numbers,
            val_frac=self.val_frac,
            val_size=self.val_size,
            split_seed=self.split_seed,
            split_strategy=self.split_strategy,
            preserve_ordering=self.preserve_ordering,
        )
        if set(train_core_demos).intersection(set(val_demos)):
            raise AssertionError("train_core and val demo IDs are not disjoint.")
        logging.info("Split train_core demos: %s", list(train_core_demos))
        logging.info("Split val demos: %s", list(val_demos))
        (
            programs,
            program_prior_log_probs_init,
            dsl_functions,
        ) = get_program_set(
            self.num_programs,
            self.base_class_name,
            self.env_factory,
            env_specs=self.env_specs,
            start_symbol=self.start_symbol,
            program_generation=self.program_generation,
            demo_numbers=train_core_demos,
            outer_feedback=outer_feedback,
            seed=self.seed_num,
            prior_version=self.prior_version,
            prior_beta=self.prior_beta,
        )
        program_prior_log_probs_opt: list[float] | None = program_prior_log_probs_init
        start_index = len(programs) + 1
        logging.info("Feature Generation is Done.")
        logging.info("%d features are genereted!", len(programs))

        demonstrations, demo_dict_train = get_demonstrations(
            self.env_factory,
            self.expert,
            demo_numbers=train_core_demos,
        )
        demo_dict_all = dict(demo_dict_train)
        if val_demos:
            _, demo_dict_val = get_demonstrations(
                self.env_factory,
                self.expert,
                demo_numbers=val_demos,
            )
            demo_dict_all.update(demo_dict_val)

        if self.program_generation is None:
            raise ValueError("program_generation config is required.")

        if self.program_generation["strategy"] == "py_feature_gen":
            feature_names = _extract_feature_names(list(programs))
            programs = [f"{name}(s, a)" for name in feature_names]

        programs_sa: list[StateActionProgram] = [
            StateActionProgram(p) for p in programs
        ]

        offline_path_name = None
        loading_cfg = (self.program_generation or {}).get("loading")
        if isinstance(loading_cfg, Mapping) and loading_cfg.get("offline"):
            offline_path_name = loading_cfg.get("offline_json_path")
        data_imbalance_cfg = (self.program_generation or {}).get("data_imbalance")

        _assert_state_disjointness(demo_dict_all, train_core_demos, val_demos)
        train_states, train_expanded = _count_states_and_expanded_examples(
            train_core_demos,
            demo_dict_all,
            data_imbalance=data_imbalance_cfg,
        )
        val_states, val_expanded = _count_states_and_expanded_examples(
            val_demos,
            demo_dict_all,
            data_imbalance=data_imbalance_cfg,
        )
        logging.info(
            "Split stats | train_core: demos=%d states=%d expanded_examples=%d",
            len(train_core_demos),
            train_states,
            train_expanded,
        )
        logging.info(
            "Split stats | val: demos=%d states=%d expanded_examples=%d",
            len(val_demos),
            val_states,
            val_expanded,
        )

        X, y, examples = run_all_programs_on_demonstrations(
            self.base_class_name,
            train_core_demos,
            programs_sa,
            demo_dict_train,
            dsl_functions,
            data_imbalance=data_imbalance_cfg,
            return_examples=True,
            offline_path_name=offline_path_name,
            demos_included=(self.program_generation or {}).get("demos_included"),
            split_tag=(
                f"seed{self.split_seed}"
                f"_train_{'-'.join(str(x) for x in train_core_demos)}"
                f"__val_{'-'.join(str(x) for x in val_demos) if val_demos else 'none'}"
                "__role_train_core"
            ),
            seed=self.seed_num,
        )

        if X is None:
            raise ValueError(
                "X is None. Ensure the program execution results are valid."
            )

        ##########################################
        ############ ANALYSIS ON DATA ############
        ##########################################

        all_zero, all_one, col_nnz = _constant_feature_cols(X)
        logging.info(f"n_examples={X.shape[0]} n_features={X.shape[1]}")
        logging.info(f"#all-zero features={len(all_zero)} indices={all_zero[:30]}")
        logging.info(f"#all-one features={len(all_one)} indices={all_one[:30]}")
        X, programs_sa, program_prior_log_probs_opt, col_nnz = (
            _filter_constant_features(X, programs_sa, program_prior_log_probs_opt)
        )

        collision_groups = log_feature_collisions(
            X,
            y,
            examples,
        )

        ################### COLLISION FEEDBACK LOOP ####################
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
            ) = _run_collision_feedback_loop(
                collision_groups=collision_groups,
                examples=examples,
                max_rounds=max_rounds,
                target_collisions=(self.collision_feedback_target_collisions),
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
                all_payload = {"collision_payloads": collision_payloads}
                out_path = (
                    collision_output_path / "py_feature_payload_collision_all.json"
                )
                out_path.write_text(json.dumps(all_payload, indent=4), encoding="utf-8")
        logging.info("Data after collision feedback loop: X shape %s", X.shape)

        n = X.shape[0]
        logging.info(f"N={n}")
        freq = col_nnz / n  # fraction of examples where feature is on
        rare = np.where(freq <= 0.05)[0]  # almost always 0
        common = np.where(freq >= 0.95)[0]  # almost always 1
        logging.info(f"Almost-always-0={len(rare)}")
        logging.info(f"Almost-always-1={len(common)}")
        assert_features_fire(X, programs_sa)
        y_bool: list[bool] = list(y.astype(bool).flatten()) if y is not None else []

        pos = sum(y_bool)
        neg = len(y_bool) - pos
        logging.info(
            f"y: n={len(y_bool)} pos={pos} ({100 * pos / len(y_bool):.2f}%) neg={neg}"
        )

        #########################################
        ############ END OF ANALYSIS ############
        #########################################

        gain = gini_gain_per_feature(X, y_bool)
        ranked_cols = np.argsort(-gain)  # descending
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
            num_dts=self.num_dts,
            program_generation_step_size=self.program_generation_step_size,
            dt_splitter=self.dt_splitter,
            cc_alpha=self.cc_alpha,
            dsl_functions=dsl_functions,
        )
        # plps, plp_priors = learn_plps(
        #     X,
        #     y_bool,
        #     programs_sa,
        #     program_prior_log_probs,
        #     num_dts=self.num_dts,
        #     program_generation_step_size=self.program_generation_step_size,
        #     dsl_functions=dsl_functions,
        # )
        if self.prior_version == "uniform":
            plp_priors = [-4.0] * len(plps)
        logging.info(f"LEN BEFORE FILTERING FALSE={len(plps)}")
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

        logging.info(f"LEN AFTER FILTERING FALSE={len(plps)}")

        valid_plps = log_plp_violation_counts(plps, demonstrations, dsl_functions)
        logging.info(f"LEN AFTER filtering violations={len(valid_plps)}")

        plps = valid_plps

        if self.permissive_filter_enabled:
            if (
                self.permissive_filter_max_avg_frac is None
                and self.permissive_filter_max_avg_count is None
            ):
                logging.info(
                    "Permissive PLP filter enabled but no thresholds provided; skipping."
                )
            else:
                set_dsl_functions(dsl_functions)
                permissive_filtered_plps: list[StateActionProgram] = []
                total_steps = len(demonstrations.steps)
                logging.info(
                    "Applying permissive PLP filter on %d PLPs over %d steps.",
                    len(plps),
                    total_steps,
                )
                for plp in plps:
                    total_allowed = 0
                    total_frac = 0.0
                    valid = True
                    for obs, _ in demonstrations.steps:
                        try:
                            rows, cols = obs.shape[:2]
                            n_actions = rows * cols
                            allowed = 0
                            for r in range(rows):
                                for c in range(cols):
                                    if plp(obs, (r, c)):
                                        allowed += 1
                            total_allowed += allowed
                            total_frac += (allowed / n_actions) if n_actions else 0.0
                        except Exception:  # pylint: disable=broad-exception-caught
                            valid = False
                            break
                    if not valid or total_steps == 0:
                        continue
                    avg_allowed = total_allowed / total_steps
                    avg_frac = total_frac / total_steps
                    if (
                        self.permissive_filter_max_avg_count is not None
                        and avg_allowed > self.permissive_filter_max_avg_count
                    ):
                        continue
                    if (
                        self.permissive_filter_max_avg_frac is not None
                        and avg_frac > self.permissive_filter_max_avg_frac
                    ):
                        continue
                    permissive_filtered_plps.append(plp)
                logging.info(
                    "Permissive PLP filter kept %d/%d PLPs.",
                    len(permissive_filtered_plps),
                    len(plps),
                )
                plps = permissive_filtered_plps

        likelihoods = compute_likelihood_plps(plps, demonstrations, dsl_functions)
        logging.info(f"LIKELIHOODS: {likelihoods}")
        logging.info(f"PRIORS: {plp_priors}")
        logging.info(
            "Structural prior cfg: alpha=%s w_clauses=%s w_literals=%s "
            "w_max_clause=%s w_depth=%s w_ops=%s",
            self.alpha,
            self.w_clauses,
            self.w_literals,
            self.w_max_clause,
            self.w_depth,
            self.w_ops,
        )
        particles = []
        particle_log_probs = []
        program_rows: list[dict[str, Any]] = []
        for plp, prior, likelihood in zip(plps, plp_priors, likelihoods):
            complexity = compute_program_structural_complexity(plp)
            struct_prior = compute_program_structural_log_prior(
                plp,
                alpha=self.alpha,
                w_clauses=self.w_clauses,
                w_literals=self.w_literals,
                w_max_clause=self.w_max_clause,
                w_depth=self.w_depth,
                w_ops=self.w_ops,
            )
            final_log_weight = prior + likelihood + struct_prior
            particles.append(plp)
            particle_log_probs.append(final_log_weight)
            program_rows.append(
                {
                    "program": str(plp),
                    "num_clauses": complexity["num_clauses"],
                    "total_literals": complexity["total_literals"],
                    "max_clause_len": complexity["max_clause_len"],
                    "depth": complexity["depth"],
                    "ops": complexity["ops"],
                    "logP_features": float(prior),
                    "logP_struct": float(struct_prior),
                    "log_likelihood": float(likelihood),
                    "log_weight": float(final_log_weight),
                }
            )

        if program_rows:
            top_rows = sorted(program_rows, key=lambda r: r["log_weight"], reverse=True)[
                :5
            ]
            for rank, row in enumerate(top_rows, start=1):
                logging.info(
                    "Top-%d program stats | clauses=%d literals=%d max_clause=%d "
                    "depth=%d ops=%d logP_features=%.4f logP_struct=%.4f "
                    "log_likelihood=%.4f final_log_weight=%.4f",
                    rank,
                    row["num_clauses"],
                    row["total_literals"],
                    row["max_clause_len"],
                    row["depth"],
                    row["ops"],
                    row["logP_features"],
                    row["logP_struct"],
                    row["log_likelihood"],
                    row["log_weight"],
                )
                logging.info("Top-%d program: %s", rank, row["program"])

        # logging.info(particle_log_probs)
        probs_arr = np.asarray(particle_log_probs)
        max_val = probs_arr.max()
        max_indices = np.flatnonzero(probs_arr == max_val)
        map_idx = int(np.random.choice(max_indices))
        logging.info(map_idx)  # MAYBE MIXTURE?
        logging.info(f"MAP program ({particle_log_probs[map_idx]}):")
        logging.info(particles[map_idx])

        top_particles, top_particle_log_probs = select_particles(
            particles, particle_log_probs, self.max_num_particles
        )
        policy: LPPPolicy
        if len(top_particle_log_probs) > 0:
            top_particle_log_probs = np.array(top_particle_log_probs) - logsumexp(
                top_particle_log_probs
            )
            top_particle_probs = np.exp(top_particle_log_probs)
            logging.info(f"top_particle_probs: {top_particle_probs}")
            # policy = LPPPolicy(top_particles, top_particle_probs)
            policy = LPPPolicy(
                top_particles,
                top_particle_probs,
                normalize_plp_actions=self.normalize_plp_actions,
            )
            policy.map_program = str(particles[map_idx])
            policy.map_posterior = particle_log_probs[map_idx]
        else:
            logging.info("no nontrivial particles found")
            # policy = LPPPolicy([StateActionProgram("False")], [1.0])
            policy = LPPPolicy(
                [StateActionProgram("False")],
                [1.0],
                normalize_plp_actions=self.normalize_plp_actions,
            )

        if os.getenv("DEBUG_LPP_FEEDBACK", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }:
            if examples is None:
                logging.info("DEBUG_LPP_FEEDBACK: examples missing; skipping feedback.")
            else:
                feature_fns = {
                    name: fn
                    for name, fn in dsl_functions.items()
                    if re.match(r"^f\d+$", name)
                }
                policy_str = (
                    policy.map_program
                    if getattr(policy, "map_program", None)
                    else str(policy)
                )
                payload = build_dnf_failure_payload(
                    policy_str=policy_str,
                    examples=examples,
                    y=y_bool,
                    feature_fns=feature_fns,
                )
                prompt = build_dnf_failure_prompt(  # pylint: disable=unused-variable
                    payload, examples
                )
                # logging.info("DEBUG_LPP_FEEDBACK payload: %s", payload)
                # logging.info("DEBUG_LPP_FEEDBACK prompt:\n%s", prompt)

        return policy

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
            result = (
                run_single_episode(
                    env,
                    self._policy,
                    max_num_steps=max_num_steps,
                    record_video=record_videos,
                    video_out_path=video_out_path,
                )
                > 0
            )
            accuracies.append(result)
            env.close()
        return accuracies

    def _get_action(self) -> _ActType:
        assert self._policy is not None, "Call reset() first."
        assert self._last_observation is not None
        # Use the logical policy to select an action
        return self._policy(self._last_observation)
