"""Program generation helpers for the LPP approach."""

import ast
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, cast

from prpl_llm_utils import models as llm_models
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel, PretrainedLargeModel

from programmatic_policy_learning.approaches.lpp_utils.utils import (
    convert_dir_lists_to_tuples,
    load_hint_text,
    load_unique_hint,
)
from programmatic_policy_learning.dsl.generators.grammar_based_generator import (
    Grammar,
    GrammarBasedProgramGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.env_specs import (
    get_env_llm_spec,
)
from programmatic_policy_learning.dsl.llm_primitives.feature_generator import (
    LLMFeatureGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.feature_priors import (
    compute_feature_log_probs,
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
from programmatic_policy_learning.learning.prior_calculation import (
    priors_from_features,
    priors_from_features_v2,
)

EnvFactory = Callable[[int | None], Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
HINTS_ROOT = (
    REPO_ROOT / "dsl" / "llm_primitives" / "hint_generation" / "llm_based" / "new_hints"
)


def _cache_path_with_model(stem: str, llm_model: str) -> Path:
    model_slug = "".join(ch if ch.isalnum() else "_" for ch in llm_model).strip("_")
    return Path(f"{stem}_{model_slug}.db")


def make_llm_client_for_model(
    llm_model: str,
    cache: SQLite3PretrainedLargeModelCache,
    *,
    use_response_model: bool | None = None,
) -> PretrainedLargeModel:
    """Create the correct OpenAI client for chat- or responses-style models."""

    if use_response_model is None:
        use_response_model = "pro" in llm_model.lower()
    if use_response_model:
        response_cls = getattr(llm_models, "OpenAIResponsesModel", None)
        if response_cls is None:
            raise ImportError(
                "OpenAIResponsesModel is not available in prpl_llm_utils. "
                "Install/upgrade the package or set use_response_model=false."
            )
        return response_cls(llm_model, cache)
    return OpenAIModel(llm_model, cache)


def _collect_full_episode_generic(
    env: Any,
    expert: Any,
    *,
    max_steps: int = 200,
    skip_rate: int = 1,
    reset_seed: int | None = None,
) -> list[tuple[Any, Any, Any]]:
    """Roll out an instantiated expert and collect sampled transitions.

    The environment is still stepped at every timestep, but only every
    ``skip_rate``-th transition is retained. The terminal transition is always
    kept so the prompt can still see how the episode ended.
    """
    if hasattr(expert, "set_env"):
        expert.set_env(env)
    try:
        reset_out = env.reset(seed=reset_seed)
    except TypeError:
        reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs, info = reset_out, {}
    expert.reset(obs, info)
    trajectory: list[tuple[Any, Any, Any]] = []
    skip_rate = max(1, int(skip_rate))
    for step_idx in range(max_steps):
        action = expert.step()
        obs_next, reward, terminated, truncated, step_info = env.step(action)
        transition = (obs, action, obs_next)
        if step_idx % skip_rate == 0:
            trajectory.append(transition)
        expert.update(obs_next, reward, terminated, step_info)
        obs = obs_next
        if terminated or truncated:
            if not trajectory or trajectory[-1] is not transition:
                trajectory.append(transition)
            break
    return trajectory


def _resolve_demo_subsets(
    demo_numbers: Sequence[int] | None,
    ensemble_cfg: dict[str, Any],
) -> list[list[int]]:
    """Resolve configured demo subsets for multi-prompt feature generation."""
    explicit_demo_subsets = ensemble_cfg.get("demo_subsets")
    if explicit_demo_subsets is None:
        if demo_numbers is None:
            return [[0]]
        return [list(dict.fromkeys(int(demo_id) for demo_id in demo_numbers))]

    demo_subsets = [
        list(dict.fromkeys(int(demo_id) for demo_id in demo_subset))
        for demo_subset in explicit_demo_subsets
    ]
    demo_subsets = [demo_subset for demo_subset in demo_subsets if demo_subset]
    if not demo_subsets:
        raise ValueError("multi_prompt_ensemble.demo_subsets cannot be empty.")
    return demo_subsets


def _canonicalize_feature_source(source: str) -> str:
    """Normalize source for deduplication across renamed feature defs."""
    tree = ast.parse(source)
    for idx, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef):
            node.name = f"feature_{idx}"
    return ast.unparse(tree).strip()


def _deduplicate_payload_features(
    payload_features: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only structurally unique features from a merged payload."""
    deduped: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for feature in payload_features:
        source = feature.get("source")
        if not isinstance(source, str) or not source.strip():
            continue
        key = _canonicalize_feature_source(source)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        deduped.append(dict(feature))
    return deduped


def _build_serialized_demo_text(
    *,
    base_class_name: str,
    env_specs: dict[str, Any] | None,
    env_factory: EnvFactory,
    expert: Any,
    demo_ids: Sequence[int],
    skip_rate: int,
    enc_method: str,
) -> str:
    """Collect and serialize demonstrations for a specific prompt instance."""
    trajectories: list[list[tuple[Any, Any, Any]]] = []
    for init_idx in demo_ids:
        env_demo = env_factory(init_idx)
        traj = _collect_full_episode_generic(
            env_demo,
            expert,
            max_steps=300,
            skip_rate=skip_rate,
            reset_seed=int(init_idx),
        )
        env_demo.close()
        trajectories.append(traj)
    llm_env_spec = get_env_llm_spec(base_class_name, env_specs)
    return llm_env_spec.serialize_demonstrations(
        trajectories,
        encoding_method=enc_method,
        max_steps=300,
    )


def _generate_py_feature_pool(
    *,
    py_generator: PyFeatureGenerator,
    prompt_path: str,
    hint_text: str,
    num_features: int,
    base_class_name: str,
    env_specs: dict[str, Any] | None,
    env_factory: EnvFactory,
    expert: Any,
    demo_numbers: Sequence[int] | None,
    program_generation: dict[str, Any],
    seed: int,
    loading_cfg: dict[str, Any],
    action_mode: str,
    generation_mode: str,
) -> tuple[list[str], dict[str, Any], list[str]]:
    """Generate one merged feature pool from configured demo subsets."""
    enc_method = str(program_generation["encoding_method"])
    demo_skip_rate = int(program_generation.get("skip_rate", 1))
    ensemble_cfg = dict(program_generation.get("multi_prompt_ensemble") or {})

    if bool(loading_cfg.get("offline", 0)):
        features, payload = py_generator.generate(
            prompt_path=prompt_path,
            hint_text=hint_text,
            num_features=num_features,
            env_name=base_class_name,
            demonstration_data=None,
            encoding_method=enc_method,
            _seed=seed,
            reprompt_checks=(
                None
                if generation_mode == "generator_script"
                else [JSONStructureRepromptCheck(required_fields=["features"])]
            ),
            loading=loading_cfg,
            action_mode=action_mode,
            generation_mode=generation_mode,
        )
        feature_display_names = []
        payload_features = payload.get("features", [])
        if isinstance(payload_features, list):
            for feature in payload_features:
                if not isinstance(feature, dict):
                    continue
                display_name = str(feature.get("name", "")).strip()
                if display_name:
                    feature_display_names.append(display_name)
        return features, payload, feature_display_names

    if expert is None:
        raise ValueError("No expert instance provided for demo serialization.")

    num_seeds_per_subset = max(1, int(ensemble_cfg.get("num_seeds_per_subset", 1)))
    demo_subsets = _resolve_demo_subsets(demo_numbers, ensemble_cfg)

    all_payload_features: list[dict[str, Any]] = []
    call_metadata: list[dict[str, Any]] = []
    next_feature_index = 1
    reprompt_checks: list[Any] | None
    if generation_mode == "generator_script":
        reprompt_checks = None
    else:
        reprompt_checks = [JSONStructureRepromptCheck(required_fields=["features"])]

    total_calls = len(demo_subsets) * num_seeds_per_subset
    logging.info(
        "Running multi-prompt feature generation with %d subsets x %d seeds = %d calls.",
        len(demo_subsets),
        num_seeds_per_subset,
        total_calls,
    )
    for subset_idx, demo_subset in enumerate(demo_subsets):
        demo_text = _build_serialized_demo_text(
            base_class_name=base_class_name,
            env_specs=env_specs,
            env_factory=env_factory,
            expert=expert,
            demo_ids=demo_subset,
            skip_rate=demo_skip_rate,
            enc_method=enc_method,
        )
        for seed_idx in range(num_seeds_per_subset):
            call_seed = int(seed + len(call_metadata))
            call_features, payload = py_generator.generate(
                prompt_path=prompt_path,
                hint_text=hint_text,
                num_features=num_features,
                env_name=base_class_name,
                demonstration_data=demo_text,
                encoding_method=enc_method,
                _seed=call_seed,
                reprompt_checks=reprompt_checks,
                loading=loading_cfg,
                action_mode=action_mode,
                generation_mode=generation_mode,
            )
            renumbered_payload = py_generator.renumber_payload_features(
                payload,
                start_index=next_feature_index,
            )
            next_feature_index += len(renumbered_payload.get("features", []))
            payload_features = renumbered_payload.get("features", [])
            if isinstance(payload_features, list):
                all_payload_features.extend(
                    feature for feature in payload_features if isinstance(feature, dict)
                )
            call_metadata.append(
                {
                    "subset_index": subset_idx,
                    "seed_index": seed_idx,
                    "llm_seed": call_seed,
                    "demo_subset": list(demo_subset),
                    "generated_features": len(call_features),
                }
            )

    deduped_features = _deduplicate_payload_features(all_payload_features)
    merged_payload = py_generator.renumber_payload_features(
        {"features": deduped_features},
        start_index=1,
    )
    merged_sources = py_generator.parse_feature_programs(merged_payload)
    if py_generator.llm_client is not None:
        py_generator.write_json(
            "py_feature_payload_ensemble_merged.json",
            {
                "metadata": {
                    "mode": "multi_prompt_ensemble",
                    "base_seed": seed,
                    "num_demo_subsets": len(demo_subsets),
                    "num_seeds_per_subset": num_seeds_per_subset,
                    "raw_feature_count": len(all_payload_features),
                    "deduped_feature_count": len(deduped_features),
                    "calls": call_metadata,
                },
                "features": merged_payload.get("features", []),
            },
        )

    feature_display_names = []
    for feature in merged_payload.get("features", []):
        if not isinstance(feature, dict):
            continue
        display_name = str(feature.get("name", "")).strip()
        if display_name:
            feature_display_names.append(display_name)
    logging.info(
        "Merged %d raw features into %d deduplicated features.",
        len(all_payload_features),
        len(merged_sources),
    )
    return merged_sources, merged_payload, feature_display_names


def _build_py_feature_functions(
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
        module_globals["math"] = math
        exec(source, module_globals)  # pylint: disable=exec-used
        for name in func_names:
            fn = module_globals.get(name)
            if not callable(fn):
                raise ValueError(f"Feature function '{name}' is not callable.")
            functions[name] = fn
    return functions


def get_program_set(
    num_programs: int,
    base_class_name: str,  # pylint: disable=unused-argument
    env_factory: EnvFactory,
    expert: Any | None = None,
    env_specs: dict[str, Any] | None = None,
    start_symbol: int = 0,
    program_generation: dict[str, Any] | None = None,
    demo_numbers: Sequence[int] | None = None,
    outer_feedback: str | None = None,
    seed: int = 0,
    prior_version: str = "v1",
    prior_beta: float = 1.0,
) -> tuple[list[Any], list[float], dict[str, Any], list[str] | None]:
    """Enumerate programs from the grammar and return programs + prior log-
    probs."""
    if program_generation is None:
        raise ValueError(
            "program_generation configuration is required for LPP approach."
        )
    strategy = program_generation["strategy"]

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
    llm_model = program_generation.get("llm_model", "gpt-4.1")

    if strategy == "feature_generator":
        cache_path = _cache_path_with_model("feature_cache", llm_model)
        cache = SQLite3PretrainedLargeModelCache(cache_path)
        llm_client = make_llm_client_for_model(
            llm_model,
            cache,
            use_response_model=program_generation.get("use_response_model"),
        )
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
        feature_display_names: list[str] = []
        payload_features = payload.get("features", [])
        if isinstance(payload_features, list):
            for feature in payload_features:
                if not isinstance(feature, dict):
                    continue
                display_name = str(feature.get("name", "")).strip()
                if display_name:
                    feature_display_names.append(display_name)
        return features, program_prior_log_probs, dsl_fns, feature_display_names

    if strategy == "py_feature_gen":
        loading_cfg = program_generation.get("loading") or {}
        offline_mode = bool(loading_cfg.get("offline", 0))
        llm_client = None
        if not offline_mode:
            cache_path = _cache_path_with_model("py_feature_cache", llm_model)
            cache = SQLite3PretrainedLargeModelCache(cache_path)
            llm_client = make_llm_client_for_model(
                llm_model,
                cache,
                use_response_model=program_generation.get("use_response_model"),
            )
        prompt_path = program_generation["py_feature_gen_prompt"]
        generation_mode = str(
            program_generation.get("py_feature_gen_mode", "feature_payload")
        )
        num_features = program_generation["num_features"]
        py_generator = PyFeatureGenerator(llm_client)
        try:
            hint_text = load_unique_hint(base_class_name, HINTS_ROOT)
        except FileNotFoundError:
            logging.warning(
                "No hint file found for env '%s' under %s; continuing with empty hints.",
                base_class_name,
                HINTS_ROOT,
            )
            hint_text = ""

        features, _payload, feature_display_names = _generate_py_feature_pool(
            py_generator=py_generator,
            prompt_path=prompt_path,
            hint_text=hint_text,
            num_features=num_features,
            base_class_name=base_class_name,
            env_specs=env_specs,
            env_factory=env_factory,
            expert=expert,
            demo_numbers=demo_numbers,
            program_generation=program_generation,
            seed=seed,
            loading_cfg=loading_cfg,
            action_mode=str((env_specs or {}).get("action_mode", "discrete")),
            generation_mode=generation_mode,
        )
        dsl_fns = get_dsl_functions_dict()
        dsl_fns.update(
            _build_py_feature_functions(features, dsl_fns)
        )  # pylint: disable=exec-used
        if prior_version == "v2":
            out_dict = priors_from_features_v2(features, beta=prior_beta)
            program_prior_log_probs = out_dict["beta_log_scores"]
        elif prior_version in {"v1", "uniform"}:  # TODO
            out_dict = priors_from_features(features)
            program_prior_log_probs = out_dict["logprobs"]
        else:
            raise ValueError(f"Unsupported prior_version: {prior_version}")
        return features, program_prior_log_probs, dsl_fns, feature_display_names

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}")

    program_generator, dsl_fns = strategies[strategy]()
    programs, program_prior_log_probs = _generate_programs(
        program_generator, num_programs
    )
    return programs, program_prior_log_probs, dsl_fns, None


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
    llm_model = program_generation.get("llm_model", "gpt-4.1")
    cache_path = _cache_path_with_model("llm_cache", llm_model)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_client = make_llm_client_for_model(
        llm_model,
        cache,
        use_response_model=program_generation.get("use_response_model"),
    )
    prompt_path = program_generation["dsl_generator_prompt"]
    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()
    removed_primitive = program_generation["removed_primitive"]
    generator = LLMPrimitivesGenerator(llm_client, removed_primitive)
    if env_specs is None:
        raise ValueError("env_specs cannot be None when indexing.")
    _, new_dsl_dict, dsl = generator.generate_and_process_grammar(
        prompt,
        env_specs["object_types"],
        env_factory,  # type: ignore[arg-type]
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
    logging.info("Generating %d programs", num_programs)
    programs: list[Any] = []
    program_prior_log_probs: list[float] = []
    gen = program_generator.generate_programs()
    for _ in range(num_programs):
        try:
            program, prior = next(gen)
        except StopIteration:
            logging.info(
                "Generator exhausted early; only produced %d programs.",
                len(programs),
            )
            break
        programs.append(program)
        program_prior_log_probs.append(prior)
    return programs, program_prior_log_probs
