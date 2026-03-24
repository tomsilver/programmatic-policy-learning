"""Program generation helpers for the LPP approach."""

import ast
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, cast

from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

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


def _resolve_demos_included(
    demo_numbers: Sequence[int] | None,
    program_generation: dict[str, Any],
) -> list[int]:
    demos_included = program_generation.get("demos_included")
    if demos_included is None:
        return list(demo_numbers) if demo_numbers is not None else [0]
    return list(demos_included)


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
) -> tuple[list[Any], list[float], dict[str, Any]]:
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
        # enc_id = enc_method.replace("enc_", "")
        num_features = program_generation["num_features"]
        num_batches = program_generation.get("num_batches")
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

        demo_text: str | None = None
        try:
            trajectories: list[list[tuple[Any, Any, Any]]] = []
            demo_ids = _resolve_demos_included(demo_numbers, program_generation)
            demo_skip_rate = int(program_generation.get("skip_rate", 1))
            if expert is None:
                raise ValueError("No expert instance provided for demo serialization.")
            for init_idx in demo_ids:
                env_demo = env_factory(init_idx)
                reset_seed = int(seed) * 1000 + int(init_idx)
                traj = _collect_full_episode_generic(
                    env_demo,
                    expert,
                    max_steps=200,
                    skip_rate=demo_skip_rate,
                    reset_seed=reset_seed,
                )
                env_demo.close()
                trajectories.append(traj)
            llm_env_spec = get_env_llm_spec(base_class_name, env_specs)
            demo_text = llm_env_spec.serialize_demonstrations(
                trajectories,
                encoding_method=enc_method,
                max_steps=50,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.info("Failed to build demonstration text: %s", exc)

        features, _payload = py_generator.generate(
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
            action_mode=str((env_specs or {}).get("action_mode", "discrete")),
        )
        dsl_fns = get_dsl_functions_dict()
        dsl_fns.update(
            _build_py_feature_functions(features, dsl_fns)
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

    program_generator, dsl_fns = strategies[strategy]()
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
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_model = program_generation.get("llm_model", "gpt-4.1")
    llm_client = OpenAIModel(llm_model, cache)
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
