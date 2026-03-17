"""Program setup helpers for the LPP approach."""

import logging
from typing import Any, Callable

from programmatic_policy_learning.dsl.state_action_program import StateActionProgram


def prepare_programs_and_dsl(
    *,
    num_programs: int,
    base_class_name: str,
    env_factory: Any,
    expert: Any,
    env_specs: dict[str, Any],
    start_symbol: int,
    program_generation: dict[str, Any] | None,
    train_demo_ids: tuple[int, ...],
    outer_feedback: str | None,
    seed_num: int,
    prior_version: str,
    prior_beta: float,
    get_program_set_fn: Callable[..., tuple[list[Any], list[float], dict[str, Any]]],
    extract_feature_names_fn: Callable[[list[str]], list[str]],
) -> tuple[
    list[StateActionProgram],
    list[float] | None,
    dict[str, Any],
    int,
]:
    """Generate candidate programs, convert to StateActionProgram, and return
    DSL."""
    (
        programs,
        program_prior_log_probs_init,
        dsl_functions,
    ) = get_program_set_fn(
        num_programs,
        base_class_name,
        env_factory,
        expert=expert,
        env_specs=env_specs,
        start_symbol=start_symbol,
        program_generation=program_generation,
        demo_numbers=train_demo_ids,
        outer_feedback=outer_feedback,
        seed=seed_num,
        prior_version=prior_version,
        prior_beta=prior_beta,
    )
    program_prior_log_probs_opt: list[float] | None = program_prior_log_probs_init
    start_index = len(programs) + 1
    logging.info("Feature Generation is Done.")
    logging.info("%d features are genereted!", len(programs))

    if program_generation is None:
        raise ValueError("program_generation config is required.")
    if program_generation["strategy"] == "py_feature_gen":
        feature_names = extract_feature_names_fn(list(programs))
        programs = [f"{name}(s, a)" for name in feature_names]

    programs_sa = [StateActionProgram(p) for p in programs]
    return programs_sa, program_prior_log_probs_opt, dsl_functions, start_index
