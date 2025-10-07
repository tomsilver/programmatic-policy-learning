"""Integration test for dataset pipeline with real environment and
generator."""

import functools
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from programmatic_policy_learning.data.dataset import (
    Trajectory,
    run_all_programs_on_demonstrations,
)
from programmatic_policy_learning.dsl.core import DSL
from programmatic_policy_learning.dsl.generators.grammar_based_generator import (
    GrammarBasedProgramGenerator,
)
from programmatic_policy_learning.dsl.primitives_sets import grid_v1
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import create_grammar
from programmatic_policy_learning.envs.providers.ggg_provider import create_ggg_env


def eval_program(state: np.ndarray, action: tuple[int, int], prog_str: str) -> Any:
    """Eval."""
    eval_globals = grid_v1.__dict__.copy()

    class MockTPN:
        """Mock TPN."""

        EMPTY = 0
        TOKEN = 1

    eval_globals.update(
        {
            "tpn": MockTPN,
            "s": state,
            "a": action,
            "state": state,
            "action": action,
            "__builtins__": __builtins__,
        }
    )
    return eval(prog_str, eval_globals)


def test_dataset_pipeline_with_real_env() -> None:
    """Test full dataset pipeline with real env and grammar-based generator."""
    # Setup environment
    base_class_name = "TwoPileNim"
    num_programs = 2
    demo_numbers = [0]
    cfg = OmegaConf.create({"provider": "ggg", "make_kwargs": {"id": "TwoPileNim0-v0"}})
    env = create_ggg_env(cfg)
    # Setup DSL and program generator
    dsl: DSL[str, Any, Any] = DSL(
        id="grid_v1", primitives={}, evaluate_fn=lambda p, i: p
    )
    program_generator = GrammarBasedProgramGenerator(
        create_grammar,
        dsl,
        env_spec={"object_types": env.get_object_types()},
        start_symbol=6,
    )
    # Generate programs using the grammar-based generator
    program_strs = []
    gen = program_generator.generate_programs()
    for _ in range(num_programs):
        prog_str, log_prior = next(gen)
        program_strs.append(prog_str)

    assert isinstance(prog_str, str)
    assert isinstance(log_prior, float)
    assert np.isfinite(log_prior)

    # Create top-level callables using functools.partial - needs work
    programs = [functools.partial(eval_program, prog_str=s) for s in program_strs]

    print("Programs:")
    for i, prog in enumerate(program_strs):
        print(f"Program {i}: {prog}")

    # Collect demonstrations (simulate a single demo)
    state = np.zeros((2, 2))
    action = (0, 1)
    demonstration: Trajectory[np.ndarray, tuple[int, int]] = Trajectory(
        steps=[(state, action)]
    )

    print("\nDemonstrations:")
    for i, (s, a) in enumerate(demonstration.steps):
        print(f"Demo {i}: state=\n{s}, action={a}")

    # Run dataset pipeline
    X, y = run_all_programs_on_demonstrations(
        base_class_name=base_class_name,
        demo_numbers=demo_numbers,
        programs=programs,
        demonstrations=demonstration,
    )
    if X is not None and y is not None:
        print("\nDataset X (features matrix):")
        print(X.toarray())
        print("\nDataset y (labels):")
        print(y)
        assert X.shape[0] == len(y)
        assert X.shape[1] == num_programs
        assert set(y.tolist()) <= {0, 1}
