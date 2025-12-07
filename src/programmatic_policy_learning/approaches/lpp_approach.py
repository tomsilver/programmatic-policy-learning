"""An approach that learns a logical programmatic policy from data."""

import logging
import signal
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Generator, Sequence, TypeVar, cast

import numpy as np
from gymnasium.spaces import Space

# from omegaconf import DictConfig
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel
from scipy.special import logsumexp

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.approaches.utils import run_single_episode
from programmatic_policy_learning.data.collect import get_demonstrations
from programmatic_policy_learning.data.dataset import run_all_programs_on_demonstrations
from programmatic_policy_learning.dsl.generators.grammar_based_generator import (
    Grammar,
    GrammarBasedProgramGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.llm_generator import (
    LLMPrimitivesGenerator,
)
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    GridInput,
    LocalProgram,
    create_grammar,
    get_dsl_functions_dict,
    make_ablated_dsl,
    make_dsl,
)
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram
from programmatic_policy_learning.learning.decision_tree_learner import learn_plps
from programmatic_policy_learning.learning.particles_utils import select_particles
from programmatic_policy_learning.learning.plp_likelihood import compute_likelihood_plps
from programmatic_policy_learning.policies.lpp_policy import LPPPolicy

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")
EnvFactory = Callable[[], Any]


class CustomTimeoutError(Exception):
    """Custom exception raised when a time limit is exceeded.

    This exception is used to indicate that a block of code has exceeded
    the allowed time limit set by the `time_limit` context manager.
    """


@contextmanager
def time_limit(seconds: int) -> Generator[None, None, None]:
    """Context manager to enforce a time limit on a block of code.

    Args:
        seconds (int): The maximum number of seconds to allow the block to run.

    Raises:
        CustomTimeoutError: If the time limit is exceeded.
    """

    def handler(signum: int, frame: FrameType | None) -> None:
        """Signal handler to raise a CustomTimeoutError when the alarm is
        triggered.

        Args:
            signum (int): The signal number.
            frame (FrameType | None): The current stack frame (unused).
        """
        raise CustomTimeoutError(f"Timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def key_fn_for_train_policy(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Build a compact run id from training-related attributes on `self`."""
    self_obj = args[0] if len(args) > 0 else kwargs.get("self")
    if self_obj is None:
        return ""
    demo_numbers = getattr(self_obj, "demo_numbers", ())
    demo_part = "-".join(str(x) for x in demo_numbers)
    parts = [
        str(getattr(self_obj, "base_class_name", "")),
        demo_part,
        str(getattr(self_obj, "program_generation_step_size", "")),
        str(getattr(self_obj, "num_programs", "")),
        str(getattr(self_obj, "num_dts", "")),
        str(getattr(self_obj, "max_num_particles", "")),
    ]
    # filter empty pieces and join
    return "-".join([p for p in parts if p != ""])


def key_fn_for_program_generation(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Return cache run id, excluding 'env_specs' which is too long."""
    # join all positional args except index 2 and all kwargs except
    # 'env_specs'.
    parts = [str(a) for i, a in enumerate(args) if i != 2]
    parts += [str(v) for k, v in kwargs.items() if k != "env_specs"]
    return "-".join(parts)


def get_program_set(
    num_programs: int,
    base_class_name: str,  # pylint: disable=unused-argument
    env_factory: EnvFactory,
    env_specs: dict[str, Any] | None = None,
    start_symbol: int = 0,
    program_generation: dict[str, Any] | None = None,
    outer_feedback: str | None = None,
) -> tuple[list, list, dict]:
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
    print(strategy)
    mode = program_generation["mode"]

    # Define strategies as a dictionary
    strategies = {
        "fixed_grid_v1": lambda: _generate_with_fixed_grid_v1(env_specs, start_symbol),
        "dsl_generator": lambda: _generate_with_dsl_generator(
            program_generation,
            env_specs,
            start_symbol,
            env_factory,
            outer_feedback,
            mode,
        ),
        "grid_v1_ablated": lambda: _generate_with_ablated_grid_v1(
            program_generation, env_specs, start_symbol
        ),
        "offline_loader": lambda: _generate_with_offline_loader(
            program_generation, env_specs, start_symbol
        ),
    }

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
    mode: str = "full",
) -> tuple[GrammarBasedProgramGenerator, dict[str, Any]]:
    """Generate programs using the DSL generator."""
    # cache_path = Path("llm_cache.db")
    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    llm_client = OpenAIModel("gpt-4.1", cache)
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
        outer_feedback,
        mode,
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
    ) -> None:
        """LPP APProach."""
        super().__init__(environment_description, observation_space, action_space, seed)
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

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        self._policy = self._train_policy()
        self._timestep = 0

    def _train_policy(self) -> LPPPolicy:
        """Train the logical programmatic policy using demonstrations."""

        outer_feedback = None
        programs, program_prior_log_probs, dsl_functions = get_program_set(
            self.num_programs,
            self.base_class_name,
            self.env_factory,
            env_specs=self.env_specs,
            start_symbol=self.start_symbol,
            program_generation=self.program_generation,
            outer_feedback=outer_feedback,  # <-- feed into grammar generator
        )

        logging.info("Programs Generation is Done.")
        programs_sa: list[StateActionProgram] = [
            StateActionProgram(p) for p in programs
        ]
        demonstrations, demo_dict = get_demonstrations(
            self.env_factory, self.expert, demo_numbers=self.demo_numbers
        )

        X, y = run_all_programs_on_demonstrations(
            self.base_class_name,
            self.demo_numbers,
            programs_sa,
            demo_dict,
            dsl_functions,
        )
        if X is None:
            raise ValueError(
                "X is None. Ensure the program execution results are valid."
            )

        # Convert y to list[bool] - short term fix
        y_bool: list[bool] = list(y.astype(bool).flatten()) if y is not None else []
        # Convert programs to list[StateActionProgram] - short term fix

        plps, plp_priors = learn_plps(
            X,
            y_bool,
            programs_sa,
            program_prior_log_probs,
            num_dts=self.num_dts,
            program_generation_step_size=self.program_generation_step_size,
            dsl_functions=dsl_functions,
        )
        likelihoods = compute_likelihood_plps(plps, demonstrations, dsl_functions)

        particles = []
        particle_log_probs = []
        for plp, prior, likelihood in zip(plps, plp_priors, likelihoods):
            particles.append(plp)
            particle_log_probs.append(prior + likelihood)

        map_idx = np.argmax(particle_log_probs).squeeze()
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
            logging.info("top_particle_probs: %s", top_particle_probs)
            policy = LPPPolicy(top_particles, top_particle_probs)
            policy.map_program = str(particles[map_idx])
            policy.map_posterior = particle_log_probs[map_idx]
        else:
            logging.info("no nontrivial particles found")
            policy = LPPPolicy([StateActionProgram("False")], [1.0])

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
