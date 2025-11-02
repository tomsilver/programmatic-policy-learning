"""An approach that learns a logical programmatic policy from data."""

import logging
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
from gymnasium.spaces import Space

# from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
# from prpl_llm_utils.models import OpenAIModel
from scipy.special import logsumexp

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.approaches.utils import run_single_episode
from programmatic_policy_learning.data.collect import get_demonstrations
from programmatic_policy_learning.data.dataset import run_all_programs_on_demonstrations
from programmatic_policy_learning.dsl.generators.grammar_based_generator import (
    GrammarBasedProgramGenerator,
)
from programmatic_policy_learning.dsl.llm_primitives.llm_generator import (
    LLMPrimitivesGenerator,
)

# from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
#     GridInput,
#     LocalProgram,
#     create_grammar,
#     get_dsl_functions_dict,
#     make_dsl,
# )
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram
from programmatic_policy_learning.learning.decision_tree_learner import learn_plps
from programmatic_policy_learning.learning.particles_utils import select_particles
from programmatic_policy_learning.learning.plp_likelihood import compute_likelihood_plps
from programmatic_policy_learning.policies.lpp_policy import LPPPolicy

# from programmatic_policy_learning.utils.cache_utils import manage_cache

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")
EnvFactory = Callable[[], Any]


def key_fn_for_train_policy(args: tuple, kwargs: dict) -> str:
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


def key_fn_for_program_generation(args: tuple, kwargs: dict) -> str:
    """Return cache run id, excluding 'env_specs' which is too long."""
    # join all positional args except index 2 and all kwargs except
    # 'env_specs'.
    parts = [str(a) for i, a in enumerate(args) if i != 2]
    parts += [str(v) for k, v in kwargs.items() if k != "env_specs"]
    return "-".join(parts)


# @manage_cache("cache", [".pkl", ".pkl"], key_fn=key_fn_for_program_generation)
def get_program_set(
    num_programs: int,
    base_class_name: str,  # pylint: disable=unused-argument
    env_specs: dict[str, Any] | None = None,
    start_symbol: int = 0,
) -> tuple[list, list, dict]:
    """Enumerate programs from the grammar and return programs + prior log-
    probs.

    This helper creates the DSL and the grammar-based generator, then
    samples `num_programs` programs from the generator. It returns a tuple of
    (programs, program_prior_log_probs).
    """
    # dsl = make_dsl()
    # new_dsl_dict = dsl

    # cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    # cache = SQLite3PretrainedLargeModelCache(cache_path)
    # llm_client = OpenAIModel("gpt-4o-mini", cache)

    # with open(
    #     "src/programmatic_policy_learning/dsl/llm_primitives/prompts/"
    #     + "one_missing_prompt.txt",
    #     # "one_missing_prompt+helpersAllowed.txt",
    #     "r",
    #     encoding="utf-8",
    # ) as file:
    #     prompt = file.read()

    # object_types = ["tpn.EMPTY", "tpn.TOKEN", "None"]

    generator = LLMPrimitivesGenerator(None)
    # grammar, updated_get_dsl_callable, new_dsl_object = (
    #     generator.generate_and_process_grammar(prompt, object_types)
    # )

    _, updated_get_dsl_callable, new_dsl_object = generator.offline_loader(
        "2025-10-30_13-39-03"
    )  # grammar, _
    dsl = new_dsl_object
    new_dsl_dict = updated_get_dsl_callable()

    program_generator = GrammarBasedProgramGenerator(
        generator.create_grammar,
        # cast(
        #     Callable[[dict[str, Any]], Grammar[LocalProgram, GridInput, Any]],
        #     create_grammar,
        # ),
        dsl,
        env_spec=env_specs if env_specs is not None else {},
        start_symbol=start_symbol,
    )

    logging.info(f"Generating {num_programs} programs")

    programs = []
    program_prior_log_probs = []
    gen = program_generator.generate_programs()
    for _ in range(num_programs):
        program, prior = next(gen)
        programs.append(program)
        program_prior_log_probs.append(prior)

    if new_dsl_dict is None:
        raise ValueError("new_dsl_dict is not being created properly.")

    return programs, program_prior_log_probs, new_dsl_dict


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

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        self._policy = self._train_policy()
        self._timestep = 0

    # @manage_cache("cache", ".pkl", key_fn=key_fn_for_train_policy)
    def _train_policy(self) -> LPPPolicy:
        """Train the logical programmatic policy using demonstrations."""
        programs, program_prior_log_probs, dsl_functions = get_program_set(
            self.num_programs,
            self.base_class_name,
            env_specs=self.env_specs,
            start_symbol=self.start_symbol,
        )
        logging.info("Programs Generation is Done.")

        programs_sa: list[StateActionProgram] = [
            StateActionProgram(p) for p in programs
        ]
        demonstrations, demo_dict = get_demonstrations(
            self.env_factory, self.expert, demo_numbers=self.demo_numbers
        )
        # dsl_functions = get_dsl_functions_dict()
        print(dsl_functions)
        X, y = run_all_programs_on_demonstrations(
            self.base_class_name,
            self.demo_numbers,
            programs_sa,
            demo_dict,
            dsl_functions,
        )
        # print(f"Shape of X: {X.shape}")
        # print(f"Shape of y: {y.shape}")

        if X is None:
            raise ValueError(
                "X is None. Ensure the program execution results are valid."
            )
        print(f"Contents of X: {X.toarray()}")
        is_all_false = np.all(X.toarray() is False)
        print(f"Is X all False? {is_all_false}")
        # print(f"Contents of y: {y}")
        # print(f"Unique classes in y: {np.unique(y)}")
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
        likelihoods = compute_likelihood_plps(plps, demonstrations)

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
