"""An approach that learns a logical programmatic policy from data."""

import logging
from typing import Any, Callable, TypeVar, cast

import numpy as np
from gymnasium.spaces import Space
from scipy.special import logsumexp

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.data.collect import get_demonstrations
from programmatic_policy_learning.data.dataset import run_all_programs_on_demonstrations
from programmatic_policy_learning.dsl.generators.grammar_based_generator import (
    Grammar,
    GrammarBasedProgramGenerator,
)
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import (
    GridInput,
    LocalProgram,
    create_grammar,
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

    def _train_policy(self) -> LPPPolicy:
        """Train the logical programmatic policy using demonstrations."""
        dsl = make_dsl()
        program_generator = GrammarBasedProgramGenerator(
            cast(
                Callable[[dict[str, Any]], Grammar[LocalProgram, GridInput, Any]],
                create_grammar,
            ),  # short-term fix
            dsl,
            env_spec=self.env_specs,
            start_symbol=self.start_symbol,
        )

        logging.info(f"Generating {self.num_programs} programs")

        # Generate programs and their priors
        programs = []
        program_prior_log_probs = []
        gen = program_generator.generate_programs()
        for _ in range(self.num_programs):
            program, prior = next(gen)
            programs.append(program)
            program_prior_log_probs.append(prior)

        programs_sa: list[StateActionProgram] = [
            StateActionProgram(p) for p in programs
        ]
        demonstrations, demo_dict = get_demonstrations(
            self.env_factory, self.expert, demo_numbers=self.demo_numbers
        )

        X, y = run_all_programs_on_demonstrations(
            self._environment_description,
            self.demo_numbers,
            programs_sa,
            demo_dict,
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

    def _get_action(self) -> _ActType:
        assert self._policy is not None, "Call reset() first."
        assert self._last_observation is not None
        # Use the logical policy to select an action
        return self._policy(self._last_observation)
