"""An approach that learns a logical programmatic policy from data."""
import numpy as np
from typing import Any, TypeVar

from gymnasium.spaces import Space
from scipy.special import logsumexp

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.dsl.state_action_program import StateActionProgram
from programmatic_policy_learning.policies.lpp_policy import LPPPolicy
from programmatic_policy_learning.learning.decision_tree_learner import learn_plps
from programmatic_policy_learning.learning.plp_likelihood import compute_likelihood_plps
from programmatic_policy_learning.learning.particles_utils import select_particles
from programmatic_policy_learning.dsl.generators.grammar_based_generator import GrammarBasedProgramGenerator
from programmatic_policy_learning.dsl.primitives_sets.grid_v1 import create_grammar
from programmatic_policy_learning.dsl.core import DSL
from programmatic_policy_learning.data.collect import collect_demo
from programmatic_policy_learning.data.demo_types import Trajectory
from programmatic_policy_learning.data.dataset import run_all_programs_on_demonstrations
_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class LogicProgrammaticPolicyApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that learns a logical programmatic policy from data."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
        base_class_name: str = "",
        demo_numbers: list[int] = [],
        program_generation_step_size: int = 10,
        num_programs: int = 100,
        num_dts: int = 5,
        max_num_particles: int = 10,
    ) -> None:
        """LPP APProach."""
        super().__init__(environment_description, observation_space, action_space, seed)
        self._policy: LPPPolicy | None = None
        self.base_class_name = base_class_name
        self.demo_numbers = demo_numbers
        self.program_generation_step_size = program_generation_step_size
        self.num_programs = num_programs
        self.num_dts = num_dts
        self.max_num_particles = max_num_particles

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        self._policy = self._train_policy()
        self._timestep = 0

    def _train_policy(self) -> LPPPolicy:
        """Train the logical programmatic policy using demonstrations."""
        dsl = DSL(...)  # Fill in with your primitives and eval function

        program_generator = GrammarBasedProgramGenerator(
            create_grammar, dsl, env_spec={}, start_symbol=...  # Fill in start symbol
        )

        # Generate programs and their priors
        programs = []
        program_prior_log_probs = []
        gen = program_generator.generate_programs()
        for _ in range(self.num_programs):
            program, prior = next(gen)
            programs.append(program)
            program_prior_log_probs.append(prior)

        traj: Trajectory = collect_demo(env_factory, expert, max_demo_length=10)
        X, y = run_all_programs_on_demonstrations(self.base_class_name, self.demo_numbers,programs, traj)
        plps, plp_priors = learn_plps(
            X, y, programs, program_prior_log_probs,
            num_dts=self.num_dts,
            program_generation_step_size=self.program_generation_step_size
        )
        # demonstrations = get_demonstrations(self.base_class_name, demo_numbers=self.demo_numbers)
        likelihoods = compute_likelihood_plps(plps, traj)

        particles = []
        particle_log_probs = []
        for plp, prior, likelihood in zip(plps, plp_priors, likelihoods):
            particles.append(plp)
            particle_log_probs.append(prior + likelihood)

        map_idx = np.argmax(particle_log_probs).squeeze()
        print(f"MAP program ({particle_log_probs[map_idx]}):")
        print(particles[map_idx])

        top_particles, top_particle_log_probs = select_particles(particles, particle_log_probs, self.max_num_particles)
        if len(top_particle_log_probs) > 0:
            top_particle_log_probs = np.array(top_particle_log_probs) - logsumexp(top_particle_log_probs)
            top_particle_probs = np.exp(top_particle_log_probs)
            print("top_particle_probs:", top_particle_probs)
            policy = LPPPolicy(top_particles, top_particle_probs)
        else:
            print("no nontrivial particles found")
            policy = LPPPolicy([StateActionProgram("False")], [1.0])

        return policy
    
    def _get_action(self) -> _ActType:
        assert self._policy is not None, "Call reset() first."
        assert self._last_observation is not None
        # Use the logical policy to select an action
        return self._policy(self._last_observation)
