"""Tests for LPP Approach."""

from omegaconf import DictConfig, OmegaConf

from programmatic_policy_learning.approaches.expert_approach import ExpertApproach
from programmatic_policy_learning.approaches.experts.grid_experts import get_grid_expert
from programmatic_policy_learning.approaches.lpp_approach import (
    LogicProgrammaticPolicyApproach,
)
from programmatic_policy_learning.envs.registry import EnvRegistry


def test_lpp_approach_real_data() -> None:
    """Test lpp approach with real_data."""
    base_class_name = "TwoPileNim"
    cfg: DictConfig = OmegaConf.create(
        {
            "provider": "ggg",
            "make_kwargs": {"id": "TwoPileNim0-v0"},
        }
    )
    registry = EnvRegistry()
    env_factory = lambda: registry.load(cfg)
    env = env_factory()  # type: ignore
    expert_fn = get_grid_expert(base_class_name)
    expert = ExpertApproach(  # type: ignore
        base_class_name,
        env.observation_space,
        env.action_space,
        seed=1,
        expert_fn=expert_fn,
    )

    # Define observation and action spaces
    observation_space = env.observation_space
    action_space = env.action_space

    # Environment specifications
    env_specs = {"object_types": env.get_object_types()}

    # Initialize the approach
    approach = LogicProgrammaticPolicyApproach(
        environment_description=base_class_name,
        observation_space=observation_space,
        action_space=action_space,
        seed=42,
        env_factory=env_factory,
        expert=expert,
        base_class_name=base_class_name,
        demo_numbers=(0, 1),
        num_programs=2,
        num_dts=1,
        max_num_particles=2,
        max_demo_length=100,
        env_specs=env_specs,
        start_symbol=0,
    )

    obs = env.reset()[0]
    info = env.reset()[1]
    approach.reset(obs, info)
    # pylint: disable=protected-access
    action = approach._get_action()
    assert isinstance(action, tuple)
    assert len(action) == 2
    assert all(isinstance(x, int) for x in action)
