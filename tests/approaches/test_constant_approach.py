# test_constant_approach.py

import gymnasium

from programmatic_policy_learning.approaches.constant_approach import ConstantApproach


def test_constant_approach():
    """Tests that the ConstantApproach always returns the same action."""

    env = gymnasium.make("LunarLander-v3")
    approach = ConstantApproach(
        "N/A", env.observation_space, env.action_space, seed=123
    )
    obs, info = env.reset()
    approach.reset(obs, info)

    # Get the first action from the approach
    first_action = approach.step()

    # Loop and and check each action is identical to the first one.
    for i in range(10):
        action = approach.step()
        assert action == first_action
        obs, reward, terminated, _, info = env.step(action)
        approach.update(obs, reward, terminated, info)