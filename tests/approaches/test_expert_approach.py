"""Tests for ExpertApproach class."""

import numpy as np

from programmatic_policy_learning.approaches.expert_approach import ExpertApproach


def test_expert_approach_step_and_reset():

    class DummySpace:
        def seed(self, seed):
            pass

    obs_space = DummySpace()
    act_space = DummySpace()

    def dummy_expert(obs):
        return np.sum(obs)

    approach = ExpertApproach(
        "DummyEnv", obs_space, act_space, seed=0, expert_fn=dummy_expert
    )
    obs = np.array([[1, 2], [3, 4]])
    info = {}
    approach.reset(obs, info)
    action = approach.step()
    assert action == np.sum(obs)
