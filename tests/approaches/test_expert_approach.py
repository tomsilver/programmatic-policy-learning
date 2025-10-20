"""Tests for ExpertApproach class."""

import numpy as np

from programmatic_policy_learning.approaches.expert_approach import ExpertApproach


def test_expert_approach_step_and_reset() -> None:
    """Test reset and step for ExpertApproach."""

    class DummySpace:
        """DummySpace."""

        def seed(self, seed: int) -> None:
            """Seed function."""
            pass  # pylint: disable=unnecessary-pass

    obs_space: DummySpace = DummySpace()
    act_space: DummySpace = DummySpace()

    def dummy_expert(obs: np.ndarray) -> float:
        return np.sum(obs)

    approach: ExpertApproach = ExpertApproach(
        "DummyEnv", obs_space, act_space, seed=0, expert_fn=dummy_expert
    )  # type: ignore[no-untyped-call]
    obs: np.ndarray = np.array([[1, 2], [3, 4]])
    info: dict = {}
    approach.reset(obs, info)  # type: ignore[no-untyped-call]
    action = approach.step()
    assert action == np.sum(obs)
