"""Tests for the Code-as-Policies baseline helpers."""

import os
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str((Path(".pytest_cache") / "matplotlib").resolve())
)

from programmatic_policy_learning.dsl.llm_primitives.baselines.llm_based.CaP_baseline import (  # pylint: disable=wrong-import-position
    _compile_policy_function,
    _evaluate_policy_function,
    _reset_policy_function_state,
)


def test_compile_policy_function_resolves_helper_functions() -> None:
    """Generated policies can call helpers defined in the same code block."""

    code = """
def helper(obs):
    return np.array([obs, math.floor(obs + 0.5)])

def policy(obs):
    return helper(obs)
"""

    fn = _compile_policy_function(code, "policy")

    np.testing.assert_array_equal(fn(1.2), np.array([1.2, 1.0]))


def test_compile_policy_function_resolves_policy_name() -> None:
    """Generated policies can refer to their own function name."""

    code = """
def policy(obs):
    if obs <= 0:
        return 0
    return policy(obs - 1) + 1
"""

    fn = _compile_policy_function(code, "policy")

    assert fn(3) == 3


def test_reset_policy_function_state_clears_generated_episode_memory() -> None:
    """Generated policies should not leak function-attribute state across
    envs."""

    code = """
def policy(obs):
    st = getattr(policy, "_st", None)
    if st is None:
        st = {"calls": 0}
        policy._st = st
    st["calls"] += 1
    policy._last_debug = {"calls": st["calls"]}
    return st["calls"]
"""

    fn = _compile_policy_function(code, "policy")
    assert fn(None) == 1
    assert fn(None) == 2

    _reset_policy_function_state(fn)

    assert fn(None) == 1


def test_evaluate_policy_function_resets_with_env_index_for_policy_and_expert() -> None:
    """CaP eval should use the same reset seeds as LPP test rollouts."""

    created_envs: list[object] = []

    class FakeActionSpace:
        def sample(self) -> int:
            return 0

    class FakeEnv:
        action_space = FakeActionSpace()

        def __init__(self, env_idx: int) -> None:
            self.env_idx = env_idx
            self.reset_seed: int | None = None

        def reset(self, seed: int | None = None) -> tuple[int, dict[str, int | None]]:
            self.reset_seed = seed
            return self.env_idx, {"seed": seed}

        def step(self, _action: int) -> tuple[int, float, bool, bool, dict[str, int]]:
            return self.env_idx, 1.0, True, False, {}

        def close(self) -> None:
            return None

    def env_builder(env_idx: int) -> FakeEnv:
        env = FakeEnv(env_idx)
        created_envs.append(env)
        return env

    cap_results, expert_results = _evaluate_policy_function(
        lambda _obs: 0,
        env_builder,
        [11, 12],
        max_num_steps=5,
        run_expert=True,
        expert_fn=lambda _obs: 0,
        env_type="grid",
    )

    assert cap_results == [True, True]
    assert expert_results == [True, True]
    assert [env.reset_seed for env in created_envs] == [11, 11, 12, 12]
