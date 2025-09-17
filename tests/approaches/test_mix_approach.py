"""Tests for mix_approach.py."""

import tempfile
from pathlib import Path

import gymnasium
import pytest
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OrderedResponseModel
from prpl_llm_utils.structs import Response

from programmatic_policy_learning.approaches.mix_approach import (
    MixApproach,
)

runllms = pytest.mark.skipif("not config.getoption('runllms')")


def test_mix_approach():
    """Tests for mix_approach.py."""

    env = gymnasium.make("LunarLander-v3")
    env.action_space.seed(123)
    constant_action = env.action_space.sample()
    environment_description = (
        "The well-known LunarLander in gymnasium, i.e., "
        'env = gymnasium.make("LunarLander-v3")'
    )

    cache_path = Path(tempfile.NamedTemporaryFile(suffix=".db").name)
    cache = SQLite3PretrainedLargeModelCache(cache_path)
    response = Response(
        f"""```python
def _policy(obs):
    return {constant_action}
```
""",
        {},
    )
    llm = OrderedResponseModel([response], cache)

    approach = MixApproach(
        environment_description,
        env.observation_space,
        env.action_space,
        seed=123,
        llm=llm,
    )

    obs, info = env.reset()
    approach.reset(obs, info)

    llm_actions = 0
    random_actions = 0

    for _ in range(20):  # Run more steps to observe randomness
        action = approach.step()
        # Most likely the constant_action fed into the LLM
        if action == constant_action:
            llm_actions += 1
        else:
            random_actions += 1
        obs, reward, terminated, _, info = env.step(action)
        approach.update(obs, reward, terminated, info)

    # Ensure both LLM-based and random actions were taken
    assert llm_actions > 0, "No LLM-based actions were taken."
    assert random_actions > 0, "No random actions were taken."
