# Programmatic Policy Learning

![workflow](https://github.com/tomsilver/programmatic-policy-learning/actions/workflows/ci.yml/badge.svg)

This is a codebase that the PRPL lab is using for multiple projects related to programmatic policy learning.

## Installation

1. Requirements: Python >=3.11 and <=3.12.
2. Recommended: use a virtual environment. For example, we like [uv](https://github.com/astral-sh/uv).
    - Install `uv`:  ```curl -LsSf https://astral.sh/uv/install.sh | sh```
    - Create the virtual environment: `uv venv --python=3.11`
    - Activate the environment (every time you start a new terminal): `source .venv/bin/activate`
3. Clone this repository and `cd` into it.
4. Install this repository and its dependencies:
    - If you are using `uv`, do ```uv pip install -e ".[develop]"```
    - Otherwise, just do ```pip install -e ".[develop]"```
5. Check the installation: ```./run_ci_checks.sh```
6. If you want to use an OpenAI LLM, make sure you have an `OPENAI_API_KEY` set (e.g., see [here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety))

## Usage Example

```python
from pathlib import Path

import gymnasium
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

from programmatic_policy_learning.approaches.ppl_approach import (
    ProgrammaticPolicyLearningApproach,
)

env = gymnasium.make("LunarLander-v3")
env.action_space.seed(123)
environment_description = (
    "The well-known LunarLander in gymnasium, i.e., "
    'env = gymnasium.make("LunarLander-v3")'
)

cache = SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
llm = OpenAIModel("gpt-4o-mini", cache)

approach = ProgrammaticPolicyLearningApproach(
    environment_description,
    env.observation_space,
    env.action_space,
    seed=123,
    llm=llm,
)

obs, info = env.reset()
approach.reset(obs, info)

print(approach._policy)

for _ in range(5):
    action = approach.step()
    assert env.action_space.contains(action)
    obs, reward, terminated, _, info = env.step(action)
    approach.update(obs, reward, terminated, info)
```

## Running Experiments

We use [hydra](https://hydra.cc/) to run experiments at scale. See `experiments/run_experiment.py`. For example:

```
python experiments/run_experiment.py -m env=lunar_lander llm=openai seed='range(0,2)'
```

## Contributing

* Ask an owner of the repository to add your GitHub username to the collaborators list
* All checks must pass before code is merged (see `./run_ci_checks.sh`)
* All code goes through the pull request review process on GitHub

## Notes

### Installing Box2D on macOS

If you encounter an error when installing dependencies (e.g., `box2d-py`) that looks like this:

```
Box2D/Box2D_wrap.cpp:3378:10: fatal error: 'string' file not found
3378 | #include <string>
      |          ^~~~~~~~
1 error generated.
error: command '/usr/bin/clang++' failed with exit code 1
```

This might mean that your macOS Command Line Tools (CLT) or SDK isn’t installed or selected correctly, and the compiler (`clang++`) cannot find the C++ standard library headers.

To fix this issue, try these steps:

1. **Reinstall or point to the correct Command Line Tools (CLT):**

   - Remove any broken or partial CLT installations:
     ```bash
     sudo rm -rf /Library/Developer/CommandLineTools
     ```

   - Reinstall the CLT (a GUI prompt will appear):
     ```bash
     xcode-select --install
     ```

2. After completing the installation, try installing the dependencies again:
   ```bash
   uv pip install -e ".[develop]"
   ```

If you are using `uv` to manage your virtual environment, you can also try installing `box2d-py` directly to verify the fix:

```bash
uv pip install box2d-py
```