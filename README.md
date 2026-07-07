# Programmatic Policy Learning

This is a codebase that the PRPL lab is using for multiple projects related to programmatic policy learning.

## Installation

Requirements: Python >=3.10 and <=3.12.

We strongly recommend [uv](https://docs.astral.sh/uv/getting-started/installation/). The steps below assume that you have `uv` installed. If you do not, just remove `uv` from the commands and the installation should still work.

```
uv pip install -e ".[develop]"
```

Check the installation: `./run_ci_checks.sh`

If you want to use an OpenAI LLM, make sure you have an `OPENAI_API_KEY` set (e.g., see [here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety))

### ARC-AGI-3 Adapter

This repo includes a thin ARC-AGI-3 environment adapter under
`programmatic_policy_learning.envs.arc_agi3`. It uses the official
`arc-agi==0.9.9` SDK and does not reimplement ARC game mechanics.

Install dependencies with the standard repo command:

```bash
uv pip install -e ".[develop]"
```

Set your ARC API key in the shell or in a local `.env` file:

```bash
cp .env.example .env
# edit .env and set ARC_API_KEY
```

List available games, launch `ls20`, reset it, and take one action:

```bash
python experiments/scripts/arc_agi3_smoke.py list
python experiments/scripts/arc_agi3_smoke.py smoke --game ls20 --action 1
```

The SDK caches downloaded game files in `environment_files/`. Once `ls20` has
been cached, offline mode can run without API access:

```bash
python experiments/scripts/arc_agi3_smoke.py list --offline
python experiments/scripts/arc_agi3_smoke.py smoke --game ls20 --action 1 --offline
```

To inspect the raw SDK observation interactively, stop in `pdb` immediately
after the game resets:

```bash
python experiments/scripts/arc_agi3_smoke.py smoke \
  --game ls20 --action 1 --offline --pdb
```

Useful debugger expressions include `type(initial_obs)`, `initial_obs`,
`initial_obs.frame`, `initial_obs.available_actions`, `env.action_space`, and
`env.step(1)`. Use `c` to let the smoke test continue, or `q` to quit.

If normal mode cannot fetch a game, register for ARC-AGI-3 access and set
`ARC_API_KEY`. Offline mode only works for games already present in the local
cache.

To collect manual expert demonstrations for later LPP work, play the game and
save repo-native `DemoRecord` pickles under `manual_demos/arc_agi3/ls20/`:

```bash
python experiments/scripts/collect_arc_agi3_manual_demos.py --game ls20 --seeds 0..2
```

Use `1` through `7` or `action1` through `action7` at the prompt, and type
`save` when the current trajectory is worth keeping. After the game is cached,
add `--offline` to collect without API access.

For `ls20` level-0 variant collection, `--randomize-initial-state` changes only
the player and rotation-switch starts by default. Seed `0` is reserved for the
official original start state; seeds `1` and above produce deterministic
position variants. This is the recommended Stage-A setup:

```bash
python experiments/scripts/collect_arc_agi3_manual_demos.py \
  --game ls20 \
  --seeds 0..9 \
  --offline \
  --randomize-initial-state \
  --demo-name shape_match_train
```

For later Stage-B experiments, add `--randomize-shape` to also randomize the
shared target/reference shape for seeds `1` and above.

ARC manual demonstrations use an object-centric processed observation. Each
saved state contains:

```python
{
    "raw": {...},                 # original SDK observation as plain data
    "grid": [[...]],              # latest 2D color grid
    "objects_by_color": {...},    # 4-connected components for every color
    "player": {...},              # game-specific semantic objects, when known
    "rotation_switch": {...},
    "current_shape": {...},
    "reference_shape": {...},
}
```

The generic component extractor lives in
`programmatic_policy_learning.envs.arc_agi3.preprocessing`. Semantic object
names are supplied by a game-specific enricher, so another ARC game can define
different keys while retaining the same `raw`, `grid`, and
`objects_by_color` interface. Existing raw ARC demonstration files are
preprocessed in memory when replayed; newly collected demonstrations persist
the processed state directly.

## Usage Example

```python
from pathlib import Path

import gymnasium
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

from programmatic_policy_learning.approaches.ppl_approach import (
    LLMPPLApproach,
)

env = gymnasium.make("LunarLander-v3")
env.action_space.seed(123)
environment_description = (
    "The well-known LunarLander in gymnasium, i.e., "
    'env = gymnasium.make("LunarLander-v3")'
)

cache = SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
llm = OpenAIModel("gpt-4o-mini", cache)

approach = LLMPPLApproach(
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

### Running the CaP Baseline

The Code-as-Policies baseline is in:

```
src/programmatic_policy_learning/dsl/llm_primitives/baselines/llm_based/CaP_baseline.py
```

Use `--demo-env-nums` to specify the exact demonstration env ids/reset seeds.
This replaces the older count-based style and makes CaP comparable to LPP runs
that use explicit `demo_numbers` or `program_generation.demos_included`.

For example, to mirror an LPP run using Chase demos `[0, 1, 2]` and evaluate on
the LPP held-out test split `[11, ..., 19]`:

```
UV_CACHE_DIR=.uv-cache uv run python src/programmatic_policy_learning/dsl/llm_primitives/baselines/llm_based/CaP_baseline.py \
  --env Chase \
  --env-type grid \
  --encodings 5 \
  --seeds 0 \
  --model gpt-4.1 \
  --demo-env-nums 0 1 2 \
  --eval-env-nums 11 12 13 14 15 16 17 18 19
```

CaP resets both the generated policy rollout and the expert rollout with
`reset_seed=env_idx`, matching the LPP test rollout convention.

CaP also uses the same demonstration-format background files as the LPP feature
generator prompts, and it serializes demonstrations through the shared
environment LLM spec. For a fair comparison, use the same encoding that LPP uses
for `program_generation.encoding_method` (for example, `5` or `enc_5`).

## Notes

### Box2D Installation on macOS

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

---

# Adding a New Environment to PPL

You can add environments in two ways:

1. **Plain Gymnasium env** (already registered via `gymnasium.make`)
2. **Provider-based env** (env lives in a separate repo and needs a small adapter)

## 1. Plain Gymnasium Env (no provider)

If the env is already registered with Gymnasium, just add a YAML under `conf/env/` and you’re done.
**Example:** `conf/env/lunarlander.yaml`

```yaml
# Passed into gymnasium.make() to create the environment.
make_kwargs:
  id: "LunarLander-v3"
  render_mode: null # "human", "rgb_array", or null

# Optional, purely descriptive.
description: "The well-known LunarLander in gymnasium, i.e., env = gymnasium.make('LunarLander-v3')"
```

**How it’s used in code:**

```python
from programmatic_policy_learning.env.registry import EnvRegistry

registry = EnvRegistry()
env = registry.load(cfg.env)  # default fallback is gymnasium.make(**make_kwargs)
```

> If you don’t specify a `provider`, `EnvRegistry` falls back to `gymnasium.make(**make_kwargs)`.

## 2. Provider-Based Env (from a separate repo)

Use this when your env lives in another repo (e.g., PRBench, GGG, custom maze env).  
You’ll: (a) create a YAML with a `provider`, (b) add a provider function, and (c) (if needed) pin the external repo in `pyproject.toml`.

### 2.1 Create the YAML (under `conf/env/`)

**Example:** `conf/env/prbench_motion2d_p1.yaml`

```yaml
make_kwargs:
  id: "prbench/Motion2D-p1-v0"
  render_mode: null

provider: prbench # <--- important

description: "PRBench Motion2D-p1. Gymnasium-style env registered by PRBench"
```

### 2.2 Register the Provider

**Edit:** `programmatic_policy_learning/env/registry.py`

Add an entry to the provider map:

```python
self._providers: dict[str, Callable[[Any], Any]] = {
    "ggg": create_ggg_env,
    "prbench": create_prbench_env,
    # "gym_maze": create_maze_env,  # example for your own provider
}
```

### 2.3 Implement the Provider Function

**File structure:**

```cpp
programmatic_policy_learning/
  env/
    providers/
      prbench_provider.py      # define create_prbench_env(cfg)
      ggg_provider.py          # define create_ggg_env(cfg)
      maze_provider.py         # define create_maze_env(cfg)  (example)
```

**Example:** `programmatic_policy_learning/env/providers/prbench_provider.py`

```python
from __future__ import annotations
from typing import Any
import gymnasium as gym

def create_prbench_env(cfg: Any):
    """Create and return a PRBench env using cfg.env.make_kwargs."""
    make_kwargs = dict(cfg.env.make_kwargs)
    env = gym.make(**make_kwargs)
    return env
```

> Your provider can do anything needed (import the external package, wrap the env, set seeds, apply wrappers, etc.). Just return the final `env`.

### 2.4 Add the External Repo to the dependencies

If your provider imports an external repo, put it in `pyproject.toml` under `dependencies = [...]`, so CI and collaborators get the same version.

**Example (GGG):**

```toml
dependencies = [
  "generalization_grid_games@git+https://github.com/zahraabashir/generalization_grid_games.git@ee0a559",
]
```

**Example (your own repo):**

```toml
dependencies = [
  "my_cool_env_pkg@git+https://github.com/your-org/my_cool_env_pkg.git@<commit-hash>"
]
```

After this, you only need to run the following command to install that dependency:

```bash
uv pip install -e ".[develop]"
```

## 3) How to Instantiate in Code

Same pattern for both plain and provider-based envs:

```python
from programmatic_policy_learning.env.registry import EnvRegistry

registry = EnvRegistry()
env = registry.load(cfg.env)  # uses provider if present, else gymnasium.make
```

- If your YAML has `provider: ...`, `EnvRegistry` routes to the matching provider function.
- If there’s **no** `provider`, it calls `gymnasium.make(**make_kwargs)`.

## Minimal Checklist

- Add `conf/env/<your_env>.yaml`
- If external repo:
  - Add dependency pin in `pyproject.toml` under `[project.optional-dependencies]`
  - Add provider entry in `EnvRegistry` (provider name → function)
  - Implement `create_<provider>_env(cfg)` in `env/providers/<provider>_provider.py`
- Instantiate with `EnvRegistry().load(cfg.env)`
  That's it!

---

## 3. Implementing Your Own Custom Env

If you want to implement an environment yourself (instead of importing it from another repo), you can follow the same provider-based structure:

- Create a new provider file under `programmatic_policy_learning/env/providers/x_provider.py`.
- Inside this file, implement your custom environment class (e.g., MyCustomEnv).
- At the end of the file, also implement a factory function (as before) like: def create_x_env(cfg: Any):
- Add a YAML under conf/env/ (as in the examples above), and register your provider in EnvRegistry.

This way, whether your env comes from an external repo or is defined locally, the process looks the same, your provider file is the single place to keep both the environment definition and the factory function.

---

## Contributing

- Ask an owner of the repository to add your GitHub username to the collaborators list
- All checks must pass before code is merged (see `./run_ci_checks.sh`)
- All code goes through the pull request review process on GitHub
