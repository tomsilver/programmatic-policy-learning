"""Tests for LPP program-generation helpers."""

from pathlib import Path
from typing import Any

from programmatic_policy_learning.approaches.lpp_utils import (
    lpp_program_generation_utils as utils,)


def test_make_llm_client_uses_responses_client_for_pro_models(monkeypatch: Any) -> None:
    """Pro models should not be sent through the chat-completions client."""

    created: dict[str, Any] = {}

    class FakeResponsesModel:
        def __init__(self, model_name: str, cache: Any) -> None:
            created["model_name"] = model_name
            created["cache"] = cache

    cache = object()
    monkeypatch.setattr(utils.llm_models, "OpenAIResponsesModel", FakeResponsesModel)

    client = utils.make_llm_client_for_model("gpt-5.2-pro", cache)  # type: ignore[arg-type]

    assert isinstance(client, FakeResponsesModel)
    assert created == {"model_name": "gpt-5.2-pro", "cache": cache}


def test_make_llm_client_uses_chat_client_for_non_pro_models(monkeypatch: Any) -> None:
    """Regular chat models should continue using OpenAIModel."""

    created: dict[str, Any] = {}

    class FakeOpenAIModel:
        def __init__(self, model_name: str, cache: Any) -> None:
            created["model_name"] = model_name
            created["cache"] = cache

    cache = object()
    monkeypatch.setattr(utils, "OpenAIModel", FakeOpenAIModel)

    client = utils.make_llm_client_for_model("gpt-4.1", cache)  # type: ignore[arg-type]

    assert isinstance(client, FakeOpenAIModel)
    assert created == {"model_name": "gpt-4.1", "cache": cache}


def test_get_program_set_merges_multi_prompt_feature_pool(monkeypatch: Any) -> None:
    """Ensemble feature generation should query all calls then deduplicate."""

    calls: list[tuple[str, int]] = []

    class FakeEnv:
        def close(self) -> None:
            return None

    class FakeEnvSpec:
        def serialize_demonstrations(
            self,
            trajectories: list[list[tuple[Any, Any, Any]]],
            *,
            encoding_method: str,
            max_steps: int,
        ) -> str:
            del encoding_method, max_steps
            return "|".join(str(traj[0][0]) for traj in trajectories)

    class FakePyFeatureGenerator(utils.PyFeatureGenerator):
        def __init__(self, llm_client: Any) -> None:
            super().__init__(llm_client)
            self.output_path = Path.cwd()

        def generate(
            self,
            prompt_path: str | Path,
            hint_text: str,
            num_features: int,
            env_name: str | None = None,
            demonstration_data: str | None = None,
            encoding_method: str | None = None,
            max_attempts: int = 3,
            _seed: int = 0,
            reprompt_checks: list[Any] | None = None,
            loading: dict[str, Any] | None = None,
            action_mode: str = "discrete",
            generation_mode: str = "feature_payload",
        ) -> tuple[list[str], dict[str, Any]]:
            del (
                prompt_path,
                hint_text,
                num_features,
                env_name,
                encoding_method,
                max_attempts,
                reprompt_checks,
                loading,
                action_mode,
                generation_mode,
            )
            assert demonstration_data is not None
            calls.append((demonstration_data, _seed))
            constant = "True" if "demo_0" in demonstration_data else "False"
            payload = {
                "features": [
                    {
                        "id": "f1",
                        "name": f"seed_{_seed}",
                        "source": f"def f1(s, a):\n    return {constant}\n",
                    }
                ]
            }
            return self.parse_feature_programs(payload), payload

    monkeypatch.setattr(utils, "PyFeatureGenerator", FakePyFeatureGenerator)
    monkeypatch.setattr(utils, "load_unique_hint", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        utils, "get_env_llm_spec", lambda *args, **kwargs: FakeEnvSpec()
    )
    monkeypatch.setattr(
        utils,
        "_collect_full_episode_generic",
        lambda env, expert, **kwargs: [(f"demo_{kwargs['reset_seed']}", "a", "n")],
    )

    program_generation = {
        "strategy": "py_feature_gen",
        "llm_model": "gpt-4.1",
        "py_feature_gen_prompt": "unused.txt",
        "py_feature_gen_mode": "feature_payload",
        "num_features": 5,
        "encoding_method": "enc_5",
        "skip_rate": 1,
        "multi_prompt_ensemble": {
            "num_seeds_per_subset": 2,
            "demo_subsets": [[0, 1], [2, 3]],
        },
        "loading": {"offline": 0},
    }

    features, priors, dsl_fns, feature_display_names = utils.get_program_set(
        num_programs=0,
        base_class_name="TestEnv",
        env_factory=lambda instance_num: FakeEnv(),
        expert=object(),
        env_specs={"action_mode": "continuous"},
        program_generation=program_generation,
        demo_numbers=[0, 1, 2, 3],
        seed=7,
        prior_version="uniform",
    )

    assert calls == [
        ("demo_0|demo_1", 7),
        ("demo_0|demo_1", 8),
        ("demo_2|demo_3", 9),
        ("demo_2|demo_3", 10),
    ]
    assert len(features) == 2
    assert "return True" in features[0]
    assert "return False" in features[1]
    assert len(priors) == 2
    assert feature_display_names == ["f1", "f2"]
    assert "f1" in dsl_fns and "f2" in dsl_fns
