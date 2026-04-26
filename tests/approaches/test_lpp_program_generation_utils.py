"""Tests for LPP program-generation helpers."""

from typing import Any, cast

from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache

from programmatic_policy_learning.approaches.lpp_utils import (
    lpp_program_generation_utils as utils,)


def test_make_llm_client_uses_responses_client_for_pro_models(
    monkeypatch: Any,
) -> None:
    """Pro models should not be sent through the chat-completions client."""

    created: dict[str, Any] = {}

    class FakeResponsesModel:
        """Test double for the responses-based LLM client."""

        def __init__(self, model_name: str, cache: Any) -> None:
            created["model_name"] = model_name
            created["cache"] = cache

    cache = cast(SQLite3PretrainedLargeModelCache, object())
    monkeypatch.setattr(utils.llm_models, "OpenAIResponsesModel", FakeResponsesModel)

    client = utils.make_llm_client_for_model("gpt-5.2-pro", cache)

    assert isinstance(client, FakeResponsesModel)
    assert created == {"model_name": "gpt-5.2-pro", "cache": cache}


def test_make_llm_client_uses_chat_client_for_non_pro_models(monkeypatch: Any) -> None:
    """Regular chat models should continue using OpenAIModel."""

    created: dict[str, Any] = {}

    class FakeOpenAIModel:
        """Test double for the standard chat-completions LLM client."""

        def __init__(self, model_name: str, cache: Any) -> None:
            created["model_name"] = model_name
            created["cache"] = cache

    cache = cast(SQLite3PretrainedLargeModelCache, object())
    monkeypatch.setattr(utils, "OpenAIModel", FakeOpenAIModel)

    client = utils.make_llm_client_for_model("gpt-4.1", cache)

    assert isinstance(client, FakeOpenAIModel)
    assert created == {"model_name": "gpt-4.1", "cache": cache}
