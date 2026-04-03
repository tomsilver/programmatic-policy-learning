"""Tests for Python feature-generation helpers."""

from __future__ import annotations

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from programmatic_policy_learning.dsl.llm_primitives.py_feature_generator import (
    PyFeatureGenerator,
)


def test_extract_python_script_removes_markdown_fences() -> None:
    """Fenced Python responses should be converted to raw script text."""
    generator = PyFeatureGenerator(None)
    # pylint: disable=protected-access
    script = generator._extract_python_script(
        "```python\n"
        "def build_feature_library():\n"
        '    return {"features": []}\n'
        "```"
    )
    assert script.startswith("def build_feature_library():")
    assert "```" not in script


def test_generate_supports_generator_script_mode(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """generator_script mode should execute build_feature_library()."""
    prompt_path = tmp_path / "generator_prompt.txt"
    prompt_path.write_text("Generate a feature script.", encoding="utf-8")

    generator = PyFeatureGenerator(None)
    generator.output_path = tmp_path
    generator.output_path.mkdir(parents=True, exist_ok=True)

    script_text = """
def build_feature_library():
    return {
        "features": [
            {
                "id": "f1",
                "name": "robot_moving_right",
                "source": "def f1(s, a):\\n    return a[0] > 0\\n",
            }
        ]
    }
"""

    monkeypatch.setattr(
        generator, "query_llm_text", lambda *args, **kwargs: script_text
    )
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")

    features, payload = generator.generate(
        prompt_path=prompt_path,
        hint_text="",
        num_features=1,
        env_name="Motion2D-p0",
        demonstration_data="demo",
        encoding_method="enc_4",
        action_mode="continuous",
        generation_mode="generator_script",
    )

    assert features == ["def f1(s, a):\n    return a[0] > 0\n"]
    assert payload["features"][0]["id"] == "f1"
    assert payload["features"][0]["name"] == "robot_moving_right"
    assert "source" in payload["features"][0]
