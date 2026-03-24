"""Tests for inference module."""
from alignrl.inference import InferenceConfig, build_prompt


class TestInferenceConfig:
    def test_defaults(self) -> None:
        cfg = InferenceConfig()
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 512

    def test_custom(self) -> None:
        cfg = InferenceConfig(temperature=0.0, max_tokens=256)
        assert cfg.temperature == 0.0


class TestBuildPrompt:
    def test_math_prompt(self) -> None:
        prompt = build_prompt("What is 2+2?", system="You are a math tutor.")
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        assert prompt[1]["content"] == "What is 2+2?"

    def test_no_system(self) -> None:
        prompt = build_prompt("Hello")
        assert prompt[0]["role"] == "user"
        assert len(prompt) == 1
