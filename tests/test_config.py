"""Tests for base configuration."""

from pathlib import Path

import pytest

from alignrl.config import CHATML_TEMPLATE, BaseTrainConfig, ensure_chat_template


class TestBaseTrainConfig:
    def test_defaults(self) -> None:
        cfg = BaseTrainConfig()
        assert cfg.model_name == "Qwen/Qwen2.5-3B"
        assert cfg.max_seq_length == 2048
        assert cfg.load_in_4bit is True
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.seed == 42

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text("model_name: test-model\nlearning_rate: 1e-5\nmax_seq_length: 4096\n")
        cfg = BaseTrainConfig.from_yaml(yaml_path)
        assert cfg.model_name == "test-model"
        assert cfg.learning_rate == 1e-5
        assert cfg.max_seq_length == 4096

    def test_from_yaml_partial(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "partial.yaml"
        yaml_path.write_text("model_name: partial-model\n")
        cfg = BaseTrainConfig.from_yaml(yaml_path)
        assert cfg.model_name == "partial-model"
        assert cfg.learning_rate == 2e-4  # default preserved

    def test_output_dir_is_path(self) -> None:
        cfg = BaseTrainConfig(output_dir="./my-output")
        assert isinstance(cfg.output_dir, Path)

    def test_lora_target_modules(self) -> None:
        cfg = BaseTrainConfig()
        assert "q_proj" in cfg.lora_target_modules
        assert "v_proj" in cfg.lora_target_modules
        assert len(cfg.lora_target_modules) == 7


class TestBaseTrainConfigValidation:
    def test_rejects_invalid_type(self) -> None:
        with pytest.raises(ValueError):
            BaseTrainConfig(learning_rate="not_a_float")

    def test_custom_lora_modules(self) -> None:
        cfg = BaseTrainConfig(lora_target_modules=["q_proj", "k_proj"])
        assert cfg.lora_target_modules == ["q_proj", "k_proj"]


class TestEnsureChatTemplate:
    def test_sets_template_when_missing(self) -> None:
        class FakeTokenizer:
            chat_template = None

        tok = FakeTokenizer()
        ensure_chat_template(tok)
        assert tok.chat_template == CHATML_TEMPLATE

    def test_preserves_existing_template(self) -> None:
        class FakeTokenizer:
            chat_template = "{% for m in messages %}custom{% endfor %}"

        tok = FakeTokenizer()
        original = tok.chat_template
        ensure_chat_template(tok)
        assert tok.chat_template == original

    def test_handles_no_attribute(self) -> None:
        class FakeTokenizer:
            pass

        tok = FakeTokenizer()
        ensure_chat_template(tok)
        assert tok.chat_template == CHATML_TEMPLATE
