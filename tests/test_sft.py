"""Tests for SFT module."""

from pathlib import Path

import pytest

from alignrl.sft import ROLE_MAP, SFTConfig, SFTRunner, format_instruction


class TestSFTConfig:
    def test_defaults(self) -> None:
        cfg = SFTConfig()
        assert cfg.dataset_name == "teknium/OpenHermes-2.5"
        assert cfg.learning_rate == 2e-4
        assert cfg.load_in_4bit is True

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "sft.yaml"
        yaml_path.write_text("model_name: test-model\ndataset_name: test-dataset\n")
        cfg = SFTConfig.from_yaml(yaml_path)
        assert cfg.model_name == "test-model"
        assert cfg.dataset_name == "test-dataset"


class TestFormatInstruction:
    def test_formats_openhermes_style(self) -> None:
        example = {
            "conversations": [
                {"from": "human", "value": "What is 2+2?"},
                {"from": "gpt", "value": "4"},
            ]
        }
        messages = format_instruction(example)
        assert messages == [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

    def test_handles_system_message(self) -> None:
        example = {
            "conversations": [
                {"from": "system", "value": "You are helpful."},
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        messages = format_instruction(example)
        assert messages[0]["role"] == "system"
        assert len(messages) == 3

    def test_empty_conversations_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            format_instruction({"conversations": []})

    def test_unknown_role_passes_through(self) -> None:
        example = {
            "conversations": [
                {"from": "tool", "value": "result"},
            ]
        }
        messages = format_instruction(example)
        assert messages[0]["role"] == "tool"

    def test_missing_conversations_key_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            format_instruction({})


class TestRoleMap:
    def test_maps_human_to_user(self) -> None:
        assert ROLE_MAP["human"] == "user"

    def test_maps_gpt_to_assistant(self) -> None:
        assert ROLE_MAP["gpt"] == "assistant"

    def test_maps_system(self) -> None:
        assert ROLE_MAP["system"] == "system"


class TestSFTRunner:
    def test_init(self) -> None:
        cfg = SFTConfig()
        runner = SFTRunner(cfg)
        assert runner.config is cfg
        assert runner._model is None
        assert runner._tokenizer is None

    def test_save_no_model(self, tmp_path: Path) -> None:
        cfg = SFTConfig()
        runner = SFTRunner(cfg)
        runner.save(tmp_path / "output")  # no raise when model is None

    def test_save_no_tokenizer(self, tmp_path: Path) -> None:
        cfg = SFTConfig()
        runner = SFTRunner(cfg)
        runner._model = None
        runner._tokenizer = None
        runner.save(tmp_path / "output")  # no raise

    def test_config_fields(self) -> None:
        cfg = SFTConfig(dataset_name="custom-ds", dataset_split="test", dataset_size=100)
        assert cfg.dataset_name == "custom-ds"
        assert cfg.dataset_split == "test"
        assert cfg.dataset_size == 100
        assert cfg.chat_template == "chatml"
