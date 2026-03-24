"""Tests for SFT module."""
from pathlib import Path

import pytest

from alignrl.config import BaseTrainConfig
from alignrl.sft import SFTConfig, format_instruction


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
