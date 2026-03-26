"""Tests for SFT module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def test_load_model(self) -> None:
        cfg = SFTConfig()
        runner = SFTRunner(cfg)

        mock_unsloth = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_unsloth.FastLanguageModel.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_unsloth.FastLanguageModel.get_peft_model.return_value = mock_model

        with patch.dict("sys.modules", {"unsloth": mock_unsloth}):
            runner._load_model()
            mock_unsloth.FastLanguageModel.from_pretrained.assert_called_once()
            mock_unsloth.FastLanguageModel.get_peft_model.assert_called_once()
            assert runner._model is mock_model

    def test_load_dataset(self) -> None:
        cfg = SFTConfig(dataset_size=5)
        runner = SFTRunner(cfg)
        runner._tokenizer = MagicMock()
        runner._tokenizer.apply_chat_template.return_value = "formatted text"

        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=100)
        mock_ds.select.return_value = mock_ds
        mock_ds.column_names = ["conversations"]
        mock_ds.map.return_value = mock_ds

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            result = runner._load_dataset()
            mock_datasets.load_dataset.assert_called_once()
            mock_ds.select.assert_called_once()

    def test_train(self, tmp_path: Path) -> None:
        cfg = SFTConfig(output_dir=str(tmp_path / "sft_output"))
        runner = SFTRunner(cfg)

        runner._load_model = MagicMock()
        runner._load_dataset = MagicMock(return_value=MagicMock())

        mock_trainer = MagicMock()
        mock_train_result = MagicMock()
        mock_train_result.training_loss = 0.5
        mock_train_result.global_step = 10
        mock_trainer.train.return_value = mock_train_result
        mock_trainer.state.log_history = [{"loss": 0.8}, {"loss": 0.5}]

        mock_trl = MagicMock()
        mock_trl.SFTTrainer.return_value = mock_trainer

        with patch.dict("sys.modules", {"trl": mock_trl}):
            result = runner.train()
            assert result.num_steps == 10
            assert result.metrics["train_loss"] == 0.5
            assert result.loss_history == [0.8, 0.5]

    def test_save_with_model(self, tmp_path: Path) -> None:
        cfg = SFTConfig()
        runner = SFTRunner(cfg)
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner.save(tmp_path / "save_output")
        runner._model.save_pretrained.assert_called_once()
        runner._tokenizer.save_pretrained.assert_called_once()

    def test_load(self, tmp_path: Path) -> None:
        cfg = SFTConfig()
        runner = SFTRunner(cfg)

        mock_unsloth = MagicMock()
        mock_unsloth.FastLanguageModel.from_pretrained.return_value = (MagicMock(), MagicMock())

        with patch.dict("sys.modules", {"unsloth": mock_unsloth}):
            runner.load(tmp_path)
            mock_unsloth.FastLanguageModel.from_pretrained.assert_called_once()
            assert runner._model is not None
