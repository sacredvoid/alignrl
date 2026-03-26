"""Tests for DPO module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from alignrl.dpo import DPOConfig, DPORunner, format_ultrafeedback


class TestDPOConfig:
    def test_defaults(self) -> None:
        cfg = DPOConfig()
        assert cfg.beta == 0.1
        assert cfg.learning_rate == 5e-7
        assert cfg.dataset_name == "HuggingFaceH4/ultrafeedback_binarized"


class TestFormatUltraFeedback:
    def test_basic_format(self) -> None:
        example = {
            "prompt": "What is Python?",
            "chosen": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ],
            "rejected": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a snake."},
            ],
        }
        result = format_ultrafeedback(example)
        assert "prompt" in result
        assert "chosen" in result
        assert "rejected" in result
        assert result["chosen"][-1]["role"] == "assistant"
        assert result["rejected"][-1]["role"] == "assistant"

    def test_single_message(self) -> None:
        example = {
            "chosen": [{"role": "assistant", "content": "Good answer"}],
            "rejected": [{"role": "assistant", "content": "Bad answer"}],
        }
        result = format_ultrafeedback(example)
        assert result["prompt"] == []
        assert len(result["chosen"]) == 1
        assert len(result["rejected"]) == 1


class TestDPORunner:
    def test_init(self) -> None:
        cfg = DPOConfig()
        runner = DPORunner(cfg)
        assert runner.config is cfg
        assert runner._model is None
        assert runner._tokenizer is None

    def test_save_no_model(self, tmp_path: Path) -> None:
        cfg = DPOConfig()
        runner = DPORunner(cfg)
        runner.save(tmp_path / "output")  # no raise when model is None

    def test_config_fields(self) -> None:
        cfg = DPOConfig(
            beta=0.2,
            max_length=2048,
            precompute_ref_log_probs=False,
            dataset_size=500,
        )
        assert cfg.beta == 0.2
        assert cfg.max_length == 2048
        assert cfg.precompute_ref_log_probs is False
        assert cfg.dataset_size == 500

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "dpo.yaml"
        yaml_path.write_text("model_name: test-model\nbeta: 0.05\n")
        cfg = DPOConfig.from_yaml(yaml_path)
        assert cfg.model_name == "test-model"
        assert cfg.beta == 0.05

    def test_load_model(self) -> None:
        cfg = DPOConfig()
        runner = DPORunner(cfg)

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

    def test_load_dataset(self) -> None:
        cfg = DPOConfig(dataset_size=10)
        runner = DPORunner(cfg)

        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=100)
        mock_ds.select.return_value = mock_ds
        mock_ds.map.return_value = mock_ds

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            result = runner._load_dataset()
            mock_ds.select.assert_called_once()
            mock_ds.map.assert_called_once()

    def test_load_dataset_no_size_limit(self) -> None:
        cfg = DPOConfig(dataset_size=None)
        runner = DPORunner(cfg)

        mock_ds = MagicMock()
        mock_ds.map.return_value = mock_ds

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            runner._load_dataset()
            mock_ds.select.assert_not_called()

    def test_load_dataset_size_zero_selects(self) -> None:
        cfg = DPOConfig(dataset_size=0)
        runner = DPORunner(cfg)

        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=100)
        mock_ds.select.return_value = mock_ds
        mock_ds.map.return_value = mock_ds

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            runner._load_dataset()
            mock_ds.select.assert_called_once_with(range(0))

    def test_train(self, tmp_path: Path) -> None:
        cfg = DPOConfig(output_dir=str(tmp_path / "dpo_output"))
        runner = DPORunner(cfg)

        runner._load_model = MagicMock()
        runner._load_dataset = MagicMock(return_value=MagicMock())

        mock_trainer = MagicMock()
        mock_result = MagicMock()
        mock_result.training_loss = 0.3
        mock_result.global_step = 20
        mock_trainer.train.return_value = mock_result
        mock_trainer.state.log_history = [{"loss": 0.6}, {"loss": 0.3}]

        mock_trl = MagicMock()
        mock_trl.DPOTrainer.return_value = mock_trainer

        with patch.dict("sys.modules", {"trl": mock_trl}):
            result = runner.train()
            assert result.num_steps == 20
            assert result.metrics["train_loss"] == 0.3
            assert result.loss_history == [0.6, 0.3]

    def test_save_with_model(self, tmp_path: Path) -> None:
        cfg = DPOConfig()
        runner = DPORunner(cfg)
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner.save(tmp_path / "output")
        runner._model.save_pretrained.assert_called_once()
        runner._tokenizer.save_pretrained.assert_called_once()

    def test_load(self, tmp_path: Path) -> None:
        cfg = DPOConfig()
        runner = DPORunner(cfg)

        mock_unsloth = MagicMock()
        mock_unsloth.FastLanguageModel.from_pretrained.return_value = (MagicMock(), MagicMock())

        with patch.dict("sys.modules", {"unsloth": mock_unsloth}):
            runner.load(tmp_path)
            assert runner._model is not None
