"""Tests for GRPO module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from alignrl.grpo import GRPOConfig, GRPORunner, _format_gsm8k_prompt


class TestGRPOConfig:
    def test_defaults(self) -> None:
        cfg = GRPOConfig()
        assert cfg.num_generations == 8
        assert cfg.beta == 0.001
        assert cfg.learning_rate == 5e-6
        assert cfg.reward_weights is None

    def test_custom_reward_weights(self) -> None:
        cfg = GRPOConfig(reward_weights=[0.7, 0.3])
        assert cfg.reward_weights == [0.7, 0.3]

    def test_dataset_config(self) -> None:
        cfg = GRPOConfig(dataset_config="socratic")
        assert cfg.dataset_config == "socratic"

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "grpo.yaml"
        yaml_path.write_text("model_name: test-model\nnum_generations: 4\nbeta: 0.01\n")
        cfg = GRPOConfig.from_yaml(yaml_path)
        assert cfg.model_name == "test-model"
        assert cfg.num_generations == 4
        assert cfg.beta == 0.01


class TestFormatGsm8k:
    def test_basic_format(self) -> None:
        example = {"question": "What is 2+2?", "answer": "2+2=4\n#### 4"}
        result = _format_gsm8k_prompt(example)
        assert result["solution"] == "4"
        assert result["prompt"][0]["role"] == "system"
        assert result["prompt"][1]["role"] == "user"
        assert result["prompt"][1]["content"] == "What is 2+2?"

    def test_strips_whitespace(self) -> None:
        example = {"question": "x?", "answer": "steps\n####  42 "}
        result = _format_gsm8k_prompt(example)
        assert result["solution"] == "42"

    def test_no_steps_before_separator(self) -> None:
        example = {"question": "q", "answer": "#### 7"}
        result = _format_gsm8k_prompt(example)
        assert result["solution"] == "7"


class TestGRPORunner:
    def test_init_default_reward_funcs(self) -> None:
        cfg = GRPOConfig()
        runner = GRPORunner(cfg)
        assert len(runner.reward_funcs) == 2

    def test_init_custom_reward_funcs(self) -> None:
        cfg = GRPOConfig()
        custom = [lambda c, **kw: [1.0]]
        runner = GRPORunner(cfg, reward_funcs=custom)
        assert runner.reward_funcs is custom

    def test_save_no_model(self, tmp_path: Path) -> None:
        cfg = GRPOConfig()
        runner = GRPORunner(cfg)
        runner.save(tmp_path / "output")  # should not raise

    def test_save_no_tokenizer(self, tmp_path: Path) -> None:
        cfg = GRPOConfig()
        runner = GRPORunner(cfg)
        runner._model = None
        runner._tokenizer = None
        runner.save(tmp_path / "output")  # should not raise

    def test_load_model(self) -> None:
        cfg = GRPOConfig()
        runner = GRPORunner(cfg)

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
        cfg = GRPOConfig(dataset_size=5)
        runner = GRPORunner(cfg)

        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=100)
        mock_ds.select.return_value = mock_ds
        mock_ds.column_names = ["question", "answer"]
        mock_ds.map.return_value = mock_ds

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            result = runner._load_dataset()
            mock_datasets.load_dataset.assert_called_once_with(
                cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split
            )
            mock_ds.select.assert_called_once()

    def test_load_dataset_no_size_limit(self) -> None:
        cfg = GRPOConfig(dataset_size=None)
        runner = GRPORunner(cfg)

        mock_ds = MagicMock()
        mock_ds.column_names = ["question", "answer"]
        mock_ds.map.return_value = mock_ds

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds

        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            runner._load_dataset()
            mock_ds.select.assert_not_called()

    def test_train(self, tmp_path: Path) -> None:
        cfg = GRPOConfig(output_dir=str(tmp_path / "grpo_output"), reward_weights=[0.7, 0.3])
        runner = GRPORunner(cfg)

        runner._load_model = MagicMock()
        runner._load_dataset = MagicMock(return_value=MagicMock())

        mock_trainer = MagicMock()
        mock_result = MagicMock()
        mock_result.training_loss = 0.4
        mock_result.global_step = 50
        mock_trainer.train.return_value = mock_result
        mock_trainer.state.log_history = [
            {"loss": 0.7, "reward": 0.2},
            {"loss": 0.4, "reward": 0.6},
        ]

        mock_trl = MagicMock()
        mock_trl.GRPOTrainer.return_value = mock_trainer

        with patch.dict("sys.modules", {"trl": mock_trl}):
            result = runner.train()
            assert result.num_steps == 50
            assert result.metrics["train_loss"] == 0.4
            assert result.loss_history == [0.7, 0.4]
            assert result.metrics["reward_history"] == [0.2, 0.6]

    def test_train_sets_reward_weights(self, tmp_path: Path) -> None:
        cfg = GRPOConfig(output_dir=str(tmp_path / "grpo"), reward_weights=[0.8, 0.2])
        runner = GRPORunner(cfg)

        runner._load_model = MagicMock()
        runner._load_dataset = MagicMock(return_value=MagicMock())

        mock_trainer = MagicMock()
        mock_result = MagicMock()
        mock_result.training_loss = 0.3
        mock_result.global_step = 10
        mock_trainer.train.return_value = mock_result
        mock_trainer.state.log_history = []

        mock_trl = MagicMock()
        mock_trl.GRPOTrainer.return_value = mock_trainer

        with patch.dict("sys.modules", {"trl": mock_trl}):
            runner.train()
            grpo_args = mock_trl.GRPOConfig.return_value
            assert grpo_args.reward_weights == [0.8, 0.2]

    def test_save_with_model(self, tmp_path: Path) -> None:
        cfg = GRPOConfig()
        runner = GRPORunner(cfg)
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner.save(tmp_path / "output")
        runner._model.save_pretrained.assert_called_once()
        runner._tokenizer.save_pretrained.assert_called_once()

    def test_load(self, tmp_path: Path) -> None:
        cfg = GRPOConfig()
        runner = GRPORunner(cfg)

        mock_unsloth = MagicMock()
        mock_unsloth.FastLanguageModel.from_pretrained.return_value = (MagicMock(), MagicMock())

        with patch.dict("sys.modules", {"unsloth": mock_unsloth}):
            runner.load(tmp_path)
            assert runner._model is not None
