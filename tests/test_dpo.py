"""Tests for DPO module."""

from pathlib import Path

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
