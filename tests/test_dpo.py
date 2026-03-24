"""Tests for DPO module."""
from alignrl.dpo import DPOConfig, format_ultrafeedback


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
