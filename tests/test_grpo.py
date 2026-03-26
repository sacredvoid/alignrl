"""Tests for GRPO module."""

from pathlib import Path

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
