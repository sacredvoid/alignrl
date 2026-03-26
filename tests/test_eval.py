"""Tests for evaluation module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from alignrl.eval import EvalConfig, EvalRunner, compare_stages, parse_results
from alignrl.types import EvalResult


class TestEvalConfig:
    def test_defaults(self) -> None:
        cfg = EvalConfig()
        assert "gsm8k" in cfg.tasks
        assert cfg.num_fewshot == 0

    def test_custom_tasks(self) -> None:
        cfg = EvalConfig(tasks=["mmlu", "hellaswag"])
        assert cfg.tasks == ["mmlu", "hellaswag"]


class TestParseResults:
    def test_parses_lm_eval_output(self) -> None:
        raw = {
            "results": {
                "gsm8k": {"exact_match,strict-match": 0.45, "exact_match,flexible-extract": 0.48},
                "arc_challenge": {"acc_norm,none": 0.52},
            }
        }
        result = parse_results(raw, model_name="test", stage="sft")
        assert isinstance(result, EvalResult)
        assert "gsm8k" in result.benchmarks
        assert result.stage == "sft"

    def test_handles_empty_results(self) -> None:
        result = parse_results({"results": {}}, model_name="test", stage="base")
        assert result.benchmarks == {}


class TestCompareStages:
    def test_generates_comparison(self) -> None:
        base = EvalResult(
            model_name="qwen",
            stage="base",
            benchmarks={"gsm8k": {"exact_match": 0.30}},
        )
        sft = EvalResult(
            model_name="qwen",
            stage="sft",
            benchmarks={"gsm8k": {"exact_match": 0.45}},
        )
        comparison = compare_stages([base, sft])
        assert "gsm8k" in comparison
        assert comparison["gsm8k"]["base"]["exact_match"] == 0.30
        assert comparison["gsm8k"]["sft"]["exact_match"] == 0.45

    def test_serializes_to_json(self, tmp_path: Path) -> None:
        base = EvalResult(
            model_name="qwen",
            stage="base",
            benchmarks={"gsm8k": {"exact_match": 0.30}},
        )
        comparison = compare_stages([base])
        out = tmp_path / "comparison.json"
        with open(out, "w") as f:
            json.dump(comparison, f)
        loaded = json.loads(out.read_text())
        assert loaded["gsm8k"]["base"]["exact_match"] == 0.30


class TestEvalRunner:
    def test_init(self) -> None:
        cfg = EvalConfig()
        runner = EvalRunner(cfg)
        assert runner.config is cfg

    def test_save_results(self, tmp_path: Path) -> None:
        base = EvalResult(
            model_name="qwen", stage="base", benchmarks={"gsm8k": {"exact_match": 0.30}}
        )
        sft = EvalResult(
            model_name="qwen", stage="sft", benchmarks={"gsm8k": {"exact_match": 0.45}}
        )

        cfg = EvalConfig()
        runner = EvalRunner(cfg)
        output_dir = tmp_path / "eval_output"
        runner.save_results([base, sft], output_dir)

        assert (output_dir / "eval_base.json").exists()
        assert (output_dir / "eval_sft.json").exists()
        assert (output_dir / "comparison.json").exists()

        comparison = json.loads((output_dir / "comparison.json").read_text())
        assert comparison["gsm8k"]["base"]["exact_match"] == 0.30
        assert comparison["gsm8k"]["sft"]["exact_match"] == 0.45

    def test_save_results_creates_dir(self, tmp_path: Path) -> None:
        result = EvalResult(
            model_name="qwen", stage="base", benchmarks={"arc": {"acc": 0.5}}
        )
        cfg = EvalConfig()
        runner = EvalRunner(cfg)
        nested = tmp_path / "deep" / "nested" / "dir"
        runner.save_results([result], nested)
        assert nested.exists()
        assert (nested / "eval_base.json").exists()

    def test_evaluate_all_stages(self) -> None:
        cfg = EvalConfig()
        runner = EvalRunner(cfg)

        mock_result_base = EvalResult(
            model_name="qwen", stage="base", benchmarks={"gsm8k": {"exact_match": 0.30}}
        )
        mock_result_sft = EvalResult(
            model_name="qwen", stage="sft", benchmarks={"gsm8k": {"exact_match": 0.45}}
        )

        with patch.object(runner, "evaluate", side_effect=[mock_result_base, mock_result_sft]):
            results = runner.evaluate_all_stages({"base": None, "sft": "./outputs/sft"})
            assert len(results) == 2
            assert results[0].stage == "base"

    def test_evaluate_all_stages_restores_config(self) -> None:
        cfg = EvalConfig(adapter_path="original")
        runner = EvalRunner(cfg)

        mock_result = EvalResult(
            model_name="qwen", stage="base", benchmarks={}
        )

        with patch.object(runner, "evaluate", return_value=mock_result):
            runner.evaluate_all_stages({"base": None, "sft": "./adapter"})
            assert cfg.adapter_path == "original"

    def test_evaluate_builds_model_args(self) -> None:
        cfg = EvalConfig(model_name="test-model", load_in_4bit=True, adapter_path="./adapter")
        runner = EvalRunner(cfg)

        mock_raw = {
            "results": {"gsm8k": {"exact_match,strict-match": 0.50}}
        }

        mock_lm = MagicMock()
        mock_lm.simple_evaluate.return_value = mock_raw

        with patch.dict("sys.modules", {"lm_eval": mock_lm}):
            result = runner.evaluate(stage="sft")
            call_kwargs = mock_lm.simple_evaluate.call_args[1]
            assert "pretrained=test-model" in call_kwargs["model_args"]
            assert "load_in_4bit=True" in call_kwargs["model_args"]
            assert "peft=./adapter" in call_kwargs["model_args"]
            assert result.stage == "sft"


class TestParseResultsEdgeCases:
    def test_filters_non_numeric(self) -> None:
        raw = {
            "results": {
                "gsm8k": {
                    "exact_match": 0.5,
                    "alias": "gsm8k_main",
                }
            }
        }
        result = parse_results(raw, model_name="test", stage="base")
        assert "exact_match" in result.benchmarks["gsm8k"]
        assert "alias" not in result.benchmarks["gsm8k"]

    def test_no_results_key(self) -> None:
        result = parse_results({}, model_name="test", stage="base")
        assert result.benchmarks == {}
