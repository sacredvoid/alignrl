"""Tests for evaluation module."""

import json
from pathlib import Path

from alignrl.eval import EvalConfig, compare_stages, parse_results
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
