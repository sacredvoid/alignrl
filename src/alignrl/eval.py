"""Evaluation harness wrapper for benchmarking across training stages."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from alignrl.config import BaseTrainConfig
from alignrl.types import EvalResult


class EvalConfig(BaseTrainConfig):
    """Evaluation configuration."""

    tasks: list[str] = Field(default=["gsm8k", "arc_challenge"])
    num_fewshot: int = 0
    batch_size: str = "auto"
    limit: int | None = None
    adapter_path: str | None = None


def parse_results(raw: dict[str, Any], model_name: str, stage: str) -> EvalResult:
    """Parse lm-evaluation-harness output into EvalResult."""
    benchmarks: dict[str, dict[str, float]] = {}
    for task_name, metrics in raw.get("results", {}).items():
        benchmarks[task_name] = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float))
        }
    return EvalResult(model_name=model_name, stage=stage, benchmarks=benchmarks)


def compare_stages(results: list[EvalResult]) -> dict[str, dict[str, dict[str, float]]]:
    """Compare eval results across training stages.

    Returns: {benchmark: {stage: {metric: score}}}
    """
    comparison: dict[str, dict[str, dict[str, float]]] = {}
    for result in results:
        for benchmark, metrics in result.benchmarks.items():
            if benchmark not in comparison:
                comparison[benchmark] = {}
            comparison[benchmark][result.stage] = metrics
    return comparison


class EvalRunner:
    """Runs evaluation benchmarks."""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def evaluate(self, stage: str = "base") -> EvalResult:
        """Run evaluation and return structured results."""
        import lm_eval

        model_args = f"pretrained={self.config.model_name}"
        if self.config.load_in_4bit:
            model_args += ",load_in_4bit=True"
        if self.config.adapter_path:
            model_args += f",peft={self.config.adapter_path}"

        raw = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            batch_size=self.config.batch_size,
            limit=self.config.limit,
        )

        return parse_results(raw, model_name=self.config.model_name, stage=stage)

    def evaluate_all_stages(
        self, adapter_paths: dict[str, str | None]
    ) -> list[EvalResult]:
        """Evaluate multiple stages and return comparison-ready results."""
        results = []
        for stage, adapter_path in adapter_paths.items():
            self.config.adapter_path = adapter_path
            result = self.evaluate(stage=stage)
            results.append(result)
        return results

    def save_results(self, results: list[EvalResult], output_dir: Path) -> None:
        """Save results as JSON for GitHub Pages consumption."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            path = output_dir / f"eval_{result.stage}.json"
            with open(path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

        comparison = compare_stages(results)
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
