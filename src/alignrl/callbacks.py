"""Weights & Biases integration for alignrl."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alignrl.types import EvalResult


def detect_wandb() -> str:
    """Return 'wandb' if wandb is installed and an API key is configured, else 'none'."""
    try:
        import wandb

        if wandb.api.api_key:
            return "wandb"
    except Exception:
        pass
    return "none"


def log_eval_to_wandb(
    results: list[EvalResult],
    project: str = "alignrl",
) -> None:
    """Log evaluation results to Weights & Biases.

    Logs each stage's benchmark metrics as W&B summary values and creates
    a comparison table across all stages.

    Args:
        results: List of EvalResult objects from different training stages
        project: W&B project name (used only if no active run exists)
    """
    import wandb

    if not wandb.run:
        wandb.init(project=project)

    for result in results:
        for bench, metrics in result.benchmarks.items():
            for metric_name, value in metrics.items():
                key = f"{result.stage}/{bench}/{metric_name}"
                wandb.run.summary[key] = value

    columns = ["stage", "benchmark", "metric", "value"]
    table = wandb.Table(columns=columns)
    for result in results:
        for bench, metrics in result.benchmarks.items():
            for name, val in metrics.items():
                table.add_data(result.stage, bench, name, val)
    wandb.log({"eval_comparison": table})
