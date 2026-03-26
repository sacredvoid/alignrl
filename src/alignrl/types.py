"""Shared types and protocols for alignrl."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Result from a training run."""

    output_dir: Path
    loss_history: list[float]
    metrics: dict[str, Any]
    num_steps: int
    num_epochs: float


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Result from an evaluation run."""

    model_name: str
    stage: str  # "base", "sft", "grpo", "dpo"
    benchmarks: dict[str, dict[str, float]]  # benchmark -> metric -> score
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "stage": self.stage,
            "benchmarks": self.benchmarks,
            "metadata": self.metadata,
        }


@runtime_checkable
class Trainer(Protocol):
    """Common interface for all trainers."""

    def train(self) -> TrainResult: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
