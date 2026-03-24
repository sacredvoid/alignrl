"""Tests for shared types."""

from pathlib import Path

import pytest

from alignrl.types import EvalResult, TrainResult


class TestTrainResult:
    def test_frozen(self) -> None:
        result = TrainResult(
            output_dir=Path("./out"),
            loss_history=[1.0, 0.5],
            metrics={"train_loss": 0.5},
            num_steps=100,
            num_epochs=1.0,
        )
        assert result.num_steps == 100
        assert result.loss_history == [1.0, 0.5]

    def test_immutable(self) -> None:
        result = TrainResult(
            output_dir=Path("./out"),
            loss_history=[],
            metrics={},
            num_steps=0,
            num_epochs=0.0,
        )
        with pytest.raises(AttributeError):
            result.num_steps = 50  # type: ignore


class TestEvalResult:
    def test_to_dict(self) -> None:
        result = EvalResult(
            model_name="test",
            stage="sft",
            benchmarks={"gsm8k": {"exact_match": 0.45}},
            metadata={"note": "test run"},
        )
        d = result.to_dict()
        assert d["model_name"] == "test"
        assert d["stage"] == "sft"
        assert d["benchmarks"]["gsm8k"]["exact_match"] == 0.45
        assert d["metadata"]["note"] == "test run"

    def test_default_metadata(self) -> None:
        result = EvalResult(
            model_name="test",
            stage="base",
            benchmarks={},
        )
        assert result.metadata == {}

    def test_frozen(self) -> None:
        result = EvalResult(model_name="x", stage="base", benchmarks={})
        with pytest.raises(AttributeError):
            result.stage = "sft"  # type: ignore
