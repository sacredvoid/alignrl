"""Tests for W&B integration callbacks."""

from unittest.mock import MagicMock, patch

from alignrl.callbacks import detect_wandb, log_eval_to_wandb
from alignrl.types import EvalResult


class TestDetectWandb:
    def test_returns_wandb_when_configured(self) -> None:
        mock_wandb = MagicMock()
        mock_wandb.api.api_key = "test-key"
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            assert detect_wandb() == "wandb"

    def test_returns_none_when_no_key(self) -> None:
        mock_wandb = MagicMock()
        mock_wandb.api.api_key = None
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            assert detect_wandb() == "none"

    def test_returns_none_when_not_installed(self) -> None:
        with patch.dict("sys.modules", {"wandb": None}):
            assert detect_wandb() == "none"


class TestLogEvalToWandb:
    def test_logs_metrics_and_table(self) -> None:
        results = [
            EvalResult(
                model_name="qwen",
                stage="base",
                benchmarks={"gsm8k": {"exact_match": 0.30}},
            ),
            EvalResult(
                model_name="qwen",
                stage="sft",
                benchmarks={"gsm8k": {"exact_match": 0.45}},
            ),
        ]

        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.summary = {}
        mock_wandb.run = mock_run
        mock_wandb.Table.return_value = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            log_eval_to_wandb(results)
            assert mock_run.summary["base/gsm8k/exact_match"] == 0.30
            assert mock_run.summary["sft/gsm8k/exact_match"] == 0.45
            mock_wandb.log.assert_called_once()
            mock_wandb.Table.assert_called_once()

    def test_inits_wandb_if_no_run(self) -> None:
        results = [
            EvalResult(
                model_name="qwen",
                stage="base",
                benchmarks={"gsm8k": {"exact_match": 0.30}},
            ),
        ]

        mock_wandb = MagicMock()
        mock_wandb.run = None
        mock_new_run = MagicMock()
        mock_wandb.init.return_value = mock_new_run

        def set_run(*args, **kwargs):
            mock_wandb.run = mock_new_run
            return mock_new_run

        mock_wandb.init.side_effect = set_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            log_eval_to_wandb(results, project="test-project")
            mock_wandb.init.assert_called_once_with(project="test-project")
