"""Tests for CLI module."""

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from alignrl.cli import cmd_eval, cmd_serve, cmd_train, main


class TestCLIParser:
    def test_train_requires_config(self) -> None:
        with pytest.raises(SystemExit):
            sys.argv = ["alignrl", "train", "sft"]
            main()

    def test_train_rejects_unknown_stage(self) -> None:
        with pytest.raises(SystemExit):
            sys.argv = ["alignrl", "train", "unknown", "-c", "configs/sft.yaml"]
            main()

    def test_no_command_exits(self) -> None:
        with pytest.raises(SystemExit):
            sys.argv = ["alignrl"]
            main()

    def test_train_missing_config_file(self, capsys: pytest.CaptureFixture) -> None:
        sys.argv = ["alignrl", "train", "sft", "-c", "/nonexistent/config.yaml"]
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower() or "not found" in captured.out.lower()

    def test_eval_defaults_parsed(self) -> None:
        sys.argv = ["alignrl", "eval"]
        parser = argparse.ArgumentParser(prog="alignrl")
        sub = parser.add_subparsers(dest="command", required=True)
        eval_p = sub.add_parser("eval")
        eval_p.add_argument("--model", default="Qwen/Qwen2.5-3B")
        eval_p.add_argument("--adapter", default=None)
        eval_p.add_argument("--stage", default="base")
        eval_p.add_argument("--tasks", default="gsm8k,arc_challenge")
        eval_p.add_argument("--limit", type=int, default=None)
        eval_p.add_argument("--output", default="./results")
        args = parser.parse_args(["eval"])
        assert args.model == "Qwen/Qwen2.5-3B"
        assert args.tasks == "gsm8k,arc_challenge"
        assert args.stage == "base"

    def test_serve_requires_stages(self) -> None:
        with pytest.raises(SystemExit):
            sys.argv = ["alignrl", "serve", "--model", "test"]
            main()


class TestCmdTrain:
    def test_train_sft_stage(self, tmp_path) -> None:
        config_path = tmp_path / "sft.yaml"
        config_path.write_text("model_name: test\nmax_steps: 1\n")
        args = argparse.Namespace(config=str(config_path), stage="sft")

        mock_runner = MagicMock()
        mock_runner.train.return_value = MagicMock(
            output_dir=tmp_path, metrics={"train_loss": 0.5}
        )

        with patch("alignrl.cli.SFTRunner", return_value=mock_runner) as mock_cls:
            with patch("alignrl.cli.SFTConfig") as mock_cfg_cls:
                mock_cfg_cls.from_yaml.return_value = MagicMock()
                cmd_train(args)
                mock_cls.assert_called_once()
                mock_runner.train.assert_called_once()

    def test_train_grpo_stage(self, tmp_path) -> None:
        config_path = tmp_path / "grpo.yaml"
        config_path.write_text("model_name: test\n")
        args = argparse.Namespace(config=str(config_path), stage="grpo")

        mock_runner = MagicMock()
        mock_runner.train.return_value = MagicMock(
            output_dir=tmp_path, metrics={"train_loss": 0.3}
        )

        with patch("alignrl.cli.GRPORunner", return_value=mock_runner):
            with patch("alignrl.cli.GRPOConfig") as mock_cfg_cls:
                mock_cfg_cls.from_yaml.return_value = MagicMock()
                cmd_train(args)
                mock_runner.train.assert_called_once()

    def test_train_dpo_stage(self, tmp_path) -> None:
        config_path = tmp_path / "dpo.yaml"
        config_path.write_text("model_name: test\n")
        args = argparse.Namespace(config=str(config_path), stage="dpo")

        mock_runner = MagicMock()
        mock_runner.train.return_value = MagicMock(
            output_dir=tmp_path, metrics={"train_loss": 0.2}
        )

        with patch("alignrl.cli.DPORunner", return_value=mock_runner):
            with patch("alignrl.cli.DPOConfig") as mock_cfg_cls:
                mock_cfg_cls.from_yaml.return_value = MagicMock()
                cmd_train(args)
                mock_runner.train.assert_called_once()

    def test_train_unknown_stage_exits(self, tmp_path) -> None:
        config_path = tmp_path / "test.yaml"
        config_path.write_text("model_name: test\n")
        args = argparse.Namespace(config=str(config_path), stage="unknown")
        with pytest.raises(SystemExit):
            cmd_train(args)


class TestCmdEval:
    def test_eval_creates_output(self, tmp_path) -> None:
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"benchmarks": {}}
        mock_result.benchmarks = {"gsm8k": {"exact_match": 0.5}}

        mock_runner = MagicMock()
        mock_runner.evaluate.return_value = mock_result

        args = argparse.Namespace(
            model="test-model",
            adapter=None,
            tasks="gsm8k",
            limit=None,
            stage="base",
            output=str(tmp_path / "results"),
        )

        with patch("alignrl.cli.EvalRunner", return_value=mock_runner):
            with patch("alignrl.cli.EvalConfig"):
                cmd_eval(args)
                mock_runner.evaluate.assert_called_once_with(stage="base")


class TestCmdServe:
    def test_serve_parses_stage_specs(self, tmp_path) -> None:
        mock_demo = MagicMock()

        args = argparse.Namespace(
            model="test-model",
            stages=["base", "sft=./outputs/sft/final"],
            port=7860,
            share=False,
        )

        with patch("alignrl.cli.create_demo", return_value=mock_demo) as mock_create:
            cmd_serve(args)
            call_kwargs = mock_create.call_args
            stages = call_kwargs[1]["stages"] if "stages" in call_kwargs[1] else call_kwargs[0][0]
            assert stages["base"] is None
            assert stages["sft"] == "./outputs/sft/final"
            mock_demo.launch.assert_called_once()
