"""Tests for CLI module."""

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from alignrl.cli import cmd_eval, cmd_serve, cmd_train, cmd_version, main


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

    def test_serve_requires_stages(self) -> None:
        with pytest.raises(SystemExit):
            sys.argv = ["alignrl", "serve", "--model", "test"]
            main()


class TestCmdTrain:
    def test_train_sft_stage(self, tmp_path) -> None:
        config_path = tmp_path / "sft.yaml"
        config_path.write_text("model_name: test\nmax_steps: 1\n")
        args = argparse.Namespace(config=str(config_path), stage="sft", push=None)

        mock_runner = MagicMock()
        mock_runner.train.return_value = MagicMock(output_dir=tmp_path, metrics={"train_loss": 0.5})

        with (
            patch("alignrl.sft.SFTRunner", return_value=mock_runner) as mock_cls,
            patch("alignrl.sft.SFTConfig") as mock_cfg_cls,
        ):
            mock_cfg_cls.from_yaml.return_value = MagicMock()
            cmd_train(args)
            mock_cls.assert_called_once()
            mock_runner.train.assert_called_once()

    def test_train_grpo_stage(self, tmp_path) -> None:
        config_path = tmp_path / "grpo.yaml"
        config_path.write_text("model_name: test\n")
        args = argparse.Namespace(config=str(config_path), stage="grpo", push=None)

        mock_runner = MagicMock()
        mock_runner.train.return_value = MagicMock(output_dir=tmp_path, metrics={"train_loss": 0.3})

        with (
            patch("alignrl.grpo.GRPORunner", return_value=mock_runner),
            patch("alignrl.grpo.GRPOConfig") as mock_cfg_cls,
        ):
            mock_cfg_cls.from_yaml.return_value = MagicMock()
            cmd_train(args)
            mock_runner.train.assert_called_once()

    def test_train_dpo_stage(self, tmp_path) -> None:
        config_path = tmp_path / "dpo.yaml"
        config_path.write_text("model_name: test\n")
        args = argparse.Namespace(config=str(config_path), stage="dpo", push=None)

        mock_runner = MagicMock()
        mock_runner.train.return_value = MagicMock(output_dir=tmp_path, metrics={"train_loss": 0.2})

        with (
            patch("alignrl.dpo.DPORunner", return_value=mock_runner),
            patch("alignrl.dpo.DPOConfig") as mock_cfg_cls,
        ):
            mock_cfg_cls.from_yaml.return_value = MagicMock()
            cmd_train(args)
            mock_runner.train.assert_called_once()

    def test_train_unknown_stage_exits(self, tmp_path) -> None:
        config_path = tmp_path / "test.yaml"
        config_path.write_text("model_name: test\n")
        args = argparse.Namespace(config=str(config_path), stage="unknown", push=None)
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
            preset=None,
            limit=None,
            stage="base",
            output=str(tmp_path / "results"),
            wandb=False,
            num_fewshot=None,
            batch_size=None,
        )

        with (
            patch("alignrl.eval.EvalRunner", return_value=mock_runner),
            patch("alignrl.eval.EvalConfig"),
        ):
            cmd_eval(args)
            mock_runner.evaluate.assert_called_once_with(stage="base")
            assert (tmp_path / "results" / "eval_base.json").exists()

    def test_eval_forwards_num_fewshot_and_batch_size(self, tmp_path) -> None:
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"benchmarks": {}}
        mock_result.benchmarks = {}
        mock_runner = MagicMock()
        mock_runner.evaluate.return_value = mock_result

        args = argparse.Namespace(
            model="test-model",
            adapter=None,
            tasks=None,
            preset=None,
            limit=None,
            stage="sft",
            output=str(tmp_path / "results"),
            wandb=False,
            num_fewshot=5,
            batch_size="8",
        )

        with (
            patch("alignrl.eval.EvalRunner", return_value=mock_runner),
            patch("alignrl.eval.EvalConfig") as mock_cfg_cls,
        ):
            cmd_eval(args)
            call_kwargs = mock_cfg_cls.call_args.kwargs
            assert call_kwargs["num_fewshot"] == 5
            assert call_kwargs["batch_size"] == "8"


class TestMainEntry:
    def test_main_module_execution(self) -> None:
        """Test if __name__ == '__main__' block in cli.py."""
        import contextlib
        import runpy

        with (
            patch("alignrl.cli.main"),
            patch("alignrl.cli.__name__", "__main__"),
            contextlib.suppress(SystemExit),
        ):
            runpy.run_module("alignrl.cli", run_name="__main__", alter_sys=True)


class TestCmdServe:
    def test_serve_parses_stage_specs(self) -> None:
        mock_demo = MagicMock()

        args = argparse.Namespace(
            model="test-model",
            stages=["base", "sft=./outputs/sft/final"],
            port=7860,
            share=False,
            temperature=None,
            max_tokens=None,
        )

        with patch("alignrl.demo.create_demo", return_value=mock_demo) as mock_create:
            cmd_serve(args)
            call_args = mock_create.call_args
            stages = call_args.kwargs.get("stages", call_args[1].get("stages"))
            assert stages["base"] is None
            assert stages["sft"] == "./outputs/sft/final"
            mock_demo.launch.assert_called_once()

    def test_serve_forwards_generation_params(self) -> None:
        mock_demo = MagicMock()

        args = argparse.Namespace(
            model="test-model",
            stages=["base"],
            port=7860,
            share=False,
            temperature=0.2,
            max_tokens=256,
        )

        with patch("alignrl.demo.create_demo", return_value=mock_demo) as mock_create:
            cmd_serve(args)
            kwargs = mock_create.call_args.kwargs
            assert kwargs["temperature"] == 0.2
            assert kwargs["max_tokens"] == 256


class TestCmdVersion:
    def test_version_subcommand_prints_version(self, capsys: pytest.CaptureFixture) -> None:
        from alignrl import __version__

        cmd_version(argparse.Namespace())
        captured = capsys.readouterr()
        assert __version__ in captured.out
        assert "alignrl" in captured.out

    def test_version_via_main(self, capsys: pytest.CaptureFixture) -> None:
        from alignrl import __version__

        sys.argv = ["alignrl", "version"]
        main()
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_version_flag(self, capsys: pytest.CaptureFixture) -> None:
        from alignrl import __version__

        sys.argv = ["alignrl", "--version"]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out
