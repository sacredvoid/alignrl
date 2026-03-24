"""Tests for CLI module."""

import pytest

from alignrl.cli import main


class TestCLIParser:
    def test_train_requires_config(self) -> None:
        with pytest.raises(SystemExit):
            import sys

            sys.argv = ["alignrl", "train", "sft"]
            main()

    def test_train_rejects_unknown_stage(self) -> None:
        with pytest.raises(SystemExit):
            import sys

            sys.argv = ["alignrl", "train", "unknown", "-c", "configs/sft.yaml"]
            main()

    def test_no_command_exits(self) -> None:
        with pytest.raises(SystemExit):
            import sys

            sys.argv = ["alignrl"]
            main()

    def test_train_missing_config_file(self, capsys: pytest.CaptureFixture) -> None:
        import sys

        sys.argv = ["alignrl", "train", "sft", "-c", "/nonexistent/config.yaml"]
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower() or "not found" in captured.out.lower()
