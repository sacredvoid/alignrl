"""Tests for demo module."""

from unittest.mock import MagicMock, patch

from alignrl.demo import MATH_SYSTEM, create_demo


class TestCreateDemo:
    def test_creates_gradio_app(self) -> None:
        mock_gr = MagicMock()
        mock_blocks = MagicMock()
        mock_gr.Blocks.return_value.__enter__ = MagicMock(return_value=mock_blocks)
        mock_gr.Blocks.return_value.__exit__ = MagicMock(return_value=False)

        mock_server = MagicMock()
        mock_server.generate.return_value = "test output"

        with (
            patch.dict("sys.modules", {"gradio": mock_gr}),
            patch("alignrl.demo.ModelServer", return_value=mock_server),
            patch("alignrl.demo.InferenceConfig"),
        ):
            mock_server.load = MagicMock()
            app = create_demo(stages={"base": None}, model_name="test-model")
            assert app is not None
            mock_server.load.assert_called_once()

    def test_multiple_stages(self) -> None:
        mock_gr = MagicMock()
        mock_blocks = MagicMock()
        mock_gr.Blocks.return_value.__enter__ = MagicMock(return_value=mock_blocks)
        mock_gr.Blocks.return_value.__exit__ = MagicMock(return_value=False)

        mock_server = MagicMock()

        with (
            patch.dict("sys.modules", {"gradio": mock_gr}),
            patch("alignrl.demo.ModelServer", return_value=mock_server),
            patch("alignrl.demo.InferenceConfig"),
        ):
            create_demo(
                stages={"base": None, "sft": "./sft", "grpo": "./grpo"},
                model_name="test-model",
            )
            assert mock_server.load.call_count == 3

    def test_math_system_prompt(self) -> None:
        assert "math" in MATH_SYSTEM.lower()
        assert "boxed" in MATH_SYSTEM.lower()

    def test_respond_and_respond_all(self) -> None:
        """Test the inner respond/respond_all functions by capturing Gradio callbacks."""
        mock_gr = MagicMock()
        mock_blocks = MagicMock()
        mock_gr.Blocks.return_value.__enter__ = MagicMock(return_value=mock_blocks)
        mock_gr.Blocks.return_value.__exit__ = MagicMock(return_value=False)

        mock_server = MagicMock()
        mock_server.generate.return_value = "test answer"

        # Capture the submit.click callback (respond_all)
        captured_fn = None
        mock_button = MagicMock()

        def capture_click(fn, **kwargs):
            nonlocal captured_fn
            captured_fn = fn

        mock_button.click.side_effect = capture_click
        mock_gr.Button.return_value = mock_button

        with (
            patch.dict("sys.modules", {"gradio": mock_gr}),
            patch("alignrl.demo.ModelServer", return_value=mock_server),
            patch("alignrl.demo.InferenceConfig"),
        ):
            create_demo(stages={"base": None, "sft": "./sft"}, model_name="test")

        # Now call the captured respond_all function
        assert captured_fn is not None
        results = captured_fn("What is 2+2?")
        assert len(results) == 2
        assert mock_server.generate.call_count >= 2
