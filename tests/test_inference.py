"""Tests for inference module."""

from unittest.mock import MagicMock, patch

from alignrl.inference import InferenceConfig, ModelServer, build_prompt


class TestInferenceConfig:
    def test_defaults(self) -> None:
        cfg = InferenceConfig()
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 512

    def test_custom(self) -> None:
        cfg = InferenceConfig(temperature=0.0, max_tokens=256)
        assert cfg.temperature == 0.0

    def test_backend_options(self) -> None:
        for backend in ("unsloth", "vllm", "mlx"):
            cfg = InferenceConfig(backend=backend)
            assert cfg.backend == backend

    def test_adapter_path(self) -> None:
        cfg = InferenceConfig(adapter_path="./my-adapter")
        assert cfg.adapter_path == "./my-adapter"


class TestBuildPrompt:
    def test_math_prompt(self) -> None:
        prompt = build_prompt("What is 2+2?", system="You are a math tutor.")
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        assert prompt[1]["content"] == "What is 2+2?"

    def test_no_system(self) -> None:
        prompt = build_prompt("Hello")
        assert prompt[0]["role"] == "user"
        assert len(prompt) == 1


class TestModelServer:
    def test_init(self) -> None:
        cfg = InferenceConfig()
        server = ModelServer(cfg)
        assert server.config is cfg
        assert server._model is None
        assert server._tokenizer is None

    def test_load_dispatches_vllm(self) -> None:
        cfg = InferenceConfig(backend="vllm")
        server = ModelServer(cfg)
        mock_llm = MagicMock()
        with patch.dict("sys.modules", {"vllm": mock_llm}):
            mock_llm.LLM.return_value = MagicMock()
            server._load_vllm = MagicMock()
            server.load()
            server._load_vllm.assert_called_once()

    def test_load_dispatches_mlx(self) -> None:
        cfg = InferenceConfig(backend="mlx")
        server = ModelServer(cfg)
        server._load_mlx = MagicMock()
        server.load()
        server._load_mlx.assert_called_once()

    def test_load_dispatches_unsloth(self) -> None:
        cfg = InferenceConfig(backend="unsloth")
        server = ModelServer(cfg)
        server._load_unsloth = MagicMock()
        server.load()
        server._load_unsloth.assert_called_once()

    def test_generate_dispatches_vllm(self) -> None:
        cfg = InferenceConfig(backend="vllm")
        server = ModelServer(cfg)
        server._generate_vllm = MagicMock(return_value="answer")
        messages = [{"role": "user", "content": "hi"}]
        result = server.generate(messages)
        server._generate_vllm.assert_called_once_with(messages)
        assert result == "answer"

    def test_generate_dispatches_mlx(self) -> None:
        cfg = InferenceConfig(backend="mlx")
        server = ModelServer(cfg)
        server._generate_mlx = MagicMock(return_value="mlx answer")
        messages = [{"role": "user", "content": "hi"}]
        result = server.generate(messages)
        server._generate_mlx.assert_called_once_with(messages)
        assert result == "mlx answer"

    def test_generate_dispatches_unsloth(self) -> None:
        cfg = InferenceConfig(backend="unsloth")
        server = ModelServer(cfg)
        server._generate_unsloth = MagicMock(return_value="unsloth answer")
        messages = [{"role": "user", "content": "hi"}]
        result = server.generate(messages)
        server._generate_unsloth.assert_called_once_with(messages)
        assert result == "unsloth answer"

    def test_load_vllm_with_adapter(self) -> None:
        cfg = InferenceConfig(backend="vllm", adapter_path="./adapter")
        server = ModelServer(cfg)

        mock_vllm = MagicMock()
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            server._load_vllm()
            call_kwargs = mock_vllm.LLM.call_args[1]
            assert call_kwargs["enable_lora"] is True

    def test_load_vllm_without_adapter(self) -> None:
        cfg = InferenceConfig(backend="vllm", adapter_path=None)
        server = ModelServer(cfg)

        mock_vllm = MagicMock()
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            server._load_vllm()
            call_kwargs = mock_vllm.LLM.call_args[1]
            assert "enable_lora" not in call_kwargs

    def test_load_mlx(self) -> None:
        cfg = InferenceConfig(backend="mlx", model_name="test-model")
        server = ModelServer(cfg)

        mock_mlx = MagicMock()
        mock_mlx.load.return_value = (MagicMock(), MagicMock())
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
            server._load_mlx()
            mock_mlx.load.assert_called_once_with("test-model")
            assert server._model is not None
            assert server._tokenizer is not None

    def test_generate_vllm(self) -> None:
        cfg = InferenceConfig(backend="vllm")
        server = ModelServer(cfg)
        server._model = MagicMock()

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="vllm result")]
        server._model.generate.return_value = [mock_output]

        mock_vllm = MagicMock()
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            result = server._generate_vllm([{"role": "user", "content": "hi"}])
            assert result == "vllm result"

    def test_generate_mlx(self) -> None:
        cfg = InferenceConfig(backend="mlx")
        server = ModelServer(cfg)
        server._model = MagicMock()
        server._tokenizer = MagicMock()
        server._tokenizer.apply_chat_template.return_value = "formatted"

        mock_mlx = MagicMock()
        mock_mlx.generate.return_value = "mlx result"
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
            result = server._generate_mlx([{"role": "user", "content": "hi"}])
            assert result == "mlx result"

    def test_load_unsloth(self) -> None:
        cfg = InferenceConfig(backend="unsloth", model_name="test-model")
        server = ModelServer(cfg)

        mock_unsloth = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_unsloth.FastLanguageModel.from_pretrained.return_value = (mock_model, mock_tokenizer)

        with patch.dict("sys.modules", {"unsloth": mock_unsloth}):
            server._load_unsloth()
            mock_unsloth.FastLanguageModel.from_pretrained.assert_called_once()
            mock_unsloth.FastLanguageModel.for_inference.assert_called_once_with(mock_model)
            assert server._model is mock_model

    def test_load_unsloth_with_adapter(self) -> None:
        cfg = InferenceConfig(backend="unsloth", adapter_path="./adapter")
        server = ModelServer(cfg)

        mock_unsloth = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "existing"
        mock_unsloth.FastLanguageModel.from_pretrained.return_value = (mock_model, mock_tokenizer)

        with patch.dict("sys.modules", {"unsloth": mock_unsloth}):
            server._load_unsloth()
            call_kwargs = mock_unsloth.FastLanguageModel.from_pretrained.call_args[1]
            assert call_kwargs["model_name"] == "./adapter"

    def test_generate_unsloth(self) -> None:
        cfg = InferenceConfig(backend="unsloth", temperature=0.5)
        server = ModelServer(cfg)
        server._model = MagicMock()
        server._tokenizer = MagicMock()

        import torch

        mock_inputs = torch.tensor([[1, 2, 3]])
        server._tokenizer.apply_chat_template.return_value = mock_inputs
        mock_inputs_on_device = MagicMock()
        mock_inputs_on_device.shape = MagicMock()
        mock_inputs_on_device.shape.__getitem__ = MagicMock(return_value=3)
        server._tokenizer.apply_chat_template.return_value = MagicMock()
        server._tokenizer.apply_chat_template.return_value.to.return_value = mock_inputs_on_device

        mock_outputs = MagicMock()
        mock_outputs.__getitem__ = MagicMock(return_value=MagicMock())
        server._model.generate.return_value = mock_outputs
        server._tokenizer.decode.return_value = "unsloth result"

        result = server._generate_unsloth([{"role": "user", "content": "hi"}])
        assert result == "unsloth result"
        server._model.generate.assert_called_once()

    def test_generate_unsloth_zero_temp(self) -> None:
        cfg = InferenceConfig(backend="unsloth", temperature=0.0)
        server = ModelServer(cfg)
        server._model = MagicMock()
        server._tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.shape = MagicMock()
        mock_inputs.shape.__getitem__ = MagicMock(return_value=3)
        server._tokenizer.apply_chat_template.return_value = MagicMock()
        server._tokenizer.apply_chat_template.return_value.to.return_value = mock_inputs

        server._model.generate.return_value = MagicMock()
        server._tokenizer.decode.return_value = "deterministic"

        result = server._generate_unsloth([{"role": "user", "content": "hi"}])
        call_kwargs = server._model.generate.call_args[1]
        assert call_kwargs["do_sample"] is False
