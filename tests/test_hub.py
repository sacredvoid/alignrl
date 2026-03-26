"""Tests for HuggingFace Hub integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from alignrl.hub import push_adapter


class TestPushAdapter:
    def test_creates_repo_and_uploads(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_text("fake")

        mock_api = MagicMock()
        mock_hf = MagicMock()
        mock_hf.HfApi.return_value = mock_api

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            url = push_adapter(adapter_dir, "user/test-model")
            assert url == "https://huggingface.co/user/test-model"
            mock_api.create_repo.assert_called_once_with(
                "user/test-model", exist_ok=True, private=False
            )
            mock_api.upload_folder.assert_called_once()

    def test_private_repo(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        mock_api = MagicMock()
        mock_hf = MagicMock()
        mock_hf.HfApi.return_value = mock_api

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            push_adapter(adapter_dir, "user/private-model", private=True)
            mock_api.create_repo.assert_called_once_with(
                "user/private-model", exist_ok=True, private=True
            )


class TestMergeAndPush:
    def test_merge_loads_and_pushes(self) -> None:
        from alignrl.hub import merge_and_push

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_unsloth = MagicMock()
        mock_unsloth.FastLanguageModel.from_pretrained.return_value = (
            mock_model,
            mock_tokenizer,
        )

        with patch.dict("sys.modules", {"unsloth": mock_unsloth}):
            url = merge_and_push(
                model_name="Qwen/Qwen2.5-3B",
                adapter_path="./outputs/sft/final",
                repo_id="user/merged-model",
            )
            assert url == "https://huggingface.co/user/merged-model"
            mock_model.push_to_hub_merged.assert_called_once()


class TestRunnerPushToHub:
    def test_sft_push_adapter(self) -> None:
        from alignrl.sft import SFTConfig, SFTRunner

        cfg = SFTConfig(output_dir="./outputs/sft")
        runner = SFTRunner(cfg)

        with patch("alignrl.hub.push_adapter", return_value="https://huggingface.co/u/m") as mock:
            url = runner.push_to_hub("u/m")
            assert url == "https://huggingface.co/u/m"
            mock.assert_called_once()

    def test_sft_push_merged(self) -> None:
        from alignrl.sft import SFTConfig, SFTRunner

        cfg = SFTConfig(output_dir="./outputs/sft")
        runner = SFTRunner(cfg)

        with patch("alignrl.hub.merge_and_push", return_value="https://huggingface.co/u/m") as mock:
            url = runner.push_to_hub("u/m", merge=True)
            assert url == "https://huggingface.co/u/m"
            mock.assert_called_once()

    def test_dpo_push_adapter(self) -> None:
        from alignrl.dpo import DPOConfig, DPORunner

        cfg = DPOConfig(output_dir="./outputs/dpo")
        runner = DPORunner(cfg)

        with patch("alignrl.hub.push_adapter", return_value="https://huggingface.co/u/m") as mock:
            runner.push_to_hub("u/m")
            mock.assert_called_once()

    def test_grpo_push_adapter(self) -> None:
        from alignrl.grpo import GRPOConfig, GRPORunner

        cfg = GRPOConfig(output_dir="./outputs/grpo")
        runner = GRPORunner(cfg)

        with patch("alignrl.hub.push_adapter", return_value="https://huggingface.co/u/m") as mock:
            runner.push_to_hub("u/m")
            mock.assert_called_once()
