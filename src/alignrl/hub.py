"""HuggingFace Hub integration for pushing trained models."""

from __future__ import annotations

from pathlib import Path


def push_adapter(
    output_dir: str | Path,
    repo_id: str,
    private: bool = False,
) -> str:
    """Push a LoRA adapter directory to the HuggingFace Hub.

    Args:
        output_dir: Path to the adapter directory (contains adapter_model.safetensors etc.)
        repo_id: HuggingFace repo ID (e.g. "user/my-sft-adapter")
        private: Whether to create a private repo

    Returns:
        The URL of the created/updated repo.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=private)
    api.upload_folder(folder_path=str(output_dir), repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


def merge_and_push(
    model_name: str,
    adapter_path: str | Path,
    repo_id: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    private: bool = False,
) -> str:
    """Merge a LoRA adapter into the base model and push to HuggingFace Hub.

    Args:
        model_name: Base model name (e.g. "Qwen/Qwen2.5-3B")
        adapter_path: Path to the LoRA adapter directory
        repo_id: HuggingFace repo ID for the merged model
        max_seq_length: Max sequence length used during training
        load_in_4bit: Whether to load base model in 4-bit
        private: Whether to create a private repo

    Returns:
        The URL of the created/updated repo.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    model.push_to_hub_merged(
        repo_id,
        tokenizer,
        save_method="merged_16bit",
        private=private,
    )
    return f"https://huggingface.co/{repo_id}"
