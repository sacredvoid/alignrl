"""Base runner with shared model loading, saving, and hub integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alignrl.config import BaseTrainConfig, ensure_chat_template

if TYPE_CHECKING:
    from pathlib import Path

    from alignrl.types import TrainResult


class BaseRunner:
    """Shared logic for SFT, GRPO, and DPO runners."""

    config: BaseTrainConfig

    def __init__(self, config: BaseTrainConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._dataset = None

    def _load_model(self) -> None:
        from unsloth import FastLanguageModel

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            dtype=None,
        )
        ensure_chat_template(self._tokenizer)
        self._model = FastLanguageModel.get_peft_model(
            self._model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            use_gradient_checkpointing="unsloth",
        )

    def train(self) -> TrainResult:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        if self._model:
            self._model.save_pretrained(str(path))
        if self._tokenizer:
            self._tokenizer.save_pretrained(str(path))

    def push_to_hub(self, repo_id: str, merge: bool = False, private: bool = False) -> str:
        """Push the trained adapter (or merged model) to HuggingFace Hub."""
        from alignrl.hub import merge_and_push, push_adapter

        if merge:
            return merge_and_push(
                model_name=self.config.model_name,
                adapter_path=str(self.config.output_dir / "final"),
                repo_id=repo_id,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                private=private,
            )
        return push_adapter(
            output_dir=str(self.config.output_dir / "final"),
            repo_id=repo_id,
            private=private,
        )

    def load(self, path: Path) -> None:
        from unsloth import FastLanguageModel

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(path),
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
        )
        ensure_chat_template(self._tokenizer)
