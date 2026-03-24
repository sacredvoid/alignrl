"""Supervised Fine-Tuning with QLoRA via Unsloth + TRL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alignrl.config import BaseTrainConfig
from alignrl.types import TrainResult

ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


class SFTConfig(BaseTrainConfig):
    """SFT-specific configuration."""

    dataset_name: str = "teknium/OpenHermes-2.5"
    dataset_split: str = "train"
    dataset_size: int | None = None
    chat_template: str = "chatml"


def format_instruction(example: dict[str, Any]) -> list[dict[str, str]]:
    """Convert OpenHermes-style conversations to chat messages."""
    convos = example.get("conversations", [])
    if not convos:
        raise ValueError("Conversations field is empty")
    return [
        {"role": ROLE_MAP.get(turn["from"], turn["from"]), "content": turn["value"]}
        for turn in convos
    ]


class SFTRunner:
    """Runs SFT training pipeline."""

    def __init__(self, config: SFTConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        from unsloth import FastLanguageModel

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            dtype=None,
        )
        self._model = FastLanguageModel.get_peft_model(
            self._model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            use_gradient_checkpointing="unsloth",
        )

    def _load_dataset(self):
        from datasets import load_dataset

        ds = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        if self.config.dataset_size:
            ds = ds.select(range(min(self.config.dataset_size, len(ds))))

        def _apply_template(example):
            messages = format_instruction(example)
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        return ds.map(_apply_template, remove_columns=ds.column_names)

    def train(self) -> TrainResult:
        from trl import SFTConfig as TRLSFTConfig
        from trl import SFTTrainer

        self._load_model()
        dataset = self._load_dataset()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TRLSFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
            optim=self.config.optim,
            max_seq_length=self.config.max_seq_length,
            seed=self.config.seed,
            report_to=self.config.report_to,
            logging_steps=self.config.logging_steps,
            save_strategy="steps",
            save_steps=100,
        )

        trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=dataset,
            args=training_args,
        )

        result = trainer.train()
        trainer.save_model(str(output_dir / "final"))

        loss_history = [log["loss"] for log in trainer.state.log_history if "loss" in log]

        train_result = TrainResult(
            output_dir=output_dir / "final",
            loss_history=loss_history,
            metrics={"train_loss": result.training_loss},
            num_steps=result.global_step,
            num_epochs=self.config.num_train_epochs,
        )

        with open(output_dir / "train_result.json", "w") as f:
            json.dump({"loss_history": loss_history, "metrics": train_result.metrics}, f, indent=2)

        return train_result

    def save(self, path: Path) -> None:
        if self._model:
            self._model.save_pretrained(str(path))
        if self._tokenizer:
            self._tokenizer.save_pretrained(str(path))

    def load(self, path: Path) -> None:
        from unsloth import FastLanguageModel

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(path),
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
        )
