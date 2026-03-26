"""Direct Preference Optimization (DPO) for alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alignrl.config import BaseTrainConfig, ensure_chat_template
from alignrl.types import TrainResult


class DPOConfig(BaseTrainConfig):
    """DPO-specific configuration."""

    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    dataset_split: str = "train_prefs"
    dataset_size: int | None = None
    learning_rate: float = 5e-7
    beta: float = 0.1
    max_length: int = 1024
    precompute_ref_log_probs: bool = True


def format_ultrafeedback(example: dict[str, Any]) -> dict[str, Any]:
    """Format UltraFeedback example for DPO training."""
    return {
        "prompt": example["chosen"][:-1],
        "chosen": example["chosen"][-1:],
        "rejected": example["rejected"][-1:],
    }


class DPORunner:
    """Runs DPO training pipeline."""

    def __init__(self, config: DPOConfig) -> None:
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
        ensure_chat_template(self._tokenizer)
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
        if self.config.dataset_size is not None:
            ds = ds.select(range(min(self.config.dataset_size, len(ds))))
        return ds.map(format_ultrafeedback)

    def train(self) -> TrainResult:
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer

        self._load_model()
        dataset = self._load_dataset()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dpo_args = TRLDPOConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
            optim=self.config.optim,
            beta=self.config.beta,
            max_length=self.config.max_length,
            precompute_ref_log_probs=self.config.precompute_ref_log_probs,
            seed=self.config.seed,
            report_to=self.config.report_to,
            logging_steps=self.config.logging_steps,
            save_strategy="steps",
            save_steps=100,
        )

        trainer = DPOTrainer(
            model=self._model,
            args=dpo_args,
            train_dataset=dataset,
            processing_class=self._tokenizer,
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
