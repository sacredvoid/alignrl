"""Direct Preference Optimization (DPO) for alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alignrl.config import BaseTrainConfig
from alignrl.runner_base import BaseRunner
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


class DPORunner(BaseRunner):
    """Runs DPO training pipeline."""

    def __init__(self, config: DPOConfig) -> None:
        super().__init__(config)

    def _load_dataset(self):
        if self._dataset is not None:
            return self._dataset

        from datasets import load_dataset

        ds = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        if self.config.dataset_size is not None:
            ds = ds.select(range(min(self.config.dataset_size, len(ds))))
        self._dataset = ds.map(format_ultrafeedback, remove_columns=ds.column_names)
        return self._dataset

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
        if not loss_history:
            loss_history = [result.training_loss]

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
