"""GRPO (Group Relative Policy Optimization) with verifiable rewards."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from alignrl.config import BaseTrainConfig
from alignrl.rewards import format_reward, math_verify_reward
from alignrl.types import TrainResult

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve problems step by step, "
    "showing your reasoning clearly. Put your final answer in \\boxed{}."
)


class GRPOConfig(BaseTrainConfig):
    """GRPO-specific configuration."""

    dataset_name: str = "openai/gsm8k"
    dataset_split: str = "train"
    dataset_config: str = "main"
    dataset_size: int | None = None
    learning_rate: float = 5e-6
    num_generations: int = 8
    max_completion_length: int = 512
    max_prompt_length: int = 256
    beta: float = 0.001
    max_steps: int = 250
    use_vllm: bool = False
    reward_weights: list[float] | None = None


def _format_gsm8k_prompt(example: dict) -> dict:
    """Convert GSM8K example to GRPO-compatible format."""
    answer = example["answer"].split("####")[-1].strip()
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "solution": answer,
    }


class GRPORunner:
    """Runs GRPO training with verifiable math rewards."""

    def __init__(
        self,
        config: GRPOConfig,
        reward_funcs: list[Callable] | None = None,
    ) -> None:
        self.config = config
        self.reward_funcs = reward_funcs or [math_verify_reward, format_reward]
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

        ds = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.dataset_split,
        )
        if self.config.dataset_size:
            ds = ds.select(range(min(self.config.dataset_size, len(ds))))
        return ds.map(_format_gsm8k_prompt, remove_columns=ds.column_names)

    def train(self) -> TrainResult:
        from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

        self._load_model()
        dataset = self._load_dataset()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        grpo_args = TRLGRPOConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
            optim=self.config.optim,
            seed=self.config.seed,
            report_to=self.config.report_to,
            logging_steps=self.config.logging_steps,
            num_generations=self.config.num_generations,
            max_completion_length=self.config.max_completion_length,
            max_prompt_length=self.config.max_prompt_length,
            beta=self.config.beta,
            save_strategy="steps",
            save_steps=50,
        )
        if self.config.reward_weights:
            grpo_args.reward_weights = self.config.reward_weights

        trainer = GRPOTrainer(
            model=self._model,
            processing_class=self._tokenizer,
            reward_funcs=self.reward_funcs,
            args=grpo_args,
            train_dataset=dataset,
        )

        result = trainer.train()
        trainer.save_model(str(output_dir / "final"))

        loss_history = [
            log["loss"] for log in trainer.state.log_history if "loss" in log
        ]
        reward_history = [
            log.get("reward", 0.0) for log in trainer.state.log_history if "reward" in log
        ]

        train_result = TrainResult(
            output_dir=output_dir / "final",
            loss_history=loss_history,
            metrics={
                "train_loss": result.training_loss,
                "reward_history": reward_history,
            },
            num_steps=result.global_step,
            num_epochs=0,
        )

        with open(output_dir / "train_result.json", "w") as f:
            json.dump(
                {"loss_history": loss_history, "reward_history": reward_history},
                f,
                indent=2,
            )

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
