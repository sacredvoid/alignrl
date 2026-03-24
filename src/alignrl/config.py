"""Configuration system using Pydantic for validation."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BaseTrainConfig(BaseModel):
    """Shared training configuration."""

    model_name: str = "Qwen/Qwen2.5-3B"
    output_dir: Path = Path("./outputs")
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_steps: int = 10
    optim: str = "adamw_8bit"
    seed: int = 42
    report_to: str = "none"
    logging_steps: int = 10

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Quantization
    load_in_4bit: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> BaseTrainConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
