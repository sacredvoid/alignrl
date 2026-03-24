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


# ChatML template used as fallback when the tokenizer doesn't have one set.
# This is the standard format for Qwen, Yi, and many other models.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
)


def ensure_chat_template(tokenizer) -> None:
    """Set a chatml chat template on the tokenizer if one isn't already set."""
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = CHATML_TEMPLATE
