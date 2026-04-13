"""Configuration system using Pydantic for validation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from typing_extensions import Self


class BaseTrainConfig(BaseModel):
    """Shared training configuration.

    All training stage configs (SFT, GRPO, DPO) inherit from this. Fields
    are validated at construction time via Pydantic, so malformed YAML
    files fail fast rather than partway through a training run.
    """

    # Use a custom config dict to forbid unknown keys. This catches typos
    # in YAML files (e.g. ``learnign_rate: 2e-4``) before any training
    # begins, which is much friendlier than a silent default.
    model_config = ConfigDict(extra="forbid")

    model_name: str = "Qwen/Qwen2.5-3B"
    output_dir: Path = Path("./outputs")
    max_seq_length: int = Field(default=2048, gt=0)
    per_device_train_batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    num_train_epochs: int = Field(default=1, ge=0)
    max_steps: int = Field(default=-1, ge=-1)
    warmup_steps: int = Field(default=10, ge=0)
    optim: str = "adamw_8bit"
    seed: int = Field(default=42, ge=0)
    report_to: str = "none"
    logging_steps: int = Field(default=10, gt=0)

    # LoRA
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    lora_target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Quantization
    load_in_4bit: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Load a config from a YAML file. Missing keys use defaults."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**(data or {}))

    def to_yaml(self, path: Path | None = None) -> str:
        """Serialize the config to YAML.

        Returns the YAML string. If ``path`` is provided, also writes the
        YAML to disk (parent directories are created as needed).
        """
        data = self.model_dump(mode="json")
        text: str = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text


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
