"""Inference utilities for serving trained models."""

from __future__ import annotations

from pydantic import BaseModel


class InferenceConfig(BaseModel):
    """Inference configuration."""

    model_name: str = "Qwen/Qwen2.5-3B"
    adapter_path: str | None = None
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    backend: str = "unsloth"  # "unsloth", "vllm", or "mlx"
    max_seq_length: int = 2048
    load_in_4bit: bool = True


def build_prompt(user_message: str, system: str | None = None) -> list[dict[str, str]]:
    """Build a chat prompt."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})
    return messages


class ModelServer:
    """Unified inference across backends."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._lora_request = None

    def load(self) -> None:
        if self.config.backend == "vllm":
            self._load_vllm()
        elif self.config.backend == "mlx":
            self._load_mlx()
        else:
            self._load_unsloth()

    def _load_unsloth(self) -> None:
        from unsloth import FastLanguageModel

        model_path = self.config.adapter_path or self.config.model_name
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
        )
        from alignrl.config import ensure_chat_template

        ensure_chat_template(self._tokenizer)
        FastLanguageModel.for_inference(self._model)

    def _load_vllm(self) -> None:
        from vllm import LLM

        kwargs = {"model": self.config.model_name, "dtype": "auto"}
        if self.config.adapter_path:
            from vllm.lora.request import LoRARequest

            kwargs["enable_lora"] = True
            self._lora_request = LoRARequest(
                lora_name="adapter",
                lora_int_id=1,
                lora_path=self.config.adapter_path,
            )
        self._model = LLM(**kwargs)

    def _load_mlx(self) -> None:
        from mlx_lm import load

        model_path = self.config.adapter_path or self.config.model_name
        self._model, self._tokenizer = load(model_path)

    def generate(self, messages: list[dict[str, str]]) -> str:
        if self.config.backend == "vllm":
            return self._generate_vllm(messages)
        elif self.config.backend == "mlx":
            return self._generate_mlx(messages)
        return self._generate_unsloth(messages)

    def _generate_unsloth(self, messages: list[dict[str, str]]) -> str:
        inputs = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        outputs = self._model.generate(
            input_ids=inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.temperature > 0,
        )
        return self._tokenizer.decode(outputs[0][inputs.shape[-1] :], skip_special_tokens=True)

    def _generate_vllm(self, messages: list[dict[str, str]]) -> str:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )
        kwargs = {"sampling_params": params}
        if self._lora_request:
            kwargs["lora_request"] = self._lora_request
        outputs = self._model.chat(messages, **kwargs)
        return outputs[0].outputs[0].text

    def _generate_mlx(self, messages: list[dict[str, str]]) -> str:
        from mlx_lm import generate

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temp=self.config.temperature,
        )
