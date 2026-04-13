"""alignrl - LLM post-training playbook."""

from __future__ import annotations

import importlib

__version__ = "0.4.0"

_LAZY_IMPORTS: dict[str, str] = {
    # Config
    "BaseTrainConfig": "alignrl.config",
    # SFT
    "SFTConfig": "alignrl.sft",
    "SFTRunner": "alignrl.sft",
    # DPO
    "DPOConfig": "alignrl.dpo",
    "DPORunner": "alignrl.dpo",
    # GRPO
    "GRPOConfig": "alignrl.grpo",
    "GRPORunner": "alignrl.grpo",
    # Evaluation
    "EvalConfig": "alignrl.eval",
    "EvalRunner": "alignrl.eval",
    "compare_stages": "alignrl.eval",
    "parse_results": "alignrl.eval",
    "BENCHMARK_PRESETS": "alignrl.eval",
    # Inference
    "InferenceConfig": "alignrl.inference",
    "ModelServer": "alignrl.inference",
    "build_prompt": "alignrl.inference",
    # Shared types / protocols
    "TrainResult": "alignrl.types",
    "EvalResult": "alignrl.types",
    "Trainer": "alignrl.types",
    # Rewards
    "math_verify_reward": "alignrl.rewards",
    "format_reward": "alignrl.rewards",
    "extract_answer": "alignrl.rewards",
    # HF Hub helpers
    "push_adapter": "alignrl.hub",
    "merge_and_push": "alignrl.hub",
    # W&B integration
    "detect_wandb": "alignrl.callbacks",
    "log_eval_to_wandb": "alignrl.callbacks",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'alignrl' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy exports in ``dir(alignrl)`` for discoverability."""
    return sorted([*_LAZY_IMPORTS, "__version__"])


__all__ = [*_LAZY_IMPORTS, "__version__"]
