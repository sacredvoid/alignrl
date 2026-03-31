"""alignrl - LLM post-training playbook."""

from __future__ import annotations

import importlib

__version__ = "0.3.0"

_LAZY_IMPORTS: dict[str, str] = {
    "BaseTrainConfig": "alignrl.config",
    "BaseRunner": "alignrl.runner_base",
    "SFTConfig": "alignrl.sft",
    "SFTRunner": "alignrl.sft",
    "DPOConfig": "alignrl.dpo",
    "DPORunner": "alignrl.dpo",
    "GRPOConfig": "alignrl.grpo",
    "GRPORunner": "alignrl.grpo",
    "EvalConfig": "alignrl.eval",
    "EvalRunner": "alignrl.eval",
    "InferenceConfig": "alignrl.inference",
    "ModelServer": "alignrl.inference",
    "build_prompt": "alignrl.inference",
    "TrainResult": "alignrl.types",
    "EvalResult": "alignrl.types",
    "push_adapter": "alignrl.hub",
    "merge_and_push": "alignrl.hub",
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


__all__ = [*_LAZY_IMPORTS, "__version__"]
