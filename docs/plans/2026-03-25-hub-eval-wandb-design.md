# Design: HF Hub Push, Eval Presets, W&B Integration, Public API

**Date:** 2026-03-25
**Status:** Approved

## Overview

Four additive features for alignrl v0.3.0:
1. HF Hub push (adapter and merged model upload)
2. Richer eval benchmark presets
3. W&B integration with custom logging
4. Public API exports in `__init__.py`

All features use lazy imports, work on Colab T4, and introduce no breaking changes.

---

## Feature 1: HF Hub Push

**New file:** `src/alignrl/hub.py`

### Functions

```python
def push_adapter(output_dir: str | Path, repo_id: str, private: bool = False) -> str:
    """Push LoRA adapter directory to HF Hub. Returns the repo URL."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=private)
    api.upload_folder(folder_path=str(output_dir), repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"

def merge_and_push(
    model_name: str,
    adapter_path: str | Path,
    repo_id: str,
    max_seq_length: int = 2048,
    private: bool = False,
) -> str:
    """Merge LoRA into base model and push merged model to HF Hub."""
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(...)
    model.save_pretrained_merged(repo_id, tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")
    return f"https://huggingface.co/{repo_id}"
```

### Runner integration

Add `push_to_hub(self, repo_id, merge=False, private=False)` to all 3 runners. Delegates to `push_adapter` or `merge_and_push`.

### CLI integration

Add `--push REPO_ID` to `alignrl train`:
```
alignrl train sft -c config.yaml --push user/my-sft-model
```

### Dependencies

`huggingface_hub` is already a transitive dep of `transformers`. No new dep needed.

---

## Feature 2: Eval Benchmark Presets

**Modified file:** `src/alignrl/eval.py`

### Preset definitions

```python
BENCHMARK_PRESETS: dict[str, list[str]] = {
    "core": ["gsm8k", "arc_challenge", "hellaswag", "mmlu", "winogrande"],
    "reasoning": ["gsm8k", "math", "arc_challenge"],
    "leaderboard": [
        "gsm8k", "arc_challenge", "hellaswag", "mmlu",
        "winogrande", "truthfulqa_mc2",
    ],
}
```

### EvalConfig changes

Add `preset: str | None = None` field. Resolution logic:
- If `tasks` is explicitly set (not the default) -> use `tasks`
- Elif `preset` is set -> resolve from `BENCHMARK_PRESETS`
- Else -> use "core" preset

Use a `model_validator` to resolve at construction time.

### CLI changes

Add `--preset` arg to `alignrl eval`:
```
alignrl eval --preset reasoning --model Qwen/Qwen2.5-3B
alignrl eval --tasks gsm8k,mmlu    # explicit override
```

---

## Feature 3: W&B Integration

**New file:** `src/alignrl/callbacks.py`

### Auto-detection

```python
def detect_wandb() -> str:
    """Return 'wandb' if wandb is installed and configured, else 'none'."""
    try:
        import wandb
        if wandb.api.api_key:
            return "wandb"
    except Exception:
        pass
    return "none"
```

### Eval logging

```python
def log_eval_to_wandb(results: list[EvalResult], project: str = "alignrl") -> None:
    """Log evaluation results to W&B as summary metrics and a comparison table."""
    import wandb
    if not wandb.run:
        wandb.init(project=project)
    # Log each stage's metrics as summary
    for result in results:
        for bench, metrics in result.benchmarks.items():
            for metric_name, value in metrics.items():
                wandb.run.summary[f"{result.stage}/{bench}/{metric_name}"] = value
    # Log comparison table
    table = wandb.Table(columns=["stage", "benchmark", "metric", "value"])
    for result in results:
        for bench, metrics in result.benchmarks.items():
            for name, val in metrics.items():
                table.add_data(result.stage, bench, name, val)
    wandb.log({"eval_comparison": table})
```

### Config change

Update `BaseTrainConfig.report_to` default: keep `"none"` as the static default (no import-time side effects). Users set `report_to="wandb"` or use CLI flag.

### CLI changes

Add `--wandb` flag to `alignrl eval`:
```
alignrl eval --preset core --wandb
```

---

## Feature 4: Public API Exports

**Modified file:** `src/alignrl/__init__.py`

Lazy imports via `__getattr__` to avoid importing heavy deps at import time:

```python
__version__ = "0.2.0"

_LAZY_IMPORTS = {
    "SFTConfig": "alignrl.sft",
    "SFTRunner": "alignrl.sft",
    "DPOConfig": "alignrl.dpo",
    ...
}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(...)

__all__ = list(_LAZY_IMPORTS) + ["__version__"]
```

---

## Implementation Order

1. Public API exports (no deps, unblocks nothing, quick win)
2. Eval benchmark presets (self-contained in eval.py)
3. HF Hub push (new module + runner changes + CLI)
4. W&B integration (new module + CLI)

Each step: implement, test, verify all tests pass.
