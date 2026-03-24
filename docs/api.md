# API Reference

## `alignrl.config`

Base configuration system using Pydantic.

### `BaseTrainConfig`

Base class for all training configurations. All fields have defaults and can be overridden via constructor kwargs or loaded from a YAML file.

```python
from alignrl.config import BaseTrainConfig
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"Qwen/Qwen2.5-3B"` | HuggingFace model ID or local path |
| `output_dir` | `Path` | `"./outputs"` | Directory for checkpoints and results |
| `max_seq_length` | `int` | `2048` | Maximum sequence length for tokenization |
| `per_device_train_batch_size` | `int` | `4` | Batch size per GPU |
| `gradient_accumulation_steps` | `int` | `4` | Steps before optimizer update |
| `learning_rate` | `float` | `2e-4` | Peak learning rate |
| `num_train_epochs` | `int` | `1` | Number of training epochs |
| `max_steps` | `int` | `-1` | Max training steps (-1 for epoch-based) |
| `warmup_steps` | `int` | `10` | Linear warmup steps |
| `optim` | `str` | `"adamw_8bit"` | Optimizer name |
| `seed` | `int` | `42` | Random seed |
| `report_to` | `str` | `"none"` | Logging backend ("none", "wandb") |
| `logging_steps` | `int` | `10` | Log metrics every N steps |
| `lora_r` | `int` | `16` | LoRA rank |
| `lora_alpha` | `int` | `32` | LoRA alpha scaling factor |
| `lora_dropout` | `float` | `0.0` | LoRA dropout probability |
| `lora_target_modules` | `list[str]` | `["q_proj", "k_proj", ...]` | Modules to apply LoRA to |
| `load_in_4bit` | `bool` | `True` | Enable 4-bit quantization |

**Class methods:**

#### `BaseTrainConfig.from_yaml(path: Path) -> BaseTrainConfig`

Load configuration from a YAML file. All subclasses inherit this method.

```python
from alignrl.sft import SFTConfig

config = SFTConfig.from_yaml(Path("configs/sft.yaml"))
```

---

## `alignrl.sft`

Supervised Fine-Tuning with QLoRA via Unsloth and TRL.

### `SFTConfig`

Extends `BaseTrainConfig` with SFT-specific fields.

```python
from alignrl.sft import SFTConfig
```

**Additional fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_name` | `str` | `"teknium/OpenHermes-2.5"` | HuggingFace dataset ID |
| `dataset_split` | `str` | `"train"` | Dataset split to use |
| `dataset_size` | `int \| None` | `None` | Limit dataset size (None for full) |
| `chat_template` | `str` | `"chatml"` | Chat template format |

### `SFTRunner`

Runs the SFT training pipeline.

```python
from alignrl.sft import SFTConfig, SFTRunner

config = SFTConfig(dataset_size=1000, max_steps=50)
runner = SFTRunner(config)
result = runner.train()
```

**Methods:**

#### `SFTRunner.__init__(config: SFTConfig) -> None`

Initialize the runner with a configuration.

#### `SFTRunner.train() -> TrainResult`

Run SFT training. Loads the model, prepares the dataset, trains with TRL's `SFTTrainer`, and saves the adapter. Returns a `TrainResult` with loss history and metrics.

#### `SFTRunner.save(path: Path) -> None`

Save the model and tokenizer to a directory.

#### `SFTRunner.load(path: Path) -> None`

Load a model from a saved adapter path.

### `format_instruction(example: dict[str, Any]) -> list[dict[str, str]]`

Convert an OpenHermes-style conversation to a list of chat messages.

```python
from alignrl.sft import format_instruction

example = {
    "conversations": [
        {"from": "human", "value": "What is 2+2?"},
        {"from": "gpt", "value": "4"},
    ]
}
messages = format_instruction(example)
# [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]
```

**Parameters:**
- `example` (`dict[str, Any]`) - Dictionary with a `"conversations"` key containing a list of turns. Each turn has `"from"` (one of "human", "gpt", "system") and `"value"` fields.

**Returns:** `list[dict[str, str]]` - List of `{"role": ..., "content": ...}` dicts.

**Raises:** `ValueError` if the conversations field is empty.

---

## `alignrl.grpo`

Group Relative Policy Optimization with verifiable rewards.

### `GRPOConfig`

Extends `BaseTrainConfig` with GRPO-specific fields.

```python
from alignrl.grpo import GRPOConfig
```

**Additional fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_name` | `str` | `"openai/gsm8k"` | HuggingFace dataset ID |
| `dataset_split` | `str` | `"train"` | Dataset split |
| `dataset_config` | `str` | `"main"` | Dataset configuration name |
| `dataset_size` | `int \| None` | `None` | Limit dataset size |
| `learning_rate` | `float` | `5e-6` | Lower than SFT (RL is sensitive) |
| `num_generations` | `int` | `8` | Completions per prompt in the group |
| `max_completion_length` | `int` | `512` | Max tokens per generation |
| `max_prompt_length` | `int` | `256` | Max prompt tokens |
| `beta` | `float` | `0.001` | KL penalty coefficient |
| `max_steps` | `int` | `250` | Training steps |
| `use_vllm` | `bool` | `False` | Use vLLM for generation |
| `reward_weights` | `list[float] \| None` | `None` | Weights for multiple reward functions |

### `GRPORunner`

Runs GRPO training with verifiable math rewards.

```python
from alignrl.grpo import GRPOConfig, GRPORunner

config = GRPOConfig(max_steps=50, num_generations=4)
runner = GRPORunner(config)
result = runner.train()
```

**Methods:**

#### `GRPORunner.__init__(config: GRPOConfig, reward_funcs: list[Callable] | None = None) -> None`

Initialize with config and optional custom reward functions. If `reward_funcs` is None, defaults to `[math_verify_reward, format_reward]`.

#### `GRPORunner.train() -> TrainResult`

Run GRPO training. The returned `TrainResult.metrics` dict includes a `"reward_history"` key with the mean reward at each logging step.

#### `GRPORunner.save(path: Path) -> None`

Save the trained adapter.

#### `GRPORunner.load(path: Path) -> None`

Load a saved adapter.

---

## `alignrl.dpo`

Direct Preference Optimization for alignment.

### `DPOConfig`

Extends `BaseTrainConfig` with DPO-specific fields.

```python
from alignrl.dpo import DPOConfig
```

**Additional fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_name` | `str` | `"HuggingFaceH4/ultrafeedback_binarized"` | HuggingFace dataset ID |
| `dataset_split` | `str` | `"train_prefs"` | Dataset split |
| `dataset_size` | `int \| None` | `None` | Limit dataset size |
| `learning_rate` | `float` | `5e-7` | Very low LR for DPO stability |
| `beta` | `float` | `0.1` | KL constraint strength |
| `max_length` | `int` | `1024` | Max sequence length |
| `precompute_ref_log_probs` | `bool` | `True` | Precompute reference log-probs to save memory |

### `DPORunner`

Runs the DPO training pipeline.

```python
from alignrl.dpo import DPOConfig, DPORunner

config = DPOConfig(max_steps=50, dataset_size=500)
runner = DPORunner(config)
result = runner.train()
```

**Methods:**

#### `DPORunner.__init__(config: DPOConfig) -> None`

Initialize with config.

#### `DPORunner.train() -> TrainResult`

Run DPO training. Uses TRL's `DPOTrainer` with precomputed reference log-probabilities.

#### `DPORunner.save(path: Path) -> None`

Save the trained adapter.

#### `DPORunner.load(path: Path) -> None`

Load a saved adapter.

### `format_ultrafeedback(example: dict[str, Any]) -> dict[str, Any]`

Format an UltraFeedback binarized example for DPO training.

```python
from alignrl.dpo import format_ultrafeedback

# Input: example with "chosen" and "rejected" as full conversation lists
# Output: {"prompt": [...], "chosen": [last_turn], "rejected": [last_turn]}
formatted = format_ultrafeedback(example)
```

**Parameters:**
- `example` (`dict[str, Any]`) - Dictionary with `"chosen"` and `"rejected"` keys, each a list of message dicts.

**Returns:** `dict[str, Any]` with `"prompt"` (all but the last turn of chosen), `"chosen"` (last turn of chosen), and `"rejected"` (last turn of rejected).

---

## `alignrl.eval`

Evaluation harness wrapper for benchmarking across training stages.

### `EvalConfig`

Extends `BaseTrainConfig` with evaluation-specific fields.

```python
from alignrl.eval import EvalConfig
```

**Additional fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tasks` | `list[str]` | `["gsm8k", "arc_challenge"]` | Benchmark tasks to run |
| `num_fewshot` | `int` | `0` | Number of few-shot examples |
| `batch_size` | `str` | `"auto"` | Eval batch size |
| `limit` | `int \| None` | `None` | Limit number of examples per task |
| `adapter_path` | `str \| None` | `None` | Path to LoRA adapter (None for base model) |

### `EvalRunner`

Runs evaluation benchmarks using `lm-evaluation-harness`.

```python
from alignrl.eval import EvalConfig, EvalRunner

config = EvalConfig(tasks=["gsm8k"], limit=100)
runner = EvalRunner(config)
result = runner.evaluate(stage="base")
```

**Methods:**

#### `EvalRunner.__init__(config: EvalConfig) -> None`

Initialize with config.

#### `EvalRunner.evaluate(stage: str = "base") -> EvalResult`

Run evaluation and return structured results.

**Parameters:**
- `stage` (`str`) - Label for this evaluation stage ("base", "sft", "grpo", "dpo").

**Returns:** `EvalResult` with benchmark scores.

#### `EvalRunner.evaluate_all_stages(adapter_paths: dict[str, str | None]) -> list[EvalResult]`

Evaluate multiple stages in sequence.

**Parameters:**
- `adapter_paths` (`dict[str, str | None]`) - Mapping of stage name to adapter path. Use `None` for the base model.

**Returns:** List of `EvalResult` objects.

```python
results = runner.evaluate_all_stages({
    "base": None,
    "sft": "./outputs/sft/final",
    "grpo": "./outputs/grpo/final",
})
```

#### `EvalRunner.save_results(results: list[EvalResult], output_dir: Path) -> None`

Save results as JSON files, including a `comparison.json` for the GitHub Pages dashboard.

### `parse_results(raw: dict[str, Any], model_name: str, stage: str) -> EvalResult`

Parse raw `lm-evaluation-harness` output into an `EvalResult`.

**Parameters:**
- `raw` (`dict[str, Any]`) - Raw output dict from `lm_eval.simple_evaluate`.
- `model_name` (`str`) - Model identifier.
- `stage` (`str`) - Training stage label.

**Returns:** `EvalResult`.

### `compare_stages(results: list[EvalResult]) -> dict[str, dict[str, dict[str, float]]]`

Compare evaluation results across training stages.

**Parameters:**
- `results` (`list[EvalResult]`) - List of evaluation results from different stages.

**Returns:** Nested dict structured as `{benchmark: {stage: {metric: score}}}`.

```python
from alignrl.eval import compare_stages

comparison = compare_stages(results)
# {"gsm8k": {"base": {"exact_match": 0.31}, "grpo": {"exact_match": 0.62}}}
```

---

## `alignrl.inference`

Unified inference across multiple backends.

### `InferenceConfig`

Configuration for model serving.

```python
from alignrl.inference import InferenceConfig
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"Qwen/Qwen2.5-3B"` | Base model ID |
| `adapter_path` | `str \| None` | `None` | LoRA adapter path |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `512` | Maximum generation tokens |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold |
| `backend` | `str` | `"unsloth"` | Backend: "unsloth", "vllm", or "mlx" |

### `ModelServer`

Unified inference server supporting multiple backends.

```python
from alignrl.inference import InferenceConfig, ModelServer

config = InferenceConfig(adapter_path="./outputs/grpo/final")
server = ModelServer(config)
server.load()
response = server.generate([{"role": "user", "content": "What is 2+2?"}])
```

**Methods:**

#### `ModelServer.__init__(config: InferenceConfig) -> None`

Initialize with config. Does not load the model until `load()` is called.

#### `ModelServer.load() -> None`

Load the model using the configured backend. Dispatches to the appropriate loader:
- `"unsloth"` - Uses `FastLanguageModel.from_pretrained` with 4-bit quantization
- `"vllm"` - Uses `vllm.LLM` with optional LoRA support
- `"mlx"` - Uses `mlx_lm.load` for Apple Silicon

#### `ModelServer.generate(messages: list[dict[str, str]]) -> str`

Generate a response from chat messages.

**Parameters:**
- `messages` (`list[dict[str, str]]`) - Chat messages in `[{"role": "user", "content": "..."}]` format.

**Returns:** `str` - The generated text.

### `build_prompt(user_message: str, system: str | None = None) -> list[dict[str, str]]`

Build a chat prompt from a user message and optional system prompt.

```python
from alignrl.inference import build_prompt

messages = build_prompt("What is 2+2?", system="You are a math tutor.")
# [{"role": "system", "content": "You are a math tutor."}, {"role": "user", "content": "What is 2+2?"}]
```

**Parameters:**
- `user_message` (`str`) - The user's message.
- `system` (`str | None`) - Optional system prompt. If None, no system message is included.

**Returns:** `list[dict[str, str]]`

---

## `alignrl.rewards`

Reward functions for GRPO training with verifiable rewards.

### `extract_answer(text: str) -> str | None`

Extract the final answer from model output. Supports multiple patterns:
1. `\boxed{...}` (takes the last match)
2. "the answer is X" / "answer: X"
3. "= X" (equation format)

```python
from alignrl.rewards import extract_answer

extract_answer(r"Therefore \boxed{42}")  # "42"
extract_answer("The answer is 7.")       # "7"
extract_answer("x = 3")                  # "3"
extract_answer("I'm not sure")           # None
```

**Parameters:**
- `text` (`str`) - Model output text.

**Returns:** `str | None` - The extracted answer, or None if no pattern matched.

### `math_verify_reward(completions, solution, **kwargs) -> list[float]`

Binary reward: 1.0 if the extracted answer matches the expected solution, 0.0 otherwise. Handles numeric equivalence (e.g., "3.0" matches "3").

```python
from alignrl.rewards import math_verify_reward

completions = [[{"content": r"\boxed{42}"}], [{"content": r"\boxed{43}"}]]
rewards = math_verify_reward(completions, solution=["42", "42"])
# [1.0, 0.0]
```

**Parameters:**
- `completions` (`list[list[dict[str, str]]]`) - Batch of completions. Each completion is a list of message dicts.
- `solution` (`list[str]`) - Expected answers, one per completion.
- `**kwargs` - Ignored (for compatibility with TRL's reward function signature).

**Returns:** `list[float]` - Rewards (0.0 or 1.0) for each completion.

### `format_reward(completions, **kwargs) -> list[float]`

Reward for following the expected output format (reasoning followed by `\boxed{answer}`).

Scoring:
- `0.0` - No `\boxed{}` found
- `0.5` - `\boxed{}` present but not near the end of the response
- `1.0` - `\boxed{}` at or near the end

```python
from alignrl.rewards import format_reward

completions = [[{"content": r"Step 1... Step 2... \boxed{42}"}]]
rewards = format_reward(completions)
# [1.0]
```

**Parameters:**
- `completions` (`list[list[dict[str, str]]]`) - Batch of completions.
- `**kwargs` - Ignored.

**Returns:** `list[float]` - Format scores for each completion.

---

## `alignrl.types`

Shared types and protocols.

### `TrainResult`

Frozen dataclass returned by all training runners.

```python
from alignrl.types import TrainResult
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `output_dir` | `Path` | Directory where the adapter was saved |
| `loss_history` | `list[float]` | Loss at each logging step |
| `metrics` | `dict[str, float]` | Training metrics (always includes `"train_loss"`) |
| `num_steps` | `int` | Total training steps completed |
| `num_epochs` | `float` | Number of epochs completed |

### `EvalResult`

Frozen dataclass returned by the evaluation runner.

```python
from alignrl.types import EvalResult
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | `str` | Model identifier |
| `stage` | `str` | Training stage ("base", "sft", "grpo", "dpo") |
| `benchmarks` | `dict[str, dict[str, float]]` | `{benchmark: {metric: score}}` |
| `metadata` | `dict[str, str]` | Optional metadata (default: empty dict) |

**Methods:**

#### `EvalResult.to_dict() -> dict`

Serialize to a plain dict for JSON output.

### `Trainer` (Protocol)

Runtime-checkable protocol that all training runners implement.

```python
from alignrl.types import Trainer

def run_any_trainer(trainer: Trainer) -> TrainResult:
    return trainer.train()
```

**Required methods:**
- `train() -> TrainResult`
- `save(path: Path) -> None`
- `load(path: Path) -> None`

---

## `alignrl.cli`

Command-line interface. Installed as the `alignrl` command.

### Commands

#### `alignrl train <stage> -c <config>`

Run a training pipeline.

```bash
alignrl train sft -c configs/sft.yaml
alignrl train grpo -c configs/grpo.yaml
alignrl train dpo -c configs/dpo.yaml
```

**Arguments:**
- `stage` - One of `sft`, `grpo`, `dpo`
- `-c / --config` - Path to a YAML config file

#### `alignrl eval`

Run evaluation benchmarks.

```bash
alignrl eval --model Qwen/Qwen2.5-3B --adapter ./outputs/sft/final --stage sft --tasks gsm8k,arc_challenge --limit 100 --output ./results
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-3B` | Base model ID |
| `--adapter` | `None` | LoRA adapter path |
| `--stage` | `base` | Stage label for output files |
| `--tasks` | `gsm8k,arc_challenge` | Comma-separated benchmark tasks |
| `--limit` | `None` | Limit examples per task |
| `--output` | `./results` | Output directory for JSON results |

#### `alignrl serve`

Launch a Gradio comparison demo.

```bash
alignrl serve --model Qwen/Qwen2.5-3B --stages base sft=./outputs/sft/final grpo=./outputs/grpo/final --port 7860
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-3B` | Base model ID |
| `--stages` | (required) | Stage specs: `base` or `name=path` |
| `--port` | `7860` | Server port |
| `--share` | `False` | Create a public Gradio share link |

---

## `alignrl.demo`

Gradio demo for comparing model stages.

### `create_demo(stages, model_name) -> gr.Blocks`

Create a Gradio app that compares model outputs across training stages side-by-side.

```python
from alignrl.demo import create_demo

demo = create_demo(
    stages={"base": None, "sft": "./outputs/sft/final"},
    model_name="Qwen/Qwen2.5-3B",
)
demo.launch()
```

**Parameters:**
- `stages` (`dict[str, str | None]`) - Mapping of stage name to adapter path (None for base model).
- `model_name` (`str`) - Base model HuggingFace ID. Default: `"Qwen/Qwen2.5-3B"`.

**Returns:** A `gr.Blocks` Gradio application.
