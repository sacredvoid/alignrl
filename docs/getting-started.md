# Getting Started

## Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/sacredvoid/alignrl.git
```

This installs the base package with only `pydantic` and `pyyaml` as dependencies. To actually run training, you need GPU extras:

```bash
pip install "alignrl[train,unsloth] @ git+https://github.com/sacredvoid/alignrl.git"
```

### From source

```bash
git clone https://github.com/sacredvoid/alignrl.git
cd alignrl
pip install -e ".[train,unsloth,dev]"
```

### On Google Colab (free T4 GPU)

Each notebook handles its own setup. Open any notebook from the [notebooks/](https://github.com/sacredvoid/alignrl/tree/main/notebooks) directory and run the first cell:

```python
!pip install "alignrl[train,unsloth] @ git+https://github.com/sacredvoid/alignrl.git"
```

### Optional dependency groups

| Extra | What it adds | When you need it |
|-------|-------------|------------------|
| `train` | torch, transformers, trl, peft, datasets, bitsandbytes, accelerate, wandb | Any GPU training |
| `unsloth` | unsloth | Memory-efficient training (recommended) |
| `eval` | lm-eval | Running benchmark evaluations |
| `serve` | vllm | High-throughput vLLM inference |
| `mlx` | mlx-lm | Apple Silicon inference |
| `gradio` | gradio | Interactive comparison demo |
| `dev` | pytest, ruff, mypy | Development and testing |
| `all` | Everything above except mlx | Full install |

---

## Your first SFT training run

Supervised Fine-Tuning teaches a base model to follow instructions. Five lines:

```python
from alignrl.sft import SFTConfig, SFTRunner

config = SFTConfig(
    model_name="Qwen/Qwen2.5-3B",
    dataset_name="teknium/OpenHermes-2.5",
    dataset_size=1000,
    max_steps=50,
    output_dir="./outputs/sft",
)

runner = SFTRunner(config)
result = runner.train()
print(f"Loss: {result.metrics['train_loss']:.4f}")
```

Or use the CLI with a YAML config:

```bash
alignrl train sft -c configs/sft.yaml
```

---

## Your first GRPO training run

GRPO uses reinforcement learning with verifiable math rewards. The model generates multiple solutions per problem, and a reward function scores each one. No critic model needed.

```python
from alignrl.grpo import GRPOConfig, GRPORunner

config = GRPOConfig(
    model_name="Qwen/Qwen2.5-3B",
    dataset_size=500,
    max_steps=50,
    num_generations=4,
    output_dir="./outputs/grpo",
)

runner = GRPORunner(config)
result = runner.train()
print(f"Loss: {result.metrics['train_loss']:.4f}")
```

The default reward functions are `math_verify_reward` (correctness) and `format_reward` (did the model use `\boxed{}`). You can pass your own via the `reward_funcs` parameter on `GRPORunner`.

---

## Your first DPO training run

DPO aligns a model using preference pairs (chosen vs. rejected responses) without training a separate reward model.

```python
from alignrl.dpo import DPOConfig, DPORunner

config = DPOConfig(
    model_name="Qwen/Qwen2.5-3B",
    dataset_size=500,
    max_steps=50,
    output_dir="./outputs/dpo",
)

runner = DPORunner(config)
result = runner.train()
print(f"Loss: {result.metrics['train_loss']:.4f}")
```

---

## Evaluating your model

Run benchmarks with `lm-evaluation-harness` across training stages:

```python
from alignrl.eval import EvalConfig, EvalRunner, compare_stages

config = EvalConfig(
    model_name="Qwen/Qwen2.5-3B",
    tasks=["gsm8k", "arc_challenge"],
    limit=50,  # subset for quick testing
)

runner = EvalRunner(config)
results = runner.evaluate_all_stages({
    "base": None,
    "sft": "./outputs/sft/final",
    "grpo": "./outputs/grpo/final",
})

comparison = compare_stages(results)
for benchmark, stages in comparison.items():
    print(f"\n{benchmark}:")
    for stage, metrics in stages.items():
        print(f"  {stage}: {metrics}")
```

Or from the CLI:

```bash
alignrl eval --model Qwen/Qwen2.5-3B --adapter ./outputs/sft/final --stage sft --tasks gsm8k,arc_challenge
```

---

## Serving your model

### Programmatic inference

```python
from alignrl.inference import InferenceConfig, ModelServer, build_prompt

config = InferenceConfig(
    model_name="Qwen/Qwen2.5-3B",
    adapter_path="./outputs/grpo/final",
    backend="unsloth",  # or "vllm" or "mlx"
)

server = ModelServer(config)
server.load()

messages = build_prompt(
    "What is 15% of 240?",
    system="Solve step by step. Put your final answer in \\boxed{}.",
)
print(server.generate(messages))
```

### Gradio comparison demo

Compare outputs across training stages side-by-side:

```bash
alignrl serve --model Qwen/Qwen2.5-3B --stages base sft=./outputs/sft/final grpo=./outputs/grpo/final
```

This launches a Gradio UI at `http://localhost:7860` where you can type a question and see how each training stage responds.

---

## Next steps

- [Concepts](concepts.md) - understand the theory behind each technique
- [Training Guide](training-guide.md) - hyperparameter tuning, hardware requirements, troubleshooting
- [Custom Rewards](custom-rewards.md) - write your own reward functions for GRPO
- [API Reference](api.md) - full documentation for every class and function
