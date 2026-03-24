# alignrl Examples

Minimal scripts demonstrating each module in the `alignrl` package.

| Script | Description |
|--------|-------------|
| [`quickstart_sft.py`](quickstart_sft.py) | Fine-tune Qwen2.5-3B on instruction data with QLoRA |
| [`quickstart_grpo.py`](quickstart_grpo.py) | Train a math reasoning model with GRPO and verifiable rewards |
| [`quickstart_dpo.py`](quickstart_dpo.py) | Align a model with human preferences using DPO |
| [`evaluate_stages.py`](evaluate_stages.py) | Evaluate and compare model performance across training stages |
| [`serve_model.py`](serve_model.py) | Serve a trained model for interactive inference |
| [`custom_rewards.py`](custom_rewards.py) | Define custom reward functions for GRPO training |
| [`launch_demo.py`](launch_demo.py) | Launch the Gradio comparison demo |

## Prerequisites

Install the package first:

```bash
pip install -e ".[dev]"
```

All examples use `report_to="none"` and small dataset subsets so they run quickly without a W&B account or large downloads.

## Recommended Order

1. **quickstart_sft.py** - supervised fine-tuning baseline
2. **quickstart_grpo.py** - reinforcement learning with math rewards
3. **quickstart_dpo.py** - preference alignment
4. **evaluate_stages.py** - compare all stages on benchmarks
5. **serve_model.py** - interact with a trained model
6. **custom_rewards.py** - extend GRPO with your own reward functions
7. **launch_demo.py** - visual side-by-side comparison
