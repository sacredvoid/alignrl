# Training Guide

## Hardware requirements

| Technique | Min VRAM | Recommended | Notes |
|-----------|----------|-------------|-------|
| SFT (QLoRA) | 12 GB | 16 GB (T4) | Fits on free Colab |
| GRPO (QLoRA) | 14 GB | 16 GB (T4) | `num_generations` affects VRAM |
| DPO (QLoRA) | 12 GB | 16 GB (T4) | `precompute_ref_log_probs=True` saves memory |
| Evaluation | 8 GB | 16 GB (T4) | Read-only, lower memory than training |
| Inference (Unsloth) | 6 GB | 8 GB | 4-bit quantized, single generation |
| Inference (vLLM) | 8 GB | 16 GB | Higher throughput, more VRAM |
| Inference (MLX) | 8 GB unified | 16 GB unified | Apple Silicon only |

All numbers are for Qwen2.5-3B with 4-bit quantization and LoRA rank 16. Larger models or higher ranks will need more memory.

---

## Colab setup

Every notebook in `notebooks/` is designed to run on a free Colab T4. Here's the general setup:

1. Open the notebook in Colab (click the badge in the README)
2. Go to Runtime > Change runtime type > T4 GPU
3. Run the first cell to install dependencies
4. Run cells top to bottom

**Tips for free Colab:**

- Sessions time out after ~90 minutes of inactivity. Keep the tab active.
- The T4 has 16 GB VRAM. If you hit OOM, reduce `per_device_train_batch_size` or `num_generations`.
- Save checkpoints to Google Drive if you want them to persist:

```python
from google.colab import drive
drive.mount('/content/drive')
# Then set output_dir="/content/drive/MyDrive/alignrl/outputs/sft"
```

- If Colab gives you an older GPU (K80, P100), the training will still work but be significantly slower. Restart the runtime to try getting a T4 again.

---

## SFT training deep dive

### What SFT does

SFT trains the model on (instruction, response) pairs. The model learns to generate responses in the expected format. Think of it as "showing the model what good outputs look like."

### Dataset

The default dataset is [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5), which contains ~1M instruction-response conversations. The `format_instruction` function converts the dataset's format to standard chat messages.

For quick experimentation, set `dataset_size=1000` to use a small subset.

### Key hyperparameters

| Parameter | Default | Range to try | What it controls |
|-----------|---------|-------------|-----------------|
| `learning_rate` | `2e-4` | `1e-5` to `5e-4` | How fast the model learns. Too high causes instability. |
| `num_train_epochs` | `1` | `1-3` | More epochs = more exposure to data. Overfitting risk above 3. |
| `per_device_train_batch_size` | `4` | `1-8` | Larger = smoother gradients but more VRAM. |
| `gradient_accumulation_steps` | `4` | `1-8` | Effective batch size = batch_size * accumulation. |
| `lora_r` | `16` | `8-64` | LoRA rank. Higher = more capacity but more VRAM and slower. |
| `max_seq_length` | `2048` | `512-4096` | Longer = handles longer conversations but uses more memory. |

### What to watch for

- **Loss should decrease steadily.** A typical SFT run on OpenHermes goes from ~2.5 to ~1.0 over one epoch.
- **Loss plateaus early?** Try increasing the learning rate or the LoRA rank.
- **Loss spikes?** Reduce the learning rate. This often happens with `lr > 5e-4`.
- **Training is slow?** Reduce `max_seq_length` or increase `per_device_train_batch_size` (if VRAM allows).

### Example config

```yaml
# configs/sft.yaml
model_name: "Qwen/Qwen2.5-3B"
output_dir: "./outputs/sft"
max_seq_length: 2048
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
num_train_epochs: 1
optim: "adamw_8bit"
lora_r: 16
lora_alpha: 32
load_in_4bit: true
report_to: "wandb"
logging_steps: 10
```

---

## GRPO training deep dive

### What GRPO does

GRPO generates multiple completions for each prompt, scores them with reward functions, then updates the model to increase the probability of high-reward completions relative to the group mean. No critic model, no reward model training.

### Dataset

The default is [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), a dataset of grade-school math problems with numerical answers. Each problem is converted to a prompt with a system message asking for step-by-step reasoning and a `\boxed{}` final answer.

### Key hyperparameters

| Parameter | Default | Range to try | What it controls |
|-----------|---------|-------------|-----------------|
| `learning_rate` | `5e-6` | `1e-6` to `2e-5` | Much lower than SFT. RL is sensitive to large updates. |
| `num_generations` | `8` | `4-16` | Completions per prompt. More = better baseline estimate but more VRAM and compute. |
| `beta` | `0.001` | `0.0001` to `0.01` | KL penalty. Higher = stays closer to reference policy. |
| `max_completion_length` | `512` | `256-1024` | Max tokens per generation. Math rarely needs >512. |
| `max_steps` | `250` | `100-500` | Total training steps. |
| `reward_weights` | `None` | `[0.7, 0.3]` etc. | Relative importance of each reward function. |

### Training dynamics

GRPO training looks different from SFT:

- **Loss is noisy.** This is normal for RL. Look at the trend over 50+ steps, not individual points.
- **Mean reward should increase.** The `reward_history` in the training result tracks this. If mean reward is flat after 100 steps, your reward function may be too hard or the learning rate too low.
- **Reward starts near 0?** Normal for math tasks. The base model might only get ~30% of GSM8K right.
- **Reward jumps to 1.0 quickly?** The task might be too easy, or the reward function is too lenient. Try harder benchmarks or stricter rewards.

### Example config

```yaml
# configs/grpo.yaml
model_name: "Qwen/Qwen2.5-3B"
output_dir: "./outputs/grpo"
max_seq_length: 2048
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 5e-6
max_steps: 250
optim: "adamw_8bit"
lora_r: 16
lora_alpha: 32
load_in_4bit: true
report_to: "wandb"
```

---

## DPO training deep dive

### What DPO does

DPO optimizes the model to prefer "chosen" over "rejected" responses using a contrastive loss. It implicitly learns a reward model and optimizes against it in a single step.

### Dataset

The default is [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), which contains preference pairs generated by multiple LLMs and ranked by GPT-4. Each example has a prompt, a chosen response, and a rejected response.

### Key hyperparameters

| Parameter | Default | Range to try | What it controls |
|-----------|---------|-------------|-----------------|
| `learning_rate` | `5e-7` | `1e-7` to `5e-6` | Very low. DPO is sensitive to large LR. |
| `beta` | `0.1` | `0.01` to `0.5` | Controls how much the model can diverge from the reference. Lower = more freedom. Higher = stays closer to original. |
| `max_length` | `1024` | `512-2048` | Max sequence length for chosen/rejected pairs. |
| `precompute_ref_log_probs` | `True` | `True/False` | Precomputing saves VRAM by not holding the reference model in memory during training. |

### What to watch for

- **DPO loss should decrease from ~0.69.** It starts near ln(2) because the model initially assigns equal probability to chosen and rejected.
- **Loss drops too fast?** The `beta` might be too low, letting the model drift too far from the reference.
- **Loss barely moves?** Try increasing the learning rate or lowering `beta`.
- **Catastrophic forgetting?** If eval scores drop on unrelated benchmarks after DPO, increase `beta` to keep the model closer to the reference policy.

### Example config

```yaml
# configs/dpo.yaml
model_name: "Qwen/Qwen2.5-3B"
output_dir: "./outputs/dpo"
max_seq_length: 1024
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5e-7
num_train_epochs: 1
optim: "adamw_8bit"
lora_r: 16
lora_alpha: 32
load_in_4bit: true
report_to: "wandb"
```

---

## Common issues and troubleshooting

### OOM (Out of Memory)

**Symptoms:** `CUDA out of memory`, `RuntimeError: CUDA error: out of memory`

**Fixes (in order of impact):**

1. Reduce `per_device_train_batch_size` to 1 or 2
2. Reduce `num_generations` (GRPO only) to 4
3. Reduce `max_seq_length` or `max_completion_length`
4. Enable `precompute_ref_log_probs=True` (DPO only)
5. Reduce `lora_r` from 16 to 8
6. Clear GPU cache before training:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Slow training

**Possible causes:**

- **CPU bottleneck on data loading.** Set `dataloader_num_workers` in the TRL config (not exposed in alignrl yet, but you can modify the runner).
- **Small batch size with many accumulation steps.** Each accumulation step has overhead. Try batch_size=4, accumulation=2 instead of batch_size=1, accumulation=8.
- **Disk I/O on Colab.** The first epoch is slow because the dataset is being downloaded and tokenized. Subsequent epochs use the cached version.

### Reward not improving (GRPO)

**Possible causes:**

- **Learning rate too low.** Try `1e-5` instead of `5e-6`.
- **Not enough generations.** With `num_generations=4`, the variance of the group baseline is high. Try 8 or 16.
- **Reward function too strict.** Check that the base model can actually solve some problems. If it can't get any right, the reward signal is all zeros and there's nothing to learn from.
- **Task too hard.** Start with an SFT model instead of the base model. SFT provides a better starting point for RL.

### Loss goes to NaN

**Possible causes:**

- Learning rate too high. Halve it and try again.
- Gradient explosion. Add gradient clipping by modifying the TRL config: `max_grad_norm=1.0`.
- Corrupted checkpoint. Start fresh from the base model.

### Adapter won't load

**Possible causes:**

- The path should point to the directory containing `adapter_config.json`, not the checkpoint directory. Usually `./outputs/<stage>/final/`.
- Model mismatch. The adapter was trained on a different base model than you're trying to load it onto.
- Missing PEFT. Install with `pip install peft`.
