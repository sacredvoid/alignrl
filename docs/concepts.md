# Concepts

## What is post-training?

Post-training is what happens after pre-training. A pre-trained language model has learned to predict the next token from trillions of tokens of internet text, but it doesn't know how to follow instructions, reason step-by-step, or avoid harmful outputs. Post-training closes that gap.

The main post-training techniques:

| Technique | What it does | Data required |
|-----------|-------------|---------------|
| **SFT** (Supervised Fine-Tuning) | Teaches the model to follow instructions | (instruction, response) pairs |
| **RLHF** (RL from Human Feedback) | Optimizes a reward model trained on human preferences | Human preference rankings |
| **DPO** (Direct Preference Optimization) | Directly optimizes on preference pairs, no reward model | (prompt, chosen, rejected) triples |
| **GRPO** (Group Relative Policy Optimization) | RL with verifiable rewards, no critic model | Problems with checkable answers |

Each technique builds on the previous one. SFT gets the model into the right format. Then RL or preference optimization pushes quality higher.

---

## The alignrl pipeline

```
Base Model (Qwen2.5-3B)
    |
    v
SFT (instruction following)
    |
    +-----> GRPO (math reasoning via RL)
    |           |
    +-----> DPO (preference alignment)
    |           |
    v           v
Evaluation (GSM8K, MATH, ARC)
    |
    v
Inference (Unsloth / vLLM / MLX)
```

The pipeline is modular. You can run SFT alone, skip straight to GRPO, or chain stages together. Each training stage produces a QLoRA adapter that can be evaluated and served independently.

**Typical workflow:**

1. **SFT** on OpenHermes-2.5 to teach instruction following
2. **GRPO** on GSM8K to train math reasoning with verifiable rewards
3. **Eval** on GSM8K, MATH, and ARC-Challenge to measure improvement
4. **Serve** with Gradio to compare outputs across stages

---

## Why QLoRA?

Full fine-tuning of a 3B parameter model requires roughly 24GB of VRAM just for model weights and optimizer states. That rules out free Colab GPUs (16GB T4).

**QLoRA** (Quantized Low-Rank Adaptation) solves this:

1. **4-bit quantization** - The base model weights are loaded in 4-bit precision, cutting memory by ~4x
2. **LoRA adapters** - Instead of updating all parameters, we train small rank-decomposition matrices injected into attention and MLP layers. For rank 16, this is ~0.5% of total parameters
3. **Gradient checkpointing** - Recompute activations during backward pass instead of storing them

The result: Qwen2.5-3B trains comfortably on a T4 with ~12GB peak VRAM.

**Tradeoffs:**

- Slightly lower quality than full fine-tuning (usually <1% on benchmarks)
- Adapters add latency at inference time unless merged
- Not all quantization formats work with all training frameworks

alignrl uses Unsloth's `FastLanguageModel` which handles quantization, LoRA injection, and gradient checkpointing in a single call.

---

## Why GRPO over PPO?

PPO (Proximal Policy Optimization) is the original RL algorithm used in RLHF. It works, but it's expensive:

| | PPO | GRPO |
|---|-----|------|
| **Critic model** | Required (same size as policy) | Not needed |
| **VRAM** | ~2x model memory | ~1x model memory |
| **Reward signal** | Learned reward model | Verifiable reward functions |
| **Stability** | Notoriously finicky | More stable (group-relative baseline) |
| **Implementation** | Complex (GAE, clipping, value loss) | Simpler (just policy gradient + KL) |

GRPO generates multiple completions per prompt (a "group"), scores each with a reward function, then uses the group mean as a baseline. This eliminates the critic entirely.

For math reasoning, where you can check if the answer is correct, GRPO is the clear choice. The reward signal is exact (not learned), training is stable, and you need half the memory.

**When PPO still makes sense:** Tasks where you can't write a verifiable reward function, like open-ended conversation quality. In those cases, you need a learned reward model, which means PPO or similar.

---

## Why DPO over RLHF?

Traditional RLHF has three stages:
1. Collect human preference data
2. Train a reward model on the preferences
3. Use PPO to optimize the policy against the reward model

DPO collapses steps 2 and 3 into a single supervised training objective. Given (prompt, chosen_response, rejected_response) triples, DPO directly updates the model to prefer the chosen response.

**Advantages:**

- **No reward model training** - One fewer model to train, debug, and store
- **No RL instability** - Supervised objective, standard loss curves
- **Less VRAM** - No need to hold a reward model in memory alongside the policy
- **Simpler** - Just another supervised training loop with a special loss function

**Tradeoffs:**

- Requires preference pairs (which are expensive to collect from humans)
- Less flexible than a trained reward model that can score arbitrary outputs
- The `beta` parameter controls how far the model can drift from the reference policy, and getting it right matters

alignrl uses the UltraFeedback dataset for DPO, which contains AI-generated preference pairs. For production use, you'd want human-annotated data.

---

## How these techniques interact

In practice, post-training is iterative:

- **SFT first** makes RL and DPO more effective. A model that already follows instructions learns faster from reward signals.
- **GRPO and DPO address different things.** GRPO is best for verifiable tasks (math, code). DPO is best for subjective quality (helpfulness, safety, tone).
- **Evaluation drives decisions.** The GSM8K score tells you if GRPO helped with math. ARC-Challenge tells you about general reasoning. You pick the technique based on what the benchmarks show.

See the [Training Guide](training-guide.md) for practical advice on choosing techniques and tuning hyperparameters.
