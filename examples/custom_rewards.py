"""Define custom reward functions for GRPO training."""
from alignrl.grpo import GRPOConfig, GRPORunner
from alignrl.rewards import extract_answer, math_verify_reward


def length_penalty_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Penalize overly long or short responses.

    Ideal length: 100-500 characters of reasoning before the answer.
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        length = len(content)
        if 100 <= length <= 500:
            rewards.append(1.0)
        elif 50 <= length <= 800:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


# Use custom + built-in rewards together
config = GRPOConfig(
    model_name="Qwen/Qwen2.5-3B",
    dataset_size=200,
    max_steps=30,
    num_generations=4,
    output_dir="./outputs/grpo-custom-rewards",
    report_to="none",
    reward_weights=[0.7, 0.3],  # weight accuracy higher than length
)

runner = GRPORunner(
    config=config,
    reward_funcs=[math_verify_reward, length_penalty_reward],
)

print("Training with custom reward functions...")
result = runner.train()
print(f"Done! Steps: {result.num_steps}, Loss: {result.metrics['train_loss']:.4f}")
