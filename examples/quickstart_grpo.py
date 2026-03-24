"""Quick start: Train a math reasoning model with GRPO and verifiable rewards."""
from alignrl.grpo import GRPOConfig, GRPORunner

config = GRPOConfig(
    model_name="Qwen/Qwen2.5-3B",
    dataset_size=500,
    max_steps=50,
    num_generations=4,  # lower for speed
    output_dir="./outputs/grpo-quickstart",
    report_to="none",
)

runner = GRPORunner(config)
result = runner.train()
print(f"Training complete in {result.num_steps} steps")
print(f"Final loss: {result.metrics['train_loss']:.4f}")
