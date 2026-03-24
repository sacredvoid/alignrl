"""Quick start: Align a model with human preferences using DPO."""
from alignrl.dpo import DPOConfig, DPORunner

config = DPOConfig(
    model_name="Qwen/Qwen2.5-3B",
    dataset_size=500,
    max_steps=50,
    output_dir="./outputs/dpo-quickstart",
    report_to="none",
)

runner = DPORunner(config)
result = runner.train()
print(f"Training complete in {result.num_steps} steps")
print(f"Final loss: {result.metrics['train_loss']:.4f}")
