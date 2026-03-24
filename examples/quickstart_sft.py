"""Quick start: Fine-tune Qwen2.5-3B on instruction data with QLoRA."""
from alignrl.sft import SFTConfig, SFTRunner

config = SFTConfig(
    model_name="Qwen/Qwen2.5-3B",
    dataset_name="teknium/OpenHermes-2.5",
    dataset_size=1000,  # small subset for demo
    max_steps=50,
    output_dir="./outputs/sft-quickstart",
    report_to="none",
)

runner = SFTRunner(config)
result = runner.train()
print(f"Training complete in {result.num_steps} steps")
print(f"Final loss: {result.metrics['train_loss']:.4f}")
print(f"Adapter saved to: {result.output_dir}")
