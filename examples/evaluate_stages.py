"""Evaluate and compare model performance across training stages."""
from pathlib import Path

from alignrl.eval import EvalConfig, EvalRunner, compare_stages

config = EvalConfig(
    model_name="Qwen/Qwen2.5-3B",
    tasks=["gsm8k", "arc_challenge"],
    limit=50,  # small subset for demo
)

runner = EvalRunner(config)

# Define which stages to evaluate (None = base model, path = LoRA adapter)
stages = {
    "base": None,
    "sft": "./outputs/sft-quickstart/final",
    # "grpo": "./outputs/grpo-quickstart/final",
    # "dpo": "./outputs/dpo-quickstart/final",
}

results = runner.evaluate_all_stages(stages)
comparison = compare_stages(results)

for benchmark, stage_scores in comparison.items():
    print(f"\n{benchmark}:")
    for stage, metrics in stage_scores.items():
        for metric, score in metrics.items():
            print(f"  {stage}: {metric} = {score:.3f}")

# Save for GitHub Pages dashboard
runner.save_results(results, Path("./results"))
print("\nResults saved to ./results/")
