"""Launch the Gradio comparison demo."""
from alignrl.demo import create_demo

# Compare base model vs trained stages
demo = create_demo(
    stages={
        "base": None,
        "sft": "./outputs/sft-quickstart/final",
        "grpo": "./outputs/grpo-quickstart/final",
    },
    model_name="Qwen/Qwen2.5-3B",
)

demo.launch(share=True)  # share=True creates a public URL
