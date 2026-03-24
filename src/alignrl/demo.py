"""Gradio demo for comparing model stages."""
from __future__ import annotations

from alignrl.inference import InferenceConfig, ModelServer, build_prompt

MATH_SYSTEM = (
    "You are a helpful math assistant. Solve problems step by step, "
    "showing your reasoning clearly. Put your final answer in \\boxed{}."
)


def create_demo(
    stages: dict[str, str | None],
    model_name: str = "Qwen/Qwen2.5-3B",
):
    """Create a Gradio demo comparing model outputs across training stages.

    Args:
        stages: {stage_name: adapter_path_or_None}
        model_name: base model name
    """
    import gradio as gr

    servers: dict[str, ModelServer] = {}
    for stage, adapter_path in stages.items():
        config = InferenceConfig(
            model_name=model_name,
            adapter_path=adapter_path,
            backend="unsloth",
        )
        server = ModelServer(config)
        server.load()
        servers[stage] = server

    def respond(message: str, stage: str) -> str:
        messages = build_prompt(message, system=MATH_SYSTEM)
        return servers[stage].generate(messages)

    def respond_all(message: str) -> list[str]:
        return [respond(message, stage) for stage in stages]

    with gr.Blocks(title="alignrl - Post-Training Comparison") as app:
        gr.Markdown("# alignrl - Compare Post-Training Stages")
        gr.Markdown("Ask a math question and see how each training stage responds.")

        with gr.Row():
            input_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What is 15% of 240?",
                lines=2,
            )
            submit = gr.Button("Compare All Stages", variant="primary")

        output_boxes = []
        with gr.Row():
            for stage in stages:
                box = gr.Textbox(label=f"Stage: {stage}", lines=10, interactive=False)
                output_boxes.append(box)

        submit.click(fn=respond_all, inputs=[input_box], outputs=output_boxes)

    return app


if __name__ == "__main__":
    demo = create_demo(
        stages={
            "base": None,
            "sft": "./outputs/sft/final",
            "grpo": "./outputs/grpo/final",
        }
    )
    demo.launch()
