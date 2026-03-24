"""Serve a trained model and compare stages interactively."""
from alignrl.inference import InferenceConfig, ModelServer, build_prompt

# Load the GRPO-trained model
config = InferenceConfig(
    model_name="Qwen/Qwen2.5-3B",
    adapter_path="./outputs/grpo-quickstart/final",
    backend="unsloth",
    temperature=0.7,
    max_tokens=512,
)

server = ModelServer(config)
server.load()

# Generate a response
messages = build_prompt(
    "A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount. "
    "How much would 7 apples cost?",
    system="Solve step by step. Put your final answer in \\boxed{}.",
)

response = server.generate(messages)
print(response)
