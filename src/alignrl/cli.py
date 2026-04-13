"""CLI entry point for alignrl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_version(args: argparse.Namespace) -> None:
    """Print the installed alignrl version."""
    from alignrl import __version__

    print(f"alignrl {__version__}")


def cmd_train(args: argparse.Namespace) -> None:
    """Run a training pipeline."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    stage = args.stage
    if stage == "sft":
        from alignrl.sft import SFTConfig, SFTRunner

        config = SFTConfig.from_yaml(config_path)
        runner = SFTRunner(config)
    elif stage == "grpo":
        from alignrl.grpo import GRPOConfig, GRPORunner

        config = GRPOConfig.from_yaml(config_path)
        runner = GRPORunner(config)
    elif stage == "dpo":
        from alignrl.dpo import DPOConfig, DPORunner

        config = DPOConfig.from_yaml(config_path)
        runner = DPORunner(config)
    else:
        print(f"Unknown stage: {stage}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting {stage} training with config: {config_path}")
    result = runner.train()
    print(f"Training complete. Output: {result.output_dir}")
    print(f"Final loss: {result.metrics.get('train_loss', 'N/A')}")

    if args.push:
        print(f"Pushing adapter to HuggingFace Hub: {args.push}")
        url = runner.push_to_hub(args.push)
        print(f"Pushed to: {url}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation benchmarks."""
    from alignrl.eval import EvalConfig, EvalRunner

    config_kwargs: dict = {
        "model_name": args.model,
        "adapter_path": args.adapter,
        "limit": args.limit,
    }
    if args.tasks:
        config_kwargs["tasks"] = args.tasks.split(",")
    if args.preset:
        config_kwargs["preset"] = args.preset
    if getattr(args, "num_fewshot", None) is not None:
        config_kwargs["num_fewshot"] = args.num_fewshot
    if getattr(args, "batch_size", None) is not None:
        config_kwargs["batch_size"] = args.batch_size

    config = EvalConfig(**config_kwargs)
    runner = EvalRunner(config)
    result = runner.evaluate(stage=args.stage)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"eval_{args.stage}.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"Evaluation complete for stage '{args.stage}':")
    for benchmark, metrics in result.benchmarks.items():
        print(f"  {benchmark}: {metrics}")

    if args.wandb:
        from alignrl.callbacks import log_eval_to_wandb

        log_eval_to_wandb([result])
        print("Results logged to Weights & Biases.")


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch Gradio demo."""
    from alignrl.demo import create_demo

    stages = {}
    for spec in args.stages:
        name, _, path = spec.partition("=")
        stages[name] = path if path else None

    demo_kwargs: dict = {"stages": stages, "model_name": args.model}
    if getattr(args, "temperature", None) is not None:
        demo_kwargs["temperature"] = args.temperature
    if getattr(args, "max_tokens", None) is not None:
        demo_kwargs["max_tokens"] = args.max_tokens

    demo = create_demo(**demo_kwargs)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


def main() -> None:
    from alignrl import __version__

    parser = argparse.ArgumentParser(prog="alignrl", description="LLM Post-Training Playbook")
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"alignrl {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Version (as a subcommand for scripting use)
    version_p = sub.add_parser("version", help="Print the installed alignrl version")
    version_p.set_defaults(func=cmd_version)

    # Train
    train_p = sub.add_parser("train", help="Run training pipeline")
    train_p.add_argument("stage", choices=["sft", "grpo", "dpo"])
    train_p.add_argument("-c", "--config", required=True, help="Path to YAML config")
    train_p.add_argument("--push", default=None, help="Push adapter to HF Hub (repo ID)")
    train_p.set_defaults(func=cmd_train)

    # Eval
    eval_p = sub.add_parser("eval", help="Run evaluation benchmarks")
    eval_p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    eval_p.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    eval_p.add_argument("--stage", default="base", help="Stage label (base/sft/grpo/dpo)")
    eval_p.add_argument(
        "--tasks", default=None, help="Comma-separated task list (overrides preset)"
    )
    eval_p.add_argument(
        "--preset",
        default=None,
        choices=["core", "reasoning", "leaderboard"],
        help="Benchmark preset (default: core)",
    )
    eval_p.add_argument(
        "--num-fewshot",
        dest="num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (default: 0)",
    )
    eval_p.add_argument(
        "--batch-size",
        dest="batch_size",
        default=None,
        help="Batch size for evaluation (e.g. 'auto', 8)",
    )
    eval_p.add_argument("--limit", type=int, default=None)
    eval_p.add_argument("--output", default="./results")
    eval_p.add_argument("--wandb", action="store_true", help="Log results to W&B")
    eval_p.set_defaults(func=cmd_eval)

    # Serve
    serve_p = sub.add_parser("serve", help="Launch Gradio comparison demo")
    serve_p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    serve_p.add_argument(
        "--stages",
        nargs="+",
        required=True,
        help="Stage specs: 'base' or 'sft=./outputs/sft/final'",
    )
    serve_p.add_argument("--port", type=int, default=7860)
    serve_p.add_argument("--share", action="store_true")
    serve_p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for generation (default: 0.7)",
    )
    serve_p.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate per response (default: 512)",
    )
    serve_p.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
