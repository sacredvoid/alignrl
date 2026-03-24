"""CLI entry point for alignrl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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


def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation benchmarks."""
    from alignrl.eval import EvalConfig, EvalRunner

    config = EvalConfig(
        model_name=args.model,
        adapter_path=args.adapter,
        tasks=args.tasks.split(","),
        limit=args.limit,
    )
    runner = EvalRunner(config)
    result = runner.evaluate(stage=args.stage)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"eval_{args.stage}.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"Evaluation complete for stage '{args.stage}':")
    for benchmark, metrics in result.benchmarks.items():
        print(f"  {benchmark}: {metrics}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch Gradio demo."""
    from alignrl.demo import create_demo

    stages = {}
    for spec in args.stages:
        name, _, path = spec.partition("=")
        stages[name] = path if path else None

    demo = create_demo(stages=stages, model_name=args.model)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


def main() -> None:
    parser = argparse.ArgumentParser(prog="alignrl", description="LLM Post-Training Playbook")
    sub = parser.add_subparsers(dest="command", required=True)

    # Train
    train_p = sub.add_parser("train", help="Run training pipeline")
    train_p.add_argument("stage", choices=["sft", "grpo", "dpo"])
    train_p.add_argument("-c", "--config", required=True, help="Path to YAML config")
    train_p.set_defaults(func=cmd_train)

    # Eval
    eval_p = sub.add_parser("eval", help="Run evaluation benchmarks")
    eval_p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    eval_p.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    eval_p.add_argument("--stage", default="base", help="Stage label (base/sft/grpo/dpo)")
    eval_p.add_argument("--tasks", default="gsm8k,arc_challenge")
    eval_p.add_argument("--limit", type=int, default=None)
    eval_p.add_argument("--output", default="./results")
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
    serve_p.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
