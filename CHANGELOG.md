# Changelog

All notable changes to `alignrl` are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-04-13

### Added

- **Expanded public API.** `alignrl` now lazily exports reward helpers
  (`math_verify_reward`, `format_reward`, `extract_answer`), evaluation
  helpers (`compare_stages`, `parse_results`, `BENCHMARK_PRESETS`), and the
  `Trainer` protocol. `dir(alignrl)` reflects every lazy export for better
  IDE discoverability.
- **`BaseTrainConfig.to_yaml()`** — serialize a validated config back to YAML
  for round-tripping from CLI overrides to committed config files. When a
  path is given, parent directories are created and the file is written.
- **`alignrl version` subcommand** and a top-level `-V` / `--version` flag
  on the CLI.
- **CLI `eval` flags**: `--num-fewshot` and `--batch-size` for configuring
  few-shot prompting and lm-eval batch size from the command line.
- **CLI `serve` flags**: `--temperature` and `--max-tokens` are now piped
  through to every `ModelServer` in the Gradio comparison demo.
- **Config validation.** `BaseTrainConfig` now uses Pydantic field
  constraints (`gt=0`, `ge=0`, etc.) for numeric fields and `extra="forbid"`
  so typos in YAML configs fail loudly at load time instead of silently
  falling back to defaults.

### Changed

- **Reward normalization is more robust.** `_normalize_numeric` now handles
  thousands separators (`1,234`), currency prefixes (`$42`, `\$42`),
  trailing percent (`50%`), and strips trailing periods. `_answers_match`
  performs case-insensitive comparison before numeric normalization.
- **`extract_answer` supports more formats.** The regex now matches
  `final answer: X` and `answer X` variants, accepts commas inside numeric
  answers, and unwraps `\text{...}` inside `\boxed{...}` groups.

### Fixed

- Trailing punctuation (`.`, `,`, `;`, `:`) is no longer carried into
  extracted answers from `"the answer is …"` / `"= …"` patterns, which
  previously caused spurious reward mismatches.

## [0.3.0] - 2026-03-25

### Added

- Public API lazy-imports surface (`alignrl.SFTConfig`, `alignrl.GRPORunner`, …).
- W&B integration: `detect_wandb`, `log_eval_to_wandb`, CLI `--wandb` flag.
- HuggingFace Hub helpers: `push_adapter`, `merge_and_push`.
- Benchmark presets for `EvalConfig` (`core`, `reasoning`, `leaderboard`).
- Docker support (Dockerfile, docker-compose) for GPU-ready workflows.

### Fixed

- Guard against empty `loss_history` in all trainers.
- Copy preset list to prevent aliasing mutation in `EvalConfig`.
- Cache lazy imports in module globals after first resolution.
- Pass LoRA adapter to vLLM via `LoRARequest` instead of silently dropping it.

[0.4.0]: https://github.com/sacredvoid/alignrl/releases/tag/v0.4.0
[0.3.0]: https://github.com/sacredvoid/alignrl/releases/tag/v0.3.0
