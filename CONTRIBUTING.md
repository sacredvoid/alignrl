# Contributing to alignrl

## Development setup

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/sacredvoid/alignrl.git
cd alignrl
pip install -e ".[dev]"
```

This installs `pytest`, `ruff`, and `mypy` alongside the base package. You do not need GPU dependencies for running tests or linting.

## Running tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_rewards.py

# Run with verbose output
pytest -v
```

Tests are in `tests/` and use `pytest`. GPU-dependent tests are skipped automatically if CUDA is not available.

The test suite covers:
- Reward functions (answer extraction, math verification, format checking)
- Config loading and validation
- Result type serialization
- Runner initialization (not full training, since that requires a GPU)

## Code style

The project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Check for lint errors
ruff check .

# Auto-fix what can be fixed
ruff check . --fix

# Format code
ruff format .
```

Configuration is in `pyproject.toml`:
- Target: Python 3.10
- Line length: 100
- Enabled rules: E, F, I, N, UP, B, SIM, TCH

Type checking with mypy:

```bash
mypy src/alignrl
```

## Project structure

```
alignrl/
  src/alignrl/       # Package source
    config.py        # BaseTrainConfig (Pydantic)
    sft.py           # SFTConfig, SFTRunner
    grpo.py          # GRPOConfig, GRPORunner
    dpo.py           # DPOConfig, DPORunner
    eval.py          # EvalConfig, EvalRunner, compare_stages
    inference.py     # InferenceConfig, ModelServer
    rewards.py       # Reward functions for GRPO
    demo.py          # Gradio comparison UI
    cli.py           # CLI entry point
    types.py         # TrainResult, EvalResult, Trainer protocol
  configs/           # YAML configs for each training stage
  docs/              # Documentation and GitHub Pages dashboard
  examples/          # Runnable example scripts
  notebooks/         # Colab-ready Jupyter notebooks
  tests/             # Test suite
  pyproject.toml     # Build config and dependencies
```

## Adding a new training technique

To add a new post-training technique (e.g., KTO, SPIN, or ORPO):

1. **Create the module** at `src/alignrl/<technique>.py`. Follow the existing pattern:
   - A config class extending `BaseTrainConfig` with technique-specific fields
   - A runner class with `train() -> TrainResult`, `save()`, and `load()` methods
   - The runner should implement the `Trainer` protocol from `alignrl.types`

2. **Register it in the CLI.** Add a new branch in `cmd_train()` in `cli.py`:
   ```python
   elif stage == "your_technique":
       from alignrl.your_technique import YourConfig, YourRunner
       config = YourConfig.from_yaml(config_path)
       runner = YourRunner(config)
   ```
   And add the stage name to the `choices` list in the argparser.

3. **Add a YAML config** in `configs/your_technique.yaml`.

4. **Write tests** in `tests/test_your_technique.py`. At minimum, test config construction and validation. If you can test the runner logic without a GPU, do that too.

5. **Add an example script** in `examples/`.

6. **Add a notebook** in `notebooks/` if the technique warrants a walkthrough.

## Submitting PRs

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run `ruff check .` and `ruff format .` to ensure code style compliance
4. Run `pytest` to make sure tests pass
5. Write a clear PR description explaining what changed and why
6. Submit the PR against `main`

Keep PRs focused. One feature or fix per PR. If your change touches multiple modules, explain the connection in the PR description.
