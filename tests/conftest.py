"""Shared test fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    return out


@pytest.fixture
def sample_config(tmp_output: Path) -> dict:
    return {
        "model_name": "Qwen/Qwen2.5-3B",
        "output_dir": str(tmp_output),
        "max_steps": 10,
        "per_device_train_batch_size": 1,
    }
