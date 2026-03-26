"""Tests for public API exports."""

import alignrl


class TestPublicAPI:
    def test_version(self) -> None:
        assert alignrl.__version__ == "0.3.0"

    def test_all_exports_listed(self) -> None:
        assert "__version__" in alignrl.__all__
        assert "SFTConfig" in alignrl.__all__
        assert "TrainResult" in alignrl.__all__

    def test_lazy_import_sft(self) -> None:
        from alignrl import SFTConfig, SFTRunner

        assert SFTConfig is not None
        assert SFTRunner is not None

    def test_lazy_import_dpo(self) -> None:
        from alignrl import DPOConfig, DPORunner

        assert DPOConfig is not None

    def test_lazy_import_grpo(self) -> None:
        from alignrl import GRPOConfig, GRPORunner

        assert GRPOConfig is not None

    def test_lazy_import_eval(self) -> None:
        from alignrl import EvalConfig, EvalRunner

        assert EvalConfig is not None

    def test_lazy_import_inference(self) -> None:
        from alignrl import InferenceConfig, ModelServer, build_prompt

        assert InferenceConfig is not None
        assert build_prompt is not None

    def test_lazy_import_types(self) -> None:
        from alignrl import EvalResult, TrainResult

        assert TrainResult is not None
        assert EvalResult is not None

    def test_invalid_attribute_raises(self) -> None:
        import pytest

        with pytest.raises(AttributeError, match="no attribute"):
            _ = alignrl.NonExistentThing
