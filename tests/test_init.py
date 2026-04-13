"""Tests for public API exports."""

import alignrl


class TestPublicAPI:
    def test_version(self) -> None:
        assert alignrl.__version__ == "0.4.0"

    def test_version_matches_package_metadata(self) -> None:
        import contextlib
        from importlib.metadata import PackageNotFoundError, version

        # Running from a source checkout without an installed dist is fine.
        with contextlib.suppress(PackageNotFoundError):
            assert version("alignrl") == alignrl.__version__

    def test_all_exports_listed(self) -> None:
        assert "__version__" in alignrl.__all__
        assert "SFTConfig" in alignrl.__all__
        assert "TrainResult" in alignrl.__all__

    def test_lazy_import_rewards(self) -> None:
        from alignrl import extract_answer, format_reward, math_verify_reward

        assert callable(math_verify_reward)
        assert callable(format_reward)
        assert callable(extract_answer)

    def test_lazy_import_eval_helpers(self) -> None:
        from alignrl import BENCHMARK_PRESETS, compare_stages, parse_results

        assert isinstance(BENCHMARK_PRESETS, dict)
        assert "core" in BENCHMARK_PRESETS
        assert callable(compare_stages)
        assert callable(parse_results)

    def test_dir_includes_lazy_exports(self) -> None:
        names = set(dir(alignrl))
        assert "SFTConfig" in names
        assert "math_verify_reward" in names
        assert "__version__" in names

    def test_lazy_import_sft(self) -> None:
        from alignrl import SFTConfig, SFTRunner

        assert SFTConfig is not None
        assert SFTRunner is not None

    def test_lazy_import_dpo(self) -> None:
        from alignrl import DPOConfig, DPORunner

        assert DPOConfig is not None
        assert DPORunner is not None

    def test_lazy_import_grpo(self) -> None:
        from alignrl import GRPOConfig, GRPORunner

        assert GRPOConfig is not None
        assert GRPORunner is not None

    def test_lazy_import_eval(self) -> None:
        from alignrl import EvalConfig, EvalRunner

        assert EvalConfig is not None
        assert EvalRunner is not None

    def test_lazy_import_inference(self) -> None:
        from alignrl import InferenceConfig, ModelServer, build_prompt

        assert InferenceConfig is not None
        assert ModelServer is not None
        assert build_prompt is not None

    def test_lazy_import_types(self) -> None:
        from alignrl import EvalResult, TrainResult

        assert TrainResult is not None
        assert EvalResult is not None

    def test_invalid_attribute_raises(self) -> None:
        import pytest

        with pytest.raises(AttributeError, match="no attribute"):
            _ = alignrl.NonExistentThing
