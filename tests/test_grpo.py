"""Tests for GRPO module."""
from alignrl.grpo import GRPOConfig


class TestGRPOConfig:
    def test_defaults(self) -> None:
        cfg = GRPOConfig()
        assert cfg.num_generations == 8
        assert cfg.beta == 0.001
        assert cfg.learning_rate == 5e-6
        assert cfg.reward_weights is None

    def test_custom_reward_weights(self) -> None:
        cfg = GRPOConfig(reward_weights=[0.7, 0.3])
        assert cfg.reward_weights == [0.7, 0.3]
