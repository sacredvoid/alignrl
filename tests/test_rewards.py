"""Tests for reward functions."""

from alignrl.rewards import extract_answer, format_reward, math_verify_reward


class TestExtractAnswer:
    def test_boxed_format(self) -> None:
        assert extract_answer(r"The answer is \boxed{42}") == "42"

    def test_boxed_with_latex(self) -> None:
        assert extract_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_answer_is_prefix(self) -> None:
        assert extract_answer("The answer is 42.") == "42"

    def test_equals_sign(self) -> None:
        assert extract_answer("x = 7") == "7"

    def test_no_answer_found(self) -> None:
        assert extract_answer("I'm not sure about this problem") is None

    def test_multiple_boxed_takes_last(self) -> None:
        assert extract_answer(r"\boxed{1} and \boxed{2}") == "2"


class TestMathVerifyReward:
    def test_correct_integer(self) -> None:
        completions = [[{"content": r"Let me think... \boxed{42}"}]]
        rewards = math_verify_reward(completions, solution=["42"])
        assert rewards == [1.0]

    def test_incorrect(self) -> None:
        completions = [[{"content": r"\boxed{43}"}]]
        rewards = math_verify_reward(completions, solution=["42"])
        assert rewards == [0.0]

    def test_no_answer_extracted(self) -> None:
        completions = [[{"content": "I don't know"}]]
        rewards = math_verify_reward(completions, solution=["42"])
        assert rewards == [0.0]

    def test_batch(self) -> None:
        completions = [
            [{"content": r"\boxed{5}"}],
            [{"content": r"\boxed{3}"}],
            [{"content": r"\boxed{5}"}],
        ]
        rewards = math_verify_reward(completions, solution=["5", "5", "5"])
        assert rewards == [1.0, 0.0, 1.0]

    def test_numeric_equivalence(self) -> None:
        completions = [[{"content": r"\boxed{3.0}"}]]
        rewards = math_verify_reward(completions, solution=["3"])
        assert rewards == [1.0]


class TestFormatReward:
    def test_correct_format(self) -> None:
        completions = [[{"content": "Let me think step by step.\n\n\\boxed{42}"}]]
        rewards = format_reward(completions)
        assert rewards[0] > 0

    def test_no_boxed(self) -> None:
        completions = [[{"content": "The answer is 42"}]]
        rewards = format_reward(completions)
        assert rewards[0] == 0.0

    def test_boxed_not_at_end_penalized(self) -> None:
        completions = [[{"content": r"\boxed{42} and then more text after that keeps going"}]]
        good = [[{"content": r"Reasoning here.\n\boxed{42}"}]]
        bad_rewards = format_reward(completions)
        good_rewards = format_reward(good)
        assert good_rewards[0] >= bad_rewards[0]
