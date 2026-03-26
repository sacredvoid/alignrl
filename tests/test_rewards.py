"""Tests for reward functions."""

from alignrl.rewards import (
    _answers_match,
    _normalize_numeric,
    extract_answer,
    format_reward,
    math_verify_reward,
)


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
        assert rewards[0] == 1.0

    def test_boxed_at_end_scores_full(self) -> None:
        completions = [[{"content": "Step 1.\n\\boxed{42}"}]]
        rewards = format_reward(completions)
        assert rewards[0] == 1.0

    def test_boxed_with_nested_braces_at_end(self) -> None:
        completions = [[{"content": "So \\boxed{\\frac{1}{2}}"}]]
        rewards = format_reward(completions)
        assert rewards[0] == 1.0

    def test_unmatched_boxed_scores_zero(self) -> None:
        completions = [[{"content": "\\boxed{42"}]]
        rewards = format_reward(completions)
        assert rewards[0] == 0.0

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

    def test_empty_completion(self) -> None:
        rewards = format_reward([[]])
        assert rewards == [0.0]


class TestNormalizeNumeric:
    def test_integer(self) -> None:
        assert _normalize_numeric("42") == "42"

    def test_float_equal_to_int(self) -> None:
        assert _normalize_numeric("3.0") == "3"

    def test_non_integer_float(self) -> None:
        assert _normalize_numeric("3.14") == "3.14"

    def test_non_numeric(self) -> None:
        assert _normalize_numeric("abc") is None

    def test_trailing_dot(self) -> None:
        assert _normalize_numeric("42.") == "42"

    def test_infinity_returns_none(self) -> None:
        assert _normalize_numeric("inf") is None
        assert _normalize_numeric("infinity") is None
        assert _normalize_numeric("-inf") is None

    def test_nan_returns_none(self) -> None:
        assert _normalize_numeric("nan") is None


class TestAnswersMatch:
    def test_exact_match(self) -> None:
        assert _answers_match("42", "42") is True

    def test_numeric_equivalence(self) -> None:
        assert _answers_match("3.0", "3") is True

    def test_non_matching_strings(self) -> None:
        assert _answers_match("hello", "world") is False

    def test_non_matching_numbers(self) -> None:
        assert _answers_match("3", "4") is False


class TestMathVerifyRewardEdgeCases:
    def test_empty_completion_list(self) -> None:
        completions = [[]]
        rewards = math_verify_reward(completions, solution=["42"])
        assert rewards == [0.0]
