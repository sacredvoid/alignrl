"""Reward functions for GRPO training with verifiable rewards."""

from __future__ import annotations

import re


def _extract_boxed_contents(text: str) -> list[str]:
    """Extract contents of all \\boxed{...} groups, handling nested braces."""
    results = []
    prefix = "\\boxed{"
    start = 0
    while True:
        idx = text.find(prefix, start)
        if idx == -1:
            break
        # Walk forward from the opening brace, tracking depth
        depth = 1
        begin = idx + len(prefix)
        pos = begin
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            results.append(text[begin : pos - 1])
        start = pos
    return results


def extract_answer(text: str) -> str | None:
    """Extract the final answer from model output.

    Supports: \\boxed{...}, 'the answer is X', 'X = Y' patterns.
    Returns the last match found (most likely the final answer).
    """
    boxed = _extract_boxed_contents(text)
    if boxed:
        return boxed[-1].strip()

    answer_match = re.findall(r"(?:the answer is|answer:)\s*([^\s.,]+)", text, re.IGNORECASE)
    if answer_match:
        return answer_match[-1].strip()

    eq_match = re.findall(r"=\s*([^\s.,=]+)", text)
    if eq_match:
        return eq_match[-1].strip()

    return None


def _normalize_numeric(s: str) -> str | None:
    """Try to parse as a number for comparison."""
    s = s.strip().rstrip(".")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return None


def _answers_match(predicted: str, expected: str) -> bool:
    """Check if two answers are equivalent."""
    if predicted.strip() == expected.strip():
        return True
    norm_pred = _normalize_numeric(predicted)
    norm_exp = _normalize_numeric(expected)
    if norm_pred and norm_exp:
        return norm_pred == norm_exp
    return False


def math_verify_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **kwargs,
) -> list[float]:
    """Binary reward: 1.0 if extracted answer matches solution, 0.0 otherwise.

    Compatible with TRL GRPOTrainer reward_funcs signature.
    """
    rewards: list[float] = []
    for completion, sol in zip(completions, solution, strict=False):
        content = completion[0]["content"] if completion else ""
        predicted = extract_answer(content)
        if predicted and _answers_match(predicted, sol):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Reward for following the expected output format (reasoning + \\boxed{answer}).

    0.0 = no boxed answer
    0.5 = boxed answer exists but not at end
    1.0 = boxed answer at or near end of response
    """
    rewards: list[float] = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        last_boxed = content.rfind("\\boxed{")
        if last_boxed == -1:
            rewards.append(0.0)
            continue
        # Walk forward from opening brace to find matching close brace
        start = last_boxed + len("\\boxed{")
        depth = 1
        pos = start
        while pos < len(content) and depth > 0:
            if content[pos] == "{":
                depth += 1
            elif content[pos] == "}":
                depth -= 1
            pos += 1
        if depth != 0:
            rewards.append(0.0)
            continue
        tail = content[pos:].strip()
        if len(tail) < 20:
            rewards.append(1.0)
        else:
            rewards.append(0.5)
    return rewards
