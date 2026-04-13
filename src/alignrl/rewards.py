"""Reward functions for GRPO training with verifiable rewards."""

from __future__ import annotations

import math
import re

# Match "the answer is X" / "final answer: X" / "answer: X" variants,
# allowing commas inside numeric answers like "1,234" and an optional
# currency/latex prefix like "$" or "\$".
_RE_ANSWER = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is|answer)\s*[:=]?\s*"
    r"\*?\*?(\\?\$?-?[\w,./\\{}%]+)",
    re.IGNORECASE,
)
_RE_EQUALS = re.compile(r"=\s*([^\s,=]+)")


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


_RE_TEXT_WRAPPER = re.compile(r"\\text\{([^{}]*)\}")


def _unwrap_latex(s: str) -> str:
    """Strip common LaTeX wrappers like ``\\text{...}`` from a boxed answer."""
    prev = None
    current = s
    while prev != current:
        prev = current
        current = _RE_TEXT_WRAPPER.sub(r"\1", current)
    return current


def extract_answer(text: str) -> str | None:
    """Extract the final answer from model output.

    Supports ``\\boxed{...}`` (with nested braces and ``\\text{}`` wrappers),
    ``the answer is X`` / ``final answer: X``, and ``X = Y`` patterns.
    Returns the last match found (most likely the final answer).
    """
    boxed = _extract_boxed_contents(text)
    if boxed:
        return _unwrap_latex(boxed[-1]).strip()

    answer_match = _RE_ANSWER.findall(text)
    if answer_match:
        return answer_match[-1].strip().rstrip(".,;:")

    eq_match = _RE_EQUALS.findall(text)
    if eq_match:
        return eq_match[-1].strip().rstrip(".,;:")

    return None


def _normalize_numeric(s: str) -> str | None:
    """Try to parse as a number for comparison.

    Handles common wrappers seen in LLM math outputs:
    - commas as thousands separators (``1,234``)
    - leading currency symbols (``$42``, ``\\$42``)
    - trailing percent sign (``50%``)
    - trailing period (``42.``)
    """
    s = s.strip()
    # Strip common latex/currency/format wrappers
    if s.startswith("\\$"):
        s = s[2:]
    s = s.lstrip("$").rstrip("%").rstrip(".")
    # Remove thousands separators (commas between digits)
    s = s.replace(",", "")
    try:
        val = float(s)
        if not math.isfinite(val):
            return None
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return None


def _answers_match(predicted: str, expected: str) -> bool:
    """Check if two answers are equivalent.

    First tries exact string match (case-insensitive), then normalized
    numeric match so that ``3.0`` == ``3`` and ``1,234`` == ``1234``.
    """
    pred = predicted.strip()
    exp = expected.strip()
    if pred == exp:
        return True
    if pred.casefold() == exp.casefold():
        return True
    norm_pred = _normalize_numeric(pred)
    norm_exp = _normalize_numeric(exp)
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
