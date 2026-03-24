# Custom Reward Functions

GRPO training uses reward functions to score model completions. alignrl ships with two built-in rewards, but you can write your own for any task where answers are verifiable.

## The reward function signature

Every reward function must follow this signature:

```python
def my_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    ...
```

**Parameters:**

- `completions` - A batch of completions. Each completion is a list of message dicts (usually a single assistant message). To get the text: `completion[0]["content"]`.
- `**kwargs` - Additional fields from the dataset. For GSM8K, this includes `solution: list[str]`. For your own datasets, whatever columns you include will be passed here.

**Returns:** A list of floats, one per completion. Higher values mean better completions.

The function is called by TRL's `GRPOTrainer`. Each call receives a batch of completions (one per prompt in the batch, times `num_generations`).

---

## Built-in rewards

### `math_verify_reward`

Binary correctness check. Extracts the answer from the completion (looking for `\boxed{}`, "the answer is", or "= X" patterns), compares it to the expected solution.

- Returns `1.0` if the answer matches (with numeric normalization, so "3.0" matches "3")
- Returns `0.0` otherwise

```python
from alignrl.rewards import math_verify_reward

completions = [[{"content": r"The sum is \boxed{42}"}]]
rewards = math_verify_reward(completions, solution=["42"])
# [1.0]
```

### `format_reward`

Checks whether the model followed the expected output format (reasoning followed by `\boxed{answer}`).

- Returns `1.0` if `\boxed{}` appears at or near the end of the response
- Returns `0.5` if `\boxed{}` is present but there's significant text after it
- Returns `0.0` if no `\boxed{}` is found

```python
from alignrl.rewards import format_reward

completions = [[{"content": r"Step 1: ... Step 2: ... \boxed{42}"}]]
rewards = format_reward(completions)
# [1.0]
```

---

## Writing your own

### Example: length penalty

Penalize responses that are too short (likely skipping reasoning) or too long (rambling):

```python
def length_penalty_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Reward responses between 100-500 characters."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        length = len(content)
        if 100 <= length <= 500:
            rewards.append(1.0)
        elif 50 <= length <= 800:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards
```

### Example: keyword presence

Reward the model for showing its work with specific reasoning markers:

```python
import re

def reasoning_markers_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Reward step-by-step reasoning patterns."""
    markers = [r"step \d", r"first,", r"therefore", r"let me", r"we can"]
    rewards = []
    for completion in completions:
        content = completion[0]["content"].lower() if completion else ""
        matches = sum(1 for m in markers if re.search(m, content))
        rewards.append(min(matches / 3.0, 1.0))  # cap at 1.0
    return rewards
```

### Example: using the solution field

The `**kwargs` receives all dataset columns. For GSM8K, that includes `solution`. For your own dataset, add whatever columns you need:

```python
def partial_credit_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **kwargs,
) -> list[float]:
    """Give partial credit for answers that are close."""
    from alignrl.rewards import extract_answer

    rewards = []
    for completion, sol in zip(completions, solution, strict=False):
        content = completion[0]["content"] if completion else ""
        predicted = extract_answer(content)
        if predicted is None:
            rewards.append(0.0)
            continue
        try:
            pred_val = float(predicted)
            sol_val = float(sol)
            if pred_val == sol_val:
                rewards.append(1.0)
            elif abs(pred_val - sol_val) / max(abs(sol_val), 1) < 0.1:
                rewards.append(0.5)  # within 10%
            else:
                rewards.append(0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards
```

---

## Combining multiple rewards with weights

Pass multiple reward functions to `GRPORunner` and optionally set weights:

```python
from alignrl.grpo import GRPOConfig, GRPORunner
from alignrl.rewards import math_verify_reward

config = GRPOConfig(
    max_steps=50,
    num_generations=4,
    reward_weights=[0.7, 0.2, 0.1],  # must match number of reward functions
)

runner = GRPORunner(
    config=config,
    reward_funcs=[
        math_verify_reward,       # weight 0.7 - correctness matters most
        format_reward,            # weight 0.2 - format is secondary
        length_penalty_reward,    # weight 0.1 - slight length preference
    ],
)

result = runner.train()
```

If `reward_weights` is `None`, all rewards are weighted equally (each gets weight 1.0).

The weighted rewards are combined by TRL's `GRPOTrainer` internally. The final reward for each completion is:

```
total_reward = sum(weight_i * reward_i for all i)
```

---

## Tips for good reward design

### Keep rewards simple and verifiable

The best GRPO rewards are binary or near-binary. "Is the answer correct?" is a better signal than "how good is the reasoning on a scale of 1-10." Continuous rewards work, but they add noise to the training signal.

### Avoid reward hacking

If your reward function has loopholes, the model will find them. Common examples:

- A length reward without a correctness check teaches the model to output the right number of characters, regardless of content.
- A format reward that only checks for `\boxed{}` can be gamed by outputting `\boxed{random_guess}` immediately.

Always include a correctness reward alongside style/format rewards, and weight it highest.

### Start with fewer reward functions

Two or three reward functions are usually enough. More rewards make the signal noisier and harder to debug. Start with correctness + format, then add more only if you see specific failure modes.

### Debug with a dry run

Before training, test your reward function on a few examples:

```python
# Sanity check: does your reward function work?
test_completions = [
    [{"content": r"Let me solve this step by step.\n\nFirst, 2 + 2 = 4.\n\n\boxed{4}"}],
    [{"content": "4"}],
    [{"content": "I don't know"}],
]
test_solutions = ["4", "4", "4"]

print(math_verify_reward(test_completions, solution=test_solutions))
# Expected: [1.0, 1.0, 0.0]

print(format_reward(test_completions))
# Expected: [1.0, 0.0, 0.0]
```

### Scale rewards to [0, 1]

GRPO uses group-relative baselines, so the absolute scale of rewards doesn't matter as much as the relative ordering. But keeping rewards in [0, 1] makes `reward_weights` intuitive and prevents one reward from dominating by scale alone.
