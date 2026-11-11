# Phase 2: Reward functions

| Reward | Score | Checks |
|---|---|---|
| `correctness_reward` | 2.0 / 0 | Number in the last `<answer>` block equals gold (same parser as the strict eval) |
| `format_reward` | 0.5 / 0 | Exactly one non-empty `<think>` then one non-empty `<answer>`, nothing else |
| `integer_reward` | 0.5 / 0 | Last `<answer>` block is a bare number (`1,200`, `18.0`, `-7` pass; `$18`, `18 dollars` fail) |

Max total 3.0. Format rewards sum to 1.0, below correctness, so a formatted guess never pays as much as a right answer.

## Decisions

- **No tags, right number = 0.** Correctness uses the strict parser. Giving lenient credit would reward skipping the format and would make the training signal disagree with the headline strict metric.
- **Shared parser.** `rewards.py` imports `extract_answer`, `has_valid_format` and `NUMBER_RE` from `eval.py`. A parser fix changes both, so reward and eval can't drift apart.
- **`integer_reward` accepts decimals.** All GSM8K golds are integers, but `18.0` is still correct, so punishing it would fight the correctness reward.
- **Units in the tag** (`$1,200`) still get correctness (2.5 total). The numeric reward nudges towards the bare number the prompt asks for.

## Hacks each reward might invite

| Reward | Possible hack | How to spot it |
|---|---|---|
| `format_reward` | Near-empty `<think>` ("ok") with a guessed answer | Think length collapses; format at max while correctness is flat |
| `integer_reward` | Always output a bare number, even a wrong one | Same as above: it only pays 0.5, so it should plateau early |
| `correctness_reward` | Copy the largest/last number from the question | Compare against a "last number in question" baseline in Phase 4 |
| all | Shorter answers to dodge truncation (not bad by itself) | Track completion length and truncated rate |

## Sanity check against the eval

```bash
python -m src.rewards results/base_eval.jsonl
```

`correctness_matches_strict_acc` and `format_matches_compliance` must both be `true`.
Smoke run (8 problems): mean total 2.0, strict acc 0.5, format compliance 1.0, both checks true.

- [x] `src/rewards.py` + `tests/test_rewards.py` passing
- [x] Score the 500 base outputs: mean total 1.721 (correctness 0.94, format 0.43, numeric 0.351); strict acc 0.47, format compliance 0.86, both checks true
