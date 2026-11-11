from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval import ANSWER_TAG_RE, NUMBER_RE, extract_answer, has_valid_format, is_correct

CORRECT_SCORE = 2.0
FORMAT_SCORE = 0.5
NUMERIC_SCORE = 0.5


def _texts(completions) -> list[str]:
    """TRL passes chat completions as [[{"role", "content"}]], plain ones as str."""
    return [c if isinstance(c, str) else c[0]["content"] for c in completions]


def correctness_reward(completions, gold, **kwargs) -> list[float]:
    """2.0 when the number in <answer> tags equals gold.

    Strict on purpose: a right number without tags earns nothing here, so the
    training signal matches the strict eval metric.
    """
    return [
        CORRECT_SCORE if is_correct(extract_answer(t), g) else 0.0
        for t, g in zip(_texts(completions), gold)
    ]


def format_reward(completions, **kwargs) -> list[float]:
    """0.5 for exactly one non-empty <think> block followed by one <answer> block."""
    return [FORMAT_SCORE if has_valid_format(t) else 0.0 for t in _texts(completions)]


def _is_bare_number(text: str) -> bool:
    return NUMBER_RE.fullmatch(text.strip()) is not None


def integer_reward(completions, **kwargs) -> list[float]:
    """0.5 when the last <answer> block holds a bare number and nothing else.

    "1,200", "18.0" and "-7" pass; "$18", "18 dollars" and "x = 5" do not.
    """
    scores = []
    for t in _texts(completions):
        blocks = ANSWER_TAG_RE.findall(t)
        scores.append(NUMERIC_SCORE if blocks and _is_bare_number(blocks[-1]) else 0.0)
    return scores


# Order matters only for logging; TRL sums them with equal weights by default.
REWARD_FUNCS = [correctness_reward, format_reward, integer_reward]


def score_rewards(completion: str, gold: str) -> dict[str, float]:
    """All rewards for one completion, plus the total."""
    scores = {f.__name__: f([completion], gold=[gold])[0] for f in REWARD_FUNCS}
    scores["total"] = sum(scores.values())
    return scores


def score_file(path: Path) -> dict:
    """Mean of each reward over a saved eval jsonl, with consistency checks
    against the eval metrics stored in the same rows."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    per_row = [score_rewards(r["completion"], r["gold"]) for r in rows]
    n = len(rows)
    means = {k: sum(s[k] for s in per_row) / n for k in per_row[0]}
    strict_acc = sum(r["strict_correct"] for r in rows) / n
    format_rate = sum(r["format_ok"] for r in rows) / n
    return {
        "n": n,
        "mean_rewards": means,
        "strict_accuracy": strict_acc,
        "format_compliance": format_rate,
        "correctness_matches_strict_acc": abs(means["correctness_reward"] / CORRECT_SCORE - strict_acc) < 1e-9,
        "format_matches_compliance": abs(means["format_reward"] / FORMAT_SCORE - format_rate) < 1e-9,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Score saved eval outputs with the training rewards.")
    p.add_argument("path", help="eval jsonl, e.g. results/base_eval.jsonl")
    args = p.parse_args(argv)
    print(json.dumps(score_file(Path(args.path)), indent=2))


if __name__ == "__main__":
    main()
