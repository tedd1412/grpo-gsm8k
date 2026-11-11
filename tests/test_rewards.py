import json

import pytest

from src.rewards import (
    CORRECT_SCORE,
    FORMAT_SCORE,
    NUMERIC_SCORE,
    correctness_reward,
    format_reward,
    integer_reward,
    score_file,
    score_rewards,
)

GOOD = "<think>\n600 * 2 = 1200\n</think>\n<answer>\n1200\n</answer>"


def test_correct_well_formatted_gets_full_score():
    assert score_rewards(GOOD, "1200") == {
        "correctness_reward": 2.0,
        "format_reward": 0.5,
        "integer_reward": 0.5,
        "total": 3.0,
    }


def test_right_number_without_tags_gets_nothing():
    # Deliberate: correctness uses the strict parser, so no tags means no credit.
    assert score_rewards("600 * 2 = 1200. The answer is 1200.", "1200")["total"] == 0.0


def test_correct_tags_wrong_number_gets_format_only():
    s = score_rewards("<think>600 + 2</think><answer>602</answer>", "1200")
    assert s["correctness_reward"] == 0.0
    assert s["total"] == 1.0


@pytest.mark.parametrize(
    "completion",
    [
        "<think>a</think><answer>1200</answer><answer>1200</answer>",
        "<think>a</think><answer></answer>",
        "<think>a</think><answer>   </answer>",
        "<think></think><answer>1200</answer>",
        "<answer>1200</answer>",
        "<answer>1200</answer><think>a</think>",
        "<think>a</think>\n<answer>1200",
    ],
)
def test_malformed_gets_no_format_credit(completion):
    assert format_reward([completion]) == [0.0]


@pytest.mark.parametrize("answer", ["1,200", "1200", "1200.0", "1200.00", " 1200 "])
def test_number_variants_count_as_same_answer(answer):
    completion = f"<think>a</think><answer>{answer}</answer>"
    assert correctness_reward([completion], gold=["1200"]) == [2.0]
    assert integer_reward([completion]) == [0.5]


@pytest.mark.parametrize(
    "content,expected",
    [
        ("42", 0.5),
        ("-7", 0.5),
        ("0.5", 0.5),
        ("1,234,567", 0.5),
        ("$42", 0.0),
        ("42 dollars", 0.0),
        ("42%", 0.0),
        ("x = 42", 0.0),
        ("", 0.0),
        ("forty-two", 0.0),
    ],
)
def test_integer_reward_requires_bare_number(content, expected):
    assert integer_reward([f"<think>a</think><answer>{content}</answer>"]) == [expected]


def test_units_in_tag_still_correct_but_lose_numeric_credit():
    s = score_rewards("<think>a</think><answer>$1,200</answer>", "1200")
    assert s["correctness_reward"] == 2.0
    assert s["integer_reward"] == 0.0
    assert s["total"] == 2.5


def test_accepts_trl_chat_format_and_batches():
    completions = [
        [{"role": "assistant", "content": GOOD}],
        [{"role": "assistant", "content": "no idea"}],
    ]
    gold = ["1200", "1200"]
    assert correctness_reward(completions, gold=gold) == [2.0, 0.0]
    assert format_reward(completions, gold=gold) == [0.5, 0.0]
    assert integer_reward(completions, gold=gold) == [0.5, 0.0]


def test_correctness_outweighs_all_format_rewards():
    # Formatting a wrong guess must never pay as much as a right answer.
    assert FORMAT_SCORE + NUMERIC_SCORE < CORRECT_SCORE


def test_score_file_tracks_eval_metrics(tmp_path):
    rows = [
        {"completion": GOOD, "gold": "1200", "strict_correct": True, "format_ok": True},
        {"completion": "The answer is 1200", "gold": "1200", "strict_correct": False, "format_ok": False},
        {"completion": "<think>a</think><answer>5</answer>", "gold": "1200", "strict_correct": False, "format_ok": True},
    ]
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = score_file(path)
    assert out["correctness_matches_strict_acc"]
    assert out["format_matches_compliance"]
    assert out["mean_rewards"]["total"] == pytest.approx((3.0 + 0.0 + 1.0) / 3)
