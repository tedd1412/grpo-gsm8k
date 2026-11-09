"""Sanity checks for the answer parser.

A parser bug can look exactly like a model improvement, so every accuracy
number in this repo depends on these tests passing. Each case is a
hand-written completion with the strict / lenient answers we expect.
"""

import pytest

from src.data import extract_gold, normalize_number
from src.eval import (
    extract_answer,
    extract_last_number,
    has_valid_format,
    is_correct,
    score_completion,
    summarize,
)

# (completion, expected_strict, expected_lenient, expected_format_ok)
CASES = [
    # 1. Perfect output
    ("<think>\n2 + 2 = 4\n</think>\n<answer>\n4\n</answer>", "4", "4", True),
    # 2. No tags at all, comma-grouped number
    ("The answer is 1,200", None, "1200", False),
    # 3. Empty answer tag: strict must NOT fall back to the reasoning
    ("<think>3 * 4 = 12</think>\n<answer></answer>", None, "12", False),
    # 4. Dollar sign, commas and trailing zeros inside the tag
    ("<think>cost</think><answer>$1,200.00</answer>", "1200", "1200", True),
    # 5. "18.0" must equal "18"
    ("<think>9 * 2</think><answer>18.0</answer>", "18", "18", True),
    # 6. Units / words inside the tag
    ("<think>...</think><answer> 18 dollars </answer>", "18", "18", True),
    # 7. Negative number
    ("<think>5 - 12</think><answer>-7</answer>", "-7", "-7", True),
    # 8. Truncated before the answer tag closes
    ("<think>so the total is 42</think>\n<answer>42", None, "42", False),
    # 9. No number anywhere
    ("I am not sure how to solve this.", None, None, False),
    # 10. Sentence-final period
    ("<think>5 + 3 = 8.</think><answer>8.</answer>", "8", "8", True),
    # 11. Two answer blocks: take the last one, and the format is invalid
    ("<think>a</think><answer>5</answer><answer>6</answer>", "6", "6", False),
    # 12. Wrong tag case is not accepted by the strict parser
    ("<think>a</think><ANSWER>5</ANSWER>", None, "5", False),
    # 13. Decimal answer with trailing zero
    ("<think>half</think><answer>0.50</answer>", "0.5", "0.5", True),
    # 14. Percent sign
    ("<think>1/4 of 100</think><answer>25%</answer>", "25", "25", True),
    # 15. Equation inside the tag: last number wins, minus after "=" kept
    ("<think>x</think><answer>3 - 5 = -2</answer>", "-2", "-2", True),
    # 16. Subtraction without spaces is not read as a negative number
    ("10-3=7", None, "7", False),
    # 17. Answer before think: strict still parses, format is invalid
    ("<answer>5</answer><think>because</think>", "5", "5", False),
    # 18. Missing think block
    ("<answer>5</answer>", "5", "5", False),
    # 19. A list "1, 2, and 3" is not one comma-grouped number
    ("I counted 1, 2, and 3", None, "3", False),
    # 20. Large comma-grouped number
    ("<think>big</think><answer>1,234,567</answer>", "1234567", "1234567", True),
]


@pytest.mark.parametrize("completion,strict,lenient,fmt", CASES)
def test_parser_cases(completion, strict, lenient, fmt):
    assert extract_answer(completion) == strict
    assert extract_last_number(completion) == lenient
    assert has_valid_format(completion) is fmt


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1,000", "1000"),
        ("$5", "5"),
        ("18.0", "18"),
        ("18.", "18"),
        ("3.50", "3.5"),
        ("-0.0", "0"),
        (" 42 ", "42"),
        ("abc", None),
        ("", None),
        (None, None),
        ("inf", None),
    ],
)
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_extract_gold():
    assert extract_gold("Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72") == "72"
    assert extract_gold("...\n#### 1,000") == "1000"
    assert extract_gold("...\n#### -5") == "-5"
    with pytest.raises(ValueError):
        extract_gold("no marker here")


def test_is_correct():
    assert is_correct("18", "18")
    assert is_correct("18", "18.0")  # gold is normalised too
    assert not is_correct(None, "18")
    assert not is_correct("17", "18")


def test_score_and_summarize():
    rows = [
        {**score_completion("<think>a</think><answer>4</answer>", "4"), "n_tokens": 10},
        {**score_completion("The answer is 4", "4"), "n_tokens": 20},
        {**score_completion("<think>a</think><answer>5</answer>", "4"), "n_tokens": 30},
        {**score_completion("", "4"), "n_tokens": 0},
    ]
    s = summarize(rows)
    assert s["n"] == 4
    assert s["strict_accuracy"] == 0.25
    assert s["lenient_accuracy"] == 0.5
    assert s["format_compliance"] == 0.5
    assert s["avg_completion_tokens"] == 15
