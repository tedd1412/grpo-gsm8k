

from __future__ import annotations

from decimal import Decimal, InvalidOperation

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"

# Fixed held-out eval subset: test split, shuffled with this seed, first N.
EVAL_SEED = 42
EVAL_SIZE = 500

SYSTEM_PROMPT = (
    "You are a careful maths tutor. Solve the problem step by step.\n"
    "Respond in exactly this format:\n"
    "<think>\n"
    "your step-by-step reasoning\n"
    "</think>\n"
    "<answer>\n"
    "the final numerical answer only, with no units\n"
    "</answer>"
)


# Number normalisation (shared by gold extraction, eval parser and rewards)

def normalize_number(text: str | None) -> str | None:
    """Canonicalise a numeric string so that equal numbers compare equal.

    "1,200" -> "1200", "$18.00" -> "18", "0.50" -> "0.5", "-0" -> "0".
    Returns None if the text is not a single number.
    """
    if text is None:
        return None
    s = text.strip().replace(",", "").replace("$", "").strip()
    s = s.rstrip(".")  # "18." at the end of a sentence
    if not s:
        return None
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    if not d.is_finite():
        return None
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def extract_gold(answer: str) -> str:
    """Return the normalised gold number after '####' in a GSM8K answer field."""
    if "####" not in answer:
        raise ValueError(f"No '####' marker in gold answer: {answer[-80:]!r}")
    raw = answer.split("####")[-1]
    gold = normalize_number(raw)
    if gold is None:
        raise ValueError(f"Gold answer is not a number: {raw!r}")
    return gold


# Dataset loading

def load_gsm8k(split: str):
    """Load a GSM8K split and add `idx` (row index in the original split) and `gold`."""
    from datasets import load_dataset  # lazy: keeps tests free of heavy deps

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    return ds.map(
        lambda ex, i: {"idx": i, "gold": extract_gold(ex["answer"])},
        with_indices=True,
    )


def get_eval_set(n: int = EVAL_SIZE, seed: int = EVAL_SEED, full: bool = False):
    """The fixed held-out eval set. `full=True` returns all 1,319 test problems."""
    ds = load_gsm8k("test")
    if full:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def get_train_set():
    """Training problems. Only ever the train split, so the eval set is never seen."""
    return load_gsm8k("train")


# Prompt formatting

def build_messages(question: str) -> list[dict[str, str]]:
    """Chat messages for one problem. Identical for base and trained models."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def format_prompt(tokenizer, question: str) -> str:
    """Render the chat template to a string, ending where the assistant starts."""
    return tokenizer.apply_chat_template(
        build_messages(question), tokenize=False, add_generation_prompt=True
    )
