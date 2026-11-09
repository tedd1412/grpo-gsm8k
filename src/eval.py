from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from src.data import (
    EVAL_SEED,
    EVAL_SIZE,
    SYSTEM_PROMPT,
    format_prompt,
    get_eval_set,
    normalize_number,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Parsing

# Strict tags: case-sensitive, exactly as the system prompt asks.
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FORMAT_RE = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.DOTALL,
)

# A number: optional minus (only if not glued to a preceding word/digit, so
# "10-3" gives 10 and 3), either comma-grouped ("1,200") or plain digits, with
# an optional decimal part. "$" and trailing "." / "%" are simply not captured.
NUMBER_RE = re.compile(
    r"(?:(?<!\w)-)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)"
)


def _last_number(text: str) -> str | None:
    matches = NUMBER_RE.findall(text)
    return normalize_number(matches[-1]) if matches else None


def extract_answer(completion: str) -> str | None:
    """Strict parser: the number inside the last <answer>...</answer> block.

    If the block holds extra words ("18 dollars"), the last number in it is used.
    Returns None when there is no closed answer tag or it contains no number.
    """
    blocks = ANSWER_TAG_RE.findall(completion)
    if not blocks:
        return None
    return _last_number(blocks[-1])


def extract_last_number(completion: str) -> str | None:
    """Lenient fallback: the last number anywhere in the completion."""
    return _last_number(completion)


def has_valid_format(completion: str) -> bool:
    """True iff the output is exactly one non-empty <think> block followed by
    one non-empty <answer> block, with nothing but whitespace around them."""
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if completion.count(tag) != 1:
            return False
    m = FORMAT_RE.match(completion)
    return bool(m and m["think"].strip() and m["answer"].strip())


def is_correct(pred: str | None, gold: str) -> bool:
    return pred is not None and pred == normalize_number(gold)


def score_completion(completion: str, gold: str) -> dict:
    strict_pred = extract_answer(completion)
    lenient_pred = extract_last_number(completion)
    return {
        "strict_pred": strict_pred,
        "lenient_pred": lenient_pred,
        "strict_correct": is_correct(strict_pred, gold),
        "lenient_correct": is_correct(lenient_pred, gold),
        "format_ok": has_valid_format(completion),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def mean(key):
        return sum(float(r[key]) for r in rows) / n

    return {
        "n": n,
        "strict_accuracy": mean("strict_correct"),
        "lenient_accuracy": mean("lenient_correct"),
        "format_compliance": mean("format_ok"),
        "avg_completion_tokens": mean("n_tokens") if "n_tokens" in rows[0] else None,
        "truncated_rate": mean("truncated") if "truncated" in rows[0] else None,
    }


# Generation

def generate_completions(
    questions: list[str],
    model_name: str = DEFAULT_MODEL,
    adapter: str | None = None,
    batch_size: int = 16,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Greedy-decode one completion per question. Returns text + token stats."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()

    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    eos_ids.add(tok.pad_token_id)

    prompts = [format_prompt(tok, q) for q in questions]
    # Batch similar-length prompts together (less padding), then restore order.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results: list[dict | None] = [None] * len(prompts)

    for start in range(0, len(order), batch_size):
        ids = order[start : start + batch_size]
        enc = tok([prompts[i] for i in ids], return_tensors="pt", padding=True).to(
            model.device
        )
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy == temperature 0
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :].tolist()
        for i, seq in zip(ids, gen):
            n_tokens = next((j for j, t in enumerate(seq) if t in eos_ids), None)
            truncated = n_tokens is None
            if truncated:
                n_tokens = len(seq)
            results[i] = {
                "completion": tok.decode(seq[:n_tokens], skip_special_tokens=True),
                "n_tokens": n_tokens,
                "truncated": truncated,
            }
        print(f"  generated {min(start + batch_size, len(order))}/{len(order)}", flush=True)

    return results  # type: ignore[return-value]


# I/O helpers

def _summary_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_name(jsonl_path.stem + "_summary.json")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _print_summary(s: dict) -> None:
    print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=2))


# Subcommands

def cmd_run(args) -> None:
    ds = get_eval_set(n=args.n, seed=args.seed, full=args.full)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Evaluating {len(ds)} problems with {args.model}"
          + (f" + adapter {args.adapter}" if args.adapter else ""))

    t0 = time.time()
    gens = generate_completions(
        list(ds["question"]),
        model_name=args.model,
        adapter=args.adapter,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.time() - t0

    rows = []
    for ex, g in zip(ds, gens):
        rows.append({
            "idx": ex["idx"],
            "question": ex["question"],
            "gold": ex["gold"],
            **g,
            **score_completion(g["completion"], ex["gold"]),
        })

    out = Path(args.out)
    write_jsonl(out, rows)

    import torch
    import transformers

    summary = summarize(rows)
    summary["config"] = {
        "model": args.model,
        "adapter": args.adapter,
        "eval_set": "full_test" if args.full else f"test.shuffle({args.seed}).select({args.n})",
        "limit": args.limit,
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "system_prompt": SYSTEM_PROMPT,
        "seconds": round(elapsed, 1),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _summary_path(out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out} and {_summary_path(out)}")
    _print_summary(summary)


def cmd_rescore(args) -> None:
    path = Path(args.path)
    rows = [{**r, **score_completion(r["completion"], r["gold"])} for r in read_jsonl(path)]
    write_jsonl(path, rows)
    summary_file = _summary_path(path)
    old = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    summary = {**summarize(rows), "config": old.get("config")}
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_summary(summary)


def cmd_show(args) -> None:
    rows = read_jsonl(Path(args.path))
    filters = {
        "all": lambda r: True,
        "wrong": lambda r: not r["lenient_correct"],
        "strict_wrong": lambda r: not r["strict_correct"],
        "format_fail": lambda r: not r["format_ok"],
        "lenient_only": lambda r: r["lenient_correct"] and not r["strict_correct"],
    }
    picked = [r for r in rows if filters[args.only](r)]
    random.Random(args.seed).shuffle(picked)
    print(f"{len(picked)} rows match '{args.only}'; showing {min(args.k, len(picked))}\n")
    for r in picked[: args.k]:
        print("=" * 80)
        print(f"idx={r['idx']}  gold={r['gold']}  strict={r['strict_pred']}  "
              f"lenient={r['lenient_pred']}  format_ok={r['format_ok']}  "
              f"tokens={r.get('n_tokens')}  truncated={r.get('truncated')}")
        print("-" * 80)
        print("Q:", r["question"])
        print("-" * 80)
        print(r["completion"])
    print("=" * 80)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="generate and score")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--adapter", default=None, help="LoRA adapter path or Hub id")
    r.add_argument("--n", type=int, default=EVAL_SIZE)
    r.add_argument("--seed", type=int, default=EVAL_SEED)
    r.add_argument("--full", action="store_true", help="all 1,319 test problems")
    r.add_argument("--limit", type=int, default=None, help="first K problems only (smoke test)")
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--max-new-tokens", type=int, default=512)
    r.add_argument("--out", default="results/base_eval.jsonl")
    r.set_defaults(func=cmd_run)

    rs = sub.add_parser("rescore", help="re-parse saved completions with the current parser")
    rs.add_argument("path")
    rs.set_defaults(func=cmd_rescore)

    s = sub.add_parser("show", help="print saved completions")
    s.add_argument("path")
    s.add_argument("--k", type=int, default=20)
    s.add_argument("--only", default="all",
                   choices=["all", "wrong", "strict_wrong", "format_fail", "lenient_only"])
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
