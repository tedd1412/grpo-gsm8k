# Phase 1 — Baseline log

## Decisions (frozen once the baseline is recorded)

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Eval set | GSM8K `test`, `shuffle(seed=42).select(range(500))` (full 1,319 once at the end) |
| Prompt | `SYSTEM_PROMPT` in `src/data.py`, Qwen chat template, `add_generation_prompt=True` |
| Decoding | greedy (`do_sample=False`), `max_new_tokens=512` |
| Strict score | last number inside the last `<answer>…</answer>` (case-sensitive) |
| Lenient score | last number anywhere in the completion |
| Format compliance | exactly one non-empty `<think>` followed by one non-empty `<answer>`, nothing else |
| Normalisation | strip `$` and `,`, trailing `.`; `18.0` → `18`; `0.50` → `0.5` |

Note: greedy results can shift slightly with batch size (padding changes fp numerics). Keep `--batch-size` the same for base and trained runs.

## Checklist

- [x] `src/data.py`: load, `extract_gold`, fixed eval set, prompt
- [x] `src/eval.py`: strict + lenient parser, format check, metrics, jsonl output
- [x] Parser sanity tests (20 hand-written completions) passing
- [ ] Smoke test on Colab (8 problems)
- [ ] Baseline run on 500 → `results/base_eval.jsonl`
- [ ] Read 20 outputs and fill in the failure notes below

## Baseline numbers

| Metric | Base |
|---|---|
| Strict accuracy | |
| Lenient accuracy | |
| Format compliance | |
| Avg completion tokens | |
| Truncated (hit 512) | |

## Failure notes (from reading outputs)

| idx | Type (arithmetic / misread / format / truncated / parser) | Note |
|---|---|---|
| | | |
