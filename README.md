# GRPO on GSM8K: does the reward going up mean the model got better?

Fine-tuning **Qwen2.5-1.5B-Instruct** with **GRPO** (LoRA) on GSM8K maths word problems, with a held-out
before/after evaluation, reward-hacking checks, and an honest account of what did and didn't work.

> Status: **Phase 1 — baseline eval** in progress.

## Results

| Model | Strict acc. | Lenient acc. | Format compliance | Avg tokens |
|---|---|---|---|---|
| Base (Qwen2.5-1.5B-Instruct) | — | — | — | — |
| + GRPO LoRA | — | — | — | — |

Fixed eval set: 500 problems from the GSM8K test split (`shuffle(seed=42)`), never used for training.
*Strict* = number inside `<answer>` tags; *lenient* = last number in the output.

## Repo layout

```
src/
  data.py      load GSM8K, gold extraction, fixed eval set, system prompt
  eval.py      answer parser + accuracy evaluation (run / rescore / show)
  rewards.py   reward functions: correctness 2.0, format 0.5, numeric 0.5
  train.py     GRPO training (Phase 3)
tests/         parser and reward tests
results/       per-problem eval outputs (jsonl), summaries, plots
notebooks/     Colab notebooks that only call src/
notes/         GRPO theory notes and the phase logs
```

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q                                   # parser + reward tests
python -m src.eval run --out results/base_eval.jsonl  # needs a GPU (T4 is fine)
python -m src.eval show results/base_eval.jsonl --only wrong --k 20
```

Or open `notebooks/01_baseline_eval.ipynb` in Colab.
