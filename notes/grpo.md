# GRPO notes

Group Relative Policy Optimization (Shao et al., 2024, *DeepSeekMath*). PPO without a value model.

## The loop

For each prompt `q`:

1. Sample a **group** of `G` completions `o_1..o_G` from the current policy (e.g. G = 8, temperature ~0.7–1.0).
2. Score each with a reward function `r_i` (here: correctness + format, programmatic, no reward model).
3. **Advantage = how much better than the group**:
   `A_i = (r_i − mean(r)) / (std(r) + ε)`
   Every token of completion `i` gets the same advantage.
4. Update the policy with a PPO-style clipped objective:
   `L = −mean_t[ min(ρ_t·A_i, clip(ρ_t, 1−ε, 1+ε)·A_i) ] + β·KL(π_θ ‖ π_ref)`
   with `ρ_t = π_θ(o_t | q, o_<t) / π_old(o_t | q, o_<t)`.

## Why it fits this project

- **No critic**: the group mean is the baseline. Saves a whole model's worth of memory — matters on a T4.
- **Verifiable rewards**: GSM8K has a single numeric gold answer, so the reward is a parser, not a learned model.
- The model already solves many problems; GRPO sharpens what it can sometimes do rather than teaching new skills.

## Things that will bite

- **Zero-variance groups**: if all G completions get the same reward (all right or all wrong), `A = 0` → no gradient
  from that prompt. Too-easy and too-hard prompts are wasted compute. Log the fraction of such groups.
- **Reward hacking**: format rewards are easy to max out without getting better at maths (e.g. empty `<think>`,
  answer tag copying a number from the question). Always check held-out *accuracy*, not training reward.
- **Length effects**: per-sequence loss normalisation can bias length (Dr. GRPO, Liu et al., 2025). Track completion
  length during training.
- **KL coefficient β**: recent work often uses β = 0 or very small; with LoRA the drift is limited anyway. Record what is used.
- **Parser = reward**: the same parser (`src/eval.py`) must back both the reward and the eval, or improvements may be artefacts.

## Open questions to answer in this project

- How much of the gain is format compliance vs. genuine accuracy (strict vs lenient)?
- Does accuracy on the held-out 500 rise along with training reward, or plateau?
- What failure types remain after training?
