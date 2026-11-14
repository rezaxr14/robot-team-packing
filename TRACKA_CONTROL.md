# Track A: what upstream did, what is feasible for us, and how to make the comparison fair

## Upstream recipe (read from the VIKI-R repo)
- **VIKI-R** = SFT on chain-of-thought plans, then GRPO. **VIKI-R-zero** = GRPO from the base model (the two L2 scripts
  differ only in `actor_rollout_ref.model.path`).
- **SFT** (`configs/viki-{1,2,3}-3b.yaml`, LLaMA-Factory): Qwen2.5-VL-3B-Instruct, full fine-tuning, vision tower and
  projector frozen, lr 1e-5, 2 epochs, cosine, warmup 0.1, batch 1 x grad-accum 2, DeepSpeed ZeRO-3. The data is a small
  CoT set registered in LLaMA-Factory's `dataset_info.json` (the 7B GRPO script points at `viki_plan_cot_495/checkpoint-31`).
  Only 3B SFT configs are in the repo.
- **GRPO** (`train/{3B,7B}GRPO/VIKI-L*/`, verl): train batch 256, max prompt 4096, max response 2048, lr 1e-6, PPO
  mini-batch 128, KL loss 0.01 (low_var_kl), rollouts per prompt n=5, vLLM rollout with tensor parallel 2, 8 GPUs, 15 epochs.
- **Reward (L2)**: format 0.1 + correctness 0.9, where correct = feasible AND not longer than the reference plan.
- **Eval scripts**: first 900 (L2) / 1000 (L1) rows only; L2 draws a random start position per asset per call.

## Why we do not retrain that on one 48 GB GPU
8-GPU full fine-tuning and verl GRPO at batch 256 x 5 rollouts is out of reach for a single card. So Table 1 is built
from three things, cheapest first:

1. **Released checkpoints through our runner** (`scripts/s0_trackA_eval.py`): upstream-style, position-robust and
   full-split numbers. No training.
2. **Eye->Brain decomposition** (`scripts/s0_trackA_pipeline_eval.py`): the same text-only Brain with oracle,
   no-Eye and real-Eye descriptions. `oracle - none` = value of perfect perception; `oracle - eye` = perception loss.
   Use any Brain (base, SFT, or ours). This is the Track A result that speaks to the architecture, and it needs no
   VIKI-specific RL.
3. **Matched-base control (only if compute allows)**: Qwen2.5-VL-3B-Instruct as the SINGLE base for both arms
   - control: upstream recipe scaled down (LoRA SFT on the CoT set, then GRPO with the upstream reward);
   - ours: same base, same data splits, same number of GRPO samples, with our additions.
   Keep everything else fixed and report the compute (samples, rollouts, GPU-hours) next to each row. Do not compare our
   Qwen3.5-4B model to released Qwen2.5-VL checkpoints and attribute the difference to the method.

## Reporting checklist (put in the table caption or appendix)
split file used and which split you treat as in-domain vs out-of-domain; full split vs first-N; seeded, exhaustive or
single-draw scoring; base model; number of rollouts and steps; whether the checkpoint is released or retrained; format
rate; number of items that failed to run (errors are reported, never dropped).
