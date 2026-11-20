# Training recipes research findings: multi-turn tool-use SFT → GRPO (k=4, k=8, KL to SFT) for Qwen3.5-4B

**Date of research / access for every external URL below: 2026-11-20.**
**Scope:** concrete, implementable recipes for the next stages of the robot-team Brain. This is a
*research/findings* document only. It changes no code and reports **no measured training result**.

## Citation convention used in this document

- Every claim that comes from outside this repository is tagged `[S#; 2026-11-20; external source, not
  reproduced here]`. `S#` resolves to the **Sources** table at the end, which carries the URL. The
  marker and the access date are therefore recorded for each claim.
- Claims taken from this repository are tagged `[internal: <path>]` — these are *not* external sources.
- Recommendations assembled from several sources, or reasoned from our own profile numbers, are tagged
  `[synthesis; not a reproduced result]`. They are proposals, not evidence.

---

## 0. Internal starting point (read first; not external)

- Benchmark matrix M1–M8 is fixed in `IMPLEMENTATION_PLAN.md` §10: M3 = 4B SFT, M4 = SFT+GRPO k=4,
  M5 = SFT+GRPO k=8; effects are M3−M1, M4/M5−M3. Held-out test seeds are `< 100000`, training seeds
  `>= 100000` `[internal: IMPLEMENTATION_PLAN.md §10; docs/AGENT_BUILD_SPEC.md §0.5]`.
- R5–R7 specify: curate verified successes + attributable failures + failure-then-replan; SFT with loss
  on assistant tokens only; GRPO where a rollout is a *full multi-turn episode*, a group is k samples of
  the **same seed**, advantage = reward − group mean normalised by group std, reward =
  success + rules + plan validity − eye cost − length, resuming while `YIELDED` penalised, format
  validity a **gate**, frozen merged SFT model as KL reference, log KL every step and stop when the
  profile budget is exceeded `[internal: docs/AGENT_RUNBOOK.md R5–R7]`.
- `train/sft_brain.py` and `train/grpo_brain.py` are both explicitly marked **UNTESTED**; the GRPO
  header adds that Qwen3.5-4B support in the installed TRL/vLLM must be verified first
  `[internal: train/sft_brain.py; train/grpo_brain.py]`. Current scripts only expose a handful of flags
  (`--model/--data/--out/--epochs/--smoke` for SFT; `--model/--data/--out/--num_generations/
  --rollout-reward/--smoke` for GRPO) — the hyperparameters proposed below do **not** exist yet.
- A notebook sketches `beta=0.04` as the KL coefficient to the frozen SFT reference, with `RUN_GRPO=False`
  and the explicit note that a group of 4 completions at 512 tokens does not fit the 6 GB 4050 and
  "nothing in this notebook reports a GRPO loss or a success delta" `[internal: notebooks/src/20_grpo_brain.py]`.
- Important gap: R7 says "stop when the budget in the profile is exceeded", but `rtp/profiles.py` has
  **no** `kl_budget`/`beta` field. A profile field must be added before R7 can be implemented as written
  `[internal: rtp/profiles.py; docs/AGENT_RUNBOOK.md R7]`.
- Measured hardware evidence so far is only: G-2 (100 steps, seq 256, 329.5 tok/s, peak **4.99 GB**)
  `[internal: reports/g2_fastpath.json]` and G-14 SFT smoke (4 steps, checkpoint pushed/resumed, peak
  **5.85 GB**) `[internal: IMPLEMENTATION_PLAN.md §12 G-14]`. Profile notes record 512 tokens fitting at
  5.42 GB but 768/1024 OOM, and a real LoRA step + optimizer fitting at 256, not 384
  `[internal: rtp/profiles.py, local_4050 notes]`.

---

## 1. Loss masking on assistant tokens only (images + text + JSON actions)

### 1.1 Accepted TRL mechanics

1. For **conversational** datasets, `SFTConfig(assistant_only_loss=True)` computes loss only on assistant
   messages and ignores user/system messages. It requires the chat template to contain
   `{% generation %}` / `{% endgeneration %}` markers; TRL auto-patches known families (including Qwen3)
   when the flag is set `[S1; 2026-11-20; external source, not reproduced here]`.
2. For **prompt–completion** datasets, loss is already completion-only by default
   (`completion_only_loss=True`); set it `False` to train on the full sequence. It can be combined with
   `assistant_only_loss=True` using a conversational prompt/completion pair
   `[S1; 2026-11-20; external source, not reproduced here]`.
3. For **pre-tokenised** data, TRL accepts `input_ids` plus `labels` where `-100` marks excluded tokens,
   or `assistant_masks` / `completion_mask` columns that are folded into labels (and then dropped).
   Padding is also `-100` by default `[S1; 2026-11-20; external source, not reproduced here]`. This is the
   same convention our `rtp/trainer.py` already uses (`labels = [-100]*len(prompt_ids) + completion_ids`)
   `[internal: rtp/trainer.py::tokenize_pairs]`.
4. **Vision data**: provide an `image` (single PIL) or `images` (list of PIL) column, and put content in
   messages as a list such as `[{"type": "image"}, {"type": "text", "text": "..."}]`. TRL processes
   vision data on the fly (`skip_prepare_dataset` is forced true), and recommends `max_length=None`
   because truncation can remove image tokens and break training. Mixing text-only and vision rows needs
   `transformers >= 4.57.0` `[S4; 2026-11-20; external source, not reproduced here]`. `assistant_only_loss`
   on a VLM requires `transformers >= 5.18.0` and a conversational *language modelling* dataset
   `[S1; 2026-11-20; external source, not reproduced here]`.
5. **Tool calling**: a dataset row is `{"messages": [...], "tools": [json_schema]}`. Assistant turns carry
   `tool_calls`; tool results are separate messages with `role: "tool"`. The `tools` JSON schema can be
   generated with `transformers.utils.get_json_schema`; with `datasets >= 4.7` store tool arguments with
   the `Json()` feature, otherwise store the schema as a JSON string
   `[S4; 2026-11-20; external source, not reproduced here]`. GRPO's tool loop additionally requires a
   *prefix-preserving* chat template (appending a tool message must not change earlier rendering); TRL
   auto-swaps a patched training template for known families such as Qwen3
   `[S2; 2026-11-20; external source, not reproduced here]`.

### 1.2 How to treat each token class in our episodes

Our JSON actions are plain assistant text, not necessarily native `tool_calls`, so we can use either
mechanism. The explicit-label route is the safest on our constrained trainer:

| Token class | Label | Why |
|---|---|---|
| system / user / observation text | `-100` | Not the Brain's behaviour to imitate `[S1; 2026-11-20; external source, not reproduced here]` |
| image placeholder tokens (`<image>` / vision pad ids inserted by the processor) | `-100` | Image pixels enter through the vision tower (`pixel_values`, `image_grid_thw`), not through the LM head; supervising placeholder ids would teach the model to emit image markers. TRL's own warning is that truncation must not remove image tokens, which implies they are inputs, not targets `[S1, S4; 2026-11-20; external source, not reproduced here]` |
| `role: "tool"` result text (eye/tool output, verifier text) | `-100` | An observation, not an action; the assistant is trained to *react* to it, not reproduce it `[S4; 2026-11-20; external source, not reproduced here]` |
| assistant reasoning/rationale (if `thinking` is trained per build spec) | train | Per WP-G "rationale field trained" `[internal: docs/AGENT_BUILD_SPEC.md §WP-G]` |
| assistant JSON action / plan text, tool-call argument JSON, EOS (`<|im_end|>`) | train | This is the supervised behaviour; align EOS with the chat template (for Qwen, `<|im_end|>`) `[S1; 2026-11-20; external source, not reproduced here]` |
| padding | `-100` | Standard ignore index `[S1; 2026-11-20; external source, not reproduced here]` |

**Truncation policy:** our `tokenize_pairs` already left-truncates the prompt and keeps the completion
`[internal: rtp/trainer.py::tokenize_pairs]`. For VLMs the external advice is not to truncate at all
(`max_length=None`), but a 6 GB card cannot honour that. Practical compromise `[synthesis; not a
reproduced result]`: **filter** episodes whose tokenised length (image placeholders included) exceeds
`profile.train_len` and log the filter rate in the dataset card; never truncate mid-image. This keeps the
"no image-token truncation" invariant while staying inside the measured 256-token training ceiling on
`local_4050` `[internal: rtp/profiles.py]`.

### 1.3 Concrete approach for this repo

- Implement `rtp/sft_format.py::tokenize_episode(messages, processor, *, max_len, ...)` that walks turns,
  applies the Qwen3.5 chat template with generation markers, and emits `input_ids`, `labels` (`-100` on
  everything except assistant spans), and the vision tensors. Reuse `rtp.loss.chunked_lm_loss` so logits
  are never materialised as `tokens × 248k` `[internal: rtp/trainer.py; IMPLEMENTATION_PLAN.md §11]`.
- If the installed TRL supports it and we move to TRL for SFT, use
  `SFTConfig(assistant_only_loss=True, completion_only_loss=True, max_length=None or filtered,
  eos_token="<|im_end|>")`; otherwise keep our custom trainer with explicit labels
  `[S1; 2026-11-20; external source, not reproduced here]`.
- Unit-test invariant: for every row, `labels[i] != -100` iff token *i* belongs to an assistant turn,
  and the count of trained tokens per episode is logged.

---

## 2. GRPO for agent / reasoning tasks

### 2.1 Group advantage

- GRPO samples `G` completions per prompt from the old policy and computes the advantage of completion
  *i* as `Â_{i,t} = (r_i − mean({r_j}_{j=1..G})) / std({r_j}_{j=1..G})`; this is the group-relative
  normalisation that gives the method its name `[S5, S7; 2026-11-20; external source, not reproduced here]`.
  TRL's default is `scale_rewards="group"` (same formula) `[S2; 2026-11-20; external source, not reproduced here]`.
- **Zero-variance groups produce no gradient.** Unsloth states the special case explicitly: with one
  generation, `(r−μ)/σ = 0/0` is undefined, so GRPO needs at least 2 generations per prompt
  `[S9; 2026-11-20; external source, not reproduced here]`. DAPO generalises this: as training proceeds,
  more prompts get accuracy 1 (all rollouts correct) or 0 (all wrong), so their group advantage is zero,
  the effective batch shrinks, and gradient variance grows; DAPO's *Dynamic Sampling* over-samples and
  filters groups whose accuracy is 0 or 1 so every kept group has a usable gradient
  `[S7 (full text); 2026-11-20; external source, not reproduced here]`. TRL exposes the monitoring metric
  `frac_reward_zero_std` for exactly this condition `[S2; 2026-11-20; external source, not reproduced here]`.
- **Std scaling is a choice, not a law.** Dr. GRPO shows std scaling introduces a question-level
  difficulty bias; TRL offers `scale_rewards=False` ("none") to disable it, and `scale_rewards="batch"`
  (local mean, global std) from the "Lite PPO" paper
  `[S2, S6; 2026-11-20; external source, not reproduced here]`. For our mixed-difficulty robot families,
  log `frac_reward_zero_std` and try `scale_rewards=False` as an ablation `[synthesis; not a reproduced result]`.

### 2.2 k = 4 vs k = 8: how to compare fairly

There is no authoritative paper that declares one canonical comparison. The fair protocol depends on the
claim, and **both normalisations should be reported** `[synthesis; not a reproduced result]`:

- **Claim A — "does more samples per seed improve update quality?"** Hold the prompt/seed set, reward
  functions, `beta`, `loss_type`, sampling temperature, and number of optimizer steps fixed; change only
  `k`. k=8 then consumes roughly 2× rollout tokens/compute. This is the direct analogue of our M4/M5 rows
  `[internal: IMPLEMENTATION_PLAN.md §10]`.
- **Claim B — "which is better per unit compute?"** Hold the total generated token/generation budget
  fixed. With k=8 the run sees roughly half as many distinct seeds at the same budget. Report success
  both per-prompt and per-rollout-token.
- In **both** regimes use exactly the same group seeds (R7: "a group is k samples of the same seed"),
  the same reward weights, and log `reward_std`, `frac_reward_zero_std`, `entropy`, `completions/mean_length`,
  `kl`, `clip_ratio/*`, and wall-clock/tokens. With k=4 the chance of an all-correct or all-wrong group
  is higher than with k=8, so `frac_reward_zero_std` is the key confound to report
  `[S2, S7, S9; 2026-11-20; external source, not reproduced here]`.
- Mechanically, `num_generations` defaults to 8 and the effective batch
  (`num_processes × per_device_batch_size × grad_accumulation_steps`) must be divisible by it;
  `generation_batch_size` and `steps_per_generation` control how many prompts are rolled out per
  optimisation step, and `num_iterations` (μ) controls repeated updates on one rollout batch
  `[S2; 2026-11-20; external source, not reproduced here]`.

### 2.3 KL / beta and the reference model

- TRL's default is `beta=0.0`, i.e. **no KL term**, motivated by Open-Reasoner-Zero, Dr. GRPO and DAPO
  finding the KL restriction unnecessary for long-CoT RL `[S2; 2026-11-20; external source, not reproduced here]`.
  DAPO explicitly drops the KL penalty because the reasoning distribution is expected to diverge
  `[S7 (full text); 2026-11-20; external source, not reproduced here]`.
- **Our plan deliberately keeps KL**, so `beta` must be set non-zero and the frozen merged SFT model used
  as reference `[internal: docs/AGENT_RUNBOOK.md R7; notebooks/src/20_grpo_brain.py]`. TRL logs the KL
  estimate using the Schulman approximator
  `KL ≈ π_ref/π_θ − log(π_ref/π_θ) − 1` as the `kl` metric when `beta != 0`
  `[S2; 2026-11-20; external source, not reproduced here]`.
- There is no universal beta. Gao et al. show the KL coefficient is the knob that trades proxy-reward
  gain against gold-reward degradation (reward overoptimization), which is precisely the failure we want
  the KL budget to catch `[S10; 2026-11-20; external source, not reproduced here]`. The repo's existing
  intended value is `beta=0.04` `[internal: notebooks/src/20_grpo_brain.py]`. Recommendation
  `[synthesis; not a reproduced result]`: start from 0.04, sweep `{0.0, 0.01, 0.04}` in smoke tests, and
  add `kl_budget` + `grpo_beta` to `rtp/profiles.py` so R7's stop rule is executable.
- Reference-model memory `[synthesis; not a reproduced result]`: with a PEFT policy, the cheapest exact
  reference is the same frozen base with adapters disabled (no second copy of the 4.66 B weights); if a
  separately *merged* SFT model is required, load it in 4-bit/NF4 or keep its forward pass offloaded.
  This must be measured, not assumed.

### 2.4 Common failure modes and the accepted mitigations

| Failure mode | What the sources say | Concrete knob for us |
|---|---|---|
| **Length blowup** | Dr. GRPO identifies an optimisation bias in GRPO that artificially increases response length, especially for incorrect outputs; Dr. GRPO removes length normalisation bias `[S6; 2026-11-20; external source, not reproduced here]`. DAPO shows sample-level loss under-penalises long low-quality outputs `[S7 (full text); 2026-11-20; external source, not reproduced here]` | `loss_type="dr_grpo"` (constant denom = `max_completion_length`) or `loss_type="dapo"` (token-level, TRL default); `mask_truncated_completions=True`; DAPO-style *soft* overlong punishment rather than a plain length subtraction `[S2, S7; 2026-11-20; external source, not reproduced here]` |
| **Entropy collapse** | DAPO reports naive GRPO entropy collapsing quickly, with near-identical samples in a group; *Clip-Higher* decouples `ε_low=0.2`, `ε_high=0.28` to let low-probability exploration tokens rise `[S7 (full text); 2026-11-20; external source, not reproduced here]` | `epsilon=0.2`, `epsilon_high=0.28`; monitor `entropy`; optional `entropy_coef` / adaptive entropy `[S2; 2026-11-20; external source, not reproduced here]` |
| **Zero-advantage groups** | See §2.1 | Dynamic sampling or seed filtering; log `frac_reward_zero_std` `[S7, S2; 2026-11-20; external source, not reproduced here]` |
| **Reward hacking / overoptimization** | Rule-based verifiable rewards are preferred to a learned reward model because the latter is hackable; Gao et al. quantify Goodhart-style overoptimization `[S7 (full text), S10; 2026-11-20; external source, not reproduced here]` | See §3 |
| **Train–inference mismatch (vLLM)** | vLLM and the training engine are mathematically equivalent but numerically different; TRL offers truncated/masked importance-sampling correction and logs `sampling/*` diagnostics `[S2; 2026-11-20; external source, not reproduced here]` | If using vLLM, keep `vllm_importance_sampling_correction=True` and watch `sampling/sampling_logp_difference/mean` `[S2; 2026-11-20; external source, not reproduced here]` |
| **Validation disconnect** | DAPO observes final training reward often correlates poorly with validation accuracy — a sign of overfitting `[S7 (full text); 2026-11-20; external source, not reproduced here]` | Evaluate on held-out seeds `<100000` only; never use them for grading during training `[internal: docs/AGENT_BUILD_SPEC.md §0.5]` |

### 2.5 Proposed commands (flags are proposals; current scripts do not accept them yet)

```bash
# k = 4 (fixed-prompt / Claim A regime)
python train/grpo_brain.py \
  --model runs/kaggle_t4x2/brain_sft/merged \
  --data runs/datagen/grpo_groups_k4.jsonl \
  --out runs/kaggle_t4x2/brain_grpo_k4 \
  --num-generations 4 --beta 0.04 --loss-type dr_grpo --scale-rewards false \
  --mask-truncated-completions --max-completion-length 512 \
  --learning-rate 1e-6 --epsilon 0.2 --epsilon-high 0.28 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --generation-batch-size 4 --num-iterations 1 \
  --vllm-gpu-memory-utilization 0.7 --vllm-enable-sleep-mode \
  --kl-budget <profile.kl_budget> --group-seeds seeds_train_ge100000

# k = 8: identical except --num-generations 8 --out .../brain_grpo_k8
# For a budget-matched (Claim B) pair, halve the number of distinct group seeds at k=8.
```
Rationale for the numeric settings: `1e-6` LR + 20-step warmup, `ε_low=0.2/ε_high=0.28`, and μ=1 are
the published DAPO settings `[S7 (full text); 2026-11-20; external source, not reproduced here]`;
`beta=0.04` is this repo's existing intent `[internal: notebooks/src/20_grpo_brain.py]`;
`num_generations=4/8` and divisibility/vLLM/sleep flags come from TRL
`[S2; 2026-11-20; external source, not reproduced here]`.

---

## 3. Preventing reward hacking in tool-use agents

1. **Prefer verifiable/rule-based rewards over a learned reward model.** DAPO replaces the reward model
   with final correctness computed by a rule (`+1` if equivalent, `−1` otherwise), explicitly because
   reward models suffer from reward hacking `[S7 (full text); 2026-11-20; external source, not reproduced here]`.
   Our symbolic `rtp/rewards.py` already follows this spirit `[internal: rtp/rewards.py]`.
2. **Make format a hard gate, not an additive 0.1 weight.** TRL shows the canonical *format reward* as a
   0/1 check that a completion matches `<think>…</think><answer>…</answer>`
   `[S2; 2026-11-20; external source, not reproduced here]`. Our current weights are
   `{format: 0.1, feasible: 0.2, rules: 0.6, length: 0.1}`, so a malformed but otherwise lucky plan can
   still collect 0.1 `[internal: rtp/rewards.py::DEFAULT_WEIGHTS]`. Recommendation `[synthesis; not a
   reproduced result]`: if `parse_plan` fails or a hard rule is violated, total reward = 0; only
   otherwise apply the weighted sum. Keep a *separate* logged format-valid metric.
3. **Process rewards / step-level supervision.** On MATH, process supervision (feedback on each
   intermediate step) significantly outperformed outcome supervision, and active learning improved it
   further `[S11; 2026-11-20; external source, not reproduced here]`. For our multi-turn episodes this
   means rewarding correct intermediate skill choices, correct eye-tool use, and correct replans after a
   failure — not just final success. This needs attributable per-step labels, which is exactly what R5's
   "attributed cause" curation provides `[internal: docs/AGENT_RUNBOOK.md R5]`.
4. **Verifier agreement and a held-out gold check.** Keep a "gold" evaluator that never contributes
   gradients — analogous to the fixed gold-standard reward model used to measure proxy overoptimization
   in Gao et al. `[S10; 2026-11-20; external source, not reproduced here]`. Concretely `[synthesis; not a
   reproduced result]`: require agreement between static `dry_run` feasibility, closed-loop symbolic
   `r_rollout`, and rule compliance before awarding the success component; report disagreement rate.
5. **Held-out seeds and manual audit.** Test seeds `<100000` never enter memory, prompts, SFT or GRPO
   `[internal: docs/AGENT_BUILD_SPEC.md §0.5]`; R7 requires printing the top-reward rollouts of each run
   for a manual reward-hacking check `[internal: docs/AGENT_RUNBOOK.md R7]`. DAPO's observation that
   training reward can rise while validation accuracy does not is the reason to treat audit + validation
   as the real signal `[S7 (full text); 2026-11-20; external source, not reproduced here]`.
6. **Penalty design caution.** R7 penalises eye cost and length and resuming while `YIELDED`
   `[internal: docs/AGENT_RUNBOOK.md R7]`. A naive per-token length subtraction fights the anti-length-bias
   fixes in §2.4; implement length control as DAPO-style soft overlong shaping plus a small *fixed* eye
   cost instead `[synthesis; not a reproduced result]`.

---

## 4. Practical constraints: 6 GB RTX 4050 and Kaggle T4 for Qwen3.5-4B LoRA SFT / GRPO

### 4.1 What others actually measured (external)

- **QLoRA** backpropagates through a frozen 4-bit NF4 base with double quantisation and paged optimizers,
  fine-tuning a 65B model on a single 48 GB GPU at 16-bit quality — the reason a 4-bit base is the only
  way a 4.66 B model fits on 6 GB `[S8; 2026-11-20; external source, not reproduced here]`.
- **Unsloth's T4 measurements for Qwen3-4B GRPO on a single T4** (vendor benchmarks, reproducible Colabs
  linked in the source):
  - `standby=True, vllm_gpu_util=0.95, num_gen=2, grad_acc=2` → **14.5 GiB**, 40 steps / 40 min;
  - `standby=True, util=0.9` → 13.8 GiB, 32 steps / 40 min;
  - `standby=False, util=0.9` **and** `util=0.8` → **OOM, even batch size 1**;
  - `standby=False, util=0.7` → ~15.1 GiB, 28 steps / 39 min;
  - `standby=True, util=0.7` → ~13 GiB (10–11 typical), 29 steps / 40 min, ~2 GiB saved
    `[S9; 2026-11-20; external source, not reproduced here]`.
  The same source reports Qwen2.5-14B on H100 80 GB at seq 32 768, 8 generations, grad-acc 4, and a
  **9 GiB peak-memory reduction** from sleep/standby (43 GB vs 52 GB), and claims vanilla TRL+LoRA only
  reached 1 024 context on that box `[S9; 2026-11-20; external source, not reproduced here]`. Treat the
  last figure as a vendor claim, not an independent measurement.
- **Why GRPO uses so much memory**: two sets of VRAM are live at once — inference engine (weights + KV
  cache) and training engine (weights + activations + gradients + optimizer states). Sharing the weight
  space and using vLLM sleep mode (`level=1` copies weights to CPU and deletes KV cache; `level=2`
  deletes both) frees memory for longer context `[S9; 2026-11-20; external source, not reproduced here]`.
- **TRL-side knobs**: activation checkpointing is on by default; `activation_offloading=True` moves
  activations to CPU; vLLM sleep mode via `vllm_enable_sleep_mode=True`; `vllm_gpu_memory_utilization`
  defaults to 0.3 for colocate; continuous batching is a lighter alternative for single-GPU,
  memory-constrained generation with `max_memory_percent` 0.3–0.4 (TRL default 0.5); and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation on long online-RL runs
  `[S3, S2; 2026-11-20; external source, not reproduced here]`. TRL also ships a HF Space
  (`trl-lib/recommend-vllm-memory`) to pick `vllm_gpu_memory_utilization`
  `[S2; 2026-11-20; external source, not reproduced here]`.

### 4.2 Translation to our profiles

| Constraint | `local_4050` (6 GB) | `kaggle_t4x2` (2×16 GB) |
|---|---|---|
| Precision / attn | QLoRA NF4 + bf16 compute, FA2 `[internal: rtp/profiles.py]` | fp16 + SDPA (T4 is sm75: no bf16/FA2), grad checkpointing `[internal: rtp/profiles.py]` |
| Sequence ceiling | forward 512, **train step only 256** `[internal: rtp/profiles.py]` | profile `max_len=8192`; measure actual GRPO ceiling before use `[internal: rtp/profiles.py]` |
| SFT | G-14 smoke passed at 5.85 GB; production is dev-only `[internal: IMPLEMENTATION_PLAN.md §12]` | headline/reportable profile `[internal: rtp/profiles.py]` |
| GRPO rollout + ref | **Do not attempt full GRPO here.** Notebook already notes k=4 × 512 tokens does not fit `[internal: notebooks/src/20_grpo_brain.py]`; 2 generations is the theoretical minimum `[S9; 2026-11-20; external source, not reproduced here]` | Viable with sleep/standby-style offload, `num_gen=4/8`, short `max_completion_length` (e.g. ≤512–1024), micro-batch 1, grad accum `[S9, S3; 2026-11-20; external source, not reproduced here]` |
| Ref model memory | keep adapters-disabled forward as reference `[synthesis; not a reproduced result]` | merged SFT reference in NF4, or adapters-disabled forward `[synthesis; not a reproduced result]` |
| Logits memory | use `rtp.loss.chunked_lm_loss`; never `tokens × 248k` logits `[internal: rtp/trainer.py; IMPLEMENTATION_PLAN.md §11]` | same `[synthesis; not a reproduced result]` |

Concrete operating recipe `[synthesis; not a reproduced result]`: do all wiring/CPU tests and the SFT
smoke on `local_4050`; do the **only** reportable GRPO k=4/k=8 runs on `kaggle_t4x2` with
`vllm_gpu_memory_utilization≈0.7` first, sleep mode on, and measure peak VRAM per step into
`reports/seq_ceiling_grpo_t4.json` before raising context length. Do not assume the external T4 numbers
transfer to our vision-augmented 4B + KL reference; re-measure.

---

## 5. Data curation

**Keep three classes** `[internal: docs/AGENT_RUNBOOK.md R5; docs/AGENT_BUILD_SPEC.md WP-G]`:
(a) verified successes; (b) verified failures **with an attributed cause**, as negatives or recovery
examples; (c) failure-then-successful-replan episodes (most valuable). Quarantine records whose rationale
contradicts the executed plan, and exclude rows whose `arm_model` is not in `skill_tables/manifest.json`.

**De-duplicate by plan pattern, not just text.** Exact and near-duplicate removal and repetitive-substring
filtering reduce verbatim memorisation by ~10×, need fewer training steps, and reduce train–test overlap
(which affected >4% of validation sets in the studied benchmarks)
`[S12; 2026-11-20; external source, not reproduced here]`. For us the natural key is a normalised hash
of the JSON action sequence plus scene family, with a near-duplicate similarity threshold on the plan
text `[synthesis; not a reproduced result]`.

**Cap per-model and per-family share** so no single teacher dominates the SFT distribution
`[internal: docs/AGENT_BUILD_SPEC.md WP-G]`; target ≥5 000 verified episodes over ≥3 models
`[internal: docs/AGENT_RUNBOOK.md R4]`.

**Prevent test-seed leakage.** Enforce seeds ≥100000 for train/datagen and <100000 for test, assert no
test seed appears in any SFT/GRPO row or RAG memory, and split by **scene** as well as seed so
near-duplicate scenes cannot straddle the split
`[internal: docs/AGENT_BUILD_SPEC.md §0.5; docs/AGENT_RUNBOOK.md R5]`; de-duplication is the external
evidence that such overlap is measurable and harmful `[S12; 2026-11-20; external source, not reproduced here]`.
Record the dataset hash and write `docs/DATASET_CARD.md` `[internal: docs/AGENT_RUNBOOK.md R5]`.

---

## Concrete plan for this repo

**New / extended files (exact names):**

| File | Action |
|---|---|
| `docs/findings/training_recipes.md` | This file (already created). |
| `rtp/sft_format.py` | **Create.** `tokenize_episode()` implementing §1.2 masks (assistant-only, `-100` on image placeholders/tool results), chat-template generation markers, EOS `<\|im_end\|>`, length filtering, and trained-token counts. |
| `rtp/trainer.py` | **Extend.** Episode collator that forwards vision tensors (`pixel_values`, `image_grid_thw`) and consumes `labels` from `sft_format`; keep `chunked_lm_loss` and exact mid-epoch resume. |
| `rtp/curate.py` | **Create.** §5 filters: fail/replan classes, plan-pattern dedup, per-model/per-family caps, seed+scene split, `<100000` leakage assertion, hash + `docs/DATASET_CARD.md`. |
| `rtp/profiles.py` | **Extend.** Add `kl_budget`, `grpo_beta`, `grpo_num_generations`, `grpo_max_completion_length` so R7's stop rule is executable. |
| `train/sft_brain.py` | **Extend.** Wire to `rtp.trainer` + `rtp.sft_format`; profile-derived `max_len`, `--assistant-only-loss`, run card `reports/runs/<run_id>.json`; print `--help`. |
| `train/grpo_brain.py` | **Extend.** Full multi-turn episode groups by seed; `num_generations` 4/8; advantage = reward−mean/std per §2.1; format gate per §3.2; `beta` to frozen SFT reference; log `kl` each step and stop on `kl_budget`; `loss_type`, `scale_rewards`, `epsilon_high`, `mask_truncated_completions`; dump top-reward rollouts; run card. |
| `ops/seq_ceiling_grpo.py` | **Create.** Measure peak VRAM/tokens-per-s for k=4 and k=8 on each profile → `reports/seq_ceiling_grpo_<profile>.json`. |
| `tests/test_sft_masks.py`, `tests/test_grpo_rewards.py`, `tests/test_curate.py` | **Create.** Assertions from §1.3, §2.1, §3.2, §5. |
| `docs/DATASET_CARD.md`, `reports/runs/<run_id>.json`, `reports/grpo_k4_smoke.json`, `reports/grpo_k8_smoke.json` | **Create** from real runs only. |
| `ops/eval_matrix.py` | **Extend** (R8) to consume run cards for M3/M4/M5. |

**Minimal smoke-test sequence (in order; nothing here is a result):**

1. `python train/sft_brain.py --help` and `python train/grpo_brain.py --help` — confirm the new flags
   exist and that Qwen3.5-4B model ID / TRL / vLLM imports resolve; print versions.
2. CPU fixture tests: `pytest tests/test_sft_masks.py tests/test_grpo_rewards.py tests/test_curate.py`.
   Assert labels `-100` on user/tool/image tokens, non-`-100` on assistant tokens, group advantage
   formula on toy rewards for both k=4 and k=8, format gate zeros bad plans, and no `<100000` seed.
3. SFT smoke on `local_4050`: `python train/sft_brain.py --model $HOME/models/Qwen3.5-4B --data data/sft/<hash>/train.jsonl --out runs/local_4050/sft_smoke --smoke` → finite loss, checkpoint,
   mid-epoch resume, peak VRAM logged (mirror G-14).
4. GRPO wiring smoke on CPU: `python train/grpo_brain.py --smoke --num-generations 4` then `... 8` → 3
   steps each; group math and format gate exercised; no real reward claims.
5. GRPO smoke on `kaggle_t4x2`: k=4 then k=8 with the **same** group seeds and reward config,
   `max_completion_length=256`, `vllm_gpu_memory_utilization=0.7`, sleep mode on, ~3–10 steps; log `kl`,
   `entropy`, `reward_std`, `frac_reward_zero_std`, `completions/mean_length`, peak VRAM, tokens/s.
6. Reward-hack audit: print top-reward rollouts for both runs; verify format gate, verifier agreement and
   held-out seed exclusion; record reviewer notes in the run card.
7. Only after all gates pass: full SFT, merged SFT reference, full k=4/k=8 runs, then M3/M4/M5 via
   `ops/eval_matrix.py`.

---

## Do not claim

- **No measured SFT or GRPO training result exists in this repo yet.** `train/sft_brain.py` and
  `train/grpo_brain.py` are marked UNTESTED; the notebook sets `RUN_GRPO=False` and states no GRPO loss or
  success delta is reported `[internal: train/sft_brain.py; train/grpo_brain.py; notebooks/src/20_grpo_brain.py]`.
- G-2 (fast-path kernel smoke) and G-14 (SFT *smoke*) are wiring/throughput evidence, **not** a trained
  model or a success-rate improvement `[internal: IMPLEMENTATION_PLAN.md §12; reports/g2_fastpath.json]`.
- No M3, M4 or M5 number has been measured; no SFT effect, no GRPO k=4 or k=8 effect, no KL number, no
  reward-hacking audit result, no format-valid/plan-pass interval from a trained model
  `[internal: IMPLEMENTATION_PLAN.md §10; docs/STATUS_HONEST.md]`.
- All numeric figures attributed to DAPO, Dr. GRPO, QLoRA, Unsloth, Gao et al., and the dedup paper are
  **external source measurements, not reproduced here** — in particular the Unsloth T4/H100 VRAM and
  context numbers and DAPO's AIME scores. Do not transfer them to our runs.
- Do not claim Qwen3.5-4B is supported by the installed TRL/vLLM until step 1 above verifies it
  `[internal: train/grpo_brain.py header]`.

---

## Sources

All URLs below were accessed **2026-11-20**. Every one is an **external source, not reproduced here**.

| ID | URL | What was used |
|---|---|---|
| S1 | https://huggingface.co/docs/trl/main/en/sft_trainer | `assistant_only_loss`, `completion_only_loss`, `max_length=None` for VLMs, template markers, pre-tokenised `labels`/`assistant_masks`, VLM/tf version requirements, adapter LR guidance |
| S2 | https://huggingface.co/docs/trl/main/en/grpo_trainer | GRPOConfig defaults (`num_generations=8`, `beta=0.0`, `scale_rewards`, `loss_type`, `epsilon_high`, `mask_truncated_completions`, `num_iterations`, vLLM settings), advantage formula, KL approximator, logged metrics, agent/tool training, entropy controls |
| S3 | https://huggingface.co/docs/trl/main/en/reducing_memory_usage | Activation checkpointing/offloading, vLLM sleep mode, `expandable_segments`, continuous batching defaults, packing/padding-free |
| S4 | https://huggingface.co/docs/trl/main/en/dataset_formats | Vision dataset schema, tool-calling schema, `Json()` feature, mixing text+vision |
| S5 | https://arxiv.org/abs/2402.03300 | DeepSeekMath — origin of GRPO and group-relative advantage |
| S6 | https://arxiv.org/abs/2503.20783 | Understanding R1-Zero-Like Training / Dr. GRPO — std-scaling difficulty bias, length bias, unbaised loss |
| S7 | https://arxiv.org/abs/2503.14476 ; full text https://arxiv.org/html/2503.14476v2 | DAPO — rule-based rewards, dynamic sampling, clip-higher (ε_low=0.2/ε_high=0.28), token-level loss, overlong shaping, training hyperparameters, entropy/length/reward monitoring |
| S8 | https://arxiv.org/abs/2305.14314 | QLoRA — NF4, double quantisation, paged optimizers, 65B on 48 GB |
| S9 | https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl.md | Measured Qwen3-4B GRPO memory on T4 and H100, KV-cache/training memory split, sleep/standby levels, ≥2 generations requirement (vendor benchmarks) |
| S10 | https://arxiv.org/abs/2210.10760 | Scaling laws for reward model overoptimization — KL coefficient controls gold-vs-proxy reward; gold evaluator as held-out check |
| S11 | https://arxiv.org/abs/2305.20050 | Let's Verify Step by Step — process supervision outperforms outcome supervision |
| S12 | https://arxiv.org/abs/2107.06499 | Deduplicating training data reduces memorisation and train–test overlap |

**External source count: 13 URLs / 12 distinct documents. Internal repository files were read for context
but are not counted as external sources.**
