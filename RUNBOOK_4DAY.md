# 4-day GPU runbook

Read this once fully before the GPU clock starts. Every command below is a stage in `ops/plan.py`, so it is logged
(`reports/stage_log.jsonl`), resumable (finished stages are skipped), and bundled for paste-back (`ops/bundle.py`).

## 0. What is and is not known (be honest with yourself about the risk)
| Piece | Status |
|---|---|
| Symbolic pipeline, scorers, tools, verification, credit, hindsight, ablation grids | built and unit-tested (CPU) |
| Track A runner, decomposition, `run_experiments` with an LLM endpoint, Eye rejection sampling, inventory | run as real processes against a fake OpenAI server and synthetic VIKI-style parquet (`ops/fake_server.py`); never against real data or a real model |
| Isaac Lab scene, scripted IK skills, bring-up scripts, demo and Eye-render collection | **written from the Isaac Lab 2.3 docs, never run**. Expect fixes. The bring-up scripts localize failures to one layer. |
| Torch arm policy, distillation, receding-horizon rollout, TRL SFT/GRPO scripts, LoRA merge | **run on CPU in the sandbox** (transformers 5.22, TRL 1.17, PEFT 0.21) with a tiny random model: the wiring works. Real-model quality is untested. |
| LLaMA-Factory Eye SFT, VLM Eye serving, Qwen3.5 support in your library versions | **never run** (gate G0). The Eye falls back to Qwen2.5-VL if needed. |
| Install commands for Isaac Sim 5.1 / Isaac Lab 2.3.1 / vLLM / TRL | from the docs, not executed. Versions are variables in the scripts. |

Deliberate change from the original plan: **arm skills are scripted IK controllers, not RL experts**. It is the fastest
route to a working Track B in four days; the same scripts produce the demonstrations that the blind arm policy is
distilled from. RL expert training is dropped unless time remains.

## 1. The GPU question (check first)
"RTX PRO 6000" is the **Blackwell** card with **96 GB**. A 48 GB card is the **RTX 6000 Ada** or the **RTX PRO 5000
Blackwell**. `python3 ops/preflight.py` detects which one you have and prints the plan. Consequences:
- **Blackwell (sm_120)**: the upstream VIKI-R pins (torch 2.6.0, vllm 0.8.4, flash-attn 2.7.4) will not run. This repo
  never uses them: every env here is built on CUDA 12.8 wheels. Driver must be recent (R570+; unverified threshold).
- **Ada (sm_89)**: everything also works.
- Not usable: A100/H100 class GPUs (no RT cores, Isaac Sim cannot render).
Isaac Lab **2.3** (Isaac Sim 5.x, Python 3.11, torch 2.7 cu128) is the line used. Isaac Lab 3.0 (Isaac Sim 6, Python 3.12,
CUDA 13, quaternions as xyzw) exists but is not used; `rtp/simbridge.py` already supports xyzw if you switch.

## 2. Before the clock starts (no GPU hours needed beyond installing)
    tar xzf robot_team.tar.gz && cd robot_team
    python3 ops/preflight.py                 # paste back reports/preflight.json
    cp ops/config.example.sh ops/config.sh   # edit later, after inventory
    python3 ops/plan.py lint                 # static check of all 60+ stages (variables, produced/consumed paths)
    python3 ops/plan.py run setup            # uv, three envs (train, serve, isaac) + LLaMA-Factory env, models, data
Expect the Isaac Sim pip install and its first launch to be slow and to download assets from NVIDIA servers (internet
required). If `env_isaac` fails, paste the last lines of `reports/logs/env_isaac.log`; do not improvise versions.
Then: `python3 ops/plan.py run smoke` and paste back `python3 ops/bundle.py`.

### Gate G0, model support (after `smoke_model`)
`reports/model_check_Qwen3.5-4B.json` must show `"ok": true`. If it fails, set in `ops/config.sh`:
`export BASE_MODEL="$FALLBACK_MODEL"` (Qwen2.5-VL-3B-Instruct, the family VIKI-R itself uses) and rerun `smoke_model`
and `smoke_vllm`. Do not debug Qwen3.5 support for more than a short attempt; the fallback keeps the comparison to
VIKI-R cleaner anyway.

Stages that need `ops/config.sh` values (Track A paths, checkpoints) refuse to start until those variables are set, and
say which ones, so a missing path cannot burn GPU time.

## 3. Memory plan (what may run together)
| VRAM | Isaac + vLLM 4B together | + training together | vLLM utilization for a 4B model |
|---|---|---|---|
| 48 GB (Ada or PRO 5000) | yes (vLLM 0.30, Isaac headless, few cameras) | **no, run training alone** | ~0.35 to 0.38 |
| 96 GB (PRO 6000 Blackwell) | yes | yes for LoRA 4B | ~0.19 |
`ops/with_server.sh` starts vLLM, waits until it answers, runs your command and always stops it. Never leave a server
running while a training stage starts on a 48 GB card.

## 4. The plan, in priority order with gates (allocations, not predictions)
Nobody can tell you how long these stages take on your machine. `stage_log.jsonl` records seconds per stage; **after
Day 1, paste the bundle back and the remaining plan gets re-budgeted from real numbers.** If time is cut, cut from the
bottom of this list, never from the top.

| Priority | Deliverable | Needs |
|---|---|---|
| P0 | Track A: released checkpoints through our runner (upstream-style and position-robust) + oracle/none decomposition | VIKI data, one checkpoint per level |
| P0 | Symbolic grids, verification ablation, credit table | CPU only |
| P1 | Isaac bring-up through episodes with the scripted Brain (Track B baseline), with and without the weigh-before-planning probe | Isaac works |
| P1 | Brain SFT then GRPO, evaluated in the symbolic loop and in Isaac | Gate G0, G1 |
| P2 | Arm distillation (flow vs MLP head), arm regression suite on Isaac | Isaac demos |
| P2 | Eye SFT and Eye credit-vs-accuracy comparison | Isaac renders, LLaMA-Factory |

### Day 1: foundations and the cheap, certain results
1. `python3 ops/plan.py run smoke`.
2. **Track A first** (it needs no Isaac):
   - `python3 ops/plan.py run trackA --only inventory`, then open `reports/inventory.txt`, decide which parquet is in-domain and
     OOD, fill `ops/config.sh` (`L1_ID L1_OOD L2_ID L2_OOD`), download checkpoints:
     `ops/13_models.sh` printed the collection; use `ops/download_model.sh <repo_id> <name>`, set `VIKI_CKPT_L1/L2`.
   - `python3 ops/plan.py run trackA` (serves each checkpoint, evaluates, shuts down; then the Eye-less decomposition).
   - Paste back `reports/trackA/*_summary.json` (they are in the bundle).
3. CPU in parallel: `python3 ops/plan.py run eval --only sym_grid`, `sym_verify`, `sym_credit`.
4. **Isaac bring-up** (`python3 ops/plan.py run isaac`), each stage is a gate:
   - **G1 scene** (`isaac_01_scene`): all PASS. FAIL here = spawner kwargs/asset path. Paste the output.
   - **G1b reach** (`02`): reach rate >= 90%. FAIL = arm placement/IK; paste the per-target lines.
   - **G1c grasp-place** (`03`): all six classes pick and place. FAIL = TCP_OFFSET / finger stiffness / approach heights.
   - **G2 force window** (`04`): every class has a non-empty success window; the CSV tells us how to calibrate
     `required_force` against the simulator.
   - **G3 episodes** (`05`, scripted Brain, oracle Eyes): >= 1 success in 10, ideally most.
   Isaac iteration is where Day 1 will go if anything is off. That is expected. Paste outputs; fixes come back as patches.
5. `python3 ops/bundle.py` and paste the summary.
**Fallback if Isaac is not working by the end of Day 1:** stop investing Isaac hours. Track B stays symbolic (the paper's
claims are then scoped to the symbolic task and Track A), and Days 2 to 4 go to Brain/Eye training, Track A and the
symbolic evidence. Do not let Isaac eat the whole window.

### Day 2: data and first training
1. `python3 ops/plan.py run data` (symbolic datasets always; `arm_demos` and `eye_renders` only if G3 passed).
2. **Brain SFT** (GPU alone): `python3 ops/plan.py run train --only brain_sft`, then `brain_merge`.
3. Evaluate the SFT Brain before GRPO: serve the merged checkpoint and run
   `scripts/run_experiments.py --llm-endpoint ...` (stage `brain_symbolic`, edit the path to `brain_sft_merged`, or run it
   manually). Paste `tables.md`. **Gate G4:** SFT Brain should beat the no-verification, one-shot baseline in the
   symbolic grid; if it produces unparseable plans, inspect `plan_parse` counts first (prompt format), not the model.
4. If Isaac works: `arm_flow` and `arm_mlp` distillation; run the arm regression suite on the result (see section 6).

### Day 3: reinforcement learning and Eye
1. `python3 ops/plan.py run train --only brain_grpo` (reward functions are the symbolic plan-level rewards).
2. Eye: `python3 ops/plan.py run train --only eye_lf_config`, then `eye_sft`, then `eye_merge` (uses the Qwen2.5-VL
   fallback by default; see `ops/make_lf_config.py` for Qwen3.5 templates). Then the credit-vs-accuracy comparison. **Recommended
   path: rejection sampling** (`eye_rft_sample_accuracy`, `eye_rft_sample_credit`, then `eye_rft_train_*` and
   `eye_rft_merge_*`): it samples K=8 reports per image from the served Eye, keeps the best by each reward and fine-tunes
   on those, so it does not depend on TRL's image-conditioned GRPO support for your VLM. The sampling step also writes
   `<out>.stats.json` with the accuracy-vs-credit **selection disagreement rate** (a paper number in itself). Evaluate the
   three Eyes (`isaac_vlm_eye_episodes`, `isaac_vlm_eye_rft_accuracy`, `isaac_vlm_eye_rft_credit`) on the same seeds.
   The GRPO route (`eye_grpo_accuracy`, `eye_grpo_credit`) is optional; the credit reward replays a symbolic episode per
   sample, so check seconds per step before committing.
3. Evaluate the GRPO Brain: stages `brain_symbolic`, `brain_symbolic_tools`, `isaac_brain_episodes`.
Sequence lengths (measured on generated data): prompt about 1.1k tokens, plan about 0.7k tokens, so SFT `max_length=4096` is
safe and GRPO uses `max_completion_length=1536`. If you change the task size, rerun the length check in
`tests/test_robust_parsing.py::test_training_rows_fit_the_context_budget`; a plan cut off mid-JSON scores zero on every reward.
Before trusting a GRPO run, watch the trainer log for `frac_reward_zero_std`: if it stays near 1, every rollout in a group
gets the same reward, so there is nothing to learn from. On a tiny random model this is what happened in my CPU smoke
test; on a real model it means the SFT Brain is not yet producing parseable plans, so fix SFT first. Add `--rollout-reward`
to `train/grpo_brain.py` for the closed-loop reward once format rewards are saturated.
**Gate G5:** if GRPO makes plans worse than SFT (more parse failures, lower success), keep the SFT Brain and report GRPO
as a negative result; do not tune for hours.

### Day 4: final numbers and buffer
1. Re-run the evaluations you will report with more seeds, each condition on the same seeds (`--seed0`).
2. Hand-label attribution: open `reports/sym/attribution_to_label.csv` (about 40 rows) and run
   `python3 -c "from rtp.attribution_eval import agreement; print(agreement('reports/sym/attribution_to_label.csv'))"`.
3. `python3 ops/plan.py run bundle` (builds `reports/results.md` with the evidence ledger and `reports/figures/*.png`, then bundles); paste the summary. The ledger labels every table as TRACK A, ISAAC, LLM-SYMBOLIC or FIXTURE: only the first three can support a claim, and FIXTURE rows never can. Keep the last half-day free for reruns of anything that failed.

## 5. Daily paste-back protocol
End of each day: `python3 ops/bundle.py`. Paste the printed summary text (stages with return codes and seconds,
failed-stage tails, Track A metrics, tables). The `paste_back_*.tar.gz` also contains every small result file.
Never paste model weights, per-item outputs or images.

## 6. Arm regression suite and the learned arm on Isaac
- `isaac_08_regression_scripted` runs 15 skill cases on the scripted skills (teleports items to a zone, maps abstract
  case objects to real items, classifies outcomes from state). Run it after ANY change to `isaac/packing_sim.py`.
  Slip/crush cases test your force calibration: if they fail, use the 04 CSV to adjust `FINGER_STIFFNESS`/`TCP_OFFSET`.
- After distillation: `isaac_regression_learned` runs the same suite with the learned policy (`isaac/learned_sim.py`),
  and `isaac_learned_flow` / `isaac_learned_mlp` run episodes on the same seeds as `isaac_scripted_arm_same_seeds`,
  so the flow-vs-MLP-vs-scripted comparison is paired (`rtp.experiments.paired_compare` on the episode records).
- A learned skill has no learned stop signal: it runs for the median demonstration length per skill (recorded by `07`).
- Handover is implemented as a table handover (put down at the centre, other arm grasps); mid-air exchange is not scripted.

## 7. Troubleshooting (symptom -> likely cause)
| Symptom | Likely cause |
|---|---|
| "no kernel image is available for execution on the device" | a PyTorch wheel without your GPU architecture (Blackwell needs cu128); never install the upstream pins |
| `isaaclab.sh` complains about pip | venv created without `--seed` (the script uses `--seed`) |
| Isaac Sim hangs at first launch | first-run shader compilation and asset download; wait, and confirm outbound internet |
| vLLM out of memory at start | lower the utilization argument or `MAXLEN`; nothing else may hold GPU memory |
| Qwen3.5 template error about system messages | the chat template needs the system message first; our prompts put it first |
| Isaac: grasps slip or objects fly away | `TCP_OFFSET`, `FINGER_STIFFNESS`, item sizes vs 8 cm gripper opening; read the 04 force-window CSV |
| `report_from_states` says `out_of_reach` for everything | `LAYOUT` zones not matching the real arm bases; compare 01 output regions |
| Track A numbers far from the paper | check split mapping (id vs ood naming mismatch upstream), the 900/1000-row truncation, and the format rate |

## 8. What to send me back after Day 1
`python3 ops/bundle.py` output, `reports/preflight.json`, `reports/model_check_*.json`, `reports/inventory.txt`, and for
every FAILED stage the tail printed in the bundle. With real stage durations I will re-plan Days 2 to 4 and patch the
Isaac code against the actual errors.
