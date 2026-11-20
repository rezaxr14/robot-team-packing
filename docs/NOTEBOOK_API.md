# Notebook API card (what a notebook may call)

Every notebook is generated from a builder script in `notebooks/src/<name>.py` using
`ops/nbbuild.py`, and the `.ipynb` is committed next to it. One notebook = one part of the paper.
No notebook may invent a function: if something below does not exist, the notebook does not call it.

Authoritative context: claim status `docs/STATUS_HONEST.md`; execution order `docs/AGENT_RUNBOOK.md`
(arms first, A1–A10 then R4–R9); multi-turn loop `docs/DATAGEN_KAGGLE.md`; benchmark matrix M1–M8
`IMPLEMENTATION_PLAN.md` §10. A notebook that needs an unbuilt stage prints `SKIPPED` (rule 1).

## Hard rules

1. **No fabricated results.** A cell that needs a GPU, a T4 session, Isaac Lab, an asset-based
   scene, a served frontier endpoint or a checkpoint that is not there must *check first* and print
   `SKIPPED: <reason>`; it must never print a made-up number or fill a pending bar with a plausible
   value. State what was measured (which gate/report) vs what is unverified. In particular:
   `RTP_SERVE_URL` unset means a `SKIPPED` for frontier CoT — notebook 60 uses **scripted teacher
   traces**, not frontier-model output; a missing results bundle, fixed test set or omni-model
   prediction means `SKIPPED`/`not run`, never a score.
2. **One profile knob.** `ACTIVE = "local_4050"` then `profile = get_profile(ACTIVE)`. Never re-type
   dtype, batch sizes, paths, or `max_len`. `ACTIVE` must be one of `rtp.profiles.PROFILE_NAMES`.
3. **Dependencies**: second cell is `nb.code(ops.nbbuild.install_cell([...]))` listing only what the
   notebook imports beyond the stdlib + `rtp`.
4. **Storage order** is local -> attached Kaggle dataset -> our private Kaggle dataset -> our private
   HF repo. Use `rtp.remotes.from_env()`; never hard-code a repo id (`store.hf_repo`), never print a
   token, never touch another repo.
5. **Paths**: only `profile.paths.*` and `assetcache.asset_dir(...)`. No `data/...` literals.
6. **Cells run top-to-bottom.** No cell may depend on a later one. Keep runtime small: gate long work
   behind `SMOKE` / `MAX_ROWS`, defaulting to a small but real number.
7. Do not modify `paper/`, `rtp/*.py`, `train/*.py`, or any test while writing notebooks.

## Building

```python
from ops.nbbuild import Notebook, install_cell
nb = Notebook("10_sft_brain")          # writes notebooks/10_sft_brain.ipynb
nb.md("# ...")
nb.code("ACTIVE = 'local_4050'")
nb.save()
```

Then check:

```bash
$V/bin/python -m pytest tests/test_notebooks.py -q
$V/bin/python ops/nbrun.py notebooks/<name>.ipynb --timeout 600
```

`jupyter nbconvert` has no working CLI in this venv (bad interpreter / missing entry point).
`ops/nbrun.py` uses `NotebookClient.execute()` and exits non-zero on the first cell error.

## Exact APIs

```python
from rtp.profiles import PROFILE_NAMES, get_profile, with_overrides, detect_caps, resolve_runtime
profile.name, profile.accel, profile.tuning, profile.dtype, profile.attn, profile.lora_r
profile.max_len, profile.micro_batch, profile.grad_accum, profile.effective_batch, profile.vram_gb
profile.paths.runs / .ckpt / .adapters / .logs / .reports / .data / .models / .hf_home
profile.ensure_dirs(); profile.config_hash(); profile.env_vars()
# runtime truth on this box, with reasons:
attn, dtype, notes = resolve_runtime(profile, detect_caps())

from rtp import assetcache as ac
spec = ac.model_spec("Qwen/Qwen3.5-4B", revision="851bf6e8", name="qwen35-4b")
ac.asset_dir(spec, profile)            # where it lives locally
ac.read_manifest(dir)                  # dict or None; has slug/kind/source/revision/files/total_bytes
ac.verify_manifest(dir, spec)          # list of problems ([] == ok)
ac.data_spec(profile, kind, content_hash)   # profile-scoped data
ac.data_spec_kind(kind, name, revision)     # profile-free data (publishing a dataset)

from rtp.remotes import from_env
store = from_env()                            # real clients, proxy configured
store = from_env(with_kaggle=False)           # HF only (Kaggle TLS is flaky here)
store.hf_repo                                 # "Reza-Nadimi/robot-team-packing"
store.resolve(spec, profile, attached="/kaggle/input")   # -> verified local dir
store.publish(root, spec, to_kaggle=True, to_hf=True)    # -> {"kaggle":..., "hf":...}
store.list_checkpoints(profile.name, "sft")   # [100, 150, 200]
store.latest_checkpoint(profile.name, "sft")  # 200 or None
store.download_checkpoint(profile.name, "sft", 200, dest)
store.push_checkpoint(root, profile.name, "sft", 201)
store.prune_checkpoints(profile.name, "sft", keep=2)

from rtp.trainer import Trainer, TrainState, StopTraining, tokenize_pairs, load_checkpoint
from rtp.checkpoints import CheckpointPolicy, fetch_latest
from rtp.brain_train import load_pairs, load_for_training, build_optimizer, run_sft, existing_targets
# run_sft(profile, model_dir=<resolved dir>, data=<jsonl|rows>, store=store, stage="sft",
#         epochs=1.0, lr=1e-4, every=50, keep=2, stop_after=None, loader=None, log=print)
#   -> SFTResult(state, out_dir, resumed_from)

from rtp.datagen import (sample_world, sample_spec, default_eyes, gen_brain_sft, gen_grpo_prompts,
                         gen_eye_sft, gen_eye_grpo_rows, gen_hindsight_sft, write_jsonl)
# gen_brain_sft(n, seed0, harness, noisy_eyes=False) -> [{prompt, completion, meta}]
# gen_grpo_prompts(n, seed0, harness, noisy_eyes=True) -> [{prompt, scene_json, spec_json, seed}]
# gen_eye_sft(n, seed0) -> [{image, target, meta}]
# gen_hindsight_sft(n, seed0, harness, noisy_eyes=True) -> (rows, stats)

from rtp.harness import Harness          # Harness("harness_files") -> .retrieve(q, k, types), .docs, .reload()
from rtp.brain import ScriptedBrain, LLMBrain, build_prompt, BrainContext
from rtp.toolbrain import ToolBrain
from rtp.loop import LoopConfig, run_episode, attribute, retrieve_docs
from rtp.perception import Eye, EyeConfig, oracle_report, merge_scenes, default_blind
from rtp.world import World
from rtp.rules import TaskSpec
from rtp.rewards import trl_reward_funcs, combined, r_format, r_feasible, r_rules, r_rollout
from rtp.eye_rewards import trl_eye_reward_funcs, parse_scene_completion, r_eye_accuracy, r_eye_credit
from rtp.experiments import (ExpConfig, run_config, write_records, read_records, summarize,
                             paired_compare, default_grid, verify_grid, record_from_result)
from rtp.tables import table_main, table_ablation, table_attribution
from rtp.figures import fig_success, fig_gap, fig_credit, fig_tokens_vs_success
from rtp.attribution_eval import episode_story, sample_for_labeling, write_label_csv, agreement
from rtp.hindsight import relabel, ReplayBrain
from rtp.vlm_eye import VLMEye, eye_messages  # legacy VLM-Eye prototype; Eyes are detector + OpenCV now (D-05) — do not use in new notebooks
```

## Additional modules a notebook may call

* `rtp.vectordb` — hybrid retrieval: the harness's BM25 ranking fused with a persistent **Qdrant**
  vector ranking by reciprocal-rank fusion (embedded local mode by default). `HashedEmbedder` is a
  labelled deterministic stand-in, not a semantic-quality claim (D-45).
* `rtp.frontier_cot` — the frontier-CoT harness exists, but **`RTP_SERVE_URL` is unset**: no frontier
  model has run. Only scripted teacher traces exist; a notebook that needs frontier CoT must print
  `SKIPPED` (D-53).
* `rtp.control.impedance` — the L1 impedance law exercised by the **simulation-only** G-6 cross-check
  (`reports/g6_impedance.json`: 0.27 % step error, push/return measured). Not a hardware controller
  and not a safety sign-off; hardware/L0/L1 sign-off is Claude's (D-22, D-58).
* `rtp.control.compliance` — the compliance supervisor (push → yield → bounded return → resume,
  plan-clock gate, leash, clamps); G-6b PASS on a **SIM-ABSTRACT** point mass only
  (`reports/g6b_compliance.json`), not the Newton articulation or hardware (D-66).
* `rtp.kaggle_chat`, `rtp.datagen_run`, `rtp.datagen_ingest`, `rtp.datagen_memory`,
  `ops.datagen_iterate` — the multi-turn datagen loop, built and fixture-tested on SIM-POSE but
  **not run on Kaggle** (`docs/DATAGEN_KAGGLE.md`). A notebook needing a real hosted run prints
  `SKIPPED`; bring-up on cheap models may run under `sim_pose_v0` and is excluded from SFT (D-62).
* `ops/fetch_simready.py` — 4,239-package catalogue and a texture-free 14-package subset exist;
  an FR3 Duo + SimReady scene was **assembled and measured** (`reports/scene_manifest.json`), not
  stepped/rendered (G-13 PARTIAL). A notebook needing a stepped asset-based scene prints `SKIPPED`.

## Training stack that is already verified here

* `transformers 5.23.0`, `peft 0.21.3`, `trl 1.18.0`, `datasets 5.2.0`, `accelerate 1.15.0`,
  `bitsandbytes 0.50.2`, `torch 2.15.0`. Qwen3.5-4B module names are in `reports/qwen35_modules.json`.
* G-2 measured 329.5 tok/s at seq 256 on the 4050 (`reports/g2_fastpath.json`). The 4050 has **two
  measured budgets**: `max_len=512` for a forward/backward pass (D-38) and `train_max_len=256` for
  SFT with adapters and 8-bit optimizer state (D-47); use `profile.train_len` for training.
* G-14 PASS (`reports/sft_smoke.json`): 4 real QLoRA steps on the 4B model with a mid-epoch resume
  from the private HF checkpoint (loss 0.876 → 0.236). It is a smoke, **not a reportable trained
  model**.
* Isaac Lab 3.0 + Newton (no Isaac Sim, no PhysX) is installed at `~/rtp_work/IsaacLab`; the venv is
  `$ISAAC_PY` from `ops/env.sh`. G-3 evidence: `reports/g3_sim.json` (4050) **and**
  `reports/g3_sim_t4.json` (Kaggle T4, D-51). Both are falling-cube rgb/depth gates on **one device
  at a time** (D-50); no asset-based manipulation scene exists.

## What is *not* true yet (say so, do not paper over it)

* No stepped/rendered Isaac Lab scene with two FR3 arms + PTZ eyes exists. An FR3 Duo + SimReady
  scene was **assembled and measured** (`reports/scene_manifest.json`, 83 bodies), but not stepped;
  `ops/g3_sim.py` is only a cube-fall rgb/depth gate. Paper simulation episodes currently come from
  `rtp/simworld.py` + `rtp/fakesim.py` (symbolic/pose), and must be labelled as such (arms first,
  D-62: certified Newton arm skills come before SFT data).
* The Kaggle T4 half of G-3 **is** now recorded (`reports/g3_sim_t4.json`, D-51), but it steps one
  of the two T4s at a time (D-50); no two-GPU training or simulation number exists.
* GRPO has not been run on this box; TRL 1.18 + Qwen3.5 support must be verified at run time, and the
  notebook must print the resolved TRL config instead of promising results.
* "Comparison with other omni models" can only be a *harness for* the comparison unless another model
  is actually run; label unrun rows `not run`. There is no fixed test set or model score.
* Claim-by-claim truth (see `docs/STATUS_HONEST.md`): a 4,239-package SimReady catalogue and a
  14-package subset exist, but G-13 is only PARTIAL (9/14 USD imports, no drop/grasp/texture tests)
  and the assembled scene is not stepped. Asset-environment training is a **SIM-POSE smoke only**
  (`reports/asset_train.json`: 45 rows, 2 steps, loss 1.081 → 0.742), not a reportable model.
  Frontier CoT has two paths: `ops/cot_trace.py` recorded 0 frontier traces (`RTP_SERVE_URL` unset;
  scripted only), while a separate Kaggle hosted task produced 8 real gemini-3.8-flash traces
  (`reports/cot_frontier.json`) that are **not replayed/scored**. No M1–M8 matrix run, no GRPO,
  no hardware/L0/L1 impedance sign-off (G-6/G-6b are simulation and SIM-ABSTRACT only). All pending
  work must be `SKIPPED`/`not run`, never filled in.
