# Decisions

Two lists. **A. User decisions** are authoritative and override anything else, including the
Claude plan when it lands. **B. Defaults we chose while building** are ours; each is one config or
one module, flip any.

Labels: **[V]** verified from a real file/source (path or URL given), **[U]** unverified, must not be
quoted as fact, **[!]** blocked on the user.

## A. User decisions (authoritative)

| id | decision | status |
|---|---|---|
| D-01 | GitHub repo is `robot-team-packing`, remote `https://github.com/rezaxr14/robot-team-packing.git`. Commit and push in meaningful increments; no micro-commit spam. First push is the architecture/decision docs. | done (remote synced, `main` pushed) |
| D-02 | `rtp` = **Robot Team Protocol**: the typed message bus + harness. | done |
| D-03 | Physics is **Isaac Lab + Newton**. Not optional, not "a limitation". | [V] `source/isaaclab_newton` (`isaaclab_newton 17.2.3`, `requires-python >=3.12`) imports `warp` + `newton`; `NewtonManager.backend_name == "newton"` |
| D-04 | **Two Eyes, two Arms.** Eyes are movable ceiling-mounted sensors, **skewed, not overhead**, placed on circular paths around the workspace, occludable by columns/objects, able to rotate, zoom, reposition, orbit, and run CV kernels. | design in `docs/ARCHITECTURE_V2.md` §2–3 |
| D-05 | Eyes are **not LLMs**. YOLO-class detectors + classical CV for items/obstacles/edges/features. They send features/images to the Brain and obey its commands. They must be **trained** to detect obstacles and objects. | design §3 |
| D-06 | **One Brain**: Qwen3.5-4B (vision-capable) commands both Arms and both Eyes. | [V] local checkpoint `~/models/Qwen3.5-4B`, `Qwen3_5ForConditionalGeneration` |
| D-07 | Train **all relevant blocks**: full attention, gated DeltaNet, vision merger, and the up/down projections — not just language layers. Verify which modules exist and are trainable. | [V] module names in `docs/ARCHITECTURE_V2.md` §7; implemented as adapter coverage (D-15) |
| D-08 | **Download and inspect the real Qwen config/model files before training.** Never guess the architecture. | [V] config + index + tokenizer downloaded to `~/models/Qwen3.5-4B`; structure table in §7 |
| D-09 | Hardware profiles: **RTX 4050 (QLoRA)**, **RTX 4090 (LoRA bf16 + FlashAttention)**, **dual T4 (LoRA fp16 + SDPA)**, plus future **NVIDIA Pro 6000** and **TPU v5e-8 / v6e-1 / v5e-1**. FlashAttention and bf16 only where genuinely supported, with fallbacks. | [V] FA2 README: Ampere/Ada/Hopper mainline, bf16 requires Ampere+, head_dim ≤256; Turing needs a separate fork; no bf16 on T4 |
| D-10 | **One config dict/selector in the notebooks.** Selecting a profile configures the whole run; no digging through code. | `rtp/profiles.py` (to write); notebook sets `ACTIVE` only |
| D-11 | Profiles are **fully isolated**: own derived folders, adapters, checkpoints, optimizer state, logs, resume cursors. They never touch each other and resume independently. | layout in §8 |
| D-12 | **Reusable installation/setup script** for every dependency, including **LangGraph**. | `ops/install.sh --profile <p>` (to write) |
| D-13 | Use a **reputable, maintained vector database alongside BM25** (hybrid). Chroma/LanceDB/FAISS-class suggestions are rejected. | **Qdrant** [V] Rust, Apache-2.0, local mode `QdrantClient(path=...)`, named + sparse vectors, payload filters |
| D-14 | **Kaggle Frontier AI API** generates task-grounded synthetic CoT traces for the VLM. Verify the real API/models/limits/workflow; do not assume. Separate from hardware profiles; works with every profile. | **[U] blocked**: `kaggle.com/docs/frontier-ai` and `/docs/api` return HTTP 403 anonymously (2026-11-16); no auth until authorised |
| D-15 | Capture **episodes**: images, eye observations/features, generated traces, tool calls/results, eye commands, arm actions, success/failure. Feed success/failure RAG into the Kaggle benchmark/AI workflows, retrieval, recovery and hindsight learning. | schema in §10 (to implement) |
| D-16 | **SFT then GRPO**, with a reward strategy and **KL regularisation against the SFT adapter**. | plan in §11 |
| D-17 | Benchmark against relevant **omni models** with defined tasks, metrics and reproducible tests. | [U] baseline list pending model servers |
| D-18 | **Work locally first. Do not authenticate or launch cloud jobs until authorised.** | in force |
| D-19 | **Record every decision in the repo docs.** | this file |
| D-20 | **Do not fabricate** facts, capabilities, compatibility, API features or results. Verify from files/sources; label unverified. | this file's labelling scheme |
| D-21 | Research and **report Newton's real GPU/software requirements with sources**. | [V] §15: Turing cc 7.5+, driver R580+ for Warp ≥1.18 (else `warp-lang==1.17.0`), Python 3.11+, MuJoCo Warp primary backend |
| D-22 | **Do not research or redesign SAC, flow matching, impedance control, or robot safety** — Claude handles that research; implement related work only as directed. | in force; no SAC/flow work started |
| D-23 | Work **one step at a time**: implement, test, document, report. Never jump ahead or make silent decisions. | in force |
| D-24 | A Claude **implementation plan** will be placed in the repo root; use it as the step-by-step checklist, but user decisions override it. | **[!] not present yet** |

### Consequences recorded from A (not new decisions)

- `tuning: qlora|lora|full` is a profile field. "Train the full attention + DeltaNet + merger +
  up/down blocks" means **adapter coverage over those block types** (plus `full` where VRAM allows),
  not "unfreeze 4.66 B params on a 6 GB card".
- `mamba_ssm_dtype: float32` [V] means the DeltaNet recurrence stays fp32 under any autocast; the
  loader asserts it and fails loudly otherwise.
- Dual T4: Newton's floor is met (cc 7.5). Whether Isaac Lab 3.0 + Newton runs on a T4 session is
  settled by gate G-3, never assumed; D-25 leaves no second simulator stack to fall back to.

## B. Defaults chosen while building (ours; flip any)

1. **Eyes:** two Eyes with different stations in the main system (occlusion makes per-Eye credit
   non-trivial); one Eye is an ablation. `default_eyes()` in `datagen.py`.
2. **Eye output:** structured geometry (`OBS`/`SCENE`: class, coarse region, size, mass/fragility
   guess, occluded, confidence) plus optional image tiles. Metric poses come from a simulator
   depth/segmentation tool, not from the VLM.
3. **Arm head:** per-arm flow-matching head; MLP head is the ablation (`ArmPolicy(head=...)`).
   A joint two-arm head for handover is not implemented.
4. **Latent messages:** out of the main system; only the typed text layer exists.
5. **Task family:** one family (backpack packing) with amendments and injected faults for Track B;
   Track A uses VIKI-Bench unchanged.
6. **Base models:** Qwen3.5-4B for Brain and Eyes per D-06; a Qwen2.5-VL-based control is required for
   Track A. `transformers 5.23.0` in the active venv exposes `qwen3_5` [V].
7. **Heavy-bottom rule** ignores the explicitly "last" item (otherwise "heavy at bottom" and
   "lunchbox last" contradict each other for light lunchboxes).
8. **Failure attribution** (Eye/Brain/Arm) is rule-based and heuristic (`loop.attribute`); report it as
   such, and validate on a hand-labelled subset before using it as a result.
9. **Eye budget** default 6 slots/round, 24/episode, with tool costs in §3 — a knob, because
   "success vs eye calls" is a paper axis.
10. **Detector** default `ultralytics` YOLO11n-seg (AGPL-3.0, flagged); BSD fallback
    `torchvision` RetinaNet/Mask R-CNN behind `rtp/eyes/detector.py`. Licence call pending.
11. **Dense embedder** default `BAAI/bge-small-en-v1.5` (384-d, CPU) with a hash fallback for offline
    tests.
12. **Physics profiles** carry `physics: isaac_newton` for every profile (D-25); the simulator is
    Isaac Lab 3.0 + Newton and there is no mjsim/PhysX branch any more.


## C. Decisions of 2026-11-17 (from the user; authoritative)

| id | decision |
|---|---|
| D-25 | Simulation is **Isaac Lab 3.0 + Newton only**. No other simulator/renderer stack. Legacy 2.3-era bring-up files are purged in Phase 0. |
| D-26 | Human safety is a property of the robot's controller: Cartesian **impedance control** — a human can push an arm, it acts as a spring, then returns to its planned pose. Object fragility is not a design driver. |
| D-27 | Kaggle-hosted frontier models (Claude Opus/Sonnet/Haiku, Gemini family, others) are the teacher/judge pool. Access and terms are authorised by the user; model catalogue and calling method come from the user-supplied Kaggle skill and CLI auth. |
| D-28 | Build from the ground up; unverified items are closed by verification gates with pass criteria, not by open questions. |
| D-33 | Local-first sequencing: everything not needing Kaggle is built and run locally first. Teacher/judge data comes from the Kaggle-hosted frontier models and starts when the user supplies the skill and CLI auth; no substitute data source. |
| D-34 | Assets: SimReady USD assets for objects/furniture/kitchen; manufacturer models for robots (Franka URDF/USD/MJCF, MuJoCo Menagerie MJCF); own CAD only for fixtures that do not exist. |

## D. Decisions taken by the v2 plan (defaults; changeable by logged decision)

| id | decision |
|---|---|
| D-29 | Arms: 2 × Franka Research 3 + Franka Hand. |
| D-30 | Eyes: Eye-A ONVIF PTZ on ring-rail carriage with 12 calibrated stations; Eye-B machine-vision + servo head as fallback. |
| D-31 | Layers: L0 hardware safety / L1 1 kHz impedance with plan-clock gate, leash, clamps / L2 planner, skills, bounded learned deltas / L3 Brain. |
| D-32 | Brain adapters, profiles, isolation, Qdrant+BM25, LangGraph unchanged from the V2 architecture. |
| D-35 | **P0 legacy purge.** Deleted with the old stack: `ops/12_env_isaac.sh`, the `OMNI_KIT_ACCEPT_EULA` export, the RT-core preflight branch, `isaac/bringup/`, the VIKI/LLaMA-Factory launcher pieces (`ops/13_models.sh`, `ops/14_env_lf.sh`, `ops/make_lf_config.py`, `ops/list_viki_collection.py`, `ops/config.example.sh`, `tests/test_cli_integration.py`), and the script-level tests of the deleted Track-A CLIs. `ops/plan.py` is now a thin registry of stages that exist (`p0`, `bundle`), not a store of paths to deleted files. `rtp/trackA*.py` stay as an inert scoring library with its (VIKI-skipped) tests; nothing in the new pipeline calls them. |
| D-36 | **Measured test baseline.** `pytest tests -q` = **192 passed, 10 skipped, 40 s** in the training venv (Python 3.12.16, torch 2.15.0+cu132, transformers 5.23.0, flash-attn 2.8.3.post2) on this 4050 laptop; the 10 skips are all "VIKI-R clone not available" (inert Track-A scorer). The plan's 137/13/15 s figure was the CPU-only sandbox before the training stack existed; this row supersedes it. |
| D-37 | **Chunked LM loss.** The causal-LM loss is computed in token chunks over the LM head (`rtp/loss.py`), so the `[tokens x 248,320 x 4 B]` float32 logits of a long sequence are never materialised (8.1 GB at 8 k tokens, 16.3 GB at 16 k). Default 128 tokens/chunk ≈ 127 MB of logits, sized for the 6 GB 4050; SFT/GRPO may raise it on 24/96 GB cards. GRPO's own log-prob memory knobs are a P8 concern; the shared helper is what the training paths call. |
| D-38 | **`max_len` is measured, and the 4050's is 512.** `local_4050` carried `max_len=8192` from the profile table; `ops/seq_ceiling.py` measured the real ceiling with one forward+backward per length at batch 1 on the 6 GB card (5.64 GB usable): 128 -> 4.94 GB, 256 -> 5.10, 384 -> 5.26, **512 -> 5.42 (fits)**, 768 OOM, 1024 OOM. The wall is the hybrid stack's per-token activation state, not the adapter: at `lora_r=8` the same 512 still fits (5.20 GB) and 1024 still OOMs. `local_4050.max_len` is therefore **512** (`reports/seq_ceiling_local_4050*.json`, `tests/test_seq_ceiling.py` refuses a profile whose `max_len` exceeds a committed ceiling). Consequence: the 4050 is a smoke/dev box; SFT at 4 k tokens runs on `local_4090` or `kaggle_t4x2`. The other profiles' `max_len` values stay unverified until the same probe runs on that hardware. |
| D-39 | **G-3 PASS on the 4050, Newton-only.** Isaac Lab v3.0.0's own `uv.lock` (`uv sync --no-dev` into the isolated `isaac_newton` venv) installed isaaclab 28.0.0, isaaclab-newton 9.1.4, newton 1.6.1, warp-lang 1.17.0, mujoco-warp 3.12.0, torch 2.11.0+cu130; **no Isaac Sim and no PhysX** are installed. `ops/g3_sim.py` ran 30 steps with `physics=newton_mjwarp` + `NewtonWarpRendererCfg`: cube z 1.0 -> 0.6832, rgb (1,120,160,3) uint8, depth (1,120,160,1) float32 with 19,200 hits, 21 steps/s, all six checks pass (`reports/g3_sim.json` + PNGs). The first run's camera saw only ground (frames identical) — the gate's `frames_differ` check caught it; a straight-down camera fixed it. Warp JIT compiles and runs a kernel on cc 8.9. Kaggle T4 remains unverified; the installer no longer hand-picks Newton versions (that got 1.8.0/warp 1.19 and diverges from the tag's lock). |
| D-40 | **Storage order is local -> private Kaggle dataset -> private HF repo, and it is live.** `rtp/remotes.py` implements it over `rtp.assetcache` (attach + atomic install + manifest verify stay in one place). Live round-trips, all through the `http://localhost:10808` proxy Kaggle requires here: (a) publish a smoke asset -> private Kaggle dataset `rezanadimikj/rtp-data-data-hub-smoke-20261118` (`dataset-metadata.json` `isPrivate: true`) **and** `Reza-Nadimi/robot-team-packing` (HF `private: true`), then `fetch` back and read the bytes from the local cache; (b) HF-only publish, then Kaggle-disabled fetch, downloaded from HF and resolved; (c) checkpoint `local_4050/sft/step-000000007` pushed, `checkpoint-latest` read back 7. Exactly one HF repo was created; `vlm-dental-models` and `thermo-agent-checkpoints` were listed before/after and are untouched. Publishing refuses a directory whose manifest does not verify, and adopts an *existing* manifest's identity instead of rewriting `source` — the first 9.4 GB attempt failed on exactly that (`local` vs `Qwen/Qwen3.5-4B`) and the guard is now a test. |
| D-41 | **Training is resumable mid-epoch, and checkpoints ship to HF.** `rtp/trainer.py` keeps `step`, `epoch`, `micro_in_epoch`, sample/token counts, loss history, optimizer state and torch RNG state; batch order is a pure function of `(seed, epoch)`, so resume rebuilds the same permutation and skips the consumed micro-batches instead of replaying the epoch. `trainer.StopTraining` ends a session at a step boundary (that is how a notebook hands over to the next machine). `rtp/checkpoints.py` saves every `every` steps, pushes to `checkpoints/<profile>/<stage>/step-%09d`, and prunes to the newest `keep` (a 4B adapter+optimizer is ~0.4–1.4 GB, so unpruned pushes would fill the repo); a failed push logs and keeps the local checkpoint rather than killing the run. Measured: interrupted at step 3 of 8 and resumed -> step 8, epoch 2, losses match the uninterrupted trajectory (tests/test_trainer.py). The loss takes `labels=` so prompt tokens are masked, and the chunked path is unchanged for unmasked input (tests/test_loss.py). |
| D-42 | **Notebooks are generated, executed, and API-checked.** Hand-editing `.ipynb` JSON is how cells get duplicate ids and dead paths, so every notebook is produced by `notebooks/src/<name>.py` through `ops/nbbuild.py` (stable per-cell ids from name+position+source hash). `ops/nbrun.py` executes a notebook headlessly: the `jupyter nbconvert` CLI is **not** installed in this venv (only the library stubs; `jupyter nbconvert` fails with `No such file or directory`), and nbclient's per-cell API asserts `self.kc is not None` when driven cell-by-cell, so `NotebookClient.execute()` is the supported path. `tests/test_notebooks.py` now refuses: a notebook without a builder, a committed notebook that its builder does not regenerate byte-for-byte, a `from rtp.x import y` naming something that does not exist, a `TODO/FIXME/XXX` marker, and a notebook with no SKIPPED path for hardware it cannot assume. `notebooks/00_setup_cache.ipynb` executed end-to-end on the 4050: cache hit on the 4B model, manifest ok (10.08 GB), `sft` latest remote step 7, G-2 fast path PASS. |
| D-43 | **The 9.4 GB model is published *from Kaggle*, and that needs one manual secret.** `ops/kaggle/publish_model/` is a private kernel (`enable_internet`, no GPU) that downloads `Qwen/Qwen3.5-4B@851bf6e8` on Kaggle's disk and creates the private dataset `rtp-model-qwen35-4b-851bf6e8` — the exact slug `RemoteStore.resolve` looks for. `ops/kaggle_model.py push` succeeded (kernel version 1), and the run then failed at credential lookup: a Kaggle kernel has no Kaggle API credentials of its own (`UserSecretsClient().get_secret("KAGGLE_API_TOKEN")` -> `ConnectionError`). Fix: add a secret named `KAGGLE_API_TOKEN` to that notebook (steps in `ops/kaggle/README.md`) and re-push. Nothing is uploaded from this machine. Also: the Kaggle proxy must **not** cover Hugging Face — routing HF through `localhost:10808` produced intermittent TLS EOFs, so `configure_kaggle_proxy()` now sets `NO_PROXY` for `huggingface.co/.hf.co/localhost` only, and `HF_USERNAME` in `.env` lets `hf_repo_id()` skip `whoami` when the link is flaky. |
| D-44 | **The Brain must be loaded as a conditional model.** G-1 records `model_type=qwen3_5`, `architectures=Qwen3_5ForConditionalGeneration`. `rtp.brain_train.load_for_training` first used `AutoModelForCausalLM`, which silently dropped the vision tower: the profile's four vision LoRA names (`qkv`, `proj`, `linear_fc1`, `linear_fc2`) were reported absent and not adapted (the gate log still records that run). It now prefers `AutoModelForImageTextToText` (falling back to CausalLM only for a text-only checkpoint), and the real run adapts all 16 target names. Two further real-run fixes: 4-bit profiles use **8-bit AdamW** (a float32 moment pair for a 65 M-parameter adapter is 0.52 GB on a 6 GB card), and a resumed checkpoint **restores the adapter/trainable weights** — the earlier `load_checkpoint` restored optimizer + RNG + cursor but not weights, which would have continued training from a fresh adapter with stale moments. Both gaps are now tests (`tests/test_brain_train.py`, `tests/test_trainer.py`). A third gap appeared on the real checkpoint: PEFT writes adapter keys without the adapter name (`...lora_A.weight`) while a live model has `...lora_A.default.weight`, so a raw `load_state_dict` reported every adapter key as missing. Resume now uses `peft.set_peft_model_state_dict`, and only *adapter* keys are required (the frozen base is legitimately absent from an adapter-only checkpoint). |
| D-45 | **Hybrid retrieval is real.** `rtp/vectordb.py` keeps BM25 from `rtp.harness` and fuses its ranking with a **Qdrant** vector ranking by reciprocal-rank fusion. Qdrant runs embedded (`QdrantClient(path=...)`), so no server is needed on the laptop; a server URL is also accepted. `HashedEmbedder` is explicitly a deterministic stand-in for plumbing tests, never a claim about semantic retrieval quality. Verified locally with qdrant-client 1.7.3: 3 docs indexed, fused top-3 all valid ids; API supports both `search` (old) and `query_points` (new). |
| D-46 | **Paper figure status is recorded without touching the SVGs.** Two stale figures got a red `\stalefig` note immediately below them in the LaTeX (`fig_signals`: Eye drawn as a VLM with LoRA; `fig_bridge`: Isaac Lab 2.x and no Newton). The exact remake instructions are in `docs/PAPER_FIGURE_STATUS.md` so the figure author can act on them. Appendix prose that named Isaac Sim 5.x / Isaac Lab 2.3 and a 137/13 CPU-only test count was corrected to the measured Newton-only stack and the training-venv count. The paper's own red-flag convention (`\todo`) was used for the prose correction. |
| D-47 | **The SFT training ceiling on the 4050 is 256 tokens, not 512.** D-38 measured a *single* forward+backward at 512 (5.42 GB). A real optimizer step adds the 8-bit AdamW moments, the adapter gradients and the checkpoint push, and the measured progression is: 512 OOM in the head on the first step; 384 takes step 1 (loss 0.6523) then OOMs on step 2; 256 completes both steps of a session (step 1 loss 0.8764, two checkpoints pushed) and resumes. The profile now carries **two** measured numbers: `max_len=512` (forward/backward) and `train_max_len=256` (SFT with adapters+optimizer, D-47). `run_sft` uses `profile.train_len`. The 6 GB card is a smoke/dev box for training; reportable SFT belongs on `local_4090` or `kaggle_t4x2`. An all-ones attention mask is dropped in `rtp.trainer` (no information, measurably more expensive flash-attn path), and the LM-head chunk on <20 GB profiles is 64 tokens. |
| D-48 | **G-14 PASS: real-weight QLoRA SFT + mid-epoch resume from HF.** `ops/sft_smoke.py --stage sft_gate8 --out reports/sft_smoke.json`: 64 teacher-validated SFT rows, Qwen3.5-4B in 4-bit nf4 with all 16 verified LoRA targets (16-bit conditional loader), 8-bit AdamW, 256 tokens, micro 1 x accum 16. Session 1 took steps 1-2 (losses 0.8764, 0.5982), stopped at micro-batch 32 of epoch 0, saved and pushed `step-000000002` to the private HF repo. Session 2 fetched that exact checkpoint from HF, restored adapter weights + optimizer + RNG + cursor, resumed at step 2 and continued to step 4 (losses 0.3792, 0.2364). Peak VRAM 5.85 GB, wall 640.8 s. All five checks pass. The run also exposed and fixed `RemoteStore.prune_checkpoints`: HuggingFace's `delete_folder` takes `path_in_repo`, not `folder_path`; the failure was correctly non-fatal (local checkpoint kept) and is now a regression test. |
| D-49 | **The 4B snapshot is now a private Kaggle dataset, published from Kaggle.** Kaggle account secrets were unreachable from this account's kernels (`UserSecretsClient.get_secret` -> HTTP 400 for every plausible label, in both script and notebook kernels), so the owner authorised a private token-carrying kernel (`ops/kaggle/push_token_kernel.py`): it reads `KAGGLE_API_TOKEN` from local `.env` at push time, writes a temporary private kernel outside the repo, and never prints or commits the token. Kernel `rezanadimikj/rtp-publish-private-token` completed. Verified: dataset `rezanadimikj/rtp-model-qwen35-4b-851bf6e8` has `isPrivate: true`, the two safetensors shards are 5,329,398,688 + 3,990,429,408 bytes (9.32 GB) plus tokenizer/config files, and its slug equals `kaggle_slug(model_spec("Qwen/Qwen3.5-4B", "851bf6e8", "qwen35-4b"))` — the exact slug `RemoteStore.resolve()` looks for. **No 9 GB was uploaded from this machine.** The dual-T4 G-3 kernel (`rezanadimikj/rtp-g3-newton-t4-gate`, `enable_gpu`) is queued/running; its result is not claimed until read back. |
| D-50 | **Kaggle exposes two T4s; current compute uses one at a time.** The `kaggle_t4x2` profile correctly records `n_gpu=2`, but `rtp.trainer` is a single-device loop and `ops/g3_sim.py`/the G-3 kernel step one scene on GPU 0. Both T4s are available in the kernel (`enable_gpu`); neither a two-GPU DDP training run nor a two-GPU simulation run is implemented or claimed. Reporting a `kaggle_t4x2` number requires either running two independent single-GPU jobs with different seeds (and saying so) or implementing DDP first. The T4 half of G-3 is running on one of the two cards. |

## E. Gates (open items closed by measurement): G-1…G-12 in `IMPLEMENTATION_PLAN.md` §12.

## F. P0 gate record (2026-11-17)

* **G-1 PASS** — `ops/g1_module_dump.py --model ~/models/Qwen3.5-4B --out reports/qwen35_modules.json`
  builds the model on the meta device (no weights) and checks the adapter map against the real graph:
  `Qwen3_5Attention{q,k,v,o_proj}`, `Qwen3_5GatedDeltaNet{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}`
  (+`A_log`,`dt_bias`, no `D`), `Qwen3_5MLP{gate,up,down}`, `Qwen3_5VisionAttention{qkv,proj}`,
  `Qwen3_5VisionMLP`/`Qwen3_5VisionPatchMerger{linear_fc1,linear_fc2}` — all present, dense (no experts).
  Checkpoint tensors, read from the safetensors headers: **738 tensors, 4,659,865,088 parameters**
  (language 4.206 B, vision 333.5 M, MTP 120.6 M); `lm_head` is tied (absent from the index); the model
  graph instantiates 5,174,964,736 parameters because it materialises an untied `lm_head` and no MTP.
* **G-2 PASS** — `ops/g2_fastpath.py` on the 4050 (cc 8.9, driver 550.54, torch 2.15.0+cu132):
  * binding: the four Qwen3.5 DeltaNet kernels resolve to
    `causal_conv1d.causal_conv1d_interface.{causal_conv1d_fn,causal_conv1d_update}` (1.7.0) and
    `fla.ops.gated_delta_rule.{chunk,fused_recurrent}.…` (0.6.0); both tiny GPU calls are finite.
  * smoke: **100 steps, seq 256, batch 1, loss chunk 128, flash_attention_2, 329.5 tokens/s,
    peak 4.99 GB VRAM, 16,232,448 trainable params, all losses finite** (0.7804 -> 5.0e-06). The
    collapse is memorisation of a fixed 8-scene cycle, not a quality result: the gate checks the
    kernel path, finiteness and the memory envelope, not generalisation.
  * `flash-attn 2.8.3.post2` is built locally from the PyPI sdist for sm_89 only (fwd+bwd verified);
    `flash_linear_attention` is absent by design (FLA is the DeltaNet provider). Building the two
    extensions needs `nvidia-*` pinned to 13.2.86 (13.5 nvvm vs 13.2 ptxas breaks), a
    `libcudart.so -> libcudart.so.13` symlink, `--no-build-isolation`, and native-arch only
    (`CAUSAL_CONV1D_ARCHS=8.9`, `FLASH_ATTN_CUDA_ARCHS=89`); flash-attn needed `-std=c++20`.
  * measured ceiling: at 6 GB, batch 2 x 256 tokens OOMs during LoRA injection (4.28 GB allocated
    before the first forward), so `local_4050` keeps `micro_batch=1`; batch 1 x 256 fits at 4.99 GB
    with the chunked loss (D-37).
* **G-3 not run** — needs the Isaac Lab 3.0 + Newton install; blocked on time/space, not on a decision.

