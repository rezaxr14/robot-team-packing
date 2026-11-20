# robot-team-packing: two Eyes, one Brain, two Arms

Research code for a two-arm, two-eye kitchen manipulation system. The Brain is one **Qwen3.5-4B**
vision-language model; the Eyes are conventional CV (detector + OpenCV kernels, never an LLM);
the simulator is **Isaac Lab 3.0 + Newton** and nothing else.

* **Honest claim-by-claim status (start here):** [`docs/STATUS_HONEST.md`](docs/STATUS_HONEST.md)
* Step-by-step plan (authoritative checklist): [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
* Every user decision and constraint: [`DECISIONS.md`](DECISIONS.md)
* V2 design (eye geometry/tools, schemas, profiles, retrieval, training): [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md)
* Paper sources and specs: `docs/{SIMULATION,HARDWARE,IMPEDANCE_SPEC,COMPATIBILITY,SCHEMAS,ASSETS,SOURCE_REGISTER}.md`
* `paper/` is user-owned: its **text** is aligned with the implemented system (D-52), while figure
  SVG/PDF are frozen and stale figures carry a red note (`docs/PAPER_FIGURE_STATUS.md`).
  `paper/scripts/` holds the older v1 Track-A scripts that used to live in `scripts/`.

`rtp` = **Robot Team Protocol** (typed bus + harness). The symbolic world is a **test fixture and
reward source, not a physics result**.

## Doc index

**Authoritative** (follow these where anything else differs):
- [`docs/STATUS_HONEST.md`](docs/STATUS_HONEST.md) — claim-by-claim status and evidence classes; no result without a report.
- [`docs/USER_PLAN_VERBATIM.md`](docs/USER_PLAN_VERBATIM.md) — the user's instruction, verbatim.
- [`DECISIONS.md`](DECISIONS.md) — every user decision and constraint (D-nn).
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — plan v3 and the exact M1–M8 benchmark matrix (§10).
- [`docs/AGENT_RUNBOOK.md`](docs/AGENT_RUNBOOK.md) — execution order: arms first (A1–A10), then R4–R9.
- [`docs/AGENT_BUILD_SPEC.md`](docs/AGENT_BUILD_SPEC.md) — work packages and interfaces.
- [`docs/DATAGEN_KAGGLE.md`](docs/DATAGEN_KAGGLE.md) — multi-turn Kaggle datagen loop, gates, cost model.
- [`docs/SCENE_DESIGN.md`](docs/SCENE_DESIGN.md) — two-arm, two-eye kitchen layout (D-65, DESIGN).
- [`docs/IMPEDANCE_SPEC.md`](docs/IMPEDANCE_SPEC.md) — compliance/impedance spec and status.
- [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) — version constraints, profile matrix, isolation.
- [`docs/RESEARCH_MEMORY.md`](docs/RESEARCH_MEMORY.md) — index/decision log for external research; detailed findings in `docs/findings/`.

**Design / reference:**
- [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md) — V2 design contract (eyes, schemas, Qwen structure, retrieval, training).
- [`docs/KAGGLE_HF_RESOURCES.md`](docs/KAGGLE_HF_RESOURCES.md) — accounts, credentials, model roster, what has run on Kaggle/HF.
- [`docs/PAPER_FIGURE_STATUS.md`](docs/PAPER_FIGURE_STATUS.md) — stale-figure red-note inventory and remake requests (`paper/` frozen).
- [`docs/NOTEBOOK_API.md`](docs/NOTEBOOK_API.md) — what a notebook may call and must mark `SKIPPED`.
- `docs/{SIMULATION,HARDWARE,SCHEMAS,ASSETS,SOURCE_REGISTER,KAGGLE_MODELS}.md` — focused specs.

## Current status (plan v3, arms first)

* **Arms first (D-62):** [`docs/AGENT_RUNBOOK.md`](docs/AGENT_RUNBOOK.md) runs A1–A10 (certify arm
  skills in Newton) before R4–R9 (SFT/GRPO/eval). Only cheap-model Kaggle bring-up (R3) may run
  early, and its records are excluded from SFT.
* **Multi-turn datagen (D-63):** GPU scene bundle → hosted CPU episodes → replay-verified ingest +
  memory rebuild → GPU replay gate; built and fixture-tested on SIM-POSE, **not run on Kaggle**
  ([`docs/DATAGEN_KAGGLE.md`](docs/DATAGEN_KAGGLE.md)).
* **Compliance supervisor (D-66):** push → yield → bounded return → resume with plan-clock gate,
  leash and clamps; G-6b PASS on a SIM-ABSTRACT point mass (`reports/g6b_compliance.json`), not on
  the Newton articulation, hardware or a safety result.
* **Two-arm, two-eye kitchen (D-65):** island 1.8 × 0.9 m, arms at x = ±0.35 m, eyes R 1.6 m /
  H 2.2 m opposed ([`docs/SCENE_DESIGN.md`](docs/SCENE_DESIGN.md); DESIGN from assumed room sizes).
* **Benchmark matrix (D-64):** exactly M1–M8 in `IMPLEMENTATION_PLAN.md` §10, held-out seeds < 100000.
* `pytest tests -q` -> **323 passed, 10 skipped, ~45 s** on the 4050 laptop (2026-11-18). The 10
  skips are all "VIKI-R clone not available" (inert Track-A scorer).
* The legacy 2.3-era Isaac launcher stack is purged (D-35): no `ops/12_env_isaac.sh`, no EULA export,
  no RT-core preflight branch, no `isaac/bringup/`, no VIKI/LLaMA-Factory launcher stages.
* Qwen3.5-4B is downloaded and sha256-verified; `reports/qwen35_modules.json` (gate G-1) and
  `reports/g2_fastpath.json` (gate G-2: 100 steps, 329.5 tok/s) are produced by `ops/`.
* The 4050's context ceiling is **measured, not assumed** (D-38): 512 tokens fit at batch 1
  (5.42 GB peak), 768 OOM - `reports/seq_ceiling_local_4050*.json`. `local_4050.max_len` is 512;
  long-context SFT runs on `local_4090` or `kaggle_t4x2`.
* **G-3 PASS on the 4050** (D-39): Isaac Lab v3.0.0 from its own `uv.lock` with Newton +
  `mujoco-warp` and **no Isaac Sim / no PhysX**; 30 steps at 21 steps/s with rgb+depth contracts
  (`ops/g3_sim.py`, `reports/g3_sim.json`) **and on Kaggle T4** via a private GPU kernel
  (`reports/g3_sim_t4.json`, D-51). It is still a cube-fall rgb/depth gate, not manipulation.
* **Storage is live** (D-40): every asset resolves **local cache -> attached Kaggle dataset ->
  our private Kaggle dataset -> our private HF repo** (`Reza-Nadimi/robot-team-packing`). Round-trips
  for publish/fetch/checkpoint are measured, not assumed.
* **G-14 PASS on real 4B weights** (D-48): four SFT steps, checkpoint pushed to HF, fetched back,
  resumed inside the epoch and continued (losses 0.88 -> 0.60 -> 0.38 -> 0.24); 5.85 GB peak on
  the 4050 at 256 tokens (`reports/sft_smoke.json`). It is a **smoke, not a reportable trained
  model** (D-57).
* **G-6 simulation-only impedance cross-check** (D-58): `reports/g6_impedance.json` — analytic step
  response error **0.27 %** (limit 5 %), 10 N push and return measured. Not Phase 3 acceptance, not
  hardware, **not a safety certification**; hardware/L0/L1 sign-off is Claude's (D-22).
* **SimReady assets: catalogue + subset present, G-13 PARTIAL**: 4,239-package catalogue, a
  14-package texture-free subset (47.18 MB), 9/14 USD packages import into Newton; an FR3 Duo +
  13 SimReady object scene was **assembled and measured** (`reports/scene_manifest.json`, 83 bodies),
  not stepped or rendered. The A1 FR3 kinematics check passed (`reports/g5_reach.json`, SIM-POSE).
* **Compute scope** (D-50, D-56): both T4s are visible in the Kaggle kernel, but the trainer and the
  G-3 scene step **one device at a time**; no DDP training or two-GPU simulation exists.
* **Explicitly pending** (`docs/STATUS_HONEST.md`): a *stepped* asset-based kitchen scene, arm
  certification (A1–A10), the multi-turn loop run on Kaggle, a fixed held-out test set, the M1–M8
  matrix, omni-model scores, GRPO and impedance control on hardware. Asset-environment training is a
  SIM-POSE smoke only (`reports/asset_train.json`); `ops/cot_trace.py` recorded 0 frontier traces
  (`RTP_SERVE_URL` unset), while a separate Kaggle hosted task produced 8 unverified gemini-3.8-flash
  traces (`reports/cot_frontier.json`). None is a reportable result; do not present them as done.
* **Training resumes mid-epoch** (D-41): checkpoints carry optimizer + RNG + batch cursor, batch order
  is a pure function of `(seed, epoch)`, and `rtp/checkpoints.py` pushes every N steps to HF and prunes
  older steps. Resumed-at-step-3-of-8 runs finish at step 8 with the same losses. A checkpoint also
  restores the **adapter weights** (a gap the toy test caught), and 8-bit AdamW is used on 4-bit
  profiles so 65 M adapter parameters do not cost 0.52 GB of optimizer state.
* **Notebooks cover the paper** (D-42): `00` setup, `10` SFT, `20` GRPO, `30` Eyes, `40` world/data,
  `50` Isaac/Newton eval, `60` harness/zero-shot/CoT, `70` hindsight + success/failure RAG, `80`
  evals + omni-model comparison harness, `90` attribution/tables. Each is generated from
  `notebooks/src/*.py`, passes `tests/test_notebooks.py`, and is executed with `ops/nbrun.py`.
* **Hybrid retrieval** (`rtp/vectordb.py`): BM25 from the harness fused by RRF with a persistent
  Qdrant index (embedded local mode by default). `HashedEmbedder` is a labelled stand-in, not a
  semantic model.
* **The 9.4 GB model is published from Kaggle**, not from this laptop (`ops/kaggle_model.py`); the
  private kernel needs a `KAGGLE_API_TOKEN` secret added once by the owner (`ops/kaggle/README.md`).

## Layout (current code)

| path | what |
|---|---|
| `rtp/messages.py` | typed messages (ABILITY, SCENE, CLAIM, ..., AMEND), schema validation, token-budget bus |
| `rtp/world.py` | symbolic packing world; force-aware skills (slip/crush), bag lock, common random numbers |
| `rtp/perception.py` | Eyes with viewpoint occlusion + fault injection; counterfactual report replacement |
| `rtp/plan.py` | plan DAG schema, robust JSON parsing, round scheduler (bag lock, soft ordering deps) |
| `rtp/rules.py` | `TaskSpec`, final-state rule checker, static feasibility, deterministic dry run |
| `rtp/harness.py` | `.md` skills/rules/notes, BM25 (+ optional embedding fusion), monitor write-back |
| `rtp/brain.py` | `ScriptedBrain` (fixture), `LLMBrain` adapter + prompt builder |
| `rtp/loop.py` | closed loop: comm modes, recovery ladder, replan, amendments, failure attribution |
| `rtp/credit.py`, `rtp/hindsight.py`, `rtp/rewards.py` | counterfactual Eye credit, hindsight relabelling, GRPO rewards |
| `rtp/datagen.py` | scenes, specs, Brain SFT / GRPO prompt data |
| `rtp/flowmath.py`, `rtp/flowhead.py` | flow-matching math (tested) and the arm policy (torch) |
| `rtp/llm_client.py` | stdlib OpenAI-compatible chat client (vLLM/HF server) |
| `rtp/experiments.py`, `rtp/tables.py`, `rtp/figures.py` | config grid → records → tables/figures (Wilson, paired bootstrap) |
| `rtp/skills.py`, `rtp/simbridge.py`, `rtp/simworld.py`, `rtp/fakesim.py` | skill spec, sim→symbolic bridge, `SimBackend`, pose-level `FakeSim` |
| `rtp/toolbrain.py` | multi-turn LLM Brain with tools (check_rule, query_scene, look, dry_run, submit_plan) |
| `rtp/profiles.py` | hardware profiles, isolated per-profile paths, runtime fallbacks (P1) |
| `rtp/assetcache.py` | per-profile asset cache: attach → local → fetch, manifest-checked, path-escape guarded (P1) |
| `rtp/loss.py` | chunked causal-LM loss: never builds `[tokens x 248k vocab]` logits; `labels=` masks prompts (D-37) |
| `rtp/trainer.py` | SFT loop with gradient accumulation, cosine LR and **exact mid-epoch resume** (D-41) |
| `rtp/checkpoints.py` | `on_checkpoint` policy: save every N steps, push to HF, prune old steps, fetch latest to resume |
| `rtp/remotes.py` | local -> Kaggle dataset -> HF resolution, private publish, resumable checkpoint transfer (D-40) |
| `ops/hub.py` | CLI: `status`, `publish`, `fetch`, `checkpoint-push`, `checkpoint-latest` |
| `isaac/` | 2.3-era modules (`packing_sim.py`, `team_env.py`, `cameras.py`, ...) awaiting the P2 port to Isaac Lab 3.0 + Newton |
| `ops/preflight.py` | GPU/arch/system report and install-branch recommendation (Newton floor, Warp pin) |
| `ops/seq_ceiling.py` | measures how many tokens a profile can train at, instead of trusting `max_len` |
| `ops/build_wheelhouse.py` | resolves the profile's package set into an offline, tag-named wheelhouse |
| `ops/install.sh` | one installer per profile (`--profile`), optional Isaac Lab 3.0 + Newton venv |
| `ops/download_model.sh` | resumable, sha256-verified HF download |
| `ops/g1_module_dump.py` | gate G-1: real module tree of the checkpoint (meta device, no weights) |
| `ops/g2_fastpath.py` | gate G-2: DeltaNet kernel binding + optional QLoRA training smoke |
| `ops/plan.py` | stage registry for the phases that exist (`p0`, `p2`, `bundle`) with logging/resume |
| `ops/g3_sim.py` | gate G-3: Newton-only Isaac Lab scene stepping with rgb+depth on this GPU |
| `ops/stage.py`, `ops/bundle.py` | stage runner (timing, logs, skip-if-done) and report bundler |
| `harness_files/brain/{rules,skills}.md` | the Brain's own rules and skills (prompt material + format fixture) |

## Quickstart

```bash
ops/install.sh --profile local_4050 --dry-run     # what would be installed, per profile
ops/install.sh --profile local_4050               # torch + training/retrieval/agents/eyes venv
ops/build_wheelhouse.py --profile kaggle_t4x2 --out ~/wheelhouses  # once, on a networked box
ops/install.sh --profile kaggle_t4x2 --wheelhouse /kaggle/input/rtp-wheels-...   # offline, no index
ops/download_model.sh Qwen/Qwen3.5-4B ~/models/Qwen3.5-4B
source ops/env.sh && python ops/plan.py list      # p0: preflight, model integrity, G-1, G-2
ops/install.sh --isaac                            # Newton-only Isaac Lab 3.0 venv (uv sync at the tag)
source ops/env.sh && python ops/plan.py run p2    # G-3: rgb+depth from the Newton Warp renderer
python -m pytest tests -q
```

No result in this repo is evidence for a paper claim until it carries a run id, config hash, git SHA
and a seed, produced by the Isaac Lab + Newton pipeline.
