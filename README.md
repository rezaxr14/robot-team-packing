# robot-team-packing: two Eyes, one Brain, two Arms

Research code for a two-arm, two-eye kitchen manipulation system. The Brain is one **Qwen3.5-4B**
vision-language model; the Eyes are conventional CV (detector + OpenCV kernels, never an LLM);
the simulator is **Isaac Lab 3.0 + Newton** and nothing else.

* Step-by-step plan (authoritative checklist): [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
* Every user decision and constraint: [`DECISIONS.md`](DECISIONS.md)
* V2 design (eye geometry/tools, schemas, profiles, retrieval, training): [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md)
* Paper sources and specs: `docs/{SIMULATION,HARDWARE,IMPEDANCE_SPEC,COMPATIBILITY,SCHEMAS,ASSETS,SOURCE_REGISTER}.md`
* `paper/` is user-owned and frozen; nothing here modifies it. `paper/scripts/` holds the older v1
  Track-A scripts that used to live in `scripts/`.

`rtp` = **Robot Team Protocol** (typed bus + harness). The symbolic world is a **test fixture and
reward source, not a physics result**.

## Current status (P2)

* `pytest tests -q` -> **237 passed, 10 skipped, ~42 s** on the 4050 laptop (2026-11-18). The 10
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
  (`ops/g3_sim.py`, `reports/g3_sim.json`). Kaggle T4 is still unverified.
* **Storage is live** (D-40): every asset resolves **local cache -> attached Kaggle dataset ->
  our private Kaggle dataset -> our private HF repo** (`Reza-Nadimi/robot-team-packing`). Round-trips
  for publish/fetch/checkpoint are measured, not assumed.
* **Training resumes mid-epoch** (D-41): checkpoints carry optimizer + RNG + batch cursor, batch order
  is a pure function of `(seed, epoch)`, and `rtp/checkpoints.py` pushes every N steps to HF and prunes
  older steps. Resumed-at-step-3-of-8 runs finish at step 8 with the same losses.

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
