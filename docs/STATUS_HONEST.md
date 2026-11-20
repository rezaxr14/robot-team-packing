# Honest status of the paper's headline claims

This file exists after an explicit correction: earlier status messages overstated what had been
done. Each row below links to the command/report that actually produces the evidence. A row without
a measured artifact is **NOT DONE**, even if a notebook or a spec file exists.

Updated: 2026-11-19 (will be amended as each gate actually passes).

| claim | state | evidence / what is missing |
|---|---|---|
| Qwen3.5-4B module map and DeltaNet fast path measured | **DONE** | `reports/qwen35_modules.json`, `reports/g2_fastpath.json` |
| Newton-only Isaac Lab 3.0 stepping with rgb+depth on 4050 **and** T4 | **DONE (gate only)** | `reports/g3_sim.json`, `reports/g3_sim_t4.json`; this is a falling cube, **not** a manipulation scene |
| Real 4B QLoRA SFT step + mid-epoch resume from HF | **DONE (4 steps)** | `reports/sft_smoke.json`, G-14; tiny smoke, not a reportable trained model |
| Private storage local → Kaggle → HF | **DONE** | `rtp/remotes.py`, `ops/hub.py`, private Kaggle dataset `rtp-model-qwen35-4b-851bf6e8` (9.32 GB, published from Kaggle), private HF repo `Reza-Nadimi/robot-team-packing` |
| SimReady assets from Hugging Face ingested | **DONE (subset)** | 14 packages, 47.18 MB texture-free USD+physics, 4,239-package catalog, licence/sha256 manifest (`assets/simready_downloaded.json`); FR3 v2 + Duo MJCF verified (`reports/g5_fr3.json`) |
| Environment built from SimReady assets | **PARTIAL** | `ops/build_scene.py` assembled FR3 Duo + ground + 13 SimReady USD objects: 83 bodies, 375 shapes, finite masses (`reports/scene_manifest.json`); NOT stepped/rendered; no PTZ eyes or kitchen layout yet |
| Chain-of-thought traces from the harness with frontier AI | **PARTIAL** | Two distinct routes. Local teacher path: `rtp/frontier_cot.py` + `ops/cot_trace.py` produced 6 `scripted-teacher` traces (`reports/cot_trace.json`), `frontier_status: unavailable: RTP_SERVE_URL is unset`, `frontier_traces: 0`. Separate Kaggle-benchmarks hosted task: 8/8 real `gemini-3.8-flash` traces, all `finish_reason: stop` (`data/local_4050/cot_frontier_traces.jsonl`, `reports/cot_frontier.json`). Those plans were **not replayed or scored**, and this is **not** the multi-turn datagen loop. |
| Proper training on that environment's data | **SMOKE DONE** | 40 SIM-POSE SFT + 5 replay-validated hindsight episodes from real SimReady worlds; 4B QLoRA 2 steps, loss 1.081 -> 0.742 (`reports/asset_train.json`); not a reportable trained model, GRPO never run |
| Hindsight relabelling recorded from real failures | **DONE on SIM-POSE** | 112 asset-backed episodes -> 6 failures -> 6 relabelled -> 5 replay-validated (`reports/asset_train.json`); symbolic fixture version in notebook 70 |
| Success/failure memory RAG | **DONE on SIM-POSE** | 40 asset-backed episodes (37 success/3 failure), BM25 separate success/failure retrieval with scores (`reports/asset_memory.json`); not yet fed into a trained Brain's decoding |
| Scored on a fixed test set | **DONE for fixture only** | `rtp/benchmark.py` + `ops/score_test.py`, seeds 0-29 (training >=100000): ScriptedBrain 26/30 = 86.7% [70.3, 94.7] (`reports/test_scores.json`); **trained Brain not scored** |
| Compared with other omni models | **HARNESS ONLY** | `reports/omni_comparison.{json,md}` has a strict prediction loader, Wilson + paired bootstrap; all competitor rows and the trained-Brain row are `not run` with no numbers |
| Impedance control making arms safe around people/children | **SIM CROSS-CHECK ONLY** | `rtp/control/impedance.py` + `ops/g6_impedance.py`: step error 0.27% (limit 5%), push deflection 0.06655 m vs F/K 0.06667 m, return error 9.6e-10 m (`reports/g6_impedance.json`). **NOT** a safety certification; D-22 assigns hardware/L0/L1 sign-off to Claude; no plan-clock gate, leash, clamps, state machine or ISO limits |

## Update 2026-11-20 — A1 (arms first) and Plan v3

| item | evidence class | state |
|---|---|---|
| Applied patch `robot-team-packing-update.patch` (`git am --3way`), patch file removed | GATE | plan v3, A1-A10 runbook, multi-turn datagen loop, compliance supervisor, eye study are now in the tree; patch deleted after applying |
| Full test suite before A1 | FIXTURE/MODEL | `428 passed, 10 skipped` on the training venv (`python -m pytest -q`) |
| A1 FR3 flange kinematics vs Newton `eval_fk` | SIM-POSE | **PASS**: 102 configurations, max position error 0.00224 mm, rotation error 0.0 deg, joint limits match; `reports/g5_reach.json`; 6 tests in `tests/test_fr3_kinematics.py` |
| A2 Newton arm API findings | SOURCE READ + VERIFIED SNIPPETS | `docs/findings/newton_arm_api.md`: all 8 questions answered with local file paths/line numbers; effort mode, state reads, gravity comp, Jacobians, wrenches, envs/substeps and contact forces verified by running snippets; MJCF->USD runtime conversion UNRESOLVED (`isaacsim` absent) |
| A3 one FR3 steps in Newton, holds a pose | SIM | **PASS** (`reports/a3_arm_smoke.json`): native Newton MJCF articulation, position-actuator L1, 5 s / 5000 steps at 1 kHz, flange drift 0.692 mm (<1 mm), 159.7 steps/s; MJCF stiffness x4 was required (default gains sagged 2.78 mm). NOT effort mode, manipulation, push or hardware |
| A4-A10 (wrench, articulation compliance, scene, skills, distillation, push robustness, certification) | — | **not run**; no reports exist |
| R3 cheap-model Kaggle bring-up | — | **not run** (allowed before A10; its records are excluded from SFT) |
| R4-R9 (scale datagen, SFT, GRPO, M1-M8 matrix) | — | **not run** |

## Update 2026-11-18 — multi-turn data generation, compliance supervisor, eye placement

| item | evidence class | state |
|---|---|---|
| Kaggle proxy transport, journal/resume, typed errors | FIXTURE | built; 11+ tests against a fake HTTP layer; **never called a real endpoint** |
| Multi-turn shard runner, hosted-task generator, ingest/replay-verify/ledger/memory loop | FIXTURE (SIM-POSE world) | built; `tests/test_multiturn_datagen.py` (no network, no torch); **not run on Kaggle** |
| Compliance supervisor (push-yield-return, plan-clock gate, leash, clamps) | SIM-ABSTRACT | G-6b PASS 8/8 rows on a point mass; Newton articulation and hardware **not done** |
| Eye placement study | DESIGN | assumed room dimensions; chosen layout in D-65; replace with measurements |
| Cost table | ESTIMATE | 10k/4k tokens per text episode assumed; first prompt measured ~1.2k tokens |
| Arm certification, GPU replay gate, scene bundle, view library | — | **not built** |

## Rules for updating this file

* Add a row only when the command's output exists in `reports/` or a committed test result.
* Mark evidence class: `FIXTURE` (symbolic), `SIM` (Newton), `GATE` (engineering), `MODEL` (trained
  weights), `HARDWARE` (real robot). Never pool classes in a paper number.
* "Notebook exists" is not evidence that the stage ran. "Spec exists" is not evidence that the
  behaviour is implemented or measured.
