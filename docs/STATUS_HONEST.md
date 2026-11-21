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
| A4 / G-7 external wrench + substeps | SIM | **PASS** (`reports/g7_wrench.json`): free-body Newton impulse error max 2e-6 across 5/10/20/30 N and substeps 1/4; Isaac Lab composer with 4 substeps gives the same velocity as 1 (ratio 1.0), consistent with the every-substep backport `5608601`. Uses a test articulation, NOT the FR3 or a push test |
| A5 compliant control on the Newton articulation | SIM | **FAIL** (`reports/g6_newton_push.json`): effort-mode `NewtonCompliantArm` (A2 effort API + Newton RNEA gravity comp + Jᵀ wrench + weak null-space PD), 100,000 simulated steps at 1 kHz. Only 2/8 rows pass: `release_return` PASS (settle 1.098 s, overshoot 8.62%, peak speed 0.107 m/s); `steady_deflection` 12/24 subrows within ±10% (worst error 30.0% at 5 N −x); `plan_pause` clock exactly frozen (0.500→0.500) but final tracking error 3.31 mm > 3 mm; `elbow_push` EE error up to 169 mm > 15 mm; `invariants` 334 return-speed violations in 100,000 steps (force/leash/torque-rate/fault = 0); `payload`, `latency_error`, `human_hand_proxy` not run. No tolerance relaxed; NOT a safety certification (D-22) |
| A6-A10 (scene, skills, distillation, push robustness, certification) | — | **not run**; no reports exist |
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

## Update 2026-11-20 — review of the arms-first run (A1–A7)

Evidence: `reports/*.json` at commit `228ac0b`. State table below is generated by `ops/arm_steps_status.py` from `reports/`, not typed.

| step | verdict | evidence |
|---|---|---|
| A1 kinematics + reach | **DONE** | `g5_reach.json` PASS, <= 1 mm over 100 poses vs Newton FK |
| A2 API findings | **DONE** | `docs/findings/newton_arm_api.md`, 10 VERIFIED, honest UNRESOLVED (the MJCF converter needs a package absent from the Newton-only venv; native `add_mjcf` is the route) |
| A3 single-arm smoke | **DONE (rung L1)** | joint-space hold, drift < 1 mm; L1 does not expose actuator effort (stated in the report) |
| A4 wrench / G-7 | **DONE** | impulse = F·dt within 2 % with 4 substeps; every-substep fix (Isaac Lab PR #8269) confirmed present |
| A5 compliant control | **FAIL** | 4 of 8 rows fail; root cause plain (not dynamically consistent) null-space projection; see the A5 section in `docs/AGENT_RUNBOOK.md` |
| A6 scene | **DONE (scene half)** | two FR3 Duo arms + table + bag + 10 objects, 14 DoF, ~90 steps/s, deterministic at 1 mm |
| A7 scripted skills | **STAND_IN** | 18/18 at 10 trials on a pose-level state machine (`SIM-POSE`); `arm_cert.yaml` requires class `SIM` and >= 200 trials |
| A8–A10 | not run | blocked by A5 and A7 |

What the agent did right and must keep doing: it wrote `result: FAIL` instead of loosening anything; rows it had not implemented were recorded as "not run, no numbers invented"; a Coulomb feed-forward it measured to be invalid was removed instead of kept; the A7 stand-in is labelled `SIM-POSE` and "NOT physical skill outcomes". The A5 rows that did run are real Newton-articulation, torque-mode results.

Process failures to avoid repeating: (1) it built A6/A7 past a FAILing A5 — the gate in rule 9 exists for this and a stand-in is never DONE; (2) `STATUS_HONEST.md` was stale for A1–A7 — regenerate the block after every step; (3) the commit message "reach 18/18" read like a result — commit messages must state class and trials (`A7 plumbing (STAND_IN, SIM-POSE, 10 trials): not a certification result`).

### Arms-first steps (generated by `ops/arm_steps_status.py` from `reports/`)

| step | state | detail |
|---|---|---|
| A1 | **DONE** | PASS |
| A2 | **DONE** | 10 VERIFIED answers |
| A3 | **DONE** | PASS |
| A4 | **DONE** | PASS |
| A5 | **FAIL** | result='FAIL'; failing checks: ['elbow_push', 'payload', 'latency_error', 'human_hand_proxy']; rows not run: ['payload', 'latency_error', 'human_hand_proxy'] |
| A6 | **DONE** | PASS |
| A7 | **STAND_IN** | kind='sim_pose_standin', evidence_class='SIM-POSE': regression suite over the A6 state-machine stand-in; NOT physical skill outcomes |
| A8 | **NOT_RUN** | no report |
| A9 | **NOT_RUN** | no report |
| A10 | **NOT_RUN** | cards missing: ['grasp', 'place', 'release', 'retreat', 'weigh', 'handover', 'push_aside'] |


A5 FAIL is real Newton torque-mode evidence for the rows that ran; A7 is `SIM-POSE` plumbing on a state-machine stand-in, **not** physical skill outcomes. Nothing after A5 is claimed.

## Rules for updating this file

* Add a row only when the command's output exists in `reports/` or a committed test result.
* Mark evidence class: `FIXTURE` (symbolic), `SIM` (Newton), `GATE` (engineering), `MODEL` (trained
  weights), `HARDWARE` (real robot). Never pool classes in a paper number.
* "Notebook exists" is not evidence that the stage ran. "Spec exists" is not evidence that the
  behaviour is implemented or measured.

## Update 2026-11-21 — seed namespaces (M3, FIXTURE)

| item | evidence class | state |
|---|---|---|
| Seed namespace rule implemented | FIXTURE | `rtp/seeds.py` + 6 tests: training seeds >= 100000, test seeds < 100000; `assert_train`/`assert_test`/`split_bounds` refuse to cross the boundary; `audit_paths` scans `seed` keys without editing files |
| Audit of existing corpora | FIXTURE | `reports/seed_audit.json` **FAIL**: 310 seed references below 100000 across 8 legacy files in `data/local_4050/` (20 files scanned). These are pre-rule smoke fixtures and are **barred from SFT/GRPO/memory** until regenerated under R5 with seeds >= 100000. Nothing deleted |
| A7 prerequisite: hand model | FIXTURE | `ops/build_hand_mjcf.py` derives `assets/derived/fr3v2_hand.xml` from the cached Menagerie FR3 v2 (source unchanged, hashes in `reports/hand_model.json`): 13 bodies, 9 DoF (7 arm + 2 prismatic fingers), palm + 2 pads, ±20 N actuator force range, 50-step smoke check finite. This is a **simplified parallel-jaw gripper, not the Franka Hand**; the cached Menagerie models are arm-only (no gripper exists in the repo) |

## Update 2026-11-21 — A7 rung S1: hand under a force cap (SIM)

| item | evidence class | state |
|---|---|---|
| S1 hand open/close on a clamped 5 cm box | SIM | **PASS** (`reports/a7_s1.json`): derived `fr3v2_hand.xml` (13 bodies, 9 DoF), 3 s at 1 kHz, position servo kp=600 N/m / kv=12 with ±20 N actuator range. Fingers travel 10.1 mm and stall at a 47.81 mm inner gap on the 50 mm box (3 mm window), mean actuator force 14.9/14.4 N in the last 0.5 s, peak 14.96 N < cap, penetration 2.19 mm (self-imposed 3 mm target), box static, 137 steps/s. Actuator force, **not** a contact-sensor measurement |
| A7 overall | — | still **STAND_IN**: S1 does not change the gate. The pose-level regression remains `SIM-POSE`; the 200-trial physical suite is not run |
| A7 skill plans | FIXTURE | `rtp/control/skill_plans.py` + 4 CPU tests: an IK joint path becomes a timed task-frame plan (`[p; rotvec(R·R_homeᵀ)]`) usable by both position and impedance backends; endpoints and bounded stepping verified in `tests/test_skill_plans.py`. Pure geometry; no physics result |
| A7 rung S2 (single seed) | SIM | **PASS** seed 0 / 0.3 kg (`reports/a7_s2.json`): free 50 mm cube lifted 95.8 mm and held 1 s after approach + close with the derived hand; xy drift 6.1 mm, 170 steps/s. Calibrated FR3 IK runs in the arm-base frame; wrist rotated so the closing axis is world x. This is one trial, **not** the >= 95 % / 50-seed rung pass. |
| A7 rung S2 smoke suite | SIM | 3/3 seeds at 0.3 kg pass (`reports/a7_s2_suite_smoke.json`), min height gain 80.2 mm. Full 50-seed × {0.1, 0.5, 1.0} kg suite launched in the background; no aggregate claim until `reports/a7_s2_suite.json` exists. |
| Held-out test-set index | FIXTURE | `rtp/testset.py` + 5 tests: 8 families × 30 seeds (< 100000), sha256-frozen `data/heldout_manifest.json` (240 episodes, `c47c5e32083c6b77…`), `verify` rejects edits, duplicate seeds, wrong namespace and forbidden train seeds. Index only; episode generation/scoring not run |
| R2 skill-table sampler | SIM-ABSTRACT (code) | `rtp/bundle_backend.py` + 4 tests: seeded, replay-deterministic `SimBackend` that samples categories only from a certified `skill_tables/<skill>.json` matching a manifest `arm_model`; missing/uncertified tables raise `BundleMissing` instead of inventing outcomes. No certified tables exist yet, so no episodes were generated |
| Full suite after M3/R2/A7 scaffolding | FIXTURE/MODEL | `487 passed, 10 skipped` (`$ISAAC_PY -m pytest -q`, 68.8 s) after seed namespaces, hand derivation, S1/S2 harnesses, skill plans, held-out index and bundle sampler. New A7 scaffolding does not change any gate: A5 FAIL, A7 STAND_IN |
