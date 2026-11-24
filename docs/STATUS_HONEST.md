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

*This is a snapshot kept for history, not the current gate output. For the current state see the 2026-11-24 update at the end of this file; re-run the generator rather than reading this table.*

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

## Update 2026-11-22 — review of A5/A7 and supervisor fix (CPU, SIM-ABSTRACT)

### Arms-first steps (generated by `ops/arm_steps_status.py` from `reports/`)

| step | state | detail |
|---|---|---|
| A1 | **DONE** | PASS |
| A2 | **DONE** | 10 VERIFIED answers |
| A3 | **DONE** | PASS |
| A4 | **DONE** | PASS |
| A5 | **FAIL** | result='FAIL'; failing checks: ['plan_pause', 'payload', 'latency_error']; ALSO criterion changed by ['D-79'] with no approval in configs/approvals.json |
| A6 | **DONE** | PASS |
| A7 | **FAIL** | skills not meeting >= 0.9 at >= 200 trials: ['place(0/0)', 'release(0/0)', 'retreat(0/0)', 'weigh(0/0)', 'handover(0/0)', 'push_aside(0/0)'] |
| A8 | **NOT_RUN** | no report |
| A9 | **NOT_RUN** | no report |
| A10 | **NOT_RUN** | cards missing: ['grasp', 'place', 'release', 'retreat', 'weigh', 'handover', 'push_aside'] |


| item | evidence class | state |
|---|---|---|
| Stiction-floor deadlock reproduced; stall-aware resume + gated bounded integral | SIM-ABSTRACT | `tests/test_compliance.py` (33 pass): without the fix RETURN is permanent and the clock never restarts at 3.25 mm; with it RESUMED, final error <= 3 mm, overshoot <= 10 %, no hunting. **Not yet run on the Newton articulation.** |
| Gate hardened (per-skill table, implausible aggregates, approvals) | FIXTURE | `tests/test_arm_steps_status.py` (7 pass) |
| D-79 elbow criterion | — | **no user approval recorded**; see runbook addendum |

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
| A7 rung S3 harness | SIM (code; one trial run) | `ops/a7_s3_trial.py` adds box friction and finger-force-factor controls; `ops/a7_s3_grid.py` aggregates a 3×3 grid into a measured outcome histogram (lifted / knocked aside / never picked up / harness fault). Single probe (friction 0.9, factor 1.0, 0.5 kg): gain 76.8 mm, drift 10.6 mm, below the 80 mm lift bound. Full grid not run yet; no slip/crush cause is claimed (no such sensor) |
| A5 elbow criterion (D-79) | — | `ops/a5_newton_push.py` previously scored `elbow_push` with a stiffness-relative bound that was PROPOSED but never approved by the user (no `configs/approvals.json` entry); the row now scores the original 15 mm bound and reports both wrench mappings (`1.25·F_task/K + 3 mm`), keeps the original 15 mm value as `original_ee_bound_m`, requires link yield > 1 mm and post-release EE <= 5 mm, and records posture error as a diagnostic only. The full run already in flight used the old code; its report will be re-run with the new criterion before any A5 gate claim |
| D-79 bound check against existing probe | SIM (measured, no new physics) | Recomputed the approved bound from `task_wrench_from_body` at the home pose: F_task = 9.82/19.64/29.47 N (link 4) and 10.02/20.04/30.06 N (link 6); D-79 bounds 27.6/52.1/76.7 and 28.1/53.1/78.2 mm vs measured EE 20.6/40.2/60.3 and 20.7/41.3/71.5 mm — all 6 subrows satisfy the bound; post-release EE 0.43–1.14 mm satisfies the 5 mm bound. Original 15 mm criterion remains recorded as FAIL in the probe report |
| A7 S2 mass sweep (old 10 cm command) | SIM | Partial 83-trial run: 0.1 kg 50/50 PASS (min gain 93.5 mm); 0.5 kg 20/30 PASS, 10 FAIL with gain ~71–72 mm < 80 mm because the box slips inside the pads; 1.0 kg not reached. The run stopped without writing an aggregate report. Thresholds unchanged. |
| A7 S2 slip-compensated trial | SIM | `ops/a7_s2_lift_v2.py` commands a 12 cm lift (still requires >= 8 cm measured gain) to compensate in-grip slip: probes 0.5 kg seeds 0/4/9 gain 109.6/110.0/116.4 mm and 1.0 kg seeds 0/4 gain 113.8/112.5 mm — 5/5 PASS. Full 50-seed × 3-mass suite restarted with this trial script into `reports/a7_s2_suite_v2.json`; no aggregate claim yet |
| S3 friction control verified | SIM (model attribute) | `ops/a7_s3_trial.py --friction {0.3,0.9}` produces `model.shape_material_mu = 0.300 / 0.900` on the item shape; the grid actually varies contact friction. Force-factor scales the two finger target gains (still clamped at the ±20 N actuator range) |
| Paper pipeline check | FIXTURE (not evidence) | `PYTHONPATH=$PWD $ISAAC_PY paper/scripts/pipeline_check.py` wrote 12 rows and stated "PIPELINE CHECK ONLY, not evidence"; the CSV was removed rather than committed as a result |
| A5 elbow probe report | SIM | `reports/g6_elbow_probe.json` (untracked until now) is the pre-D-79 6-subrow elbow measurement: EE 20.6/40.2/60.3 mm (link 4) and 20.7/41.3/71.5 mm (link 6) at 10/20/30 N, K=500 N/m, all `ee_error_le_15mm` false; used as the measured basis for D-79 |

## Update 2026-11-21 — R1 scene bundle + eye view library (code only)

| item | evidence class | state |
|---|---|---|
| R1 `ops/render_views.py` + CPU manifest tests | FIXTURE (code, not run) | CLI, deterministic sha256 manifest, tamper detection and the honest MJCF->USD fallback are written; `tests/test_render_views_manifest.py` passes on CPU. **Not run on GPU**: `isaacsim` is absent in the Newton-only venv, so no USD scene builds, no `views/*.jpg` are written and no fps is measured; the >= 10 fps acceptance (G-8) and bundle-hash stability across two runs stay unmeasured. Fallback writes `result: FAIL`, `reason: "MJCF->USD conversion unavailable in Newton-only venv"`, `fps_per_eye: null` |

## Update 2026-11-21 — A7 ladder S4/S5/S6 fixtures (code only, NOT run)

| item | evidence class | state |
|---|---|---|
| A7 rungs S4/S5/S6 trial fixtures | FIXTURE (code, NOT run) | `ops/a7_s4_place.py`, `ops/a7_s5_weigh.py`, `ops/a7_s6_handover.py` written as single-seeded Newton trials that will write `reports/a7_s4.json` / `a7_s5.json` / `a7_s6.json` when invoked later under `$ISAAC_PY`; `evidence_class: SIM` only after a real solver step, explicit `result: FAIL` + reason on build failure. Pure helpers tested on CPU (`tests/test_a7_ladder_helpers.py`, 16 passed). **NOT run** (GPU reserved for other jobs): no Newton stepping, no place/weigh/handover result and no pass rate exists; no certification claim. Bounds from `configs/arm_cert.yaml` / runbook unchanged |

## Update 2026-11-21 — R5 curation + SFT formatting (code only)

| item | evidence class | state |
|---|---|---|
| R5 `rtp/curate.py` + `rtp/sft_format.py` + CPU tests | FIXTURE (code, not run on real data) | `/home/reza/venvs/rtp/bin/python -m pytest tests/test_curate.py -q`: 11 passed. Synthetic records only; covers certified/uncertified arm models, test-seed and replay-failure exclusion, dedupe, per-model cap, failure→replan ranking, deterministic chat rows and canonical-hash tamper detection. No certified skill tables, certified-arm episodes or SFT dataset exist: **no dataset was built**, no train/val jsonl written, no threshold changed |
| Full suite after R1/R5/S4-S6 code | FIXTURE/MODEL | `520 passed, 10 skipped` (`$ISAAC_PY -m pytest -q`, 69.9 s). All new modules are code fixtures; this does not change A5 FAIL or A7 STAND_IN |

## Update 2026-11-21 — A7 S5 weigh probe (SIM, single seed)

| item | evidence class | state |
|---|---|---|
| S5 mass-from-gravity-torque | SIM | First probe (0.5 kg) measured -0.5 kg because the estimator used the opposite sign to Newton's **holding-torque** convention; fixed `m = F_z/g` with matching CPU tests (16 passed). Re-run seeds 0 at 0.1 / 0.5 / 1.0 / 3.0 kg: estimates 0.1 / 0.5 / 1.0 / 3.00001 kg, relative error 0.0 in all four (`reports/a7_s5_probe*.json`). This is a welded-payload single-seed probe, **not** the >= 90 % / >= 200-trial rung pass and not a certification |

## Update 2026-11-21 — A7 rung S4/S5/S6 single-seed probes (SIM, ungraded)

| item | evidence class | state |
|---|---|---|
| S4 place into a compartment | SIM | `reports/a7_s4_probe.json` **PASS** seed 0: box centre in `compartment_1`, hand clear, penetration 0 mm. One trial; >= 90 % / >= 200 trials not run |
| S5 weigh | SIM | `reports/a7_s5_probe*.json` **PASS** at 0.1 / 0.5 / 1.0 / 3.0 kg with the holding-torque sign fix (D-83), 0 % relative error. One seed each, welded payload; ungraded |
| S6 push_aside | SIM | `reports/a7_s6_push_probe.json` **PASS** seed 0: blocker moved 93.1 mm (> 50 mm), tilt 3.0 deg (< 45 deg). One trial; ungraded |
| S6 handover | SIM | `reports/a7_s6_handover_probe.json` **FAIL**: during L's carry the box fell back toward the table; at R close, box at (-0.109, 0.029, 0.935) vs expected (0, 0, 1.045), 157.3 mm away. No transfer occurred; no threshold relaxed. Diagnosed as a real grasp/carry failure, not a scoring artifact |
| S6 handover drop timing (SIM) | SIM | `reports/a7_s6_handover_probe.json` now includes the box-z trace sampled every 200 steps: L lifts the box to 1.037–1.040 m and holds it through the carry, then it drops during R's approach/close to ~0.95 m. The failure is at the transfer phase, not at L's grasp; no cause beyond the measured timing is claimed |

## Update 2026-11-21 — A7 S2 graded suite, first aggregate (SIM, FAIL)

| item | evidence class | state |
|---|---|---|
| S2 50 seeds × 3 masses, 12 cm command | SIM | `reports/a7_s2_suite_v2.json` **FAIL**: 147/150 overall (98 %) but per-mass 0.1 kg 50/50, 0.5 kg 50/50, **1.0 kg 47/50 = 94 % < 95 %**; three failures at 78.7/78.9 mm and one near-zero gain with 50 mm drift. Original 80 mm bound untouched. All three failing seeds pass with a 14 cm commanded lift (100.5/98.9/131.8 mm), so a 1.0 kg-only 50-seed suite with `--lift-delta 0.14` is running into `reports/a7_s2_v2_heavy.json`; no pass claimed yet |

## Update 2026-11-22 — A5 row fixes and A7 S2 final rung (SIM)

| item | evidence class | state |
|---|---|---|
| A5 `human_hand_proxy` kinematic box | SIM | `SolverMuJoCo` does not populate `contacts.rigid_contact_force` on this build (0 N throughout), so the row now reports a **model-based external wrench estimate** (`M qdd + C + g − τ`), clearly labelled as an estimate, not a contact sensor: peak 7.50 N, targeted probe `reports/a5_hand_probe.json` PASS |
| A5 `release_return_estimated` | SIM | Release signalled to the estimator (as a wrist contact sensor would) plus `speed_limiter_gain=8000` and `ComplianceConfig(zeta=1.2)`: 0.5/1.0 N noise both PASS, overshoot 6.7/7.0 % (<10 %), peak 0.0989 m/s (<0.11). No threshold changed; `reports/a5_est_probe.json` |
| A7 S2 final rung aggregate | SIM | `reports/a7_s2_final.json` **PASS**: 200 trials, 196 pass (98 %, Wilson [94.97, 99.22]); per mass 0.1 kg 50/50, 0.5 kg 50/50, 1.0 kg 96/100 (the 12 cm and 14 cm commanded-lift sources are both recorded). NOT the A7 certification gate: the full >=200-trial regression must cover all 7 skills |
| S6 handover drop phase (SIM) | SIM | Phase-labelled box positions (`reports/a7_s6_handover_probe.json`): with the original R hover the box is knocked off during R's **pre-approach** (z 1.037 → 0.941 m); with a 25 cm higher approach waypoint it survives the pre-approach (1.037 → 1.020 m) but is knocked off during R's **descent** (→ 0.929 m). This is a real R motion-planning/contact failure, not a proxy artifact; handover remains FAIL and no threshold was changed |

## Update 2026-11-22 — A7 physical regression driver (gate bug fixed)

| item | evidence class | state |
|---|---|---|
| `ops/a7_regression_physical.py` | SIM (aggregator; trials from real Newton runs) | Imported the 200 graded grasp trials (`reports/a7_physical/grasp/`, 196 PASS) and wrote `reports/skill_regression_scripted.json`. First aggregate exposed a **false DONE**: raw trial counts (196 passes) were divided by the 7-skill `total`, so the gate read 196/7 >= 0.90. Fixed the aggregator to report `trials` = minimum per-skill trials, `passed` = number of graded skills, `total` = 7. Gate now correctly reports **A7 FAIL** (`trials=0 < 200`) because place/release/retreat/weigh/handover/push_aside are ungraded. The old SIM-POSE stand-in report is preserved at `reports/skill_regression_standin.json` |
| A7 place seed 1 probe | SIM | `reports/a7_s4_probe_s1.json` PASS (compartment containment, hand clear, 0 mm penetration). A 200-seed graded `place` suite is running into `reports/a7_physical/place/`; no pass claim until the aggregate exists |
| S6 handover experiments (not adopted) | SIM | Tried a 90°-rotated R wrist, perpendicular grippers, a +0.25 m higher approach and 12 incremental Cartesian descent waypoints. All still dropped the box or made DLS IK diverge; none improved on the measured FAIL, so all were reverted to the committed version. No threshold changed; handover remains FAIL |
| A7 release / retreat trial scripts | SIM (single seed) | `ops/a7_s7_release_retreat.py` added and registered in the physical regression driver; seed 0 probes PASS for both skills (`reports/a7_release_probe.json`, `reports/a7_retreat_probe.json`): box lands in the table band, lift observed, hand clear >= 80 mm. No 200-trial aggregate yet; not a certification |
| S3 friction × force grid | SIM | `reports/a7_s3_grid.json` PASS (all 27 cells ran): 27/27 `lifted`, mean gain 115.1/111.4/98.5 mm at force factors 0.7/1.0/1.3 (0.5 kg, friction 0.3/0.6/0.9). The histogram does **not** show the runbook's expected slip/crush differentiation; with the 14 cm commanded lift and the derived gripper all cells were lifted, so no slip/crush cause is claimed. One outcome class only; not a certification |

## Update 2026-11-22 — definitive A5 full run (SIM): 7 PASS / 3 FAIL

Command: `source ops/env.sh && $ISAAC_PY ops/a5_newton_push.py --invariant-steps 100000 --out reports/g6_newton_push.json`
Report: `reports/g6_newton_push.json` (225,723 bytes). Gate: `python ops/arm_steps_status.py --require A5` exits **1**.

| row | state | measured evidence |
|---|---|---|
| `steady_deflection` | **PASS** | ±10 % bound met after the identified-mass/friction compensation |
| `release_return` | **PASS** | release and bounded-speed return within the unchanged runbook bounds |
| `plan_pause` | **FAIL** | plan clock frozen during yield (0.500 → 0.500), state returns to TRACK, but the clock **never advances after resume** (final 0.500) and final tracking error is **3.241 mm > 3 mm**; no RESUMED event. Single root cause for the two grid failures below |
| `elbow_push` | **FAIL** (superseded text: it was once marked PASS here) | The PASS recorded in this historical row used the D-79 stiffness-relative bound, which the user never approved. D-79 is PROPOSED ONLY and `configs/approvals.json` is `{}`. Scored on the **original 15 mm bound**, the measured EE errors are 19.33/39.01/59.16 mm (link 4) and 19.65/39.63/63.32 mm (link 6) at 10/20/30 N for **both** wrench mappings (`reports/g6_newton_push.json`, `reports/a5_elbow_two_mapping.json`), so the row FAILS |
| `invariants` | **PASS** | 100,000 random-push steps, 63,979 RETURN ticks, **zero** force / moment / leash / torque-rate / return-speed / fault violations (predictive limiter + residual-comp gating) |
| `human_hand_proxy` | **PASS** | virtual spring hand peak 14.66 N; kinematic box peak 7.50 N from a labelled **model-based force estimate** (Newton does not populate `rigid_contact_force` here) |
| `steady_deflection_estimated`, `release_return_estimated` | **PASS** | 0.5 N and 1 N estimator noise, release signalling: overshoot 6.7/7.0 %, peak 0.0989 m/s < 0.11 |
| `payload` | **FAIL** | 0/1/2/3 kg cells all pass `steady_deflection`, `release_return` and `elbow_push`; **every cell fails only because of `plan_pause`** (final tracking error 3.241 / 3.006 / 2.825 / 2.671 mm) |
| `latency_error` | **FAIL** | all 8 cells (0–3 tick delay, ±20 % mass/inertia/friction/armature) pass the other three rows; **every cell fails only because of `plan_pause`** (final tracking error 2.53–5.59 mm) |

Diagnosis for the next patch: a single defect in plan-clock resume/tracking after a push (the supervisor enters TRACK but the clock stays frozen and the controller settles 0.24 mm outside the 3 mm bound). No threshold, tolerance, clamp or test was changed. This is simulation evidence only (D-22); no safety claim. A7 remains `FAIL` until its 7-skill 200-trial regression exists.

## Update 2026-11-22 — plan-resume integral experiment (not adopted)

| item | evidence class | state |
|---|---|---|
| Bounded Cartesian/rotational integral in `NewtonCompliantArm` | SIM | Tried to remove the stiction floor that prevents the supervisor's 3 mm resume criterion: with gain 150 / max 8 N and rot max 2 Nm, `plan_pause` PASSED (clock 5.345, RESUMED, error 2.70 mm) but `release_return` overshot 24.2 % and `release_return_estimated` 29.3 %. With gain 100 / max 1.5 N and rot max 1 Nm, `release_return` PASSED (8.7 % overshoot) but `plan_pause` still failed at 3.54 mm. No setting passed every row, so the controller was reverted to the committed baseline. No threshold/tolerance changed; A5 remains FAIL on `plan_pause` (3.241 mm) and the grids that use it. Targeted evidence: `reports/a5_plan_probe.json`, `reports/a5_targeted_after_integral.json` |

## Update 2026-11-23 — review-2 response (A5 resume, elbow mappings, A7 grasp rules)

| item | evidence class | state |
|---|---|---|
| A5 plan resume (stiction deadlock) | SIM | The supervisor patch resumes via `RESUMED ... via=stalled`; on the Newton plant the plan clock now advances (5.05) and the final tracking error is **0.45 mm** with the existing residual-wrench compensation also active in free-space TRACK. `release_return` overshoot is unchanged at **0.9 %** (bound 10 %). Targeted probe `reports/a5_trackbias_probe.json`; definitive full run in flight |
| A5 elbow two-mapping test | SIM | Dynamic `F_task = J̄ᵀ τ_ext` and flange-equivalent `(J_f⁺)ᵀ J_bᵀ F` give the **same** EE deflection: 19.37/38.94/58.92 mm (link 4) and 19.70/39.71/61.13 mm (link 6) at 10/20/30 N (`reports/a5_elbow_two_mapping.json`), i.e. both are `F/K`. The dynamically consistent mapping does **not** reduce the Newton plant's IE error, so the original **15 mm bound is not met** and the row FAILS |
| D-79 | — | Wording corrected to **“proposed; user approval not recorded.”** No `configs/approvals.json` entry exists; the row is scored on the original 15 mm bound and does not cite D-79. The stiffness-relative number is kept only as `d79_proposal_bound_m` for the user’s decision |
| A7 grasp S2, earlier 12 cm/14 cm numbers | SIM (exploration) | **Relabelled exploration, not a rung result.** `lift_delta` was chosen after seeing failing seeds and evaluated on the same seeds; the 1 kg figure pooled two commands. They must not be quoted as a certified rung |
| A7 grasp single lift rule | SIM | `lift_delta_for_mass(m) = 0.10 + 0.04·clip(m, 0, 1)` → 0.104/0.12/0.14 m at 0.1/0.5/1.0 kg, implemented once in `ops/a7_s2_lift_v2.py` (`--rule`) and used by every reported trial. Constants are being checked on the tuning split only (`seeds >= 900000`, `configs/arm_cert.yaml: tuning_seeds_min`); the reported evaluation uses fresh seeds `>= 100000`. **Not yet run** |
| A7 rung S3b (force boundary) | SIM | `ops/a7_s3b_boundary.py` runs mass {0.1,1,2,3} × mu {0.1,0.3,0.9} × force factor {0.1,0.3,0.5,1.0}, measures the per-pad squeeze from the finger position servo (model-based estimate, not a contact sensor), and compares the sim boundary with `2·mu·N >= m·g`. **Grid running; no result yet** |
| Crush definition | — | No rigid-body crush sensor on this hand. Defined as peak finger-force estimate > `rtp.simbridge.crush_force_for(fragility)`; because that includes the +15 N simulator margin (23 N at fragility 1) and the hand's actuator range is 20 N, the raw `rtp.world.crush_limit` (8 N at fragility 1) is recorded too. The named-threshold outcome is reported honestly (expected: 1.3× does **not** reach 23 N) |
| A7 S3b contact-model diagnosis | SIM | Measured one setting at a time (`reports/a7_grasp_diagnosis.json`): with the derived hand's 1.2 pad friction the object mu never entered the contact (S3: 27/27 lifted, `reports/a7_s3_grid.json`). Setting the pad/palm mu to the tested object mu makes the boundary cell behave physically: 0.1 kg, mu 0.1, factor 0.1 changed from lifted (101.8 mm at 1.52 N pad force) to slipped (0.02 mm; `reports/a7_s3b_contact_fix_smoke.json`). No threshold changed |

## Update 2026-11-23 (night) — A7 S3b + tuning measured; compliant bridge blocked (all SIM)

Nothing in this update is a rung result. No threshold, tolerance, clamp or test was changed.

| item | evidence class | state |
|---|---|---|
| A7 S3b force-boundary grid | SIM | **FAIL** (`reports/a7_s3b_boundary.json`, 48 cells + 2 crush probes). `boundary_matches_analytic_within_one_step=false`: 11 of 12 (mass, mu) rows agree with `2·mu·N >= m·g` within one grid step, one does not (0.1 kg, mu 0.3: the sim never lifts at any factor, analytic needs factor 0.3). Most agreements are trivial "never lifts" rows: the pad-force **estimate** collapses at contact (measured 0.03–0.28 N against nominal 1.5–15.3 N), because `kp_eff·(ctrl−q)` goes to zero once the position servo reaches its target; the contact force is carried by the constraint, not the servo error. **Corrected 2026-11-24:** the 0.03–0.28 N readings were taken *after* the lift; in a slipping cell the fingers then close onto the command and the servo error goes to zero. The grasp-phase force is 1.5–14.3 N and tracks the commanded factor; see the S3b row in the 2026-11-24 update. 2 kg and 3 kg never lift at any factor in this harness. The force axis is therefore not actually instrumented and the boundary cannot be validated this way |
| A7 S3b crush probes | SIM | 0.3× peak 4.56 N (no crush) PASS; 1.3× peak 19.09 N, below the named `crush_force_for(1)=23 N` so the named-threshold check is **false**, while the raw `rtp.world.crush_limit(1)=8 N` is exceeded (`crushed_under_raw_limit=true`). The `crush_1p3x_crushes_fragility_1` check fails as predicted by the 20 N actuator ceiling |
| A7 grasp lift-rule tuning | SIM | `reports/a7_s2_rule_tuning.json` (tuning seeds ≥ 900000 only): 39/40 pass. The 0.5 kg / 0.12 m cell had one drop (min gain 2e-05 m), so `lift_delta_for_mass(0.5)=0.12` is **not** clean even on the tuning split. The one-shot evaluation on fresh seeds ≥ 100000 was **not run** |
| A7 grading through `NewtonCompliantArm` | SIM (probe only) | Bridge written (`ops/a7_compliant.py`): zeroes only the arm servo gains before solver construction, drives the A5 controller with time-parameterised plans that start at the measured pose, phase profiles approach/grasp/transport. **Measured blocker:** with `approach` (300 N/m) the move ends in TRACK with **17–22 mm** static error (joint stiction floor; the TRACK residual compensation does not remove it), and the next move then crosses `e_pause` (30 mm), yields, and deadlocks in RETURN because the error is above `eps_stall` (8 mm). Descend did not reach the box and the grasp height gain was **0.0 m**. **0 compliant graded trials exist**; A7 stays ungraded |
| A5 definitive full run | SIM | Restarted 00:38 with the two-mapping elbow row and grid optimisation; still running at stop time and **no report written yet**. Gate still reads A5 FAIL |

Commits are local; nothing was pushed (no push approval).

## Update 2026-11-24 — half-turn bug, A5 re-run, S3b force axis (all SIM)

No threshold, tolerance, clamp, pre-registered test or `configs/arm_cert.yaml` line was changed.

| item | evidence class | state |
|---|---|---|
| `rotvec_from_matrix` half-turn bug | SIM / MODEL | **Fixed.** The helper returned `0` for an exact pi rotation (the antisymmetric part is zero there). The FR3 home flange orientation is exactly pi, so the controller's orientation reference collapsed to identity and a bridge plan expressed *relative to home* was applied as absolute (the A7 hand arrived upside down). The axis now comes from the symmetric part `A=(R+I)/2`; round trip `max |R - exp([w])| = 1.5e-15` at home and 1.0e-15 for the 135-degree bridge target. New test `test_rotation_helper_half_turn_is_not_identity`; 17/17 in `tests/test_newton_arm.py`. A5 plans use `rot=0`, which is the correct "hold home" under the fixed convention, so A5 had to be re-run rather than argued |
| A5 definitive re-run after the rotvec fix | SIM | **FAIL**, one run, `reports/g6_newton_push.json` (100,000 steps). PASS `steady_deflection` (worst 6.07 %), `release_return` (overshoot 0.60 %, peak 0.0971 m/s), `plan_pause` (clock frozen 0.500 in YIELD, resumed via settled at 2.428 s, final error **0.249 mm**), `invariants` (0 violations / 39,086 RETURN ticks), `human_hand_proxy` (14.57 N), `release_return_estimated`. FAIL `elbow_push` at the original 15 mm bound: link 4 19.33/39.01/59.16 mm and link 6 19.65/39.63/63.32 mm at 10/20/30 N (both wrench mappings); `payload` and `latency_error` fail **only** through that elbow subrow. **New:** `steady_deflection_estimated` fails at 12.02 % / 11.38 % (0.5 / 1 N noise) vs the unchanged 10 % bound, because the TRACK residual-bias compensation (gain 4) multiplies the estimator noise; that coupling was introduced after D-89. Gate `--require A5` exits 1 |
| A7 S3b force axis + boundary re-measured | SIM | Force estimate is now the actuator model `ke*(ctrl-q) - kd*qdot` (clamped to the hand's 20 N `forcerange`), sampled over the 0.2 s **before** the lift. Measured grasp force at mu 0.9 is 1.52/4.55/7.49/14.31 N at factors 0.1/0.3/0.5/1.0 (nominal 1.53–15.3), so the force axis is live. `reports/a7_s3b_boundary.json`: **still FAIL**, `boundary_matches_analytic_within_one_step=false`, but now **10/12** (mass, mu) rows match (was 11/12 with an uninstrumented force axis). The two misses are both simulator-more-conservative: (0.1 kg, mu 0.1) and (2.0 kg, mu 0.9) never lift although `2*mu*N >= m*g` is met at factor 1.0. Cause measured in the first: with mu 0.1 the box is pushed out during the **closing** transient (grasp force 5.7 N instead of 14.3 N; the box is no longer between the pads), which the quasi-static bound does not model. 2 kg and 3 kg never lift at any factor. Crush: 0.3x peak 4.75 N (no crush), 1.3x peak 19.33 N, below the named 23 N but above the raw 8 N limit; the 20 N actuator range makes the named check unreachable. Re-run kept the old report as `reports/a7_s3b_boundary_pre_force_fix.json` |

## Update 2026-11-24 (evening) — A7 compliant descent root cause (all SIM)

No threshold, tolerance, clamp, profile constant, pre-registered test or `configs/arm_cert.yaml`
line was changed. `configs/approvals.json` stayed `{}`. **A7 is still FAIL: the compliant trials
graded so far are 4 probe seeds, not 200.**

| item | evidence class | state |
|---|---|---|
| A7 compliant descent root cause | SIM | **Found and fixed in the plan, not in the controller.** The Cartesian plan was built through the FK poses of the **raw `solve_ik` iterate path**. That path is not a trajectory: consecutive DLS iterates move 35 % of the remaining error, so near convergence they overshoot (S2 `grasp` path, measured: flange z 0.22043 -> 0.15042 -> 0.15684 -> 0.12116, orientation `rotvec` dipping to `[-0.162, +0.133, 2.358]`). The plan therefore commanded a ~0.19 rad orientation swing in ~0.3 s. With `k_rot=25 Nm/rad` and the unchanged `m_max=4 Nm` cap the wrist lagged up to **0.30 rad** (17 deg), the pads dragged the box and the 30 mm `e_pause` fired (trace: `er_true` 0.008 -> 0.235 rad, box z still 0.9249, i.e. before any contact). `ops/a7_compliant.py:path_plan` now uses only the converged end pose of the path (goal pose and IK tolerances unchanged): a straight Cartesian line, orientation along the geodesic `R0 @ exp(u log(R0^T R1))`, and the null posture reference as a straight joint-space line from the measured posture. Same S-curve clock. `ops/a7_s2_lift_v2.py --trace PATH` records the per-tick diagnosis (default off; the trial result is unchanged) |
| A7 compliant probe after the fix | SIM, **not a rung result** | 4 seeds x 3 masses, `reports/a7_descent_probe.json` (seed 0, 0.5 kg, full trace at `reports/a7_descent_trace.json`) plus `/tmp` reports for seeds 1-3: **4/4 PASS**. Height gain 0.1097 / 0.0995 / 0.1254 / 0.1088 m (threshold 0.08), box xy drift 2.1 / 4.0 / 0.6 / 2.1 mm (bound 20 mm), plan rotation target now constant (`rv_ref` = [0, 0, 2.3557] rad), `er_true` <= 0.10 rad, zero YIELD ticks. A repeat of seed 0 differs by 2e-5 m, so runs are numerically stable but not bit-identical |
| A5 full re-run (item 3) | SIM | **Still running at the time of this update** (`ops/a5_newton_push.py --invariant-steps 100000`, PID 19485, log `reports/a5_full_review3.log`); `reports/g6_newton_push.json` has **not** been rewritten yet, so this update claims no A5 number. The previous report is preserved at `reports/g6_newton_push_2026-11-23.json` |
| A7 grasp rule evaluation on fresh seeds | SIM | **Not run.** The driver (`ops/a7_s2_rule.py eval`) does not yet pass `--compliant`, so any number it produced would be position-servo evidence and cannot be a cert number. The 196/200 remains discarded |
| A7 other six skills through `NewtonCompliantArm` | SIM | **Not run / not wired.** Only `ops/a7_s2_lift_v2.py` has the compliant branch; `place`, `weigh`, `release`, `retreat`, `handover`, `push_aside` still drive MJCF position servos. Handover stays FAIL |

| A7 grading contract | — | The aggregator (`ops/a7_regression_physical.py aggregate`) now counts a trial only if its report's `controller` starts with `NewtonCompliantArm`; non-compliant reports in the same directory are counted in `n_reports`/`excluded_noncompliant` but never in `n`/`passed`. `run --skill ... --compliant` refuses a skill that has no compliant driver (`NO_COMPLIANT_DRIVER`, exit 1) instead of silently running position servos. `ops/a7_s2_rule.py` gained `--compliant` and records the controller in its reports. This tightening cannot create a false DONE |
