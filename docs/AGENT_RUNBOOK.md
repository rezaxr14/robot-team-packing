# Agent runbook — ARMS FIRST, then everything else

You are a coding agent working in this repository on the user's machine. Follow the steps IN ORDER. Do not skip a step. Do not start a step before the previous step's "Done when" is true. This file is the instruction; `docs/USER_PLAN_VERBATIM.md` is the user's authority; `docs/AGENT_BUILD_SPEC.md` explains why; `docs/STATUS_HONEST.md` says what is already true.

## 0. Rules you must obey every single time

1. **Arms first (decision D-62, the user's order).** Steps A1–A10 train and certify the arms. Steps R1–R9 (data generation with frontier models, SFT, GRPO, evaluation) come AFTER A10. The only R-step you may run earlier is the Kaggle bring-up on cheap models (R3), and its output is never used for training.
2. **Simulation is Isaac Lab 3.0 + Newton only** (D-25). Use the Newton-only environment: `source ops/env.sh` then `$ISAAC_PY` (it points to the single merged venv `$VENVS/rtp`). Do not install or call any other simulator or renderer.
3. **Never claim more than you measured.** Every script you write must write a JSON report to `reports/` containing: `evidence_class` (`SIM` for Newton runs, `SIM-ABSTRACT` for numpy, `HARDWARE` for the real robot), library versions, seeds, the raw numbers, and `checks` with true/false for each acceptance row, and `result: PASS|FAIL`. Copy the pattern of `ops/g3_sim.py`, `ops/g5_fr3_load.py`, `ops/g6_impedance.py`.
4. **If a step fails, do not weaken the test.** Write the failure into the report, write what you tried into `docs/STATUS_HONEST.md` ("not done: …"), and stop. Never raise a clamp, loosen a tolerance or delete a test to make a check pass.
5. **The Brain (language model) never sets stiffness, speed, force limits or joint values.** Only the skill layer picks a stiffness profile from `rtp/control/compliance.py::PROFILES`.
6. **Do not rewrite code that works.** Extend it behind tests. Files you may extend: `rtp/loop.py`, `rtp/toolbrain.py`, `rtp/skills.py`, `rtp/simworld.py`, `rtp/simbridge.py`, `rtp/flowhead.py`, `rtp/policy_rollout.py`, `rtp/control/*`, `train/*`. Do not touch `paper/`.
7. **One step = one commit** with message `A3: …` / `R2: …`. After each step: update `docs/STATUS_HONEST.md` (what is now true, with its evidence class) and add a line to `DECISIONS.md` if you made a design choice.
8. **Seeds.** Training seeds are >= 100000. Held-out test seeds are < 100000 and are never used for training, memory or prompts.
9. **Before every step run:** `source ops/env.sh && python -m pytest -q` (training venv). It must be green before you start and green when you finish.
10. **Gate before every arm step.** Run `python ops/arm_steps_status.py --require A<k-1>` before starting step A<k>. If it exits 1, work on the first step that is not DONE; skipping ahead is forbidden. `STAND_IN` (a pose-level or fixture stand-in) is never DONE. After each step regenerate the status block with `python ops/arm_steps_status.py --markdown` and paste it into `docs/STATUS_HONEST.md`.
11. If you are unsure about a library function name, **look it up in the local checkout** (`$ISAACLAB_DIR`, and Newton inside `$ISAAC_PY`'s site-packages) with the `grep` commands given in the step. Do not guess names from memory.

## 1. The ten arm steps (A1–A10) at a glance

| step | what you build | proof file | done when |
|---|---|---|---|
| A1 | correct FR3 kinematics + reach test | `reports/g5_reach.json` | Newton FK vs our FK within 1 mm on 100 random poses |
| A2 | API findings for arms in Isaac Lab 3.0 + Newton | `docs/findings/newton_arm_api.md` | all 8 questions answered with file paths |
| A3 | one FR3 stepping in Newton, holds a pose | `reports/a3_arm_smoke.json` | drift < 1 mm over 5 s, gravity held |
| A4 | external wrench injection | `reports/g7_wrench.json` | impulse = F·dt within 2 % with substeps > 1 |
| A5 | compliant control on the articulation | `reports/g6_newton_push.json` | all push rows pass |
| A6 | two-arm packing scene as a `SimBackend` | `reports/a6_scene.json` | scene resets, steps, reports object states |
| A7 | scripted skills pass the regression suite | `reports/skill_regression_scripted.json` | pass rates meet thresholds |
| A8 | expert demos → blind policy (distill) | `reports/skill_regression_learned.json` | learned >= threshold, no regressions vs scripted |
| A9 | push-perturbed evaluation (and residual RL only if needed) | `reports/a9_push_robust.json` | success under pushes meets threshold |
| A10 | certification cards + outcome tables | `reports/arm_cert_<skill>.json` | cards exist, hashes pinned |

Thresholds (write them to `configs/arm_cert.yaml` in A7 BEFORE you look at any result, then do not change them): nominal success >= 0.90 per skill, push-perturbed success >= 0.80, clamp/leash violations = 0, per-skill >= 200 seeded trials for A7 and >= 500 for A10.

---

## A1 — Correct FR3 kinematics and the reach test

**Why.** `rtp/control/impedance.py` uses generic link lengths ("NOT a calibrated FR3 model", line 60). Impedance on a wrong kinematic chain is meaningless. `reports/g5_fr3.json` lists "zero-pose reach test and kinematics agreement within 1 mm" as pending.

**Precondition.** `ls assets/cache/menagerie/franka_fr3_v2/fr3v2.xml assets/cache/menagerie/franka_fr3_duo/scene.xml`. If a file is missing: `python ops/fetch_robots.py --help`, run it as documented, then re-run `$ISAAC_PY ops/g5_fr3_load.py` (must print `PASS`). The cache is local and is not committed.

**Do.**
1. Create `rtp/control/fr3_kinematics.py` with `fk(q) -> (pos[3], rot[3x3])` and `jacobian(q) -> [6x7]` in torch, using this modified-DH table as a *starting hypothesis* (public Franka parameters; you will test it, not trust it):
   `a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]`, `d = [0.333, 0, 0.316, 0, 0.384, 0, 0]`, `alpha = [0, -pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2]`, flange `d = 0.107`. Use the Craig (modified) convention: `T_i = RotX(alpha_{i-1}) · TransX(a_{i-1}) · RotZ(theta_i) · TransZ(d_i)` — be careful that `a, alpha` belong to the previous link.
2. Create `ops/g5_reach.py` (run with `$ISAAC_PY`): load `assets/cache/menagerie/franka_fr3_v2/fr3v2.xml` exactly as `ops/g5_fr3_load.py` does (`builder.add_mjcf(entry)` after `chdir` to the file's folder, `model = builder.finalize()`), then for 100 random joint configurations inside the joint limits: set `joint_q`, run Newton forward kinematics (find the function with `$ISAAC_PY -c "import newton; print([n for n in dir(newton) if 'fk' in n.lower() or 'eval' in n.lower()])"`), read the pose of the last arm body (the link before the hand/flange; print `model.body_key` to find its name), and compare with `fk(q)`.
3. If the position error is > 1 mm: print per-joint offsets, fix the DH table (the usual mistake is the sign of `a4/a5` or the flange offset), repeat. If after honest effort it does not agree, STOP and report; do not switch to a fudge factor.
4. Also record in the report: the MJCF joint limits (`model.joint_limit_lower/upper`) and total mass (already 17.31 kg for v2), marked "datasheet comparison pending".
5. Add `tests/test_fr3_kinematics.py`: analytic checks (zero pose reach = sum of known offsets within 2 mm of Newton's value copied as a constant into the test, Jacobian vs finite difference within 1e-5).
6. Replace the generic chain inside `rtp/control/impedance.py::forward_kinematics` by `fr3_kinematics.fk` behind a flag `ImpedanceConfig.kinematics = "fr3"` (default stays generic until all existing tests pass with `"fr3"`; then make `"fr3"` the default and update `tests/test_impedance.py` expectations only where they encode the old generic lengths — say so in the commit message).

**Done when.** `reports/g5_reach.json` has `max_pos_err_mm <= 1.0` over 100 poses, `result: PASS`; tests green.

**If it fails.** Write `result: FAIL` with the worst pose and offsets. Do not continue to A3.

---

## A2 — Discover the real Isaac Lab 3.0 + Newton API for arms (no guessing)

**Why.** `isaac/DESIGN.md` says accessors changed between Isaac Lab 2.x and 3.x. The only reference that runs is `ops/g3_sim.py` (a cube + camera). You must find, in the local checkout, the exact way to do the arm things below, and write the answers down.

**Do.** Create `docs/findings/newton_arm_api.md`. For each question write: the answer (3–10 lines of code), the file path and line number in `$ISAACLAB_DIR` or the Newton package where you found it, and "VERIFIED by running <script>" only after you actually ran it.

Questions and where to look (run each grep, open the files, read):
1. **Spawn an arm from MJCF/URDF as an Articulation under `physics=newton_mjwarp`.**
   `grep -rn "Mjcf\|Urdf" $ISAACLAB_DIR/source/isaaclab/isaaclab/sim/spawners | head -30`; `ls $ISAACLAB_DIR/source/isaaclab_newton`; `grep -rln "Articulation" $ISAACLAB_DIR/source/isaaclab_newton | head`.
2. **Control mode:** how to command joint **effort** (torque) and how to switch off the built-in position actuator.
   `grep -rn "effort" $ISAACLAB_DIR/source/isaaclab_newton/isaaclab_newton --include=*.py | head -40`; also in Newton: `$ISAAC_PY -c "import newton; print(newton.JointTargetMode.__members__)"`.
3. **Read state:** joint positions/velocities and body poses as torch tensors (look for `.data.joint_pos`, `.data.body_pos_w`, and whether they need `.torch`). `grep -rn "def joint_pos\|body_pos_w\|body_link_pose" $ISAACLAB_DIR/source/isaaclab_newton --include=*.py | head`.
4. **Gravity/bias compensation:** how to get the gravity (+Coriolis) torque for the current state (MuJoCo-Warp `qfrc_bias`? an Isaac Lab method? Newton function?). `grep -rn "qfrc_bias\|gravity_comp\|bias" $ISAACLAB_DIR/source/isaaclab_newton --include=*.py | head`; `$ISAAC_PY -c "import mujoco_warp as m; print([n for n in dir(m) if 'bias' in n or 'qfrc' in n or 'rne' in n])"`.
5. **Jacobian:** is one exposed? If not, you will use `rtp/control/fr3_kinematics.jacobian` (A1). `grep -rn "jacobian" $ISAACLAB_DIR/source/isaaclab_newton $ISAACLAB_DIR/source/isaaclab/isaaclab/controllers --include=*.py | head`.
6. **External wrench on a body** (the "human push"): `grep -rn "permanent_wrench_composer\|instantaneous_wrench_composer\|body_f" $ISAACLAB_DIR/source --include=*.py | head -30`. Also find out if wrenches are applied on EVERY solver substep (look for the PR/commit text "every solver substep" in `$ISAACLAB_DIR` git log: `git -C $ISAACLAB_DIR log --oneline | grep -i wrench | head`).
7. **Parallel environments:** how to clone N copies (`num_envs`) and how `decimation`/substeps are set for a 1 kHz inner loop (`dt`). `grep -rn "decimation\|num_substeps\|substeps" $ISAACLAB_DIR/source/isaaclab_newton --include=*.py | head`.
8. **Contact forces on fingertips:** which sensor works on Newton (the docs say net/friction forces work, contact *poses* raise `NotImplementedError`). `ls $ISAACLAB_DIR/source/isaaclab/isaaclab/sensors/contact_sensor`.

**Done when.** The file exists with all eight answers; for each, a 10-line runnable snippet; items you could not resolve are listed under "UNRESOLVED" with the commands you ran. **Do not start A3 until questions 1, 2, 3 and 6 have VERIFIED answers.**

---

## A3 — One FR3 steps in Newton and holds a pose

**Do.** Create `ops/a3_arm_smoke.py` (run with `$ISAAC_PY`), modelled on `ops/g3_sim.py` (copy its `versions()`, `SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))` usage and report writing).
1. Spawn one FR3 (from the A2 answer 1), fixed base, at the home pose `q_home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]`.
2. **Ladder (do L1 first, only go to L2 if L1 works):**
   * **L1 joint-space hold:** position targets with stiffness/damping from A2 answer 2; step 5 s at the chosen `dt`; record max joint drift.
   * **L2 torque mode:** switch to effort mode, command `tau = gravity_comp(q) + Kp(q_home-q) - Kd dq` where `gravity_comp` comes from A2 answer 4 (if Newton/MuJoCo-Warp exposes none, compute it from the model inertias with a recursive Newton–Euler routine in `rtp/control/fr3_dynamics.py` and test it against Newton's own free-fall behaviour).
3. Report: steps/s, max drift in mm at the end effector, peak torque, whether gravity was held (end-effector sag < 1 mm without the stiffness doing the work: run once with `Kp = 0` and gravity comp only for 1 s; it must not fall).

**Done when.** `reports/a3_arm_smoke.json` `result: PASS`: drift < 1 mm over 5 s in L2 (or L1 only, with `mode: "L1"` written explicitly if L2 was impossible; then A5 must use the L1 ladder rung below).

---

## A4 — External wrench injection (the "human push") and gate G-7

**Do.** Create `ops/g7_wrench.py`:
1. Use the A2 answer 6 method to apply a constant force `F` (try 5, 10, 20, 30 N; directions ±x, ±y, ±z) at the end-effector link and at link 4 (elbow) for 0.5 s on a **free-floating test body** first (a single box with gravity off) and measure its momentum change `Δp`.
2. Pass condition: `|Δp − F·t| / (F·t) <= 0.02`, with the solver substeps set to **at least 4** (this proves the wrench is applied on every substep). Also run with substeps = 1 and write the number.
3. Then repeat on the FR3 with the A3 controller holding and measure the deflection direction/sign is right.

**Done when.** `reports/g7_wrench.json` `result: PASS` for every direction and both link choices. If the error is ~1/substeps, the wrench is only applied on the first substep: record it, find the fix in the checkout (A2 answer 6), do not hide it.

---

## A5 — Compliant control on the articulation (the impedance behaviour you were asked for)

**What it must do.** A person pushes any part of the arm; the arm yields like a spring; the planned trajectory pauses; on release it returns at bounded speed and the trajectory resumes. Spec: `docs/IMPEDANCE_SPEC.md`. The state machine already exists in `rtp/control/compliance.py` (numpy, tested on a point mass, `reports/g6b_compliance.json`).

**Do.**
1. Create `rtp/control/newton_arm.py` with `class NewtonCompliantArm` that each control tick: reads `q, dq`; computes end-effector pose and velocity with `fr3_kinematics`; computes the pose error to `x_ref` from `ComplianceSupervisor.step(...)`; maps the returned 6-vector wrench to joint torque with `tau = Jᵀ F`; adds a weak null-space PD toward `q_home` (`k_null <= 20 Nm/rad`, project with `I − Jᵀ(J⁺)ᵀ`); adds gravity compensation (A3); applies **torque limits** and the **rate limit 1 Nm/ms** from `rtp/control/impedance.py::rate_limit`; writes the torque in effort mode.
   * **Null-space must be dynamically consistent** (the first A5 run used the plain `I−Jᵀ(J⁺)ᵀ` and the elbow push was transmitted to the end-effector, EE error ≈ F/K): get `M(q)` from Newton inverse dynamics (column i = `ID(q,0,e_i) − ID(q,0,0)`), verify `M` symmetric positive definite, then `Λ=(J M⁻¹ Jᵀ)⁻¹`, `J̄=M⁻¹ Jᵀ Λ`, `N=I−Jᵀ J̄ᵀ`, `τ = Jᵀ F + N τ_null`, with `k_null` 15–20 Nm/rad and critical damping. Unit-test `N·N ≈ N` and `J M⁻¹ Nᵀ ≈ 0` (1e-6).
   * **Elbow criterion (rule 6).** If `elbow_push` EE error at 10 N is still ≈ F/K after the null-space fix, the pre-registered ≤ 15 mm bound may be unreachable at that stiffness: write a `DECISIONS.md` entry with the measured numbers proposing a stiffness-relative bound, and **leave the row FAIL until the user approves**. Never edit the bound silently.
   * Orientation error: use a rotation-vector error (`R_err = R · R_refᵀ → rotvec`), never subtract quaternions.
   * The wrench estimate for push detection: in simulation use the injected wrench plus the controller's own contact estimate (`tau_ext_hat = τ_measured − τ_model`, mapped with `J⁺ᵀ`); log both and compare.
   * **Fallback ladder** if A3 only reached L1: use joint stiffness `K_q = Jᵀ K J` (computed each tick) as the joint `ke` with critical damping and `Δq_target` from the plan; the state machine still runs; mark `mode: "L1"` in the report and say so in STATUS.
2. Create `isaac/tests/push_test.py` (run with `$ISAAC_PY`). Rows (each is a function returning numbers + bool):
   * `steady_deflection`: F ∈ {5,10,20,30} N, 6 directions, hold 1.5 s → |δ| within ±10 % of F/K.
   * `release_return`: release after 1 s at δ ≈ 0.10 m → within 5 mm in ≤ 2.5 s, overshoot ≤ 10 % of δ, peak return speed ≤ 1.1 × `v_return`.
   * `plan_pause`: push during a moving plan → plan clock frozen exactly; arm completes the trajectory within 3 mm after resume.
   * `elbow_push`: 10–30 N on link 4 and link 6 → link yields, end-effector error ≤ 15 mm, null posture returns ≤ 3 s.
   * `payload`: 0, 1, 2, 3 kg attached to the hand → all rows still pass.
   * `latency_error`: +1–3 ms command delay and ±20 % inertia/friction error → stable, rows pass with the relaxed bounds written in the report.
   * `invariants`: ≥ 10⁵ steps of random pushes (35 N max, torques ≤ 1.5 Nm, idle gaps 0.5–3 s as in `tests/test_compliance.py`) → zero violations of force cap, leash, torque rate, speed cap in RETURN.
   * `human_hand_proxy`: first a virtual spring-damper hand (wrench through the composer, label it `virtual_spring_hand`), then a kinematic contact body; record the peak force.
   * `*_estimated` rows: repeat `steady_deflection` and `release_return` with the supervisor fed the inverse-dynamics wrench **estimate** (noise 0.5 N then 1 N, filtered) instead of the injected wrench — the real arm only has an estimate.
3. Write `reports/g6_newton_push.json` with every row's numbers and `evidence_class: "SIM"`.

**Done when.** every row `pass: true` AND `python ops/arm_steps_status.py --require A5` exits 0 (rows marked "not run" count as FAIL). Remember: this is simulation evidence, **not** a safety certification; force/speed caps in `ComplianceConfig` are placeholders until ISO 10218-2:2025 is read (hardware step J, later).

---

## A6 — Two-arm packing scene as a `SimBackend`

**Contracts you must implement (already defined, do not change them).** `rtp/simworld.py::SimBackend` has `reset(seed)`, `run_skill(arm, skill, obj, params) -> SkillReport`, `object_states() -> list[ObjState]`, plus `arm_zones`, `payload`, `force_limit`. Outcome categories come from `rtp/simbridge.report_from_states` — do not re-implement success logic. `isaac/packing_sim.py` is a LEGACY Isaac Lab 2.3 attempt that was never run: read it for the scene layout and skill structure, but write the new file `isaac/newton_packing_sim.py` from scratch on the verified API.

**Do.**
1. Scene: `assets/cache/menagerie/franka_fr3_duo/scene.xml` (two arms, 14 DoF; if the arms are not on the island centre line at x = ±0.35 m, add the offset when building: `docs/SCENE_DESIGN.md` D-65), a table top at z = 0.90, a three-compartment bag (laptop slot at the back, main, front pocket; static boxes), and 10 rigid items. Item geometry/mass/friction come from `reports/scene_manifest.json` and `assets/selection_downloaded.json` (SimReady packages ingested by `ops/build_scene.py`); randomise item set/poses/mass/friction from the seed with `rtp.datagen.CLASSES` ranges.
2. `NewtonPackingSim(SimBackend)`: `reset(seed)` rebuilds/reseeds the scene deterministically; `object_states()` returns `ObjState` (pose, mass, fragility, held flag, peak contact force) — contact peaks from the A2 answer 8 sensor or from contact impulse; `run_skill` is implemented in A7.
3. Determinism test: same seed → identical `object_states()` hash after 500 steps.
4. Speed: record steps/s on this GPU in the report (so later steps can plan runtimes).

**Done when.** `reports/a6_scene.json` `PASS`: reset/step/state all work for 20 seeds; determinism passes; an MP4 or PNG sequence of both arms moving to scripted poses is saved under `reports/a6_scene/` (use the viewer/recorder found in A2 or save rendered frames as in `ops/g3_sim.py`).

---

## A7 — Scripted skills (no learning) and the regression suite

**Do.**
1. In `isaac/newton_packing_sim.py` implement `run_skill` for `grasp, place, release, retreat, weigh, handover, push_aside` with **scripted differential IK** (`rtp/control/fr3_kinematics.jacobian`, damped least squares) producing `x_ref(t)` for `NewtonCompliantArm`; pick the stiffness profile by phase from `PROFILES` (`approach` → `grasp` → `transport` → `place` → `retreat`; `handover` for handovers). Grasp force: `need = 4 + 6·mass` N (rule from `rtp/rules`), clipped below the crush limit `simbridge.crush_force_for(fragility)`.
2. Create `isaac/regression_newton.py` adapting `isaac/regression.py::IsaacExecutor` to `NewtonPackingSim` (it maps abstract case objects onto real items). Run `rtp.skills.run_regression(ex, cases, n_trials=200, pass_rate=...)`.
3. Write `configs/arm_cert.yaml` FIRST (thresholds from section 1 above), commit it, then run.
4. Report: `reports/skill_regression_scripted.json` with per-skill pass rate, per-case outcome categories, failure causes (`slip, crush, out_of_reach, no_space, ...`), Wilson intervals (`rtp/stats.py`).

**Physical skills only.** A pose-level state machine on the Newton scene is `SIM-POSE` plumbing and does not count. Climb this ladder, one rung per commit, each with its own report, and do not skip rungs; then run >= 200 seeded trials per skill through `NewtonCompliantArm` with evidence class `SIM`.

Two prerequisites were measured on the way to S1/S2 and must not be rediscovered: (1) the cached Menagerie FR3 v2 / Duo MJCF models are **arm-only** — use the derived `assets/derived/fr3v2_hand.xml` (D-76), a simplified parallel-jaw hand, never call it the Franka Hand; (2) DLS IK targets are in the **arm-base frame**, not world coordinates, and the wrist must be rolled so the finger closing axis is parallel to a box face (D-78).

| rung | build | pass |
|---|---|---|
| S1 | hand opens/closes under position control with a force cap; close on a fixed 5 cm box held in a static clamp | finger contact force readable, box does not move |
| S2 | free box on the table: approach with the impedance arm (profile `approach`), close, lift 10 cm, hold 1 s | box height gain >= 8 cm at the end of the hold, in >= 95 % of 50 seeds for masses 0.1–1 kg |
| S3 | vary friction {0.3, 0.6, 0.9} and grasp force `need = 4 + 6·mass` ×{0.7, 1, 1.3} | failure causes match the rules: ×0.7 → `slip`, high force on fragile → `crush` (cause histogram in the report) |
| S4 | place into the three bag compartments; release; retreat | object ends inside the compartment volume, hand clear, <= 10 mm penetration |
| S5 | `weigh`: estimate mass from the joint-torque/gravity difference while holding still | mass estimate within 15 % over 0.1–3 kg (needs effort-mode torque readout from A5's L2) |
| S6 | `handover` between the two arms (profile `handover`), `push_aside` | object transferred without ground contact; blocker moved >= 5 cm without toppling |

If grasps do not hold in Newton's MuJoCo-Warp solver, diagnose in this order and record each setting and its effect in `reports/a7_grasp_diagnosis.json`: (1) contact friction on fingers and object (set both explicitly; use the friction-cone `condim` that supports torsional friction); (2) finger collision shapes (print `model.shape_*` for the finger bodies; the Menagerie hand must have usable pads); (3) solver settings (`impratio`, iterations, timestep); (4) object collision geometry (a convex box first; SimReady meshes only after S3 passes); (5) force control of the finger joints in effort mode instead of position targets. The stand-in stays in the repo only as `SIM-POSE` plumbing and is never counted.

**Done when.** `python ops/arm_steps_status.py --require A7` exits 0 (class `SIM`, >= 200 trials, pass rate >= 0.90), or the report lists exactly which cases fail and why (then fix the skill, not the threshold).

---

## A8 — Expert demonstrations → blind policy (distillation)

**Do.**
1. Create `isaac/collect_demos.py`: run the scripted skills over ≥ 5,000 randomised cases (seeds ≥ 100000), record per control step: `proprio[15]` (7 joint pos, 7 joint vel, 1 gripper width), `contact[8]` (fingertip forces, contact flags, slip flag), `skill[int]`, `target[int]`, `params[4]`, and the expert action chunk `actions[H, 8]` (7 joint targets + gripper). Save `.npz` with exactly the keys `train/distill_arm.py` expects: `proprio, contact, skill, target, params, actions`. Keep only successful expert rollouts.
2. **First** run the pipeline on synthetic data to prove the trainer works on this GPU: `python train/distill_arm.py --synthetic --head flow --steps 300 --out runs/arm_distill_synth`.
3. Then real: `python train/distill_arm.py --data data/arm_demos.npz --head flow --steps 20000 --out runs/arm_distill`. Compare `--head mlp` as an ablation row.
4. Evaluate the learned policy **in the sim** with `isaac/learned_sim.py` (it uses `rtp/policy_rollout.rollout_skill`; adapt it to `NewtonPackingSim` — its docstring says it was never run) through the same regression suite → `reports/skill_regression_learned.json`; then `rtp.skills.compare(old=scripted, new=learned)`.
5. The learned arm acts only through the bounded-delta impedance interface (clamps identical to the scripted arm).

**Done when.** learned success ≥ threshold per skill and no skill regresses against scripted by more than the interval; otherwise go to A9's residual RL for the failing skills.

---

## A9 — Push-perturbed evaluation (and residual RL only if needed)

**Do.**
1. Add `inject_push(arm, link, wrench, duration)` to `NewtonPackingSim` (A4 method) and an event log `PUSHED / YIELDED / RESUMED / FAULT` (from the supervisor).
2. Evaluate scripted and learned skills with a random push in 30 % of trials (pulse 8–30 N, 0.3–1.5 s, random link, random time inside the skill). Correct behaviour: the skill pauses, resumes, and the final outcome is still correct; a push must not cause crush, a drop, or a clamp violation.
3. If push-perturbed success < 0.80 for a skill: train a **residual SAC** policy (bounded delta ≤ the clamps, stiffness scale in [0.5, 1.5]) on top of the distilled policy **with pushes during training**. Before writing any SAC code, read `$ISAACLAB_DIR` for an existing SAC/RL runner you can reuse (`grep -rln "sac\|SAC" $ISAACLAB_DIR/source --include=*.py | head`); write your finding in `docs/findings/newton_arm_api.md` under "RL runner". Do not start SAC if A8 + pushes already pass.
4. Report `reports/a9_push_robust.json`.

**Done when.** every skill ≥ 0.80 push-perturbed, zero clamp/leash violations.

---

## A10 — Certify the arm and export the outcome tables

**Do.** Create `ops/arm_certify.py`:
1. For each skill run ≥ 500 seeded trials per condition grid: object mass {0.1, 0.5, 1, 2, 3 kg} × friction {0.3, 0.6, 0.9} × fragility {0, 0.5, 1} × grasp force {need·0.7, need, need·1.3} × position noise {0, 5, 10 mm}, nominal and push-perturbed (use a fixed sample of the grid if the full product is too slow; write which).
2. Write `reports/arm_cert_<skill>.json`: success rate per cell with Wilson intervals, failure-cause histogram, peak force/deflection/yield/resume statistics, compliance rows (A5), code git SHA, versions, `arm_model: "newton_cert_<8-hex hash of the card>"`.
3. Export `skill_tables/<skill>.json`: for each parameter cell, the empirical distribution of `{ok, cause, peak_force, wall_s}` (the table the CPU episodes will sample from). Pin the hash in `skill_tables/manifest.json`.
4. Update `rtp/datagen_ingest.py::sft_eligible` call sites to pass the certified `arm_model` set read from `skill_tables/manifest.json`.

**Done when.** all cards exist, thresholds met, manifest written, `docs/STATUS_HONEST.md` says which skills are certified and at what evidence class (`SIM`). **Only now may Steps R4–R9 start.**

---

# After the arms

## R1 — Scene bundle and eye view library
Use `reports/eye_placement.json` (chosen layout: eyes on a ring R = 1.6 m, H = 2.2 m, 12 stations; two eyes opposed). Create `ops/render_views.py` (`$ISAAC_PY`, API from `ops/g3_sim.py`): for each scene seed render rgb+depth for stations × zoom {1, 2, 4} × a pan/tilt grid, with arms in their start pose and in a reach pose; write `scene_<seed>/views/*.jpg`, `visibility.json`, `objects.json` and a `manifest.json` with sha256 of everything and the Newton/Isaac Lab versions. Acceptance: ≥ 10 fps per eye at 640×480 (G-8, write the measured number), bundle hash stable across two runs of the same seed.

## R2 — Bundle-backed backend for the CPU episodes
Create `rtp/env/bundle_backend.py` implementing `SimBackend`: `run_skill` samples the outcome from `skill_tables/<skill>.json` given the call's parameters (seeded, so replay is deterministic); `object_states` evolves from the outcomes; eye tools answer from the cached views (nearest station/zoom). Extend `rtp/datagen_run.py::run_one` with a `backend` argument so episodes can use it. Tests: the backend satisfies the existing `SimBackend` tests (`tests/test_simworld.py` pattern) and replay verification still passes (`rtp/datagen_ingest.py::verify_record`). Acceptance: for 100 seeds, outcome frequencies from the bundle backend match a direct Newton replay within the tolerance you write down in the report.

## R3 — Kaggle bring-up on cheap models (allowed before A10; output never used for SFT)
```bash
python ops/datagen_iterate.py build --iteration 0 --shard haiku-it0 --seeds 100000:100010 --models claude-haiku-5-5
git push                      # the hosted worker clones the default branch
kaggle benchmarks tasks push rtp-multiturn-haiku-it0 -f ops/kaggle/multiturn/task.py
kaggle benchmarks tasks run  rtp-multiturn-haiku-it0 -m claude-haiku-5-5
kaggle benchmarks tasks download rtp-multiturn-haiku-it0 -o runs/datagen/downloads -m claude-haiku-5-5 -s -f
python ops/datagen_iterate.py ingest --shard haiku-it0 --iteration 0 --episodes <downloaded>/episodes.jsonl
```
Repeat with `-m gpt-5.6-luna` (resolve the exact slug with the skill's model list). Pass criteria (DG-1 in `docs/DATAGEN_KAGGLE.md`): manifest `status: ok`; ≥ 90 % of turns parse; `finish_reason` is not `length`; replay verification 100 % on genuine records; cost per episode within 2× of the table in the doc. Local access goes through `export RTP_HTTP_PROXY=http://localhost:10808`.

## R4 — Scale-up data generation (after A10 and R2)
Set `--arm-model newton_cert_<hash>` in `build`. Run iterations: (cheap models, 100 seeds each) → ingest → memory → next iteration → … then strong models (Opus 5.5, Sonnet 5.5, Gemini 4 Argon / 3.8, Astra) on hard-case slices. Each iteration: `build` → push/run/download → `ingest`. Keep `runs/datagen/ledger.jsonl`; check `python ops/datagen_iterate.py status`. Target: ≥ 5,000 verified episodes over ≥ 3 models (adjust after the ledger shows cost per accepted example). Include a share of push-interruption episodes (A9 events inside episodes via the bundle backend).

## R5 — Curate and build the SFT dataset
Create `rtp/curate.py` and `rtp/sft_format.py` (spec in `docs/AGENT_BUILD_SPEC.md` WP-G and §2.5): keep verified successes, verified failures with attributed cause (as negatives), and failure-then-successful-replan episodes (most valuable); drop duplicate plan patterns; cap each model's share; split by seed/scene; exclude any record whose `arm_model` is not in `skill_tables/manifest.json`; write `data/sft/<hash>/{train,val}.jsonl` and `docs/DATASET_CARD.md`. Tests: no test seed (< 100000) appears anywhere; hash recorded.

## R6 — SFT (Qwen3.5-4B)
Read `train/sft_brain.py` and `rtp/trainer.py` first (print their `--help`). Use your machine's profile from `rtp/profiles.py` (`local_4050` QLoRA with the 256-token ceiling, `local_4090` BF16 LoRA, Kaggle T4 fp16). Train on `data/sft/<hash>/train.jsonl` with loss on assistant tokens only, validate on `val.jsonl` (format-valid rate, plan-pass rate). Write a run card `reports/runs/<run_id>.json`. Acceptance: validation plan-pass rate and format-valid rate reported with intervals; the adapter hash is in the card.

## R7 — GRPO (k = 4 and k = 8, KL to SFT)
Read `train/grpo_brain.py`. Extend it so a rollout is a full multi-turn episode on the bundle backend (R2); a group is k samples of the same seed; advantage = reward − group mean (normalised by group std); reward = success + rules satisfied + plan validity − eye cost − length; resuming while `YIELDED` is penalised; format validity is a gate. Keep the frozen merged SFT model as the KL reference; log KL every step and **stop** when the budget in the profile is exceeded. Run k = 4 and k = 8 as two runs. Each run: run card + the top-reward rollouts printed for manual review (reward-hacking check).

## R8 — Evaluation matrix M1–M8 and industry comparison
Create `ops/eval_matrix.py` that reads run cards and prints the tables. Rows (D-64): M1 base 4B zero-shot, M2 base 9B zero-shot, M3 4B SFT, M4 SFT+GRPO k=4, M5 SFT+GRPO k=8, M6 frontier zero-shot, M7 frontier in harness, M8 base 4B in harness. Test set: held-out seeds < 100000, ≥ 30 per family (packing, heavy/fragile, occlusion-heavy, distractors, handover, push-interruption, kitchen-object tasks, unseen layouts/objects). Frontier rows: hosted tasks with `mode="eval"` (`ops/kaggle/multiturn/build_task.py --mode eval`). Industry VLA baselines (LeRobot: ACT, Diffusion, π0, π0.5, SmolVLA, GR00T): export demonstrations with a new `ops/export_lerobot.py` (LeRobotDataset v3) from the certified scripted arm in the bundle/Newton env, fine-tune with LeRobot's defaults, evaluate on the same seeds with the same `final_check`. State in the report that VLAs act on low-level end-effector deltas through the same bounded impedance interface while the Brain plans skills.

## R9 — Ablations and report
Run the ablation rows in `docs/AGENT_BUILD_SPEC.md` WP-I (one eye vs two, memory off, verifier off, adapter coverage, GRPO k, certified vs stand-in arm). Generate every table from run cards. No number is typed by hand.

---

## J — Hardware (parallel, do not block on it)
Order two FR3 arms + Franka Hand; read ISO 10218-1/-2:2025 and replace the placeholder caps in `ComplianceConfig`; write the C++ copy of the impedance law (`cpp/rtp_impedance/`) with a golden-vector parity test against `rtp/control/compliance.py`; hardware push tests with a force gauge only after a written risk assessment. Nothing from this track is claimed until it has run.

## Reporting template (copy into `docs/STATUS_HONEST.md` after each step)
```
### <step id> — <title> (<date>)
evidence class: SIM | SIM-ABSTRACT | HARDWARE | FIXTURE
what is now true: <one sentence with the number>
report: reports/<file>.json (result PASS|FAIL)
not done / known limits: <list>
```
