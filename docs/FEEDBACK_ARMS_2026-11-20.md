# Feedback on the arms-first run (A1–A7), 2026-11-20

Source of truth for every number below: `reports/*.json` at commit `228ac0b`. State table: `python ops/arm_steps_status.py` (generated, not typed).

## 1. Verdict

| step | verdict | evidence |
|---|---|---|
| A1 kinematics + reach | **DONE** | `g5_reach.json` PASS, <= 1 mm over 100 poses vs Newton FK |
| A2 API findings | **DONE, good** | `docs/findings/newton_arm_api.md`, 10 VERIFIED, honest UNRESOLVED (the MJCF converter needs a package that is absent from the Newton-only venv; Newton-native `add_mjcf` is the route) |
| A3 single-arm smoke | **DONE (rung L1)** | joint-space hold, drift < 1 mm; L1 does not expose actuator effort (stated in the report) |
| A4 wrench / G-7 | **DONE** | impulse = F·dt within 2 % with 4 substeps; the every-substep fix (Isaac Lab PR #8269) confirmed present |
| A5 compliant control | **NOT DONE (FAIL)** | 4 of 8 rows fail, see section 3 |
| A6 scene | **DONE (scene half)** | two FR3 Duo arms + table + bag + 10 objects, 14 DoF, 93 steps/s, deterministic at 1 mm |
| A7 scripted skills | **NOT DONE (STAND_IN)** | 18/18 at **10 trials** on a pose-level state machine (`SIM-POSE`); `arm_cert.yaml` requires evidence class `SIM` and >= 200 trials |
| A8–A10 | not run | blocked by A5 and A7 |

What the agent did right and must keep doing: it wrote `result: FAIL` instead of loosening anything; it recorded rows it had not implemented as "not run, no numbers invented"; it removed a Coulomb feed-forward it measured to be invalid instead of keeping it; it labelled the A7 stand-in `SIM-POSE` and "NOT physical skill outcomes". The A5 rows that did run are real Newton-articulation, torque-mode results: steady deflection worst error 5.6 % (limit 10 %), release-return settle 1.05 s with peak return speed 0.107 m/s (cap 0.11), plan clock frozen while yielding, and 100,000 random-push steps with zero force/moment/leash/torque-rate/return-speed/fault violations.

## 2. What went wrong in the process (fix the process, not only the code)

1. **It went past a failing step.** A5 was FAIL and the agent built A6/A7 anyway. The runbook says do not start a step before the previous one is DONE. From now on run `python ops/arm_steps_status.py --require A<k-1>` before starting step A<k>; if it exits 1, work on the first step that is not DONE. A stand-in is not DONE.
2. **`docs/STATUS_HONEST.md` was not updated** for A1–A7 (its rules say every claim needs a row). A generated block is added by this patch; regenerate it with `python ops/arm_steps_status.py --markdown` and paste it after each step.
3. **The commit message "A6/A7 … reach 18/18" reads like a result.** It is plumbing on a stand-in. Write commit messages as `A7 plumbing (STAND_IN, SIM-POSE, 10 trials): not a certification result`.

## 3. A5: why four rows fail and how to fix each (do these in order)

**Elbow push (`elbow_push`, 6 sub-rows fail).** Measured: end-effector error 20.5 / 40.1 / 59.7 mm for 10 / 20 / 30 N at link 4 (and 20.6 / 41.3 / 71.5 mm at link 6) with K = 500 N/m. That is almost exactly **F/K**, and the link's own displacement (19.9 mm) is almost the same as the end-effector error. Meaning: the elbow push is being transmitted to the end-effector as if it were pushed there; the arm is not yielding in its null space. Cause: the null-space term uses the plain projection `I − Jᵀ(J⁺)ᵀ`, which is **not dynamically consistent** (it ignores the joint inertia), and `k_null` is only 5 Nm/rad. Also the posture does not come back (`final_posture_error` 0.14–0.15 rad, `null_return_time_s` null): 5 Nm/rad × 0.15 rad = 0.75 Nm, which is inside the 0.05–0.6 Nm stiction the agent itself measured.
Fix, in this order:
1. Get the joint-space mass matrix `M(q)`. Newton already has inverse dynamics in the venv (`newton._src.sim.inverse_dynamics` loads in the log). Column `i` of `M` is `ID(q, 0, e_i) − ID(q, 0, 0)` (acceleration = unit vector, velocity 0, subtract the gravity-only call), 7 calls for 7 joints. Verify: `M` symmetric positive definite; compare `qᵀMq̈` energy against a free-swing test.
2. Use the dynamically consistent projection: `Λ = (J M⁻¹ Jᵀ)⁻¹`, `J̄ = M⁻¹ Jᵀ Λ`, `N = I − Jᵀ J̄ᵀ`, `τ = Jᵀ F + N τ_null`. Add `tests/test_newton_arm.py` rows: `N` is idempotent (`N·N ≈ N`) and `J M⁻¹ Nᵀ ≈ 0` to 1e-6.
3. Raise `k_null` to 15–20 Nm/rad (the spec cap is 20) with critical damping; the posture row needs the return torque to exceed stiction.
4. Re-run `elbow_push`. If the end-effector error at 10 N is still ≈ F/K after steps 1–3, the pre-registered bound (<= 15 mm at 10–30 N) is not reachable at that stiffness. Then, and only then, record a decision in `DECISIONS.md` with the measured numbers proposing a stiffness-relative bound (`ee_error <= 0.5·F/K + 2 mm`) and **leave the row FAIL until the user approves**. Do not change the bound silently.

**`payload` (not run).** Rebuild the Newton model per payload: before `builder.finalize()` add a rigid box to the hand with mass 1, 2 and 3 kg (or edit the hand's body mass/inertia in the builder, whichever Newton exposes; look in `ModelBuilder` for `body_mass` / `add_shape_box(density=...)`), run all earlier rows again. The supervisor's identified mass stays 3 kg on purpose (that is the robustness test).

**`latency_error` (not run).** (a) command delay: a ring buffer on the torque command of 1, 2, 3 ticks; (b) model error: scale body masses/inertias and joint friction/armature by 0.8 and 1.2 in the builder before `finalize()`. Run `steady_deflection`, `release_return`, `plan_pause` for the 2 × 3 grid; the row passes if all stay stable and meet the bounds written in the report.

**`human_hand_proxy` (not run).** Two honest levels, report which one you did: (i) `virtual_spring_hand`: an external wrench `F = k_c(x_hand − x_ee) − d_c(ẋ_ee − ẋ_hand)` through the wrench composer with a scripted `x_hand(t)` (no contact geometry; label it so); (ii) contact: a kinematic body with a box shape pushed along a scripted path, reading the contact force. Do (i) first, report peak force.

**Wrench estimator.** The supervisor uses the **injected ground-truth wrench** (the report says so). On the real arm only an estimate exists. Add rows `steady_deflection_estimated` and `release_return_estimated` that feed the supervisor the acceleration-based inverse-dynamics estimate with Gaussian noise (0.5 N, then 1 N) and a first-order filter; if push detection (`F_push` 8 N / `F_release` 4 N) chatters, add hysteresis time to the filter, not a bigger threshold.

When these are done: `python ops/g6_newton_push.py` (or the agent's script) must write `result: PASS`, then `python ops/arm_steps_status.py --require A5` must exit 0.

## 4. A7: what "done" means and the physical-grasp ladder

The current A7 report says "plumbing only". The next work is **physical** skills through `NewtonCompliantArm` in the Newton scene, evidence class `SIM`, >= 200 seeded trials per skill, per-skill success >= 0.90 (`configs/arm_cert.yaml`). The grasp is the hard part; climb this ladder, one rung per commit, each with its own report, and do not skip rungs:

| rung | build | pass |
|---|---|---|
| S1 | hand opens/closes under position control with a force cap; close on a fixed 5 cm box held in a static clamp | finger contact force readable, box does not move |
| S2 | free box on the table: approach with the impedance arm (profile `approach`), close, lift 10 cm, hold 1 s | box height gain >= 8 cm at the end of the hold, in >= 95 % of 50 seeds for masses 0.1–1 kg |
| S3 | vary friction {0.3, 0.6, 0.9} and grasp force `need = 4 + 6·mass` ×{0.7, 1, 1.3} | failure causes match the rules: ×0.7 -> `slip`, high force on fragile -> `crush` (the cause histogram is part of the report) |
| S4 | place into the three bag compartments; release; retreat | object ends inside the compartment volume, hand clear, <= 10 mm penetration |
| S5 | `weigh`: estimate mass from the joint-torque/gravity difference while holding still | mass estimate within 15 % over 0.1–3 kg (this needs effort-mode torque readout, available from A5's L2) |
| S6 | `handover` between the two arms (profile `handover`), `push_aside` | object transferred without ground contact; blocker moved >= 5 cm without toppling |

If grasps do not hold in Newton's MuJoCo-Warp solver, diagnose in this order and write each finding down: (1) contact friction on fingers and object (set both explicitly, use the friction-cone condim that supports torsional friction); (2) finger collision shapes (the Menagerie hand must have usable pads; print `model.shape_*` for the finger bodies); (3) solver settings (`impratio`, iterations, timestep) via the solver cfg; (4) object collision geometry (a convex box first, SimReady meshes only after S3 passes); (5) force control of the finger joints in effort mode instead of position targets. Record every setting you changed and its effect in `reports/a7_grasp_diagnosis.json`. Only after S1–S6 pass do you run the 200-trial regression through the physical backend; the stand-in stays in the repo only as `SIM-POSE` plumbing and is never counted.

## 5. Exact next actions for the agent (in this order)

1. `git pull`, apply this patch, run `python ops/arm_steps_status.py` and read the table.
2. A5: dynamically consistent null-space (section 3, steps 1–3), re-run `elbow_push`; then payload, latency/model-error, hand proxy, estimator rows. Gate: `--require A5` exits 0.
3. Add the generated state block to `docs/STATUS_HONEST.md` after every step.
4. A7 rungs S1–S6 (section 4). Gate: `--require A7` exits 0.
5. Only then A8 (`train/distill_arm.py --synthetic`, then real), A9, A10 exactly as in `docs/AGENT_RUNBOOK.md`.
6. Do not start R4–R9 until `python ops/arm_steps_status.py --require A10` exits 0. R3 (Haiku/Luna bring-up) may run in parallel; its records never go to SFT.
