# Handoff for review — two-arm certification, 2026-11-24

**Purpose.** This file is a complete, self-contained status report written for an outside reviewer
("Claude"). The operator will read your feedback and relay it. Nothing here is new evidence: every
number is copied from a file under `reports/`, and each row states its evidence class. Where
something was not run, it says **not run**.

**One-line state.** A1–A4, A6 are DONE; **A5 is FAIL** on four rows (elbow bound, its two dependent
rows, and one estimated-deflection row), and the definitive 2026-11-25 re-run reproduces the previous
run **exactly**; **A7 is FAIL** with 0 certifiable trials, but the compliant-descent bug that blocked
it is now fixed and probes pass; A8–A10, Kaggle, SFT/GRPO, the industry eval and the paper are
**not run**. The ladder is blocked at A5 by a **criterion decision only the operator can make**
(D-79).

---

## 1. Repository state

| item | value |
|---|---|
| repo | `github.com:rezaxr14/robot-team-packing`, branch `main` |
| HEAD | `9f64984` — **pushed**; `main` is level with `origin/main` |
| commits in this push | `741bad3 … 9f64984` (11 commits, `c103e62..9f64984`) |
| evidence committed with this revision | `reports/g6_newton_push.json` (new A5 run), `reports/g6_newton_push_2026-11-23.json` (preserved previous A5 report), `reports/a5_full_review3.log`, `reports/a7_probe_2026-11-24/`, `docs/USER_PROMPTS_VERBATIM.md` (U062/U063/U064) |
| A5 process | **finished 2026-11-25 00:53**; `reports/g6_newton_push.json` was rewritten (100,000 steps, 7,496.8 s) and every measured number equals the 2026-11-23 run; gate `--require A5` exits 1 |

Authority order (do not reorder): `docs/USER_PLAN_VERBATIM.md`, `.agent/RULES.md`,
`docs/AGENT_RUNBOOK.md` (including "Addendum 2026-11-22"), `IMPLEMENTATION_PLAN.md`, `DECISIONS.md`.
`docs/STATUS_HONEST.md` is the only truth source for state.

Standing constraints: no threshold / tolerance / clamp / pre-registered test changes without the
operator; `configs/arm_cert.yaml` frozen except the authorized `tuning_seeds_min: 900000`;
`configs/approvals.json` is `{}`; no push without per-push approval; one step = one commit; every
number carries an evidence class; simulation is not safety (D-22) — never claim the arms are safe
around people.

---

## 2. Ladder

| step | state | why |
|---|---|---|
| A1–A4 | DONE | PASS / 10 VERIFIED answers / PASS |
| A5 | **FAIL** | `elbow_push` fails the original 15 mm bound at both wrench mappings; `payload` and `latency_error` fail only through that same subrow; `steady_deflection_estimated` fails 12.02 % / 11.38 % vs 10 % |
| A6 | DONE | PASS |
| A7 | **FAIL** | 0 skills have 200 compliant graded trials. Six skills have no compliant driver at all |
| A8–A10 | NOT RUN | gated on A7 |

Gate command: `python ops/arm_steps_status.py --require A5` (and `--require A6`, …). It exits 1 at A5.

---

## 3. A5 — the open criterion (this is the decision)

### 3.1 What ran, and reproducibility

`source ops/env.sh && $ISAAC_PY ops/a5_newton_push.py --invariant-steps 100000 --out
reports/g6_newton_push.json`, 2026-11-24 22:17 -> 2026-11-25 00:53. The run took 7,496.8 s of wall
time (13.3 steps/s) instead of 7,677.6 s, because two A7 compliance probes shared the GPU for part
of it; every criterion is sim-time based. **Every measured value matches the 2026-11-23 run to the
reported precision** — including the four failing rows. Only `wall_s` differs. So the harness is
reproducible and the FAIL is a design/criterion result, not a flaky measurement. The previous
report is preserved at `reports/g6_newton_push_2026-11-23.json`, the log is
`reports/a5_full_review3.log`.

### 3.2 The ten rows (2026-11-25 run; identical to 2026-11-23)

Source: `reports/g6_newton_push.json` (previous: `reports/g6_newton_push_2026-11-23.json`),
100,000 simulated steps, 7,496.8 s wall, software versions pinned inside the report.

| row | state | measured |
|---|---|---|
| `steady_deflection` | PASS | 24/24 subrows, worst 6.07 % vs the 10 % bound |
| `release_return` | PASS | release delta 96.8 mm, settle 1.104 s, final 0.95 mm, overshoot 0.60 %, peak 0.0971 m/s < 0.11 cap |
| `plan_pause` | PASS | clock frozen at 0.500 in YIELD, `RESUMED err=0.000856 via=settled` at 2.428 s, final tracking error **0.249 mm** |
| `invariants` | PASS | 100,000 steps, 0 violations in 39,086 RETURN ticks |
| `human_hand_proxy` | PASS | 14.57 N virtual-spring peak, 59.7 mm excursion |
| `release_return_estimated` | PASS | same as above through the estimator |
| `elbow_push` | **FAIL** | see 3.3; certified mapping is the dynamic one, the flange-equivalent is within 0.25 % |
| `payload` | **FAIL** | 4 cells (0/1/2/3 kg); each cell's own steady/release/plan checks pass, it fails only through the `elbow_push` subrow |
| `latency_error` | **FAIL** | 8 cells (delay 0-3 ticks, mass and armature +/-20 %); fails only through the `elbow_push` subrow |
| `steady_deflection_estimated` | **FAIL** | 12.02 % (0.5 N estimator noise) and 11.38 % (1 N) vs 10 % |

So **four of the ten rows are FAIL** (6 PASS), and all four reduce to **two independent
questions**: the elbow bound (3.3, which also carries `payload` and `latency_error` through its
subrow) and the estimator-noise coupling (3.5).

### 3.3 Both elbow wrench mappings vs the original 15 mm bound

Two mappings were measured on the same body pushes, same seeds, same controller:
(a) the flange-equivalent of the body push, `(J_f⁺)ᵀ J_bᵀ F`, and (b) the dynamically consistent
task wrench `Λ J M⁻¹ τ_ext`. Source: `reports/a5_elbow_two_mapping.json` (4,000 steps, SIM).

| link | force | (a) flange-equivalent | (b) dynamic | 15 mm bound |
|---|---|---|---|---|
| `fr3v2_link4` | 10 N | 19.355 mm | 19.372 mm | FAIL |
| `fr3v2_link4` | 20 N | 38.899 mm | 38.935 mm | FAIL |
| `fr3v2_link4` | 30 N | 58.870 mm | 58.918 mm | FAIL |
| `fr3v2_link6` | 10 N | 19.732 mm | 19.699 mm | FAIL |
| `fr3v2_link6` | 20 N | 39.802 mm | 39.709 mm | FAIL |
| `fr3v2_link6` | 30 N | 61.295 mm | 61.131 mm | FAIL |

The two mappings differ by ≤ 0.25 %. The full A5 run reports the same numbers (19.33/39.01/59.16 and
19.65/39.63/63.32 at 10/20/30 N). **The dynamic mapping does not rescue the 15 mm bound**, so D-79
(the proposed criterion change) cannot be withdrawn on physics grounds.

### 3.4 Why 15 mm is unreachable at this stiffness

The harness maps a body push to a flange-equivalent wrench and feeds it to the compliance
supervisor. The task spring then **deliberately** yields `F/K_lin` at the flange. With
`K_lin = 500 N/m` (`grasp` profile) and 10/20/30 N that is 20/40/60 mm — the measured values almost
exactly. A null-space projector cannot cancel a command the controller itself issued; the measured
`|EE(b) − EE(a)|` above confirms it. So the original absolute 15 mm bound is incompatible with the
pre-registered 500 N/m task spring for 10–30 N pushes. That is a **criterion** question, not a bug.

D-79 (proposed, **not approved**, no `approvals.json` entry) would replace it with a
stiffness-relative bound `F/K_lin + 3 mm` (28.8/53/78 mm at 10/20/30 N for 500 N/m), plus link
displacement > 1 mm, post-recovery EE error ≤ 5 mm and zero cap/leash/rate violations. The
`final_posture_error_rad` (0.135–0.569 rad) is recorded as a diagnostic only, because a 7-DoF arm
with a 6-DoF task has a 1-D null space so exact `q_home` recovery is not defined by the plan.

### 3.5 The estimator-noise coupling

`steady_deflection_estimated` fails at 12.02 % / 11.38 % because the TRACK residual-bias
compensation (`residual_comp_gain = 4`, `residual_beta = 0.1`) multiplies the estimator noise. That
coupling was introduced after D-89; the gain is a **controller constant**, not a test. Making this
row pass requires a full A5 re-run with a different gain — i.e. another authorization.

### 3.6 The options (nothing has been applied)

| option | consequence |
|---|---|
| A. Keep the 15 mm bound | A5 stays FAIL permanently as currently designed; A7–A10 cannot be certified; no code change |
| B. Operator adds a D-79 entry to `configs/approvals.json` | the row is re-scored against the stiffness-relative bound; A5 can become DONE if the other rows pass; requires a full re-run to record the new criterion in the report |
| C. Change `PROFILES["grasp"].K_lin` (e.g. 2000 N/m) | 30 N → 15 mm exactly at 2000 N/m — but that is a controller-constant change, needs authorization, and may break the human-hand/deflection rows |
| D. Keep the bound and change what `elbow_push` measures (joint displacement + recovery) | a criterion change like B; needs approval |

---

## 4. A7 — what was actually wrong

### 4.1 Root cause (found, measured, fixed)

The compliant bridge built its Cartesian plan through the FK poses of the **raw `solve_ik` iterate
path**. `solve_ik` returns its DLS iterations, each moving 35 % of the remaining error, so near
convergence the path overshoots. Measured for the S2 grasp path (`--seed 0 --mass 0.5`):

```
waypoint  flange z (m)   rotvec rel. home (rad)
0         0.22043        [-0.0016,  0.0013,  2.3555]
1         0.15042        [-0.1617,  0.1326,  2.3578]
2         0.15684        [-0.0872,  0.0232,  2.3610]   <- goes back up
3         0.12116        [ 0.0039, -0.0305,  2.3473]
4         0.12066        [ 0.0001, -0.0006,  2.3557]
```

The plan therefore commanded a ~0.19 rad orientation swing in ~0.3 s. With `k_rot = 25 Nm/rad` and
the unchanged `m_max = 4 Nm` cap, the wrist lagged up to **0.30 rad (17°)** *before any box
contact*; the pads then dragged the box and the 30 mm `e_pause` fired. Trace (`reports/
a7_descent_trace.json`, sampled every 10 ticks): `er_true` 0.008 → 0.235 rad while box z stayed
0.9249, then `State.YIELD` at s≈6.64, `RETURN` parked at z≈0.14. This is why **0 compliant graded
trials** existed.

Important nuance: the box was *not* the cause. The rotation error grew with descent speed, the
commanded moment was pegged at the 4 Nm cap, and the null-space reference error (`er_null`) stayed
≤ 0.06 rad — the posture PD was not fighting the task spring.

### 4.2 The fix

`ops/a7_compliant.py:SkillArm.path_plan` now uses **only the converged end pose** of the path (goal
pose and IK tolerances unchanged):

* position: straight line from the measured pose to `FK(path_q[-1])`;
* orientation: geodesic `R0 · exp(u · log(R0ᵀ R1))` (π-robust helper);
* null posture reference: straight joint-space line from the measured posture to `path_q[-1]`;
* same S-curve clock, so the supervisor still sees a zero-velocity start/end.

`ops/a7_s2_lift_v2.py --trace PATH` was added (default off) to record per-tick pose error, plan
target, torque components and box height.

### 4.3 Probe results (SIM — **not a rung result, n=5**)

| trial | evidence file | result |
|---|---|---|
| grasp seed 0, 0.5 kg | `reports/a7_descent_probe.json`, `reports/a7_probe_2026-11-24/grasp_probe_s0_m0.5.json` | PASS, gain 0.10966 m, drift 2.1 mm |
| grasp seed 1, 0.1 kg | `reports/a7_probe_2026-11-24/grasp_probe_s1_m0.1.json` | PASS, gain 0.09948 m, drift 4.0 mm |
| grasp seed 2, 1.0 kg | `reports/a7_probe_2026-11-24/grasp_probe_s2_m1.0.json` | PASS, gain 0.12544 m, drift 0.6 mm |
| grasp seed 3, 0.5 kg | `reports/a7_probe_2026-11-24/grasp_probe_s3_m0.5.json` | PASS, gain 0.10875 m, drift 2.1 mm |
| place seed 0, 0.3 kg | `reports/a7_probe_2026-11-24/place_probe_s0_m0.3.json` | PASS, compartment_1, 0.0 mm penetration, hand clear |

Probe seeds 0–3 are deliberately neither the tuning split (≥ 900000) nor the evaluation split
(≥ 100000), so they cannot contaminate either. In the descent the plan's rotation target is now
constant (`rv_ref = [0, 0, 2.3557]`), `er_true ≤ 0.10 rad` and there are zero YIELD ticks.

### 4.4 Grading contract (D-99)

`ops/a7_regression_physical.py aggregate` now counts a trial only if its report's `controller`
starts with `NewtonCompliantArm`; other reports in the same directory are listed as
`excluded_noncompliant`. `run --compliant` refuses a skill with no compliant branch
(`NO_COMPLIANT_DRIVER`, exit 1) instead of silently running servos. This is a **tightening**: it can
only turn a PASS into a FAIL, never the reverse. The old 196/200 position-servo number is discarded.

### 4.5 S3b (item 5) and crush

`reports/a7_s3b_boundary.json`, 48 cells + 2 crush probes, force axis now the actuator model
`ke·(ctrl−q) − kd·q̇` sampled 0.2 s before the lift (1.52/4.55/7.49/14.31 N at factors
0.1/0.3/0.5/1.0 vs nominal 1.53–15.3 N). **FAIL**: `boundary_matches_analytic_within_one_step =
false`, but **10/12** (mass, μ) rows match `2μN ≥ mg` within one grid step (was 11/12 before the
force fix). Both misses are simulator-more-conservative: (0.1 kg, 0.1) and (2.0 kg, 0.9) never lift
although the analytic bound is met at factor 1.0; the first was traced to the box being pushed out
during the **closing** transient (grasp force 5.7 N instead of 14.3 N), which a quasi-static bound
does not model. 2 kg and 3 kg never lift at any factor.

Crush: 0.3× peak 4.75 N (no crush); 1.3× peak 19.33 N — **below** the named
`crush_force_for(fragility=1) = 23 N`, **above** the raw `rtp.world.crush_limit = 8 N`. Because the
hand's actuator range is ±20 N, the named 23 N check is **structurally unreachable**; the harness
reports it honestly as false.

### 4.6 What A7 still needs

1. `ops/a7_s2_rule.py eval --compliant --seeds 210` on fresh seeds ≥ 100000 → the single lift rule
   `lift_delta_for_mass(m) = 0.10 + 0.04·clip(m,0,1)` evaluated once, ~1.5–3 h. **Not run.**
2. The same rule should first be confirmed on the tuning split through the compliant controller
   (the existing 39/40 tuning is position-servo evidence).
3. Compliant branches for `weigh`, `release`, `retreat`, `handover`, `push_aside` (grasp and place
   are done). `handover` has two arms and needs two runners — the largest job.
4. Then `a7_regression_physical.py run --skill <s> --compliant` × 200 each, `aggregate`, and
   `--require A7`.

---

## 5. Known risks / caveats

1. **Numerical reproducibility.** Two runs of the identical grasp trial (seed 0, 0.5 kg) differ by
   2e-5 m in height gain. Outcomes are stable; the runs are not bit-identical. If "deterministic
   seed" is meant bit-exactly, this needs a look (GPU solver).
2. **GPU contention.** Two Newton processes share one RTX 4050 and each runs at ~50 % throughput.
   The A5 re-run's `wall_s` will therefore be inflated; all A5 checks are sim-time based, but the
   recorded wall time will not match the 7,677.6 s of the 2026-11-23 run.
3. **Residual compensation gain 4.** In free space `f_ext = 0`, so the residual is the spring force
   itself and the correction is a ×5 positive feedback, limited by `f_lin_max`/`m_max`. It is
   responsible for the `steady_deflection_estimated` failure and it interacts with the 4 Nm moment
   cap during fast descents. It is a controller constant; not touched.
4. **4 Nm moment cap** (`ComplianceConfig.m_max`, marked PLACEHOLDER in the source) is what limits
   orientation tracking during the descent; the wrist needed ~6 Nm at 0.235 rad error and was
   capped at 4 Nm.
5. **The straight-line plan assumes no obstacle.** `place` now moves straight from the pre-pose to
   the place pose. The bag has a wall; seed 0 clears it, but a straight line may clip the wall on
   some seeds. A path that keeps the IK postures but smooths them (rather than discarding them)
   would be safer. This is an open design question (Q3).
6. **Simulation is not safety.** Nothing here is a safety result; no ISO 10218 limit has been read;
   no hardware.

---

## 6. Questions for the reviewer

1. **D-79.** Is the stiffness-relative bound `F/K + 3 mm` the right resolution, or is the real error
   that an elbow push is being mapped to a flange wrench at all? A joint-space elbow metric
   (displacement + recovery) might be more faithful than either flange bound.
2. **A5 `steady_deflection_estimated`.** Should the residual-bias gain be reduced (controller change,
   another full re-run) or should the estimated row be reported as informational while the injected
   row gates?
3. **Plan shape.** We discarded the IK iterates and use a straight Cartesian line + geodesic
   orientation. For obstacle-free grasps that is clean; for `place` (bag wall) is a smoothed
   posture-aware path (e.g. fit a quintic through decimated, monotone waypoints) better? We have not
   seen a failure yet, but one seed is not evidence.
4. **4 Nm cap vs 25 Nm/rad.** `m_max = 4 Nm` is a placeholder; the grasp profile can only correct
   `4/25 = 0.16 rad` before saturating. Is increasing it (a safety-limit question, not a tuning
   question) the right fix, or should the task spring be softer?
5. **Determinism.** Is 2e-5 m run-to-run acceptable for a "deterministic seed" claim, or does
   certification need a bit-exact replay?
6. **Crush.** The named threshold (23 N) exceeds the actuator range (20 N), so it can never fire.
   Should the card report "not crushable within the actuator range" instead of a false check?
7. **Gate ordering.** The runbook requires `--require A<k-1>` before each step, so A7 certification
   is blocked behind A5's FAIL — even though the A7 work is independent. Is that intended, or should
   A7 evidence be allowed to accumulate while the A5 criterion decision is pending?

---

## 7. Exact commands

```bash
cd ~/Projects/robot_team && source ops/env.sh

# gate
$ISAAC_PY ops/arm_steps_status.py --require A5     # exits 1 today

# A5 full run (~2.1 h)
$ISAAC_PY ops/a5_newton_push.py --invariant-steps 100000 --out reports/g6_newton_push.json

# A7 compliant probes
$ISAAC_PY ops/a7_s2_lift_v2.py --compliant --seed 0 --mass 0.5 \
    --trace reports/a7_descent_trace.json --out reports/a7_descent_probe.json
$ISAAC_PY ops/a7_s4_place.py --compliant --seed 0 --mass 0.3 --out reports/a7_s4.json

# A7 single-rule evaluation (NOT yet run; fresh seeds, ~1.5-3 h)
$ISAAC_PY ops/a7_s2_rule.py eval --compliant --seeds 210

# A7 grading (only compliant trials count now)
$ISAAC_PY ops/a7_regression_physical.py aggregate
```

---

## 8. Not done (explicit)

* Nothing further on A5: the re-run landed and reproduces the previous result; the four FAIL rows need the criterion decisions in section 3.6.
* 200-trial compliant grasp evaluation; compliant confirmation on the tuning split.
* Compliant branches for weigh / release / retreat / handover / push_aside.
* 7-skill grading, A8, A9, A10.
* Kaggle R3 (needs the local proxy `http://localhost:10808`, never HF; Kaggle records never used
  for SFT), Qwen3.5-4B SFT/GRPO, industry eval, paper.
* R4+ (requires `--require A10` to exit 0).
