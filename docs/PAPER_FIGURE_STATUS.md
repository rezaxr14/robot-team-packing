# Figure remake requests (for the figure author)

These are **not** edits to the SVGs/PDFs: the drawings are untouched. Each one has a short red label
immediately below it in the LaTeX source (`\stalefig`), so the request is visible in the PDF and can
be removed once the figure is redrawn. Current inventory (2026-11-19): **four** red notes —
`fig_signals`, `fig_bridge` (existing), `fig_overview`, `fig_design` (new). No other figure carries
one: the remaining figures either are explicitly labelled FIXTURE/synthetic/pending or show no
stale role/stack claim.

When a figure is redrawn, align it with the current design: arms-first order and the two-arm,
two-eye kitchen (`docs/AGENT_RUNBOOK.md`, `docs/SCENE_DESIGN.md` D-65), the compliance supervisor
(`docs/IMPEDANCE_SPEC.md`, D-66), and the exact M1–M8 benchmark matrix (`IMPLEMENTATION_PLAN.md` §10,
D-64). Evidence classes are authoritative in `docs/STATUS_HONEST.md`; no class may be pooled into a
paper number.

## `fig_signals.svg` — stale

The Eye row says **"Eye (VLM)"** and **"LoRA fine-tuning"**. That was the old design. The Eyes are
conventional CV — a detector plus OpenCV geometry, never a language model (D-01, D-02).

Remake that row as:

> rendered table images + SCENE JSON targets → conventional detector → credit-weighted selection
> among the detector's candidate readings

Also name the Brain row **Qwen3.5-4B** (one model commanding both arms and both eyes) and the Arm
row **FR3 + Franka Hand**. The reward formula in the lower panel is read from the code and stays
as it is.

## `fig_bridge.svg` — stale

The physics column and the quaternion note come from the old stack. The simulator is **Isaac Lab
3.0 + Newton (`mujoco-warp`)**, with **no Isaac Sim and no PhysX** (D-25, D-39). Gate G-3 is
measured on the RTX 4050: 30 steps at 21 steps/s with rgb + depth contracts.

Remake the column as "Isaac Lab 3.0 + Newton", drop the "Isaac Lab 2.x" half of the quaternion note
(only the 3.0 convention is used), and draw the two Eyes as pan-tilt cameras on a ring rail rather
than fixed points. Keep the "one success definition across three worlds" structure and the
"validates the integration, not the physics" caveat.

## `fig_overview.svg` — stale (new note, 2026-11-19)

The lower band still assigns the old roles: the Eye box says **"LoRA fine-tuning"** and the Brain is
framed as a text-only planner. The Eyes are conventional CV (detector + OpenCV, never a language
model), and the Brain is **one Qwen3.5-4B vision-language model** that takes typed claims only (no
pixels) and commands both arms and both Eyes (D-05, D-06, D-52).

Remake the lower band as:

> Eye: rendered table images + SCENE JSON targets → detector training → credit-weighted selection
> among candidate readings (no LoRA). Brain: Qwen3.5-4B, typed claims in, plan out. Arms: FR3 +
> Franka Hand.

Keep the typed-bus, message-stream and force-window panels; they match the text.

## `fig_design.svg` — stale (new note, 2026-11-19)

The evidence-class column still uses the pre-measurement classes: the **ISAAC** row is drawn as if
Isaac Lab manipulation episodes exist. The measured items are the falling-cube rgb/depth gate G-3
(`reports/g3_sim.json`, `reports/g3_sim_t4.json`), the simulation-only impedance cross-check G-6
(`reports/g6_impedance.json`), the numpy compliance supervisor G-6b (SIM-ABSTRACT,
`reports/g6b_compliance.json`), the A1 FR3 kinematics check (SIM-POSE, `reports/g5_reach.json`)
and G-14's four-step SFT smoke (`reports/sft_smoke.json`). There is no stepped asset-based scene.

Remake the evidence classes using the authoritative classes in `docs/STATUS_HONEST.md` —
`FIXTURE` / `SIM-POSE` / `SIM` / `SIM-ABSTRACT` / `GATE` / `MODEL` / `HARDWARE` — with the measured
gates (G-1/G-2/G-3/G-6/G-6b/G-14 and the A1 kinematics check) and what each does **not** show, plus
a **SIM/MODEL** class (planned, not run). Mark the asset-based scene and arm certification as not
built. Keep "every outcome is pending" and "evidence classes are never pooled".

## Prose corrected in this pass (no figure work needed)

* Appendix "Environment and tests": the planned Isaac Sim 5.x / Isaac Lab 2.3 pairing was replaced
  with the measured Newton-only stack, and the stale 137/13 CPU-only test count with the measured
  training-environment count.
* Evaluation protocol: evidence class ISAAC now says Isaac Lab 3.0 with Newton, and the compute
  paragraph records the measured 4050 budgets (D-38, D-47) and the G-3 result on **both** the 4050
  and a Kaggle T4 (D-39, D-51), with the one-device-at-a-time scope (D-50).
* Abstract and introduction (2026-11-19): the Eyes are no longer called vision-language models and
  the Brain is no longer described as text-only; the shared checker and Isaac Lab results are
  marked as a protocol/target rather than a measurement. New "status of the evidence" paragraphs in
  the introduction and evaluation section distinguish the measured gates (including G-6,
  simulation-only) from pending results.
* `07_appendix.tex` "Environment and tests": test count is no longer quoted as a stale number; it
  is deferred to the gate record at submission time.
