# Figure remake requests (for the figure author)

These are **not** edits to the SVGs: the drawings are untouched. Each one has a short red label
immediately below it in the LaTeX source (`\stalefig`), so the request is visible in the PDF and can
be removed once the figure is redrawn.

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

## Prose corrected in this pass (no figure work needed)

* Appendix "Environment and tests": the planned Isaac Sim 5.x / Isaac Lab 2.3 pairing was replaced
  with the measured Newton-only stack, and the stale 137/13 CPU-only test count with the measured
  training-environment count.
* Evaluation protocol: evidence class ISAAC now says Isaac Lab 3.0 with Newton, and the compute
  paragraph records the measured 4050 ceiling (D-38) and G-3 result (D-39) while leaving the T4
  half unverified.
