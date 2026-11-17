# Simulation architecture: Isaac Lab 3.0 + Newton

Decision D-25: this is the only simulation stack. Labels as in `IMPLEMENTATION_PLAN.md`.

## 1. Roles
| piece | role | evidence |
|---|---|---|
| Isaac Lab 3.0 (GA) | task API, scenes, sensors, RL environments, batching; multi-backend factory; Python 3.12 | [V:S-10, S-15] |
| Newton | GPU physics on Warp; MuJoCo-Warp as primary solver; XPBD, VBD, MPM, Featherstone, SemiImplicit also available; imports URDF/MJCF/USD; `Control.joint_f` effort mode | [V:S-11, S-17, S-18] |
| Newton Warp renderer | lightweight tiled ray tracer for eye cameras; `rgb` + `depth` only | [V:S-12, S-13] |
| OVRTX (optional) | RTX-quality rendering via `isaaclab_ov`, ovrtx v0.5.1; GPU requirements **[G]** | [V:S-12] |
| Visualizers | `newton_gl` (also headless via EGL), Rerun, Viser (browser); `newton_rtx` cannot capture headlessly | [V:S-14] |
| our controller | `rtp/control/impedance.py` batched torch; Isaac Lab `OperationalSpaceController` is the cross-check | [V:S-16] |

Select physics/renderer/visualizer with the task's config (`physics=newton_mjwarp`, renderer cfg, `--viz newton_gl`) so the same task runs on every profile [V:S-10, S-14].

## 2. Install policy
* One environment, **Newton-only Isaac Lab install**, Python 3.12, pinned by Isaac Lab's own requirement files at the chosen tag; versions logged in every run (`reports/env_<profile>.json`).
* Newton GPU floor: Turing (cc 7.5+); Warp 1.18 needs driver R580+, otherwise pin `warp-lang==1.17.0` [V:S-17, S-19] (gate G-3 decides per machine/Kaggle image).
* Prebuilt wheels and the Isaac Lab checkout go into the wheelhouse/cache datasets (`docs/COMPATIBILITY.md`).

## 3. Scene specification (`isaac/scenes/kitchen.py`)
* Frame: Z up, metres. Table/workbench carrying both arm bases (spacing from the P2 reachability study), counters, cabinets, columns, a bag/packing zone, a utensil zone.
* Dimensions from the physical room measurements; until measured, parametric defaults in `configs/kitchen_default.yaml` (a column and a cabinet that deliberately occlude parts of the table).
* Objects, furniture, kitchen structure: SimReady USD packages ingested per `docs/ASSETS.md` (G-13); measured mass/size overrides for real objects we own.
* Arms: 2 × FR3 + Franka Hand from Franka's official FR3/FR3 Duo models or Menagerie MJCF through Newton `add_urdf`/`add_mjcf` (G-5); FR3 hardware version (v1 vs v2/v2.1) must match the model version. Per-arm joint armature/friction identified later (P12).
* Eyes: articulated ring-rail carriage + pan/tilt + zoom camera model with 12 stations; cameras rendered by the Newton Warp renderer; occluders include the arms.
* Human proxies: (a) external wrench on any link point via `asset.permanent_wrench_composer.set_forces_and_torques(...)` / `instantaneous_wrench_composer` (Newton: written to `state.body_f`, applied every substep; G-7); (b) kinematic hand/forearm body driven by a spring-damper to push; (c) a static mannequin volume for separation tests.
* Lighting/visual randomisation within what the Warp renderer supports; OVRTX used only for a realism check.

## 4. Control loop timing in sim
Physics substeps and the controller run at a fixed rate equal to the hardware loop (target 1 kHz for the impedance loop, policy at 10–30 Hz via decimation). Same clamps, same latency model as the real L1 interface; gate G-6 checks stability at the chosen dt.

## 5. Inspection workflows (what you can see and do)
* Local window: `--viz newton_gl`. Headless video: `--viz newton_gl --headless` (EGL) + the video recorder [V:S-14].
* Browser: Viser; time-series and 3-D logs: Rerun. Tiled eye views can be added to the Newton visualizers [V:S-11].
* Interactive probes: pause, step, apply a push to any link, toggle contact display, replay a logged episode.
* Not available through Newton Warp: instance segmentation/normals as renderer outputs → eyes compute segmentation with their own CV; ground truth for training comes from the simulator state [V:S-13].

## 6. Validation and evidence
Per-asset checklist (mass/inertia, limits, zero-pose reach, interpenetration, step response, licence); per-scene determinism test (same seed → same trajectory hash); every figure in any report must carry run id + config hash + git SHA.

## 7. Performance gates
G-3 (stack runs per profile), G-8 (≥ 10 fps/eye at 640×480 on 4090-class), env-count scaling curve per profile recorded in `reports/sim_scaling_<profile>.json` (steps/s vs number of envs, with and without cameras).
