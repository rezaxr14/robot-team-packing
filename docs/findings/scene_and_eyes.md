# Scene bundle and two-eye view library — research findings

**Evidence class of this document: RESEARCH/DESIGN.** It records what outside projects and
vendor docs do and what this repo currently contains. It does **not** record a measured
render, detector, push, safety or hardware result. Every external item is listed in
[Sources](#sources) with URL, access date and the marker *external source, not reproduced here*.
Local claims cite a repo path (and line where useful).

Method note (2026-11-20): the `websearch` integration returned HTTP 402 on every call, so
external material was obtained with `webfetch` from primary URLs and the arXiv API
(`export.arxiv.org/api/query`). Where an arXiv ID in the task prompt did not resolve to the
named paper, the correct record was found by API title search (VLMbench = 2206.08522,
BEHAVIOR = 2108.03332, LeRobot = 2602.22818, Qwen2-VL = 2409.12191). The robotics
"Ravens" benchmark ID could **not** be confirmed by `ti:Ravens` (it returned a neuroprocessor
paper), so **no claim is made here about Ravens' file layout**. R2R/ScanNet-style view
libraries are represented by the R2R results cited inside [S9] and by IR2R [S10].

## 0. What is already true in this repo (local facts only)

* The scene bundle schema is *specified*, not built: `docs/AGENT_BUILD_SPEC.md:56-57` defines
  `scene_<seed>/{scene.json, objects.json, views/<station>_<zoom>_<pan>_<tilt>.jpg,
  depth/*.npz, visibility.json}` plus `skill_tables/<skill>.json` and `manifest.json`
  (sha256, SimReady asset ids/revisions, git SHA, library versions). WP-D at
  `docs/AGENT_BUILD_SPEC.md:103-106` calls for `ops/render_views.py` and a detector; R1/R2 at
  `docs/AGENT_RUNBOOK.md:193-197` call for the bundle and a CPU `SimBackend`.
* Eye placement is a **DESIGN/SIM-POSE** study, not a render: `docs/SCENE_DESIGN.md:1-4`,
  `reports/eye_placement.json` (990 workspace cells, hFOV 70°/vFOV 50°, 12 stations).
  Its machine-readable recommendation is `R=2.0, H=2.2`, eye pair `[3, 9]`,
  union worst-case coverage `0.9853`, both-see `0.5314`, 12 cells never seen
  (`reports/eye_placement.json:823-978`). Note a **text/data discrepancy**:
  `docs/SCENE_DESIGN.md:9` and `docs/AGENT_RUNBOOK.md:194` say ring R = 1.6 m while the
  JSON `recommended` block and `docs/SCENE_DESIGN.md:29-36` favour R = 2.0 m. Pin one in
  config and log the choice; do not silently pick.
* The only rendered scene evidence is a stage-1 preview: `ops/preview_scene.py:179-238`
  renders five fixed views (`oblique, front, top, eye_left, eye_right`) at 640×480 with
  `newton.sensors.SensorTiledCamera` and 55° pinhole rays. Its report says
  "rendered and measured; NOT stepped, NOT a manipulation/safety result"
  (`reports/scene_preview.json`). Preceding manifest: `reports/scene_manifest.json:2-3`
  ("assembled and measured; NOT stepped, rendered or safety-tested").
* Measured Warp render+physics speed in this repo: `reports/g3_sim.json:16-21`
  (160×120, 1 env, 30 steps, **21.0 steps/s**, RTX 4050 Laptop) and
  `reports/g3_sim_t4.json:16-21` (**0.8 steps/s**, 36.851 s, Kaggle T4). These are
  combined step rates at 160×120, **not** fps at the G-8 target 640×480.
* USD ingestion is partial: `reports/g13_usd_ingest.json:96-98` imported 9/14 attempted
  packages (`result: PARTIAL`); orphan-joint errors for the bottle, two cabinets and the
  coffee station; a USD open error for the paper bag. `ops/preview_scene.py:146-149` works
  around orphan joints with `ignore_paths=["/RootNode/Joints"]`.
* SimReady ingestion policy, licences and the G-13 gate are in `docs/ASSETS.md:1-58`; the
  impedance spec and its "simulation-only, not a safety certificate" status are in
  `docs/IMPEDANCE_SPEC.md:5-13, 97-101`.

## 1. Hashable scene-bundle format for CPU multi-turn episodes

### What comparable projects do (external)

* **RoboTwin 2.0** [S1, S2] separates an object library (RoboTwin-OD: 731 instances,
  147 categories, semantic + manipulation annotations) from task programs and from
  trajectories. Episode files are HDF5 under
  `data/<task_config>/<task>/<embodiment>/data/episode_0000000.hdf5`; a seed is searched
  once and then replayed to collect data. Cameras are stored as *encoded image bits*, and
  the README insists on one decode path (`decode_image_bit`): legacy bits were produced by
  handing RGB to `cv2.imencode` (channel-reversed), standard bits carry an `XPL-RGB1` JPEG
  marker. Conversion to LeRobot v2.1/v3 lives in separate scripts. Domain randomisation is
  five named axes (clutter, background, lighting, table height, language) and task config
  names (`demo_clean`, `demo_randomized`). **Lesson for us:** pin a codec/decoder contract,
  keep randomisation profiles as named configs, make seed→replay the collection primitive,
  and keep object annotations outside episode blobs.
* **LeRobot** [S3] defines `LeRobotDataset` as a self-contained, multimodal, streaming
  schema (task text, embodiment, capture fps, sensor types) designed to scale to millions
  of trajectories; this is the format our Tier-2 VLA export targets
  (`docs/AGENT_BUILD_SPEC.md:150`, `ops/export_lerobot.py` to be built). **Lesson:** put
  self-describing metadata (fps, sensor/camera list, task text, provenance) in the bundle
  header, not in filenames.
* **VLMbench** [S4] generates demonstrations from modular rule-based task templates with
  language instructions and uses a keypoint model on **multi-view** observations.
  **BEHAVIOR** [S5, S6] uses an object-centric predicate-logic language for initial/goal
  conditions, lists simulator-agnostic features an environment must expose, and evaluates
  task progress against human demonstrations. **Lesson:** symbolic goal/visibility ground
  truth from sim state is a first-class artefact, not a by-product of rendering.
* **R2R/ScanNet-style view libraries** [S9, S10]: R2R agents consume a fixed, revisit-able
  set of navigable viewpoints; IR2R keeps memory across tours of up to 100 episodes in
  80 persistent scenes. **Lesson:** a fixed, addressable view set plus persistent memory is
  a legitimate evaluation substrate; `view_id` must be stable across runs.

### Recommended format for this repo (design, not yet implemented)

Extend — do not replace — the schema at `docs/AGENT_BUILD_SPEC.md:56-57`:

```
bundles/scene_<seed>/
  manifest.json          # canonical JSON; the only hashed entry point
  scene.json             # immutable geometry: island, cabinets, columns, arm bases, ring
  objects.json           # per-seed instances: id, SimReady pkg+revision, pose, mass, friction, fragility
  views/index.json       # view_id -> station, zoom, pan, tilt, arm_pose, intrinsics, extrinsics, fov
  views/<view_id>.jpg    # one fixed encoder + quality + colour convention, declared in manifest
  depth/<view_id>.npz    # float32 depth + depth_scale + ray model
  visibility.json        # per-cell station visibility from the SAME analytic ray-cast as design_eyes.py
  occlusion.json         # optional: per-view occluder labels from sim shape-index/depth
  skill_tables/<skill>.json  # empirical outcome distributions, arm_model hash pinned
```

Hashability rules: (a) canonical JSON (`sort_keys=True`, fixed separators, UTF-8, LF) for
every `.json`; (b) `manifest.json` lists `sha256` + byte size of every file except itself
and a `bundle_sha256` = sha256 of the canonical manifest body; (c) manifest also carries
`schema_version`, `seed`, `scene_family`, `config_hash`, `git_sha`, `newton`/`isaaclab`/`warp`
versions, asset id+revision+sha, the generation command and the RNG seed list; (d) view
filenames are derived from a fixed `view_id` string, never from float formatting that can
change between platforms. This is the local analogue of RoboTwin's "seed → replay" and
LeRobot's self-describing header. Delivery to the hosted CPU worker remains an unsecreted
public CC-BY HF dataset repo *or* the repo itself, with the manifest hash compared
(DG-6 at `docs/DATAGEN_KAGGLE.md:123-124`).

Two separate hashes matter and must not be confused: `bundle_sha256` (scene + views) and
`arm_model = newton_cert_<hash>` (`docs/AGENT_BUILD_SPEC.md:98-100`). The bundle *pins* the
latter; it must not recompute skill outcomes.

## 2. Rendering and caching multi-camera views with pan/tilt/zoom

### Rendering in this stack (local facts)

The current renderer is `SensorTiledCamera` in `ops/preview_scene.py:210-232`: pinhole rays
via `compute_camera_rays_pinhole(width, height, camera_fovs=...)`, one transform per view,
`model.bvh_refit_shapes(state)` before `sensor.update(...)`, output `uint8` RGB and float32
depth. The Newton docs [S13] list `SensorTiledCamera` as **deprecated** in favour of
`SensorCamera` (which adds normal/albedo/shape-index and MSAA/SSAA); our pinned Newton is
1.6.1 (`reports/g3_sim.json:6-9`), so keep the working API and record the version. Likely
migration item: `SensorCamera` + `compute_camera_rays_pinhole_usd` when the lock is bumped.
Zoom should be implemented as a **declared FOV narrowing** on the same ray helper
(fov / zoom), not as new optics; `docs/SCENE_DESIGN.md:41` explicitly says zoomed FOV is
*not covered* by the placement study, so record it as an assumption. Pan/tilt are applied
to the look-at direction in `ops/preview_scene.py::look_quat`.

### Budgeting the library (design arithmetic, not measured)

With 12 stations × zoom {1,2,4} × pan {−15,0,15} × tilt {−10,0,10} × arm poses {home,reach}
= **648 views/seed** at 640×480; the minimal 6-station set
`[3,9,8,10,1,5]` (`reports/eye_placement.json:969-977`) halves this to 324. Both are
proposals; measure JPEG size and render time before fixing a hosting budget. Depth at
640×480 float32 is ~1.23 MB/view uncompressed, so store compressed/quantised `.npz` and
keep raw depth only for the evaluation set. Define one RGB byte convention explicitly —
RoboTwin [S2] shows how a silent channel reversal can survive a whole dataset.

### Image tokens and latency (external + local derivation)

Qwen2-VL [S9] uses naive dynamic resolution: ViT patches and a 2×2 merge; a 224×224 image
with patch 14 becomes **66 tokens** before the LLM (tokens ≈ (H/p)(W/p)/m²). Applying the
**local** Qwen3.5-4B config (`vision_patch_size = 16`, `vision_spatial_merge_size = 2`,
`reports/qwen35_modules.json:25-28`) gives ~300 tokens for a 640×480 image and ~600 for two
eyes — a derived estimate, not a measured tokenizer output. Hosted-episode assumptions in
`docs/DATAGEN_KAGGLE.md:86-87` budget 60k input / 6k output tokens with images and a
5 MB `image_url` cap (DG-5, `:123`); the 4050's "256-token ceiling"
(`docs/AGENT_BUILD_SPEC.md:185`) is a **training** context cap, not an inference image cap —
do not conflate them. LeRobot's own latency table [S3] (RTX 4090, fp32: ACT 5.0 ms,
Diffusion 369.8 ms, π0 209.4 ms, SmolVLA 99.2 ms per forward pass) is a separate policy
budget, not a render budget.

### Active vision evidence (external)

BAVO-Bench [S7] controls external visibility (Clean / Stage Occlusion / Random-time
Occlusion) and reports that active recovery beats passive baselines under occlusion;
AV-ALOHA [S8] uses a dedicated 7-DoF camera-only arm and finds large gains over fixed
cameras when visibility is limited; AVR [S11] reports 5–17% simulation and >25% real gains
from ROI-conditioned viewpoint change with aspect-preserving crops and motorised zoom;
Cheng et al. [S12] show active vision/manipulation policies outperform static cameras in
clutter. This directly motivates the WP-I metrics "success vs eye-budget curve vs
fixed-view baseline at equal cost" and "ask-another-eye rate"
(`docs/AGENT_BUILD_SPEC.md:153`).

## 3. Measuring eye/view quality (occlusion, visibility, mAP, fps gates)

1. **Visibility ground truth from sim.** Compute per-cell, per-station, per-arm-pose
   visibility with the same analytic ray-cast used by `ops/design_eyes.py`, and store it in
   `visibility.json`. Existing DESIGN numbers to reproduce (not to quote as measured):
   990 cells, 12 never seen, 14 seen by <2 stations, best pair `[3,9]` coverage 0.9853,
   both-see 0.5314 (`reports/eye_placement.json:979-1005`). These are box-obstacle,
   pose-level numbers, not a Warp render measurement.
2. **Occlusion maps.** For each cached view, render depth plus, where available, shape-index
   (Newton `SensorCamera` supports shape-index/albedo/normal [S13]) so occluder identity
   (arm / column / cabinet / object) is known per pixel. Compare against the analytic
   `visibility.json` label for the sampled cell; disagreement is a measurable simulator
   consistency gate. The renderer has no built-in segmentation
   (`docs/AGENT_BUILD_SPEC.md:105`), so own CV produces masks; split held-out **stations and
   objects**, not just seeds.
3. **Detector mAP with intervals.** Report COCO-style mAP@[.5:.95] and per-class AP on
   held-out stations/objects, with Wilson intervals for per-station detection rates and
   paired bootstrap (`rtp/stats.py:7,17`) for mAP differences between station sets. Store
   the raw per-image detections so the number can be regenerated; never hand-type it.
4. **Latency / fps gates.** G-8 is ≥10 fps/eye at 640×480 on a 4090-class GPU
   (`docs/AGENT_BUILD_SPEC.md:106`). The only local numbers are 21.0 steps/s (4050) and
   0.8 steps/s (T4) at **160×120 combined physics+render**
   (`reports/g3_sim.json`, `reports/g3_sim_t4.json`); the ~26× gap means a T4 gate must be
   measured, never inferred from the 4050 number. No published RTX 4050/T4 Warp
   ray-tracing fps figure was retrieved in this session — treat all target-resolution fps
   as **not measured**. Report separately: ray-trace/s, encode/s, and end-to-end
   `render_views` wall time.

## 4. External pushes and compliance in Isaac Lab / Newton (simulation-only)

**Control surface.** Use explicit effort mode (`IdealPDActuatorCfg` with zero gains →
`JointTargetMode.EFFORT`) and publish commands with `write_data_to_sim()` before stepping,
as already verified in `docs/findings/newton_arm_api.md:39-56`. Run the impedance law from
`docs/IMPEDANCE_SPEC.md:39-47` through `rtp/control/compliance.py` and write torques at the
hardware loop rate via substeps; the Brain never sets gains, forces or speeds
(`docs/AGENT_BUILD_SPEC.md:8-16`).

**Wrench composers.** Isaac Lab exposes `permanent_wrench_composer` and
`instantaneous_wrench_composer` on articulations, rigid objects and collections
(local checkout: `source/isaaclab/isaaclab/assets/articulation/base_articulation.py:464,476`,
`.../rigid_object/base_rigid_object.py:93,105`,
`.../mdp/events.py:1209`). PR #7536 [S14] moved frame selection into `WrenchComposer`:
local content stays in body frame; world-at-CoM can be submitted directly by PhysX/OvPhysX;
**Newton keeps body-frame submission** (`supports_world_at_com=False`), so a world-frame
push must be composed through link poses before it reaches MJWarp. The same PR documents a
non-zero-COM-offset caveat for world-at-CoM submission and per-step view-allocation costs;
neither is a safety issue, both are correctness/perf checks for our push harness.

**First-substep-only force — the critical pitfall.** PR #8269 [S15] fixes a real bug:
external wrenches were written once to `state_0.body_f`, while
`NewtonManager._run_solver_substeps` cleared forces after each substep, so a constant
wrench delivered only `1/num_substeps` of its impulse (and `1/(decimation × num_substeps)`
with folded decimation). The fix stages `body_f` at step start and re-applies it before
every substep; regression test `test_staged_body_force_acts_on_every_solver_substep`
(2 substeps MJWarp, 3 substeps + decimation 2 Kamino) went from 2 failed to 2 passed, and
the repro printed 0.100 m/s for 1, 2 and 4 substeps before the fix. It was merged
2026-10-08 and backported to `release/3.0.0` as commit `5608601`. This is exactly gate G-7
(`docs/AGENT_RUNBOOK.md:93-100`, impulse = F·dt within 2 % with substeps > 1). Our local
checkout (`<repo>/.isaac/IsaacLab`, HEAD `e3f7d9ff1`) contains the staged-force code
(`source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py:1674-1711`), so the fix is
present; still **verify by running G-7 at substeps 1 and 4**, and write the backport commit
into the report.

Newton issue #4545 / PR #5492 [S16] (accepted 2026-11-20) makes `State.body_f` /
`State.particle_f` strictly input-only for solvers, so "write once, read on every substep"
becomes a documented contract rather than an implementation accident; the audit found
SemiImplicit and Featherstone wrote those buffers, and Kamino copied to `state_out`. Our
pinned Newton 1.6.1 may predate the contract, so rely on the staging fix and the impulse
gate, not on the new contract.

**Other pitfalls to gate explicitly.** Contact instability when `dt`/substeps are too
coarse for the push stiffness; mass/inertia mismatch from authored assets — locally,
`reports/g13_usd_ingest.json:34-46` shows a "cart_computer" package at 116.83 kg and a mug
at 0.1 kg with 182 shapes, so `docs/ASSETS.md` §3 step 7 (`:45`) requires measured real-twin
overrides; force applied at a link point vs at CoM produces a different moment (the PR
#7536 review found missing COM-offset integration coverage); and the force caps in
`docs/IMPEDANCE_SPEC.md:59-61` are **placeholders** pending ISO 10218-2:2025.

**Safety boundary.** All of the above is a simulation cross-check of *behaviour* only.
`docs/IMPEDANCE_SPEC.md:7-13,100` and `docs/AGENT_BUILD_SPEC.md:90` state plainly: not a
full acceptance table, not hardware, not a safety certification; sign-off is Claude's
(D-22/D-58) and hardware validation is WP-J.

## 5. SimReady USD ingestion into Newton

Newton's `ModelBuilder.add_usd()` parses standard `UsdPhysics` (rigid bodies, collision,
joints, `MaterialAPI`, scene settings) and uses **schema resolvers** to bridge solver-specific
PhysX/MuJoCo attributes, with priority-based resolution when values conflict; it collects
solver-native attributes instead of discarding them and converts UsdPhysics degrees to
radians [S17]. Custom `newton:` attributes can be declared on the `PhysicsScene` and assigned
per prim [S18] — a good place to carry asset revision, licence id and `collision_mode` into
the model alongside physics data.

Known limits relevant to SimReady packs [S17]: deformable import covers only a narrow
AOUSD-proposal subset; native OmniPhysics/PhysX deformables that do not apply the AOUSD
simulation APIs are **not recognised as deformables and import as static geometry**; vendor
namespaces are only a fallback for deformable *material* attributes (applied schemas,
renamed attributes and hierarchy conventions are not translated); `startsAsleep` and
`simulationOwner` are not read; per-element `GeomSubset` material bindings warn and are
ignored; dynamic/static friction is not mapped onto the deformable collision
approximation; collision groups are not applied to deformables. Also note float32 live
points vs float64 rest geometry at large world offsets. These are reasons to treat
furniture/appliances as static simplified colliders rather than trusting authored
deformables (the policy in `docs/ASSETS.md:43`).

Local, concrete workarounds: orphan joints (`ValueError: Found N joint(s) not belonging to
any articulation`) in `reports/g13_usd_ingest.json:25,63,83,86` → either
`ignore_paths=["/RootNode/Joints"]` as in `ops/preview_scene.py:146-149`, or call
`add_articulation()` for the authored joints; a broken USD layer for `paperbag_a01`
(`:65-67`) → exclude/re-fetch; G-13 is still only `PARTIAL`
(`reports/g13_usd_ingest.json:2-4,96-98`), so no pack is validated yet.

Attribution/licensing: the NVIDIA SimReady Assets dataset card declares `cc-by-4.0` [S19];
`docs/ASSETS.md:24-32` records per-source licences (CC-BY-4.0 for the general/kitchen/scanned
sets, *gated non-commercial research* for SimReady Homes, per-package licence in
`com.nvidia.simready.packaging.json` for the `imagineio` packs, Franka/Menagerie terms for
robots). Requirement: generate `assets/ATTRIBUTION.md` from package metadata, pin
id+revision+sha in the manifest, and localise `https://` references before loading Homes
(`docs/ASSETS.md:29,42,46`). Bundle delivery to Kaggle must use a public CC-BY repo with
attribution included (DG-6, `docs/AGENT_BUILD_SPEC.md:118`).

## Sources

All rows accessed **2026-11-20**. Every external row is an **external source, not reproduced
here**; only short factual summaries and derived arithmetic appear above.

| id | title | URL | date | status |
|---|---|---|---|---|
| S1 | RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation (arXiv:2506.18088v2) | https://arxiv.org/html/2506.18088v2 | 2026-11-20 | external source, not reproduced here |
| S2 | RoboTwin GitHub README (XPolicyLab format, `decode_image_bit`, HDF5 episodes, LeRobot conversion) | https://github.com/robotwin-Platform/robotwin | 2026-11-20 | external source, not reproduced here |
| S3 | LeRobot: An Open-Source Library for End-to-End Robot Learning (arXiv:2602.22818v1) | https://arxiv.org/html/2602.22818v1 | 2026-11-20 | external source, not reproduced here |
| S4 | VLMbench: A Compositional Benchmark for Vision-and-Language Manipulation (arXiv:2206.08522v2) | https://arxiv.org/abs/2206.08522 | 2026-11-20 | external source, not reproduced here |
| S5 | BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments (arXiv:2108.03332v1) | http://export.arxiv.org/api/query?id_list=2108.03332 | 2026-11-20 | external source, not reproduced here |
| S6 | BEHAVIOR Robot Suite (arXiv:2503.05652v2) | http://export.arxiv.org/api/query?search_query=all:BEHAVIOR+AND+all:household+AND+all:benchmark | 2026-11-20 | external source, not reproduced here |
| S7 | Recovering the View: Benchmarking Physical Active Vision for Occlusion Recovery in Robotic Manipulation / BAVO-Bench (arXiv:2609.37292v1) | http://export.arxiv.org/api/query?search_query=all:%22active+vision%22+AND+all:manipulation | 2026-11-20 | external source, not reproduced here |
| S8 | Active Vision Might Be All You Need: Exploring Active Vision in Bimanual Robotic Manipulation / AV-ALOHA (arXiv:2409.17435v2) | same API query as S7 | 2026-11-20 | external source, not reproduced here |
| S9 | Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (arXiv:2409.12191v2) — dynamic-resolution token rule, R2R agent results | https://arxiv.org/html/2409.12191v2 | 2026-11-20 | external source, not reproduced here |
| S10 | Iterative Vision-and-Language Navigation (arXiv:2210.03087v3) — persistent R2R tours in 80 scenes | http://export.arxiv.org/api/query?search_query=ti:ScanNet+OR+ti:Qwen2-VL+OR+ti:%22Vision-and-Language+Navigation%22 | 2026-11-20 | external source, not reproduced here |
| S11 | AVR: Active Vision-Driven Precise Robot Manipulation with Viewpoint and Focal Length Optimization (arXiv:2503.01439v4) | same API query as S7 | 2026-11-20 | external source, not reproduced here |
| S12 | Reinforcement Learning of Active Vision for Manipulating Objects under Occlusions (arXiv:1811.08067v2) | same API query as S7 | 2026-11-20 | external source, not reproduced here |
| S13 | Newton documentation — Sensors (`SensorCamera`, deprecated `SensorTiledCamera`, shape-index, ray helpers) | https://newton-physics.github.io/newton/stable/concepts/sensors.html | 2026-11-20 | external source, not reproduced here |
| S14 | Isaac Lab PR #7536 — Move external wrench frame selection into WrenchComposer (Newton stays body-frame) | https://github.com/isaac-sim/IsaacLab/pull/7536 | 2026-11-20 | external source, not reproduced here |
| S15 | Isaac Lab PR #8269 — Apply Newton external wrenches on every solver substep (bug + backport 5608601) | https://github.com/isaac-sim/IsaacLab/pull/8269 | 2026-11-20 | external source, not reproduced here |
| S16 | Newton issue #4545 / PR #5492 — solvers must treat `body_f`/`particle_f` as input-only (per-solver audit) | https://github.com/newton-physics/newton/issues/4545 | 2026-11-20 | external source, not reproduced here |
| S17 | Newton documentation — USD Parsing and Schema Resolver System (PhysX/MuJoCo resolvers, deformable subset + limitations) | https://newton-physics.github.io/newton/stable/concepts/usd_parsing.html | 2026-11-20 | external source, not reproduced here |
| S18 | Newton documentation — Custom Attributes (`newton:` namespace on PhysicsScene, USD/MJCF/URDF) | https://newton-physics.github.io/newton/stable/concepts/custom_attributes.html | 2026-11-20 | external source, not reproduced here |
| S19 | NVIDIA SimReady Assets dataset card (`license: cc-by-4.0`) | https://huggingface.co/datasets/nvidia/simready-assets/raw/main/README.md | 2026-11-20 | external source, not reproduced here |

Local sources (not in the table above) are repository files already cited by path, plus the
local Isaac Lab checkout `<repo>/.isaac/IsaacLab` at git `e3f7d9ff1` (wrench composer
and staged substep-force code paths listed in §4).

## Concrete plan for this repo

**Files to add or extend** (all consistent with `docs/AGENT_BUILD_SPEC.md` WP-A/WP-D and
`docs/AGENT_RUNBOOK.md` R1/R2; no code is changed by this research document):

1. `configs/views.yaml` (new) — pin ring R/H (resolve the 1.6 vs 2.0 discrepancy from
   `reports/eye_placement.json` and record it), station list, zoom set, pan/tilt grid, arm
   poses, FOV-at-zoom-1, resolution, JPEG encoder/quality, depth dtype/scale, view_id scheme.
2. `ops/render_views.py` (new, R1/WP-D) — extend `ops/preview_scene.py::build/look_quat`;
   iterate `configs/views.yaml`; call `model.bvh_refit_shapes(state)` per view; render RGB
   + depth through `SensorTiledCamera`; write `reports/r1_render_views.json` with per-view
   wall time, fps/eye and versions. Analysis-only ray-cast visibility and occlusion labels
   from `ops/design_eyes.py` logic; no detector training in this step.
3. `ops/make_scene_bundle.py` (new) — assemble `scene.json`, `objects.json`,
   `views/index.json`, `visibility.json`, links to `skill_tables/`, and canonical
   `manifest.json` with `bundle_sha256`; verify every referenced hash before writing.
4. `rtp/eyes/gateway.py` (extend, WP-D) — implement `orbit/pan/tilt/zoom/detect/...` from
   `docs/SCHEMAS.md`, returning `view_id` + `image_url` parts (DG-5, ≤5 MB) resolved through
   the bundle; nearest-cached-view policy recorded in the episode.
5. `rtp/env/bundle_backend.py` (new, R2/WP-E) — `SimBackend` that seeds outcome sampling
   from `skill_tables/<skill>.json` and answers eye tools from cached views; must pass the
   `tests/test_simworld.py` conformance pattern and replay verification.
6. `ops/eye_eval.py` (new) — visibility/occlusion agreement, detector mAP with Wilson +
   paired bootstrap using `rtp/stats.py`, and the fps/encode latency gates →
   `reports/r1_eye_eval.json`.
7. `tests/test_scene_bundle.py`, `tests/test_bundle_backend.py`, `tests/test_eye_tools.py`
   (new) — canonical hashing, missing/corrupt file detection, seeded replay, tool schema.
8. `ops/g7_wrench.py` / `isaac/tests/push_test.py` (per runbook A4/A5) — add a report field
   recording the Isaac Lab commit containing the every-substep fix, and run substeps 1 and 4
   so the 1/n failure mode is visible rather than hidden.

**Minimal smoke test (all numbers written to a report; none assumed):**

```bash
source ops/env.sh && python -m pytest -q            # green before and after
python ops/render_views.py --seed 100000 --stations 3,9 --zoom 1,2,4 \
  --pan -15,0,15 --tilt -10,0,10 --arm-poses home,reach \
  --width 640 --height 480 --out bundles/scene_100000
python ops/make_scene_bundle.py --seed 100000 --views bundles/scene_100000 \
  --out bundles/scene_100000
python -m pytest -q tests/test_scene_bundle.py tests/test_bundle_backend.py tests/test_eye_tools.py
python ops/eye_eval.py --bundle bundles/scene_100000 --out reports/r1_eye_eval.json
python ops/g7_wrench.py --substeps 1,4 --out reports/g7_wrench.json
```

Pass conditions for the smoke test: two consecutive `render_views`+`make_scene_bundle`
runs of the same seed yield the same `bundle_sha256`; every manifest hash re-verifies;
view files and `visibility.json` exist for all requested `view_id`s; the CPU backend passes
conformance and a recorded episode replays; G-7 shows `|Δp − F·t|/(F·t) ≤ 0.02` at
substeps ≥ 4 and reports the substeps=1 value. Anything else is recorded as FAIL, not
smoothed.

## Do not claim

* No **manipulation result** exists here: no grasp, place, handover or packing success has
  been measured on the two-arm Newton scene (`reports/scene_manifest.json:132-135`,
  `docs/ASSETS.md:17-21`). `docs/AGENT_RUNBOOK.md` requires arms certified (A10) before R4–R9.
* No **safety result** exists: G-6/G-7-style simulation cross-checks are behaviour checks
  only; force/speed caps are placeholders; ISO 10218-2:2025 limits, hardware push tests,
  L0/L1 sign-off and the cell risk assessment do not exist
  (`docs/IMPEDANCE_SPEC.md:5-13,100`, D-22/D-58, WP-J).
* No **hardware result** exists: no FR3, force gauge, PTZ rig or measured room dimensions;
  all eye numbers are DESIGN/SIM-POSE and all arm numbers are SIM-ABSTRACT/SIM at best.
* No **detector mAP**, **occlusion-map accuracy**, **bundle hash stability**, **bundle-vs-
  Newton outcome agreement**, or **640×480 fps on 4050/T4** has been measured in this repo.
  The only render rates on record are 21.0 steps/s (4050) and 0.8 steps/s (T4) at 160×120.
* G-13 is not passed: 9/14 packages imported, no drop/grasp/mass-vs-metadata check
  (`reports/g13_usd_ingest.json`). Assets are not yet validated for physics.
* The view library, scene bundle, gateways and CPU backend described above are a **plan**;
  no file except this document was created or modified by this research task.
