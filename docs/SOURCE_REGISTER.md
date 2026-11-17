# Source register

Accessed 2026-11-17. Reliability: **P** primary (vendor/maintainer/standard), **S** secondary, **R** repo. A fact is [V] only if backed by a row here.

| id | source | rel. | supports |
|---|---|---|---|
| S-01 | repo `docs/ARCHITECTURE_V2.md`, `DECISIONS.md`, `isaac/DESIGN.md` @ `a50347f` | R | prior decisions, module table, profiles |
| S-02 | https://kaggle.com/llms.txt | P | Kaggle product list |
| S-03 | Kaggle Community Benchmarks announcement (kaggle.com/product-announcements/667898; blog.google Kaggle Community Benchmarks) | P | free quota-limited frontier-model access, image input |
| S-04 | https://github.com/Kaggle/kaggle-benchmarks | P | SDK, Apache-2.0, environment-dependent models |
| S-05 | https://pypi.org/project/kaggle_benchmarks/0.2.0/ | P | `MODEL_PROXY_API_KEY`, `LLMS_AVAILABLE` |
| S-08 | Hugging Face Qwen3.5-4B model cards (upstream card copies) | S | architecture, license, thinking mode |
| S-09 | https://github.com/huggingface/transformers/pull/45034; HF cards; SkyRL PR | P/S | `flash-linear-attention` + `causal-conv1d` fast path |
| S-10 | https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0 and release notes (docs.nvidia.com/isaaclab/references/release-notes) | P | Isaac Lab 3.0 GA, Newton backend, multi-backend, quaternion order |
| S-11 | https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/refs/release_notes.html | P | Newton solvers/sensors, tiled camera in Newton visualizers |
| S-12 | https://docs.nvidia.com/isaaclab/concepts/renderers.md ; isaac-sim.github.io IsaacLab main renderers page | P | Newton Warp vs OVRTX renderers |
| S-13 | Isaac Lab camera sensor docs (release/3.0.0-beta2) | P | Newton Warp supports only rgb+depth; tiled rendering |
| S-14 | https://docs.nvidia.com/isaaclab/isaac-lab/migrating-to-3-0 | P | `--viz`, headless EGL, newton_rtx limit, quaternion change |
| S-15 | https://docs.nvidia.com/isaaclab/isaac-lab/installation.md | P | Isaac Lab 3.0 install, Python 3.12 |
| S-16 | Isaac Lab `isaaclab.controllers.operational_space_cfg` source (v2.x–3.0 beta2) | P | OSC impedance modes, damping formula |
| S-17 | https://raw.githubusercontent.com/newton-physics/newton/main/README.md ; PyPI `newton` | P | Newton requirements, governance |
| S-18 | Newton docs: overview, tutorials, migration guide (newton-physics.github.io) | P | importers, Control API (`joint_f`, targets), solvers, actuators |
| S-19 | https://github.com/NVIDIA/warp/releases/tag/v1.17.0 | P | CUDA 13 / R580 / cc 7.5 floor |
| S-20 | `franka_example_controllers` source (docs.ros.org noetic): cartesian_impedance_example_controller, dual_arm data container | P | impedance law, damping, torque-rate limit, filter, dual-arm example |
| S-21 | libfranka-rs / franka_rust examples (docs.rs, rust-digger) mirroring libfranka examples | S | stiffness 150/10, firmware impedance 3000/300, collision behaviour outside loop, user-stop warning |
| S-22 | `serl_franka_controllers` source (docs.ros.org) | P | impedance with clipping and integral term for HIL-SERL |
| S-23 | https://franka.de/research ; FR3 datasheet snippets; reseller listings | P/S | FR3 specs, torque sensors, FCI; price (S) |
| S-24 | Kinova Gen3 one-pager 2026; RobotShop | P/S | Gen3 control modes, 1 kHz |
| S-25 | flexiv.com Rizon pages; automate.org; kurabo; robodk | P/S | Rizon specs and safety functions; reach conflict |
| S-26 | The Robot Report; arXiv 2602.17822; ibf-solutions; evsint; automate.org (A3 R15.06-2025) | S/P | ISO 10218:2025, ISO/TS 15066 absorbed, EU references 7 Sep 2026 (S) |
| S-27 | NVIDIA T4 datasheet | P | T4: Turing, 40 RT cores, 16 GB |
| S-28 | arXiv 2506.18088 (RoboTwin 2.0); HF LeRobot docs | P | dual-arm benchmark, baselines, LeRobot v3 data |
| S-29 | ros-drivers/axis_camera; index.ros.org `axis_msgs`, `ptz_action_server_msgs`; ONVIF wrappers | P/S | PTZ interfaces and caveats |
| S-30 | Packt *The Kaggle Book*; remote-ssh-kaggle repo | S | `/kaggle/working` ≈ 20 GB; private datasets ≈ 100 GB |
| S-31 | Isaac Lab: PR #8269 (Newton external wrenches every substep), PR #7536 (WrenchComposer), release notes (wrench composers), `isaaclab.assets` API docs | P | exact external-force API and Newton substep behaviour |
| S-32 | https://franka.de/3d-assets ; https://franka.de/models-and-simulation | P | official FR3/FR3 Duo/accessory models in CAD, URDF, USD, MJCF; v1/v2/v2.1 |
| S-33 | https://github.com/google-deepmind/mujoco_menagerie (+PR #364 FR3 Duo; PickNik notice) | P | `franka_fr3`, `franka_fr3_v2`, `franka_fr3_duo`, Apache-2.0, from franka_description v2.9.0 |
| S-34 | HF dataset `aravindsuresh16/cathlab_fr3_instruments_v2_5` | S | FR3 Duo + Robotiq 2F-85 simulated on Newton 1.6 / MuJoCo-Warp (existence proof) |
| S-35 | HF: `nvidia/simready-assets`, `imagineio/PhysicalAI-SimReady-Assets`, `imagine-io/PhysicalAI-Simready-Kitchen-Assets`, `imagineio/PhysicalAI-SimReady-Homes`, `ALLSIDES/simready-scanned-objects` | P/S | SimReady catalogues, licences, formats |

## Not researched yet (do not cite as fact)
LangGraph specifics; Qdrant hybrid in local mode; OVRTX GPU requirements; gripper/F-T datasheets; Dynamixel/machine-vision options; torque-interface gravity convention and `O_F_ext_hat_K` exposure in libfranka; TPU support for Qwen3.5; ISO 10218-2:2025 contact limits (standard not yet acquired); SAC/flow-matching/HIL-SERL paper citations (background knowledge only).
