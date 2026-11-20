# A2 — Isaac Lab 3.0 + Newton arm API findings

Sources: local checkout `$ISAACLAB_DIR` = `<repo>/.isaac/IsaacLab` (release/3.0.0, git HEAD `e3f7d9ff1`), Newton 1.6.1 in `$ISAAC_PY`, Warp 1.17.0, torch 2.11.0+cu130, GPU RTX 4050 Laptop.
"VERIFIED by running …" means the command was executed in this session and its output is recorded below. Everything else is source reading only.
Sections are in the order they were landed (1, 2, 3, 6 first — A3's dependencies — then 4, 5, 7, 8).

## 1. Spawn an arm from MJCF/URDF as an `Articulation` under `physics=newton_mjwarp`

**Answer.** In an `InteractiveScene`, the arm is an `ArticulationCfg` whose `spawn` is a file spawner; the physics backend is selected by `SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))`. `MjcfFileCfg`/`UrdfFileCfg` exist (`from_files_cfg.py:289,311`) but their converters import `isaacsim`, which is **absent from the Newton-only venv**, so they cannot convert at runtime here (see UNRESOLVED). The route that works in this venv is a **pre-built USD** articulation (`UsdFileCfg`), exactly as the Isaac Lab Newton tests do at `test/envs/test_events.py:73-81`. For the FR3 MJCF that is already cached locally, the *working* Newton-native load is `newton.ModelBuilder.add_mjcf()` (no Isaac Lab `Articulation` wrapper), `ops/g5_fr3_load.py:45-46`.

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

USD = f"{ISAACLAB_DIR}/source/isaaclab_newton/test/assets/data/articulation_ordering_branching.usda"
sim_cfg = SimulationCfg(device="cuda:0", dt=1.0/500,
                        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=4,
                                          use_cuda_graph=False))
scene_cfg = InteractiveSceneCfg(num_envs=2, env_spacing=3.0)
scene_cfg.robot = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot", spawn=sim_utils.UsdFileCfg(usd_path=USD),
    actuators={"torque": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
)
with build_simulation_context(sim_cfg=sim_cfg) as sim:
    scene = InteractiveScene(scene_cfg); robot = scene["robot"]; sim.reset()
    print(type(robot), robot.data.joint_names, robot.num_instances)
```

**Where found.** `source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py:289` (`UrdfFileCfg`), `:311` (`MjcfFileCfg`); `source/isaaclab_newton/test/envs/test_events.py:73-81` (scene + `UsdFileCfg` + `NewtonCfg`); `source/isaaclab_assets/isaaclab_assets/robots/franka.py:99-110` (`FRANKA_PANDA_CFG`, `variants={"Physics": "physx"}` — set `{"Physics": "mujoco"}` for Newton/MJWarp as done in `test/assets/test_articulation.py:83-84`); `ops/g5_fr3_load.py:45` (`builder.add_mjcf(entry)`).

**VERIFIED by running** the snippet above with `$ISAAC_PY`: output
`<class 'isaaclab_newton.assets.articulation.articulation.Articulation'> num_envs 2`,
`JOINT NAMES ['left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow']`,
`is_fixed_base False num_joints 4`, and `joint_target_mode` `[...,4,4,4,4,...]` (4 = EFFORT) for both envs.
Also **VERIFIED by running** `source ops/env.sh && $ISAAC_PY ops/g5_fr3_load.py`: `result: PASS`, FR3 v2 `joint_dof 7`, `total_mass_kg 17.31`.

## 2. Control mode: command joint effort/torque and disable the built-in position actuator

**Answer.** Use an **explicit** actuator group (`IdealPDActuatorCfg`) with `stiffness=0.0, damping=0.0`: `_configure_joint_target_modes` maps non-implicit actuator groups to `JointTargetMode.EFFORT` (`articulation.py:226-231`), so the solver's implicit position/velocity drive is off. Command torque with `robot.actuators.target_command.set_effort_index(value=..., full_data=True)` (`actuator_collection.py:755`); the old `set_joint_effort_target` is deprecated (`base_articulation.py:3276`). **Critical:** call `robot.write_data_to_sim()` explicitly before `sim.step()` — `Articulation.update(dt)` only refreshes data (`articulation.py:454-458`), it does not publish commands, and `InteractiveScene.update()` calls only `update()`.

```python
# after scene creation and sim.reset(), with IdealPDActuatorCfg(stiffness=0, damping=0)
tau = torch.full((robot.num_instances, robot.num_joints), 0.5, device=sim.device)
robot.actuators.target_command.set_effort_index(value=tau, full_data=True)
for _ in range(20):
    robot.write_data_to_sim()          # publishes effort to control.joint_f
    sim.step(render=False)
    scene.update(sim.get_physics_dt())
print(robot.data.joint_vel.torch)      # non-zero => torque reached the solver
```

**Where found.** `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py:55-59` (`_target_mode_from_gains`), `:177-231` (`_configure_joint_target_modes`; EFFORT for non-implicit), `:388-451` (`write_data_to_sim`), `:454-458` (`update`); `source/isaaclab/isaaclab/actuators/actuator_collection.py:138` (`target_command`), `:681,755` (`set_effort_index`); `source/isaaclab/isaaclab/assets/articulation/base_articulation.py:3276-3290` (deprecated setter); `source/isaaclab_newton/isaaclab_newton/assets/articulation/actuator_control.py:140-186` (effort published to `_sim_bind_joint_effort` / `joint_act`); `newton.JointTargetMode.__members__` printed `{'NONE':0, 'POSITION':1, 'VELOCITY':2, 'POSITION_VELOCITY':3, 'EFFORT':4}`.

**VERIFIED by running** the snippet above with `$ISAAC_PY`: `joint_target_mode` for the 4 actuated DoFs was `4` (EFFORT), `control.joint_f[:12]` became `[0,0,0,0,0,0,0.5,0.5,0.5,0.5,...]` after `write_data_to_sim()`, and after 20 steps `joint_vel` was `[[25.9851, 12.9926, 25.9851, 12.9926], ...]` and `joint_pos` `[[0.5262, 0.2631, ...]]`. A control run **without** `write_data_to_sim()` left `joint_f` and `joint_vel` at zero — the explicit publish call is required.

## 3. Read joint positions/velocities and body poses as torch tensors

**Answer.** All state fields are `ProxyArray`; access the torch view with **`.torch`** (e.g. `robot.data.joint_pos.torch`, not `robot.data.joint_pos` — the latter is a `ProxyArray`, not a tensor). `joint_pos`/`joint_vel` are `(num_envs, num_joints)`; `body_link_pose_w` is `(num_envs, num_bodies, 7)` with `[x,y,z,qx,qy,qz,qw]`. Body poses trigger a lazy FK refresh on read.

```python
q  = robot.data.joint_pos.torch              # (N, J) float32, CUDA
dq = robot.data.joint_vel.torch              # (N, J) float32, CUDA
pose = robot.data.body_link_pose_w.torch     # (N, B, 7) [pos, quat(x,y,z,w)]
pos_w = pose[..., :3]; quat_w = pose[..., 3:]
# alternatives: robot.data.body_pos_w.torch / body_quat_w.torch,
#               robot.data.root_link_pose_w.torch
```

**Where found.** `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py:1065` (`joint_pos`), `:1074` (`joint_vel`), `:736-746` (`body_link_pose_w`, shape/dtype/quaternion order), `:1366,1380` (`body_pos_w`/`body_quat_w` derive from it), `:38-42` (Tier-1 fields list).

**VERIFIED by running** the A2 scene with `$ISAAC_PY`: `joint_pos.torch` → `tensor([[0.,0.,0.,0.],[0.,0.,0.,0.]], device='cuda:0')`, `joint_vel.torch` similarly, `body_link_pose_w.torch[0,:2]` → `tensor([[1.5,0.,0.,0.,0.,0.,1.],[1.5,0.,0.,0.,0.,0.,1.]], device='cuda:0')`; after effort steps the same accessors returned non-zero values.

## 6. External wrench on a body (the "human push") + every-substep behaviour

**Answer.** Use `robot.permanent_wrench_composer` (persists across steps) or `robot.instantaneous_wrench_composer` (one step). Reset, then `add_forces_and_torques_index(forces, torques, body_ids)` (or `set_forces_and_torques_index`), and call `robot.write_data_to_sim()` before `sim.step()`. Under Newton the composer writes into the shared `state.body_f` view (`articulation_data.py:1569`), which `NewtonManager` **stages once and re-applies before every solver substep** (fix PR #8269, commit `5608601d4`, `newton_manager.py:1680-1714`). A write therefore lasts exactly one `step()` call and delivers the full impulse regardless of `num_substeps`.

```python
w = robot.permanent_wrench_composer
w.reset()
w.add_forces_and_torques_index(
    forces=torch.tensor([[0., 0., 9.81]] * robot.num_instances, device=sim.device),
    torques=torch.zeros((robot.num_instances, 3), device=sim.device),
    body_ids=torch.tensor([2], device=sim.device),
)
robot.write_data_to_sim()   # scatters into state.body_f; do this EVERY step
sim.step(render=False)
# Direct escape hatch: robot.data._sim_bind_body_external_wrench is the raw wp array
# backed by state_0.body_f[:, 0] (private; composer is the supported API).
```

**Where found.** `source/isaaclab/isaaclab/assets/articulation/base_articulation.py:464-487` (composer properties/docstrings); `source/isaaclab/isaaclab/utils/wrench_composer.py:230` (`add_forces_and_torques_index`), `:294` (`set_forces_and_torques_index`), `:592` (`reset`); `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py:388-440` (`write_data_to_sim` scatter); `articulation_data.py:1569` (`get_attribute("body_f", state_0)`); `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py:1680-1714` (`_stage_body_forces` / `_apply_staged_body_forces` in `_run_solver_substeps`). Git: `git -C $ISAACLAB_DIR log --oneline | grep -i wrench` → `5608601d4 Apply Newton external wrenches on every solver substep (#8269)`; `git merge-base --is-ancestor 5608601d4 HEAD` → exit 0 (present in checkout). The commit message explicitly states the old bug delivered `1/num_substeps` of the impulse and that the fix copies `state_0.body_f` at the start of each step into every substep's input state.

**VERIFIED by running** the snippet above with `$ISAAC_PY`: after `w.reset()` + `add_forces_and_torques_index` + `write_data_to_sim()`, `wp.to_torch(robot.data._sim_bind_body_external_wrench)[0]` showed `9.81` in the z row of body 2 and zeros elsewhere; after `sim.step()` the buffer was cleared back to zero, as the manager contract says. The numeric "full impulse with substeps=4" check is the A4 gate (`ops/g7_wrench.py`) and has not been run as part of A2.

## 4. Gravity / bias compensation torque (`qfrc_bias` equivalent)

**Answer.** Use `robot.data.gravity_compensation_forces.torch`. On Newton this calls `ArticulationView.eval_inverse_dynamics_passive` (Newton's RNEA "passive forces") on the current FK state, so it is the gravity + velocity-dependent bias term `g(q) + C(q,dq)dq` (not just static gravity). Shape `(num_envs, num_joints + num_base_dofs)`; for a fixed-base arm it is `(N, num_joints)`, and joint `j` maps to column `j + num_base_dofs`. MuJoCo-Warp's `qfrc_bias` is **not** exposed (`dir(mujoco_warp)` has only `rne`, `rne_postconstraint`), so do not go looking for it. Controller usage pattern: `tau = g + Kp(q_des - q) - Kd*dq`.

```python
Kp, Kd = 80.0, 4.0                        # example gains, set by the skill layer
q_des = robot.data.joint_pos.torch.clone()          # (N, J); hold current pose
g = robot.data.gravity_compensation_forces.torch    # (N, J+base_dofs)
tau = g[:, robot.num_base_dofs:] + Kp * (q_des - robot.data.joint_pos.torch) \
      - Kd * robot.data.joint_vel.torch
robot.actuators.target_command.set_effort_index(value=tau, full_data=True)
robot.write_data_to_sim(); sim.step(render=False)
```

**Where found.** `source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py:943-955` (API contract / shape / equation); `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py:1012-1058` (Newton implementation calls `eval_inverse_dynamics_passive` then `gather_dof_force_rows`); `source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py:662` and `pink_task_space_actions.py:335` (real controller call sites using `.torch`). Probed with `$ISAAC_PY -c "import mujoco_warp as m; print([n for n in dir(m) if 'bias' in n or 'qfrc' in n or 'rne' in n])"` → `['rne', 'rne_postconstraint']`.

**VERIFIED by running** the 5-body, 5 kg branching articulation in Newton: `gravity_compensation_forces.torch` → `[[0, 0, 49.05, 0, 0, 0, 0, 0, 0, 0]]`, i.e. exactly 5 kg × 9.81 in the free-base z slot. `hasattr(robot._root_view, 'eval_inverse_dynamics_passive')` → `True`.

## 5. Jacobian exposure

**Answer.** Yes — Isaac Lab 3.0 exposes Jacobian-like quantities directly on `ArticulationData`, backed by Newton's `ArticulationView.eval_jacobian`: `body_com_jacobian_w` `(N, B, 6, J+base_dofs)` (COM-referenced), `body_link_jacobian_w` `(N, B, 6, J+base_dofs)` (origin-referenced; use this one for task-space control / fingertip IK), and `mass_matrix` `(N, J+base_dofs, J+base_dofs)`. Body `b`, row `r`, joint `j` → `[env, b, r, j + num_base_dofs]`. `rtp/control/fr3_kinematics.jacobian` (A1) is still useful as an independent analytic check, but is not required just to get a Jacobian.

```python
J = robot.data.body_link_jacobian_w.torch     # (N, B, 6, J+base_dofs)
fingertip_id = robot.data.body_names.index("left_tip")
J_tip = J[:, fingertip_id, :, :]              # (N, 6, J+base_dofs)
H = robot.data.mass_matrix.torch              # (N, J+base_dofs, J+base_dofs)
# Newton-side primitives also available: robot._root_view.eval_jacobian(state, J=..., joint_S_s=...)
```

**Where found.** `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py:888-927` (`body_com_jacobian_w`, `eval_jacobian`), `:929-957` (`body_link_jacobian_w`, COM→origin shift), `:959-1010` (`mass_matrix`); `source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py` (base API). Controllers that consume it: `source/isaaclab/isaaclab/controllers/operational_space.py` and `source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py`. No `jacobian` hits in `isaaclab/controllers` beyond those consumers.

**VERIFIED by running** the A2 scene: `body_com_jacobian_w.torch` shape `(2, 5, 6, 10)`, non-zero entries 36, body-2 slice showed `[1,0,...]`/`[0,0,0,0,0,1,1,1,...]` rows; `body_link_jacobian_w.torch` shape `(2, 5, 6, 10)`; `mass_matrix.torch` shape `(2, 10, 10)` with positive diagonal.

## 7. Parallel envs + decimation/substeps for a 1 kHz inner loop

**Answer.** `InteractiveSceneCfg(num_envs=N, env_spacing=...)` clones the articulated asset N times (regex `prim_path="{ENV_REGEX_NS}/Robot"`). `SimulationCfg(dt=...)` sets the physics step; `NewtonCfg(num_substeps=k)` subdivides it, so solver dt = `dt / k`. Decimation is set on the env (`ManagerBasedEnvCfg.decimation`) and forwarded to the backend via `set_decimation`; when all actuators are graphable `sim.step()` runs the whole `decimation × (actuators + num_substeps)` loop internally and `handles_decimation()` is True. For 1 kHz control: `dt=1/1000`, `num_substeps=4` gives a 4 kHz solver tick; policy decimation then multiplies the step size.

```python
sim_cfg = SimulationCfg(device="cuda:0", dt=1.0/1000,
    physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=4))
scene_cfg = InteractiveSceneCfg(num_envs=4, env_spacing=3.0)   # 4 cloned arms
# env: ManagerBasedEnvCfg(sim=sim_cfg, scene=scene_cfg, decimation=4)
# or, without an env:
sim.physics_manager.set_decimation(4)
robot.write_data_to_sim(); sim.step(render=False); scene.update(sim.get_physics_dt())
```

**Where found.** `source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py:171-175` (`num_substeps`, `collision_decimation`); `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py:1600-1605` (solver dt = physics dt / substeps), `:690-760` (`step()` all-graphable path), `:1992-2014` (`set_decimation`, `handles_decimation`); `source/isaaclab/isaaclab/envs/manager_based_env.py:203-206,552-554` (env decimation forwarding / step count); `source/isaaclab/isaaclab/envs/direct_marl_env_cfg.py:54-57` (decimation semantics).

**VERIFIED by running** the A2 scene with `num_envs=2`, `dt=1/1000`, `num_substeps=4`: printed `solver_dt 0.00025`, `handles_decimation True`; after `NM.set_decimation(4)` five `sim.step()` calls advanced `physics_manager.get_simulation_time()` by exactly `0.02` s (= 5 × 4 × 0.001), and `NM._decimation == 4`.

## 8. Contact force sensing on fingertips (what works on Newton)

**Answer.** Use `isaaclab.sensors.ContactSensorCfg` (body mode) on the fingertip prim: `data.net_forces_w.torch` (N, sensors, 3) and `net_friction_forces_w.torch` work; with `filter_prim_paths_expr` you also get `force_matrix_w` / `contact_pos_w`. Newton also has `NewtonContactSensorCfg` (shape mode) in `isaaclab_newton.sensors.contact_sensor`. `track_pose=True` / `data.pose_w` is **not** supported — it raises `NotImplementedError`. For a Newton contact sensor, `track_contact_points=True` requires filter objects or it is silently disabled (`contact_sensor_cfg.py:35-41`). Use `update_period=0.0` and call `sensor.update(dt, force_recompute=True)` after `sim.step()` when not using lazy sensor updates.

```python
from isaaclab.sensors import ContactSensorCfg
scene_cfg.tip_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Finger", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    track_friction_forces=True, track_contact_points=True, update_period=0.0, history_length=1)
# after sim.reset():
sensor = scene["tip_contact"]
for _ in range(90):
    sim.step(render=False); scene.update(sim.get_physics_dt())
    sensor.update(sim.get_physics_dt(), force_recompute=True)
print(sensor.data.net_forces_w.torch)   # (N, 1, 3), world-frame force on the fingertip
```

**Where found.** `source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py:19-45` (body mode, friction, filters); `source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor/contact_sensor_data.py:48` (`pose_w` raises), `:83,112,208,249` (net forces, matrix, friction, contact positions); `contact_sensor_cfg.py:35-41` (track_contact_points needs filters); `source/isaaclab_newton/test/sensors/test_contact_sensor.py:362-430` (`test_resting_object_contact_force`), `:727` (`test_finger_contact_sensor_isolation`), `:484` (`test_filtered_contact_forces_and_points`).

**VERIFIED by running** a 2-env Newton scene with a 2 kg box ("Finger") resting on a ground plane, `ContactSensorCfg(prim_path=..., track_friction_forces=True, update_period=0.0, history_length=1)`, 90 steps at dt=1/120: `sensor_names ['Finger']`, `net_forces_w.torch` → `[[[1.19e-07, -2.38e-07, 19.6200]], ...]` for both envs (= 2 kg × 9.81 upward), `net_friction_forces_w` ≈ 0, `net_forces_w_history.shape == (2, 1, 1, 3)`, and `pose_w` raised `NotImplementedError: pose_w is not supported by the Newton contact sensor.` A filtered variant on the ground plane failed with `ValueError: No bodies matched the counterpart pattern(s)` — a `GroundPlaneCfg` is not a body in the Newton model; filter against an actual `RigidObjectCfg` body.

## UNRESOLVED

### MJCF/URDF runtime conversion in the Newton-only venv
Commands run:
- `$ISAAC_PY -c "from isaaclab.sim.spawners.from_files.from_files_cfg import MjcfFileCfg; from isaaclab.sim.converters import MjcfConverter; ... MjcfConverter(MjcfFileCfg(asset_path=..., usd_dir=...)).convert()"` → `ModuleNotFoundError: No module named 'isaacsim'` (raised at `isaaclab/sim/converters/mjcf_converter.py:71`).
- `grep -rn "MjcfFileCfg" $ISAACLAB_DIR/source --include=*.py` found only the cfg definition and `spawn_from_mjcf`; no test uses it under Newton.
Conclusion: `MjcfFileCfg`/`UrdfFileCfg` cannot convert in this venv. For the cached FR3 MJCF use `newton.ModelBuilder.add_mjcf()` (`ops/g5_fr3_load.py:45`) or pre-convert the MJCF/URDF to USD with an Isaac Sim installation and spawn that USD with `UsdFileCfg`.

### Not numerically measured during A2 (deferred to A4)
- The "wrench is applied on EVERY solver substep" claim is established from the source (`newton_manager.py:1680-1714`) and the backported PR text of commit `5608601d4`, plus the observed `body_f` write/clear across one step. I did **not** measure `Δp` vs `F·t` at `num_substeps=4`; that is gate G-7 in `ops/g7_wrench.py`. Commands to run it: `source ops/env.sh && $ISAAC_PY ops/g7_wrench.py --out reports/g7_wrench.json` (script is A4, not written during A2).
- A `ContactSensorCfg` filter aimed at `/World/Ground` (a `GroundPlaneCfg`) failed with `ValueError: No bodies matched the counterpart pattern(s)`; the pose/friction checks above used an unfiltered sensor against the ground. Filtered force-matrix sensing needs a real `RigidObjectCfg` counterpart body — the pattern is exercised by `test_contact_sensor.py:484` and `:922`, not re-verified here.
