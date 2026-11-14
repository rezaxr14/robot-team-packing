# Isaac Lab design for Track B (two blind arms, bag, eight items, two distractors)

Status: DESIGN + partially grounded skeleton. Nothing in this folder has been run. Items are tagged
VERIFIED (read in the Isaac Lab docs while building) or UNVERIFIED (from memory; check against your install).

## Version pin
The API moved between Isaac Lab 2.x and 3.x (docs show `data.joint_pos` in 2.2.x and `data.joint_pos.torch` plus
`set_joint_effort_target_index(...)` on develop/3.0). Pick ONE version at Stage 0, record it in the lockfile, and
adapt the two accessor helpers at the top of `team_env.py`; everything else goes through them.

## What the docs confirm (VERIFIED)
- Multi-agent direct tasks use `isaaclab.envs.DirectMARLEnv` / `DirectMARLEnvCfg`; cfg fields: sim, scene,
  decimation, episode_length_s, possible_agents, observation_spaces, action_spaces, state_space.
- Direct-workflow hooks: `_setup_scene`, `_pre_physics_step`, `_apply_action`, `_get_observations`,
  `_get_rewards`, `_get_dones`, `_reset_idx` (documented for DirectRLEnv; MARL variants return per-agent dicts,
  UNVERIFIED in detail).
- Scene pattern: `Articulation(cfg)`, `spawn_ground_plane("/World/ground", cfg=GroundPlaneCfg())`,
  `scene.clone_environments(copy_from_source=False)`, `scene.filter_collisions(global_prim_paths=[])` for PhysX,
  register with `scene.articulations["name"] = asset`, light via `sim_utils.DomeLightCfg(...).func(...)`.

## Entities
- Two 7-DoF arms with parallel grippers on opposite sides of the table (robot config: UNVERIFIED, e.g. the Franka
  config shipped in isaaclab_assets; confirm the import path and gripper joint names in your version).
- Table with three zones along x: left < -0.20 m, center, right > 0.20 m (see `rtp.simbridge.Layout`).
- Bag: open box at the table's far edge, three compartments by depth (laptop slot at the back, main, front pocket).
- Eight items (2 books, 2 notebooks, pencil case, bottle, lunchbox, laptop) + 2 distractors (mug, plate), rigid
  bodies with randomised mass, friction, size from `rtp.datagen.CLASSES`. Fragility is a task-level attribute
  enforced by the crush threshold `simbridge.crush_force_for(fragility)` on peak contact force.
- Two fixed cameras (Eye A and Eye B) with different viewpoints so each leaves a region occluded.

## Observations (per arm; NO pixels, NO instruction)
proprio 15 (joint pos/vel, gripper width), contact 8 (fingertip forces, contact flags, slip flag), skill command
(skill id one-hot, target id, params: force, speed, orientation). Messages enter through the skill command only.

## Skills as controllers
Each skill is a policy conditioned on the command. Stage 1: privileged-state RL experts (they may see object poses),
reward shaped per skill. Stage 2: distill to the blind policy with `train/distill_arm.py` on expert rollouts.
Stage 3: online RL on the blind policy. After every stage run `rtp.skills.run_regression` through an
executor that classifies outcomes with `simbridge.classify_skill_outcome`, and `compare()` against the previous report.

## Shared definition of success
Simulator state -> `simbridge.ObjState` -> `simbridge.world_from_sim` -> `rules.check_final`. The same checker scores
symbolic episodes, GRPO dry runs and Isaac episodes, so a plan-vs-physical gap is a gap between predicted and
simulated outcomes of the SAME rule set.

## Episode driver (outside the env)
`rtp.loop.run_episode` is the closed loop and is simulator-agnostic through `rtp.simworld.SimWorld(backend)`.
A backend implements `reset`, `run_skill`, `object_states` (see `isaac/backend.py`); the outcome category of each
skill is derived from state before/after by `simbridge.report_from_states`. This whole path is validated with
`rtp.fakesim.FakeSim` (poses only): the loop gives identical outcomes through it as directly on the symbolic world
(tests/test_simworld.py), and the skill regression suite passes through it. What remains untested is only what
real physics changes: contact peaks, drops, controller failures, skill termination.
`Eye` becomes `rtp.perception.VLMEye` with `env.render_eye(view)`.

## Open items
Gripper/contact sensor config, crush detection through contact forces vs. deformation proxy, drop detection, bag
collision geometry for compartments, domain randomisation events, camera extrinsics, skill termination criteria.
