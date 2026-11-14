"""Isaac Lab environment skeleton for the two-arm packing task.  >>> NOT RUNNABLE AS-IS <<<

See isaac/DESIGN.md for what is VERIFIED against the docs and what is UNVERIFIED. Lines marked `VERIFY` are
from memory. Everything that depends on a specific robot asset raises NotImplementedError on purpose.

The simulator-independent half (poses -> regions -> World -> rules.check_final; outcome classification) lives in
rtp/simbridge.py and IS tested; this file only fills `ObjState` from simulator tensors.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtp.simbridge import Layout, ObjState, world_from_sim

try:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from isaaclab.utils import configclass
    HAVE_ISAAC = True
except ImportError:                       # lets the file be imported for review / unit tests without Isaac Lab
    HAVE_ISAAC = False
    DirectMARLEnv = object
    DirectMARLEnvCfg = object
    configclass = lambda c: c

ARM_PROPRIO_DIM, ARM_CONTACT_DIM, ARM_ACT_DIM = 15, 8, 8
SKILL_DIM = 8 + 32 + 4
ARMS = ("arm_L", "arm_R")


# --- version-sensitive accessors: adapt these two for your Isaac Lab version (2.x vs 3.x differ) -------------
def joint_pos(asset):
    d = asset.data.joint_pos              # VERIFIED for 2.2.x; on 3.0/develop the docs show `.joint_pos.torch`
    return getattr(d, "torch", d)


def root_pose_w(obj):
    d = obj.data                          # VERIFY: root_pos_w / root_quat_w names and tensor-vs-wrapper access
    pos, quat = d.root_pos_w, d.root_quat_w
    return getattr(pos, "torch", pos), getattr(quat, "torch", quat)
# ------------------------------------------------------------------------------------------------------------


@configclass
class TeamPackEnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 30.0
    possible_agents = list(ARMS)
    action_spaces = {a: ARM_ACT_DIM for a in ARMS}
    observation_spaces = {a: ARM_PROPRIO_DIM + ARM_CONTACT_DIM + SKILL_DIM for a in ARMS}
    state_space = 0
    # VERIFIED field names; values below are placeholders to tune
    # sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0, replicate_physics=True)
    layout = Layout()


class TeamPackEnv(DirectMARLEnv):
    cfg: TeamPackEnvCfg

    def _setup_scene(self):
        # Pattern VERIFIED from the cartpole tutorial; asset specifics are not.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.arms = {a: self._spawn_arm(a) for a in ARMS}            # NotImplementedError until a robot is chosen
        self.items = self._spawn_items()                              # NotImplementedError: rigid bodies per class
        self.bag = self._spawn_bag()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])            # VERIFIED needed for PhysX replication
        for a, art in self.arms.items():
            self.scene.articulations[a] = art
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ---- asset spawners: intentionally unimplemented ----
    def _spawn_arm(self, name):
        raise NotImplementedError("choose the arm config (e.g. from isaaclab_assets) and offset it to its side of the table")

    def _spawn_items(self):
        raise NotImplementedError("RigidObject per item with randomised mass/friction/size from rtp.datagen.CLASSES")

    def _spawn_bag(self):
        raise NotImplementedError("open box with three compartments; match rtp.simbridge.Layout")

    # ---- control hooks ----
    def set_skill(self, arm: str, skill: str, target: str, params: dict):
        raise NotImplementedError("store the skill command; it is part of each arm's observation")

    def _pre_physics_step(self, actions):
        raise NotImplementedError

    def _apply_action(self):
        raise NotImplementedError

    def _get_observations(self):
        raise NotImplementedError("per-arm dict: proprio, contact/force, skill command; NO camera input")

    def _get_rewards(self):
        raise NotImplementedError("skill-level shaped rewards for expert RL (privileged state allowed there only)")

    def _get_dones(self):
        raise NotImplementedError

    def _reset_idx(self, env_ids):
        raise NotImplementedError

    def render_eye(self, view: str):
        raise NotImplementedError("return an RGB image for Eye A or B; feeds rtp.perception.VLMEye")

    # ---- bridge to the symbolic world (the conversion logic is tested in tests/test_simbridge.py) ----
    def object_states(self, env_id: int, init_regions: dict | None = None, touched: set | None = None,
                      peaks: dict | None = None, held: dict | None = None) -> list:
        out = []
        for name, obj in self.items.items():
            pos, quat = root_pose_w(obj)
            out.append(ObjState(name=name, cls=self.item_meta[name]["cls"], pos=tuple(pos[env_id].tolist()),
                                quat=tuple(quat[env_id].tolist()), mass=self.item_meta[name]["mass"],
                                fragility=self.item_meta[name]["fragility"], size=self.item_meta[name]["size"],
                                friction=self.item_meta[name]["friction"], held_by=(held or {}).get(name),
                                peak_contact_force=(peaks or {}).get(name, 0.0), touched=name in (touched or set()),
                                init_region=(init_regions or {}).get(name, ""),
                                is_distractor=self.item_meta[name].get("is_distractor", False)))
        return out

    def to_symbolic_world(self, env_id: int, pack_order=None, **kw):
        return world_from_sim(self.object_states(env_id, **kw), self.cfg.layout, pack_order)
