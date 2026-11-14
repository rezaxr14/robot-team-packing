"""PackingSim: two Franka arms, a table, a three-compartment bag and ten rigid items in Isaac Lab 2.3.x.

IMPLEMENTS rtp.simworld.SimBackend with SCRIPTED differential-IK skills (grasp, place, release, retreat), so the
closed loop (rtp.loop.run_episode through rtp.simworld.SimWorld) runs on real physics without RL. The same
skills generate demonstrations for distilling the blind arm policy (isaac/collect_demos.py).

STATUS: written against the Isaac Lab 2.3 docs and tutorials; NEVER RUN (the authoring sandbox has no Isaac Sim).
Use isaac/bringup/01..06 in order; each checks one layer and prints diagnostics.
VERIFIED in docs: DifferentialIKController(Cfg), FRANKA_PANDA_HIGH_PD_CFG (+ .replace), SceneEntityCfg(joint_names,
  body_names).resolve(scene), robot.set_joint_position_target(.., joint_ids=..), subtract_frame_transforms, SimulationCfg,
  InteractiveScene, sim.reset(). VERIFY on first run (from memory): shape-spawner kwargs, RigidObject write_* methods,
  root_physx_view.get_jacobians / set_masses, Camera API, TCP offset and finger stiffness, `.data` field names.
Quaternions: Isaac Lab 2.x is (w, x, y, z). For 3.0 (x, y, z, w) set QUAT_ORDER and fix the two helpers below.

Import AFTER the simulation app is launched (see bringup scripts).
"""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from rtp.datagen import sample_world
from rtp.simbridge import Layout, ObjState, report_from_states, SIM_FORCE_MARGIN_N
from rtp.simworld import SkillReport

QUAT_ORDER = "wxyz"
TOP_DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)          # hand z-axis pointing down (wxyz); fingers then open along world y
TCP_OFFSET = 0.1034                            # panda_hand frame -> fingertip centre (VERIFY)
FINGER_OPEN = 0.04
FINGER_STIFFNESS = 2.0e3                       # N/m, ImplicitActuator on the hand (VERIFY against the cfg actuators)
HOVER = 0.16                                   # clearance above items/bag (m)
LIFT_OK = 0.04                                 # object must rise this much to count as held

# kind, dims (box: x,y,z; cylinder: radius,height). Long box axis along x, short (grasped) axis along y.
SIM_DIMS = {"book": ("box", (0.10, 0.065, 0.03)), "notebook": ("box", (0.08, 0.055, 0.015)),
            "pencilcase": ("box", (0.10, 0.04, 0.03)), "bottle": ("cyl", (0.03, 0.12)),
            "lunchbox": ("box", (0.09, 0.06, 0.05)), "laptop": ("box", (0.14, 0.07, 0.012)),
            "mug": ("cyl", (0.035, 0.07)), "plate": ("cyl", (0.05, 0.01))}
ITEM_NAMES = ["book1", "book2", "notebook1", "notebook2", "pencilcase1", "bottle1", "lunchbox1", "laptop1", "mug", "plate"]
ITEM_CLASS = {n: n.rstrip("0123456789") for n in ITEM_NAMES}
ZONE_X = {"table_left": -0.34, "table_center": 0.0, "table_right": 0.34}
ROW_Y = (-0.02, 0.07, 0.16)                    # rows used to spread items inside a zone
ARM_BASE = {"L": (-0.46, -0.30, 0.0), "R": (0.46, -0.30, 0.0)}
HOME_Q = {"panda_joint1": 0.0, "panda_joint2": -0.5, "panda_joint3": 0.0, "panda_joint4": -2.3,
          "panda_joint5": 0.0, "panda_joint6": 1.9, "panda_joint7": 0.8, "panda_finger_joint.*": FINGER_OPEN}
LAYOUT = Layout(zone_axis=0, left_max=-0.17, right_min=0.17, table_z=0.0, bag_center=(0.0, 0.42), bag_half=(0.20, 0.17),
                bag_floor_z=0.0, bag_height=0.30, laptop_slot_depth=0.11, front_pocket_depth=0.08)


def _vis(color):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color)


def _spawn(cls_name, mass, friction=1.0):
    kind, d = SIM_DIMS[cls_name]
    common = dict(rigid_props=sim_utils.RigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                  collision_props=sim_utils.CollisionPropertiesCfg(),
                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
                  visual_material=_vis((0.6, 0.5, 0.3)))
    if kind == "box":
        return sim_utils.CuboidCfg(size=d, **common)
    return sim_utils.CylinderCfg(radius=d[0], height=d[1], axis="Z", **common)


def _item(name, mass=0.5):
    cls_name = ITEM_CLASS[name]
    return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/" + name, spawn=_spawn(cls_name, mass),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.5, 0.2)))   # parked until reset


def _static_box(name, size, pos, color=(0.35, 0.3, 0.25)):
    return AssetBaseCfg(prim_path="{ENV_REGEX_NS}/" + name,
                        spawn=sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                                                  visual_material=_vis(color)),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=pos))


def _arm_cfg(arm):
    cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot" + arm)
    yaw = math.pi / 2
    cfg.init_state = cfg.init_state.replace(pos=ARM_BASE[arm], rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
                                           joint_pos=HOME_Q)
    return cfg


@configclass
class PackSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)))
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8)))
    table = _static_box("Table", (1.4, 1.0, 0.05), (0.0, 0.1, -0.025), (0.55, 0.45, 0.35))
    bag_floor = _static_box("BagFloor", (0.40, 0.34, 0.01), (0.0, 0.42, 0.005))
    bag_wall_n = _static_box("BagWallN", (0.40, 0.01, 0.12), (0.0, 0.595, 0.06))
    bag_wall_s = _static_box("BagWallS", (0.40, 0.01, 0.12), (0.0, 0.245, 0.06))
    bag_wall_e = _static_box("BagWallE", (0.01, 0.34, 0.12), (0.205, 0.42, 0.06))
    bag_wall_w = _static_box("BagWallW", (0.01, 0.34, 0.12), (-0.205, 0.42, 0.06))
    div_laptop = _static_box("DivLaptop", (0.40, 0.006, 0.08), (0.0, 0.42 - 0.17 + 0.11, 0.04))
    div_front = _static_box("DivFront", (0.40, 0.006, 0.08), (0.0, 0.42 + 0.17 - 0.08, 0.04))
    robot_L = _arm_cfg("L")
    robot_R = _arm_cfg("R")
    book1 = _item("book1"); book2 = _item("book2")
    notebook1 = _item("notebook1"); notebook2 = _item("notebook2")
    pencilcase1 = _item("pencilcase1"); bottle1 = _item("bottle1")
    lunchbox1 = _item("lunchbox1"); laptop1 = _item("laptop1")
    mug = _item("mug"); plate = _item("plate")


def _t(x):
    """Unwrap Isaac Lab 3.0 data wrappers (.torch) if present."""
    return getattr(x, "torch", x)


class PackingSim:
    """SimBackend: reset(seed), run_skill(arm, skill, obj, params) -> SkillReport, object_states()."""

    arm_zones = {"L": {"table_left", "table_center", "bag"}, "R": {"table_right", "table_center", "bag"}}
    payload = {"L": 2.5, "R": 2.5}
    force_limit = {"L": 35.0, "R": 35.0}

    def __init__(self, sim, scene, layout: Layout = LAYOUT, step_substeps: int = 1):
        self.sim, self.scene, self.layout = sim, scene, layout
        self.dt = sim.get_physics_dt()
        self.arms = {a: scene["robot_" + a] for a in ("L", "R")}
        self.items = {n: scene[n] for n in ITEM_NAMES}
        self.ent = {a: SceneEntityCfg("robot_" + a, joint_names=["panda_joint.*"], body_names=["panda_hand"]) for a in "LR"}
        self.fingers = {a: SceneEntityCfg("robot_" + a, joint_names=["panda_finger_joint.*"]) for a in "LR"}
        for c in list(self.ent.values()) + list(self.fingers.values()):
            c.resolve(scene)
        self.ik = {a: DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                                                           ik_method="dls"), num_envs=scene.num_envs,
                                               device=sim.device) for a in "LR"}
        self.held, self.touched, self.peaks, self.truth = {}, set(), {}, None
        self.init_regions = {}
        self.grip_open = {a: FINGER_OPEN for a in "LR"}
        self.log = []
        self.trace = None                      # list to record control steps (07_collect_demos.py sets it)
        self.last_q = {}                       # last joint target per arm (held during gripper phases)
        self.lock_holder = None                # arm that currently owns the bag lock (regression cases set it)

    # ------------------------------------------------------------------ stepping / control primitives
    def _step(self, n=1):
        for _ in range(n):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.dt)

    def _ee_pose_b(self, arm):
        r, e = self.arms[arm], self.ent[arm]
        ee_w = _t(r.data.body_state_w)[:, e.body_ids[0], 0:7]
        root = _t(r.data.root_state_w)[:, 0:7]
        return subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])

    def _set_fingers(self, arm, half_width):
        r, f = self.arms[arm], self.fingers[arm]
        tgt = torch.full((self.scene.num_envs, len(f.joint_ids)), float(half_width), device=self.sim.device)
        r.set_joint_position_target(tgt, joint_ids=f.joint_ids)

    def goto(self, arm, pos_w, quat_w=TOP_DOWN_QUAT, max_steps=240, tol=0.006):
        """Drive the hand to a world pose with differential IK. Returns (reached, final_position_error)."""
        r, e = self.arms[arm], self.ent[arm]
        root = _t(r.data.root_state_w)[:, 0:7]
        tgt_pos = torch.tensor([pos_w], device=self.sim.device, dtype=torch.float32).repeat(self.scene.num_envs, 1)
        tgt_quat = torch.tensor([quat_w], device=self.sim.device, dtype=torch.float32).repeat(self.scene.num_envs, 1)
        pos_b, quat_b = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], tgt_pos, tgt_quat)
        cmd = torch.cat([pos_b, quat_b], dim=-1)
        self.ik[arm].reset()
        self.ik[arm].set_command(cmd)
        jac_idx = e.body_ids[0] - 1                                    # fixed-base articulation: no root body row
        err = float("inf")
        for _ in range(max_steps):
            ee_pos_b, ee_quat_b = self._ee_pose_b(arm)
            jac = r.root_physx_view.get_jacobians()[:, jac_idx, :, e.joint_ids]
            q = _t(r.data.joint_pos)[:, e.joint_ids]
            q_des = self.ik[arm].compute(ee_pos_b, ee_quat_b, jac, q)
            r.set_joint_position_target(q_des, joint_ids=e.joint_ids)
            self.last_q[arm] = q_des.clone()
            self._set_fingers(arm, self.grip_open[arm])
            if self.trace is not None:
                self.trace.append({"arm": arm, "proprio": self._proprio(arm), "contact": self._contact(arm),
                                   "action": q_des[0].tolist() + [self.grip_open[arm]]})
            self._step()
            err = float((self._ee_pose_b(arm)[0] - pos_b).norm())
            if err < tol:
                return True, err
        return False, err

    def _proprio(self, arm):
        """15-d proprioception: 7 joint pos, 7 joint vel, gripper width (matches train/distill_arm.py)."""
        r, e, f = self.arms[arm], self.ent[arm], self.fingers[arm]
        q = _t(r.data.joint_pos)[0, e.joint_ids]
        qd = _t(r.data.joint_vel)[0, e.joint_ids]
        w = _t(r.data.joint_pos)[0, f.joint_ids].sum()
        return torch.cat([q, qd, w.reshape(1)]).tolist()

    def _contact(self, arm):
        """8-d contact/force proxy from finger joints: pos(2), vel(2), applied effort(2), holding flag, pad. VERIFY the
        applied_torque field name; replace by a ContactSensor reading if you add one."""
        r, f = self.arms[arm], self.fingers[arm]
        pos = _t(r.data.joint_pos)[0, f.joint_ids].tolist()
        vel = _t(r.data.joint_vel)[0, f.joint_ids].tolist()
        try:
            eff = _t(r.data.applied_torque)[0, f.joint_ids].tolist()
        except AttributeError:
            eff = [0.0, 0.0]
        return (pos + vel + eff + [1.0 if arm in self.held.values() else 0.0, 0.0])[:8]

    def _hold(self, arm, n):
        """Hold the last joint target for n steps (gripper closing/opening phases), recording them like goto steps."""
        r, e = self.arms[arm], self.ent[arm]
        q = self.last_q.get(arm)
        for _ in range(n):
            if q is not None:
                r.set_joint_position_target(q, joint_ids=e.joint_ids)
            self._set_fingers(arm, self.grip_open[arm])
            if self.trace is not None and q is not None:
                self.trace.append({"arm": arm, "proprio": self._proprio(arm), "contact": self._contact(arm),
                                   "action": q[0].tolist() + [self.grip_open[arm]]})
            self._step()

    def render_eye(self, view: str) -> bytes:
        """PNG bytes of Eye camera `view` ('A' or 'B') for the current scene state (build(with_cameras=True))."""
        import io
        from PIL import Image
        from isaac.cameras import EYE_POSES
        cam = self.scene["camera_" + view]
        eye, tgt = EYE_POSES[view]
        cam.set_world_poses_from_view(torch.tensor([eye], device=self.sim.device), torch.tensor([tgt], device=self.sim.device))
        self._step(3)
        cam.update(self.dt)
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()

    def _obj_pos(self, name):
        return tuple(float(v) for v in _t(self.items[name].data.root_state_w)[0, 0:3])

    # ------------------------------------------------------------------ state for the bridge
    def object_states(self):
        out = []
        w = self.truth
        for n in ITEM_NAMES:
            d = _t(self.items[n].data.root_state_w)[0]
            pos, quat = tuple(float(v) for v in d[0:3]), tuple(float(v) for v in d[3:7])
            o = w.objs[n]
            out.append(ObjState(name=n, cls=o.cls, pos=pos, quat=quat, quat_order=QUAT_ORDER, mass=o.mass,
                                fragility=o.fragility, size=o.size, friction=o.friction, held_by=self.held.get(n),
                                peak_contact_force=self.peaks.get(n, 0.0), touched=n in self.touched,
                                init_region=self.init_regions.get(n, ""), is_distractor=o.is_distractor))
        return out

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int):
        w = sample_world(seed, p_arm_noise=0.0)
        self.truth = w
        self.held, self.touched, self.peaks = {}, set(), {}
        rng = random.Random(f"layout|{seed}")
        count = {z: 0 for z in ZONE_X}
        for n in ITEM_NAMES:
            o = w.objs[n]
            z = o.region if o.region in ZONE_X else "table_center"
            row = count[z] % len(ROW_Y)
            col = count[z] // len(ROW_Y)
            count[z] += 1
            kind, d = SIM_DIMS[ITEM_CLASS[n]]
            h = d[2] if kind == "box" else d[1]
            x = ZONE_X[z] + (-0.06 if col == 0 else 0.07) + rng.uniform(-0.01, 0.01)
            y = ROW_Y[row] + rng.uniform(-0.005, 0.005)
            pose = torch.tensor([[x, y, h / 2 + 0.002, 1.0, 0.0, 0.0, 0.0]], device=self.sim.device)
            self.items[n].write_root_pose_to_sim(pose)
            self.items[n].write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))
            try:
                self.items[n].root_physx_view.set_masses(torch.tensor([[o.mass]]), torch.tensor([0], dtype=torch.int32))
            except Exception as e:                                       # noqa: BLE001
                self.log.append(f"set_masses failed for {n}: {e}")
        for a in "LR":
            r = self.arms[a]
            r.write_joint_state_to_sim(_t(r.data.default_joint_pos).clone(), _t(r.data.default_joint_vel).clone())
            self.grip_open[a] = FINGER_OPEN
        self.scene.reset()
        self._step(60)                                                   # settle
        self.init_regions = {s.name: _region(s, self.layout) for s in self.object_states()}
        for s in self.object_states():
            self.truth.objs[s.name].init_region = self.init_regions[s.name]
        for a in "LR":
            self.log.append(f"{a} home ee {[round(float(v), 3) for v in self._ee_pose_b(a)[0][0]]}")

    # ------------------------------------------------------------------ skills
    def run_skill(self, arm, skill, obj, params):
        pre = self.object_states()
        fn = getattr(self, f"_sk_{skill}", None)
        if fn is not None and (obj is None or obj in self.items):
            try:
                fn(arm, obj, params or {}, pre)
            except Exception as e:                                       # noqa: BLE001
                self.log.append(f"{skill} {arm} {obj} raised {type(e).__name__}: {e}")
        post = self.object_states()
        cat, touch = report_from_states(arm, skill, obj, params or {}, pre, post, self.layout, self.arm_zones,
                                        self.payload, lock_holder=self.lock_holder)
        return SkillReport(cat, touch)

    def _feasible_grasp(self, arm, o_pre):
        reg = _region(o_pre, self.layout)
        return (arm not in self.held.values() and reg in self.arm_zones[arm] and o_pre.mass <= self.payload[arm])

    def _sk_grasp(self, arm, obj, p, pre):
        o = {s.name: s for s in pre}[obj]
        if not self._feasible_grasp(arm, o):
            return
        force = min(float(p.get("force", 10.0)), self.force_limit[arm])
        kind, d = SIM_DIMS[ITEM_CLASS[obj]]
        width = d[1] if kind == "box" else 2 * d[0]                       # grasped across world y
        height = d[2] if kind == "box" else d[1]
        x, y, z = o.pos
        grasp_z = z + TCP_OFFSET - min(0.004, height / 4)
        self.grip_open[arm] = FINGER_OPEN
        ok1, _ = self.goto(arm, (x, y, z + HOVER))
        ok2, _ = self.goto(arm, (x, y, grasp_z), max_steps=200)
        if not (ok1 and ok2):
            return
        delta = force / FINGER_STIFFNESS
        self.grip_open[arm] = max(0.0, width / 2 - delta)
        self._hold(arm, 40)
        self.touched.add(obj)
        self.peaks[obj] = max(self.peaks.get(obj, 0.0), force + SIM_FORCE_MARGIN_N)     # proxy: replace by measured effort
        self.goto(arm, (x, y, z + HOVER), max_steps=160)
        self._hold(arm, 30)
        if self._obj_pos(obj)[2] > z + LIFT_OK and not _crushed(self, obj):
            self.held[obj] = arm
        else:
            self.grip_open[arm] = FINGER_OPEN
            self._hold(arm, 30)

    def _sk_place(self, arm, obj, p, pre):
        if self.held.get(obj) != arm:
            return
        comp = p.get("compartment", "main")
        cx, cy = self.layout.bag_center
        hy = self.layout.bag_half[1]
        cyl = {"laptop": cy - hy + self.layout.laptop_slot_depth / 2,
               "front": cy + hy - self.layout.front_pocket_depth / 2,
               "main": cy + (self.layout.laptop_slot_depth - self.layout.front_pocket_depth) / 2}[comp]
        stack = sum(1 for s in pre if _region(s, self.layout) == f"bag:{comp}")
        kind, d = SIM_DIMS[ITEM_CLASS[obj]]
        h = d[2] if kind == "box" else d[1]
        drop_z = 0.02 + 0.035 * stack + h / 2 + TCP_OFFSET
        self.goto(arm, (cx, cyl, drop_z + HOVER))
        self.goto(arm, (cx, cyl, drop_z), max_steps=160)
        self.grip_open[arm] = FINGER_OPEN
        self._hold(arm, 40)
        self.held.pop(obj, None)
        if p.get("orientation") == "flat" and ITEM_CLASS[obj] == "bottle":
            pos = self._obj_pos(obj)                                      # SIMULATED laying-down: teleport to a lying pose
            q = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
            self.items[obj].write_root_pose_to_sim(torch.tensor([[pos[0], pos[1], pos[2], *q]], device=self.sim.device))
        self.goto(arm, (cx, cyl, drop_z + HOVER), max_steps=120)
        self._hold(arm, 20)

    def _sk_weigh(self, arm, obj, p, pre):
        """Grasp gently, lift a few cm, put the item back. The 'load cell' reading is the simulator's mass for the item
        (report_from_states returns it); replace by an estimate from finger joint effort for a physical-sensor claim."""
        o = {s.name: s for s in pre}[obj]
        if not self._feasible_grasp(arm, o):
            return
        kind, d = SIM_DIMS[ITEM_CLASS[obj]]
        width = d[1] if kind == "box" else 2 * d[0]
        height = d[2] if kind == "box" else d[1]
        x, y, z = o.pos
        grasp_z = z + TCP_OFFSET - min(0.004, height / 4)
        self.grip_open[arm] = FINGER_OPEN
        self.goto(arm, (x, y, z + HOVER))
        self.goto(arm, (x, y, grasp_z), max_steps=200)
        self.grip_open[arm] = max(0.0, width / 2 - 10.0 / FINGER_STIFFNESS)
        self._hold(arm, 30)
        self.touched.add(obj)
        self.goto(arm, (x, y, grasp_z + 0.05), max_steps=120)
        self._hold(arm, 20)
        self.goto(arm, (x, y, grasp_z), max_steps=120)
        self.grip_open[arm] = FINGER_OPEN
        self._hold(arm, 30)
        self.goto(arm, (x, y, z + HOVER), max_steps=120)

    def _sk_release(self, arm, obj, p, pre):
        if self.held.get(obj) == arm:
            self.grip_open[arm] = FINGER_OPEN
            self._hold(arm, 40)
            self.held.pop(obj, None)

    def _sk_retreat(self, arm, obj, p, pre):
        r = self.arms[arm]
        r.set_joint_position_target(_t(r.data.default_joint_pos).clone())
        self._step(90)

    def _sk_reach(self, arm, obj, p, pre):
        x, y, z = {s.name: s for s in pre}[obj].pos
        self.goto(arm, (x, y, z + HOVER))

    def _sk_push_aside(self, arm, obj, p, pre):
        """Sweep the closed gripper across the object along +x at fingertip height (moves it ~0.1 m)."""
        o = {s.name: s for s in pre}[obj]
        if _region(o, self.layout) not in self.arm_zones[arm]:
            return
        x, y, z = o.pos
        hz = z + TCP_OFFSET - 0.01
        self.grip_open[arm] = 0.0
        self.goto(arm, (x - 0.08, y, hz + HOVER))
        self.goto(arm, (x - 0.08, y, hz), max_steps=160)
        self.goto(arm, (x + 0.06, y, hz), max_steps=160)
        self.touched.add(obj)
        self.goto(arm, (x + 0.06, y, hz + HOVER))
        self.grip_open[arm] = FINGER_OPEN

    def _sk_handover(self, arm, obj, p, pre):
        """Table handover: put the object down at table_center, retreat, and the other arm grasps it there.
        (A mid-air two-gripper exchange is not scripted; this keeps the same outcome the symbolic world models.)"""
        to = p.get("to")
        if self.held.get(obj) != arm or to not in ("L", "R") or to == arm or to in self.held.values():
            return
        cx, cy = 0.0, 0.07
        kind, d = SIM_DIMS[ITEM_CLASS[obj]]
        h = d[2] if kind == "box" else d[1]
        self.goto(arm, (cx, cy, h / 2 + TCP_OFFSET + HOVER))
        self.goto(arm, (cx, cy, h / 2 + 0.004 + TCP_OFFSET), max_steps=160)
        self.grip_open[arm] = FINGER_OPEN
        self._hold(arm, 40)
        self.held.pop(obj, None)
        self.goto(arm, (cx, cy, h / 2 + TCP_OFFSET + HOVER), max_steps=120)
        self._sk_retreat(arm, None, {}, [])
        self._sk_grasp(to, obj, {"force": float(p.get("force", 12.0))}, self.object_states())

    # ---- helpers for regression cases and parking ----
    def teleport(self, name, zone, slot=0):
        """Put an item at a table zone slot (regression setup). Not a skill: no bookkeeping side effects."""
        kind, d = SIM_DIMS[ITEM_CLASS[name]]
        h = d[2] if kind == "box" else d[1]
        x = ZONE_X[zone] + (-0.06 if slot % 2 == 0 else 0.07)
        y = ROW_Y[(slot // 2) % len(ROW_Y)]
        self.items[name].write_root_pose_to_sim(torch.tensor([[x, y, h / 2 + 0.002, 1.0, 0.0, 0.0, 0.0]], device=self.sim.device))
        self.items[name].write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))

    def park(self, names):
        """Move unused items to the far corners of the table, out of every zone and the bag."""
        for i, n in enumerate(names):
            kind, d = SIM_DIMS[ITEM_CLASS[n]]
            h = d[2] if kind == "box" else d[1]
            x = (-0.62 if i % 2 == 0 else 0.62)
            y = 0.30 + 0.08 * (i // 2)
            self.items[n].write_root_pose_to_sim(torch.tensor([[x, y, h / 2 + 0.002, 1.0, 0.0, 0.0, 0.0]], device=self.sim.device))
            self.items[n].write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))


def _region(s: ObjState, lay: Layout) -> str:
    from rtp.simbridge import region_of
    return region_of(s, lay)


def _crushed(sim: PackingSim, obj: str) -> bool:
    from rtp.simbridge import crush_force_for
    o = sim.truth.objs[obj]
    return sim.peaks.get(obj, 0.0) > crush_force_for(o.fragility)


def build(sim_cfg_dt=1 / 120, device="cuda:0", num_envs=1, with_cameras=False):
    """Create SimulationContext + scene + PackingSim. Call after the Kit app is running.
    with_cameras=True adds the two Eye cameras (launch the app with --enable_cameras)."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=sim_cfg_dt, device=device))
    sim.set_camera_view([1.2, -1.0, 0.9], [0.0, 0.2, 0.0])
    if with_cameras:
        from isaac.cameras import PackSceneWithCamsCfg
        scene = InteractiveScene(PackSceneWithCamsCfg(num_envs=num_envs, env_spacing=3.0))
    else:
        scene = InteractiveScene(PackSceneCfg(num_envs=num_envs, env_spacing=3.0))
    sim.reset()
    return sim, scene, PackingSim(sim, scene)
