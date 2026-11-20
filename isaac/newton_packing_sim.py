"""A6 backend: `SimBackend` over the Newton two-arm packing scene.

Status (honest): the scene is stepped in Newton (SIM), but `run_skill` here is an explicit
**state-machine stand-in** (SIM-POSE) used to wire `SimWorld` / the harness to the scene before
A7 scripted skills are implemented. It does not claim physical grasp/contact outcomes and must
never be used as arm-certification evidence. A7 replaces `run_skill` with scripted IK paths and
measured Newton outcomes; the interface stays identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from rtp.simbridge import Layout, ObjState, report_from_states
from rtp.simworld import SkillReport

LAYOUT = Layout(table_z=0.90, bag_center=(0.0, 0.25), bag_half=(0.22, 0.14))


class NewtonPackingSim:
    kind = "sim_pose_standin"
    evidence_class = "SIM-POSE"   # scene is SIM; skill outcomes are not yet physical

    def __init__(self, layout: Layout | None = None):
        self.layout = layout or LAYOUT
        self.arm_zones = {"L": {"table_left", "table_center", "bag"},
                          "R": {"table_right", "table_center", "bag"}}
        self.payload = {"L": 2.5, "R": 2.5}  # 3 kg FR3 rating minus gripper
        self.force_limit = {"L": 40.0, "R": 40.0}
        self._model = self._state = self._control = None
        self._items = []
        self._states: list[ObjState] = []
        self._solver = self._cp = self._contacts = None

    # -- SimBackend -------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        self._seed = int(seed)
        import newton
        from ops.a6_scene import build

        (self._model, self._state, self._control, self._items,
         _offsets, _bodies) = build(int(seed))
        self._solver = newton.solvers.SolverMuJoCo(self._model, njmax=256, nconmax=256)
        self._cp = newton.CollisionPipeline(self._model)
        self._contacts = self._cp.contacts()
        self._states = []
        for name, x, y, hx, hy, hz in self._items:
            region = "table_left" if x < -0.20 else ("table_right" if x > 0.20 else "table_center")
            self._states.append(ObjState(name=name, cls="object",
                                         pos=(float(x), float(y), 0.90 + hz),
                                         mass=0.1, fragility=0.3, size=max(hx, hy, hz) * 2,
                                         friction=0.6, init_region=region))

    def step(self, n: int = 1, dt: float = 0.001) -> None:
        for _ in range(n):
            self._state.clear_forces()
            self._cp.collide(self._state, self._contacts)
            self._solver.step(self._state, self._state, self._control, self._contacts, dt)

    def object_states(self) -> list:
        return [ObjState(**s.__dict__) for s in self._states]

    def run_skill(self, arm: str, skill: str, obj: str | None, params: dict) -> SkillReport:
        # Reuse the canonical, already-tested skill rules instead of re-implementing them.
        # The scripted IK path (rtp.control.scripted_ik) is checked for every physical skill;
        # an unreachable target is reported as out_of_reach before the symbolic outcome.
        from rtp.skills import CATEGORIES
        from rtp.world import Obj, World, crush_limit, required_force
        from rtp.control.scripted_ik import solve_ik
        from rtp.control.fr3_kinematics import Q_HOME

        pre = self.object_states()
        by_name = {s.name: s for s in self._states}
        target = by_name.get(obj or "")
        if target is not None and skill in ("grasp", "weigh", "push_aside", "place"):
            if target.held_by is None:
                ik = solve_ik(Q_HOME, (target.pos[0], target.pos[1], target.pos[2]),
                              max_iters=120, pos_tol=5e-3)
                if not ik.converged:
                    return SkillReport(category="out_of_reach", touch={},
                                       detail="scripted IK could not reach target (SIM-POSE stand-in)")

        if skill == "grasp" and target is not None:
            if target.mass > self.payload.get(arm, 2.5):
                return SkillReport(category="overload", touch={}, detail="mass above payload")
            force = float(params.get("force", 10.0))
            probe = Obj(name=target.name, cls=target.cls, mass=target.mass,
                        fragility=target.fragility, size=target.size,
                        friction=target.friction, region=target.init_region)
            if force > crush_limit(probe):
                return SkillReport(category="crush", touch={"peak_force": force},
                                   detail="commanded force above crush limit")
            if force < required_force(probe):
                return SkillReport(category="slip", touch={"peak_force": force},
                                   detail="commanded force below slip threshold")

        objs = []
        for s in self._states:
            region = ("held:" + s.held_by) if s.held_by else s.init_region
            objs.append(Obj(name=s.name, cls=s.cls, mass=s.mass, fragility=s.fragility,
                            size=s.size, region=region, friction=s.friction,
                            is_distractor=s.is_distractor, init_region=s.init_region))
        w = World(objs, seed=self._seed, deterministic=True)
        w.holding = {a: next((s.name for s in self._states if s.held_by == a), None)
                     for a in ("L", "R")}
        w.lock_holder = getattr(self, "lock_holder", None)
        w.bag = {c: [] for c in ("main", "front", "laptop")}
        for s in self._states:
            if s.init_region.startswith("bag:"):
                w.bag[s.init_region.split(":", 1)[1]].append(s.name)
        res = w.execute(arm, skill, obj, params)
        # sync state
        world_by = w.objs
        for s in self._states:
            o = world_by[s.name]
            s.init_region = o.region
            s.held_by = o.region.split(":", 1)[1] if o.region.startswith("held:") else None
            s.touched = s.touched or o.moved
            if o.region.startswith("bag:"):
                s.pos = (self.layout.bag_center[0], self.layout.bag_center[1],
                         self.layout.bag_floor_z + s.size / 2)
        category = "success" if res.ok else (res.cause if res.cause in CATEGORIES else "other_failure")
        return SkillReport(category=category, touch=dict(res.touch),
                           detail="sim_pose_standin with canonical rules; not physical skill evidence")
