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
        self.payload = {"L": 3.0, "R": 3.0}
        self.force_limit = {"L": 40.0, "R": 40.0}
        self._model = self._state = self._control = None
        self._items = []
        self._states: list[ObjState] = []
        self._solver = self._cp = self._contacts = None

    # -- SimBackend -------------------------------------------------------------------
    def reset(self, seed: int) -> None:
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
        pre = self.object_states()
        by_name = {s.name: s for s in self._states}
        target = by_name.get(obj or "")
        touch = {}
        if target is not None:
            target.touched = True
            if skill == "grasp":
                target.held_by = arm
                touch["grasp_force_N"] = float(params.get("force", 5.0))
            elif skill in ("place", "push_aside"):
                target.held_by = None
                if skill == "place":
                    target.pos = (self.layout.bag_center[0], self.layout.bag_center[1],
                                  self.layout.bag_floor_z + target.size / 2)
                else:
                    target.pos = (target.pos[0] + 0.05, target.pos[1], target.pos[2])
            elif skill == "release":
                target.held_by = None
        elif skill == "handover":
            holder = next((s for s in self._states if s.held_by == arm), None)
            if holder is not None:
                holder.held_by = "R" if arm == "L" else "L"
        post = self.object_states()
        category, derived_touch = report_from_states(
            arm, skill, obj, params, pre, post, self.layout,
            self.arm_zones, self.payload,
            getattr(self, "lock_holder", None),
        )
        touch.update(derived_touch)
        return SkillReport(category=category, touch=touch,
                           detail="sim_pose_standin (A6); not physical skill evidence")
