"""A pose-level fake simulator implementing SimBackend.

Decisions come from an internal symbolic World (so semantics match), but everything the loop sees comes back as
POSES, grip peaks and held flags, forcing the whole sim path (poses -> regions/compartments -> World mirror ->
rules) to be exercised. It is an integration-test double, not physics.
"""
from __future__ import annotations

from .world import World, FORCE_MAX
from .simbridge import Layout, ObjState, SIM_FORCE_MARGIN_N
from .simworld import SkillReport
from .skills import CATEGORIES

UP = (1.0, 0.0, 0.0, 0.0)
SIDE = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
ZONE_X = {"table_left": -0.40, "table_center": 0.0, "table_right": 0.40}


class FakeSim:
    def __init__(self, make_world, layout: Layout | None = None):
        self.make_world = make_world           # seed -> symbolic World (ground truth + stochastic skills)
        self.layout = layout or Layout()
        self.w = None

    def reset(self, seed: int):
        self.w = self.make_world(seed)
        self.arm_zones = {a: set(z) for a, z in self.w.zones.items()}
        self.payload, self.force_limit = dict(self.w.payload), dict(self.w.force_limit)
        self.peak, self.touched = {}, set()
        self.init = {n: o.region for n, o in self.w.objs.items()}

    def run_skill(self, arm, skill, obj, params):
        r = self.w.execute(arm, skill, obj, params)
        if skill == "grasp" and obj in self.w.objs and r.cause in ("", "slip", "crush", "arm_noise"):
            f = min(float(params.get("force", 10.0)), self.w.force_limit[arm], FORCE_MAX)
            self.peak[obj] = max(self.peak.get(obj, 0.0), f + SIM_FORCE_MARGIN_N)
        if skill in ("grasp", "push_aside") and obj in self.w.objs and (r.ok or r.cause in ("slip", "crush", "arm_noise")):
            self.touched.add(obj)
        cat = "success" if r.ok else (r.cause if r.cause in CATEGORIES else "other_failure")
        return SkillReport(cat, r.touch, detail=r.cause if not r.ok else "")

    def object_states(self):
        lay, out = self.layout, []
        bx, by = lay.bag_center
        hy = lay.bag_half[1]
        comp_y = {"laptop": by - hy + 0.02, "main": by, "front": by + hy - 0.02}
        for n, o in self.w.objs.items():
            held_by, quat = None, (UP if o.upright else SIDE)
            if o.region.startswith("held:"):
                held_by, pos = o.region.split(":")[1], (0.0, 0.2, 0.5)
            elif o.region.startswith("bag:"):
                c = o.region.split(":")[1]
                idx = self.w.bag[c].index(n)
                pos = (bx, comp_y[c], lay.bag_floor_z + 0.03 + 0.04 * idx)
            else:
                pos = (ZONE_X.get(o.region, 0.0), 0.0, lay.table_z + 0.05)
            out.append(ObjState(name=n, cls=o.cls, pos=pos, quat=quat, mass=o.mass, fragility=o.fragility,
                                size=o.size, friction=o.friction, held_by=held_by,
                                peak_contact_force=self.peak.get(n, 0.0), touched=n in self.touched,
                                init_region=self.init[n], is_distractor=o.is_distractor))
        return out
