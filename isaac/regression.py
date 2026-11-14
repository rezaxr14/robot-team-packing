"""IsaacExecutor: runs the executor-agnostic skill regression suite (rtp.skills) on PackingSim / LearnedPackingSim.

Each Case's abstract objects (x, y, ...) are mapped to real items of the same class (or any free item), teleported to the
case's zone, and given the case's mass/fragility/friction/size in the symbolic truth. Outcomes are classified by the same
state-diff code as in episodes. Slip and crush cases test your force calibration (see 04_force_window).
"""
from __future__ import annotations

from isaac.packing_sim import ITEM_NAMES, ITEM_CLASS
from rtp.skills import CATEGORIES


class IsaacExecutor:
    def __init__(self, ps):
        self.ps, self.map = ps, {}

    def reset(self, objects, seed, lock_to):
        ps = self.ps
        ps.reset(seed)
        used, self.map = set(), {}
        for spec in objects:
            pool = [n for n in ITEM_NAMES if n not in used and ITEM_CLASS[n] == spec["cls"]] or \
                   [n for n in ITEM_NAMES if n not in used and not ps.truth.objs[n].is_distractor]
            item = pool[0]
            used.add(item)
            self.map[spec["name"]] = item
            o = ps.truth.objs[item]
            o.mass, o.fragility, o.size, o.friction = spec["mass"], spec["fragility"], spec["size"], spec["friction"]
            slot = sum(1 for v in self.map.values() if v != item and ps.truth.objs[v].init_region == spec["region"])
            ps.teleport(item, spec["region"], slot)
            try:
                import torch
                ps.items[item].root_physx_view.set_masses(torch.tensor([[spec["mass"]]]), torch.tensor([0], dtype=torch.int32))
            except Exception as e:                                       # noqa: BLE001
                ps.log.append(f"set_masses failed in regression: {e}")
            o.init_region = spec["region"]
        ps.park([n for n in ITEM_NAMES if n not in used])
        ps._step(60)
        for s in ps.object_states():
            ps.init_regions[s.name] = ps.truth.objs[s.name].init_region if s.name in used else ps.init_regions.get(s.name, "")
        ps.lock_holder = lock_to

    def run(self, arm, skill, obj, params):
        rep = self.ps.run_skill(arm, skill, self.map.get(obj, obj), params)
        return rep.category if rep.category in CATEGORIES else "other_failure"

    def state(self):
        inv = {v: k for k, v in self.map.items()}
        return {"bag": {c: [inv.get(n, n) for n in names] for c, names in self._bag().items()},
                "holding": {a: inv.get(n, n) if n else None for a, n in self._holding().items()}}

    def _bag(self):
        from rtp.simbridge import region_of
        out = {"main": [], "front": [], "laptop": []}
        for s in sorted(self.ps.object_states(), key=lambda s: s.pos[2]):
            r = region_of(s, self.ps.layout)
            if r.startswith("bag:"):
                out[r.split(":")[1]].append(s.name)
        return out

    def _holding(self):
        h = {"L": None, "R": None}
        for n, a in self.ps.held.items():
            h[a] = n
        return h
