"""Symbolic backpack-packing world with stochastic, force-aware skills.

Purpose: a cheap, deterministic-given-seed stand-in for the Isaac Lab scene. It is used for
(1) testing the Eye/Brain/Arm loop, (2) symbolic GRPO rewards and dry runs,
(3) credit-assignment and relabeling logic. It is NOT a physics result and must never be
reported as one.

Common random numbers: every stochastic draw is keyed by (seed, key, per-key counter), so a
replay of the same (arm, skill, object) attempt sequence reuses the same draws regardless of
what else happened in the episode. Counterfactual credit relies on this.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field, replace

COMPARTMENTS = ("main", "front", "laptop")
CAP = {"main": 12.0, "front": 4.0, "laptop": 3.0}
SLOT_CLASSES = {"laptop": {"laptop", "notebook"}}  # restricted slots
FORCE_MAX = 40.0


@dataclass
class Obj:
    name: str
    cls: str
    mass: float
    fragility: float
    size: float
    region: str                      # table_left | table_right | table_center | held:<arm> | bag:<comp>
    friction: float = 1.0
    upright: bool = True
    is_distractor: bool = False
    crushed: bool = False
    moved: bool = False
    pack_index: int = -1
    init_region: str = ""

    def __post_init__(self):
        if not self.init_region:
            self.init_region = self.region


def required_force(o: Obj) -> float:
    """Minimum grip force (N) to avoid slip."""
    return (4.0 + 6.0 * o.mass) / o.friction


def crush_limit(o: Obj) -> float:
    """Force (N) above which the object is crushed."""
    return 30.0 - 22.0 * o.fragility


@dataclass
class Result:
    ok: bool
    cause: str = ""
    touch: dict = field(default_factory=dict)
    arm: str = ""
    skill: str = ""
    obj: str = ""
    detail: str = ""                 # optional prose from the simulator (kept out of `cause`: the loop matches on cause)


class World:
    ARMS = ("L", "R")

    def __init__(self, objs, seed: int = 0, p_arm_noise: float = 0.03, deterministic: bool = False):
        self.objs = {o.name: replace(o) for o in objs}
        self.seed = seed
        self.p_arm_noise = p_arm_noise
        self.deterministic = deterministic   # dry-run mode: no crush/slip/noise failures
        self._cnt = {}
        self.holding = {a: None for a in self.ARMS}
        self.bag = {c: [] for c in COMPARTMENTS}
        self.zones = {"L": {"table_left", "table_center", "bag"},
                      "R": {"table_right", "table_center", "bag"}}
        self.payload = {"L": 2.5, "R": 2.5}
        self.force_limit = {"L": 35.0, "R": 35.0}
        self.lock_holder = None
        self.pack_counter = 0
        self.t = 0
        self.history = []

    # ---- randomness (common random numbers) ----
    def u(self, key: str) -> float:
        n = self._cnt.get(key, 0)
        self._cnt[key] = n + 1
        h = hashlib.sha256(f"{self.seed}|{key}|{n}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2 ** 64

    # ---- views ----
    def packed(self):
        return [n for c in COMPARTMENTS for n in self.bag[c]]

    def compartment_of(self, name: str):
        r = self.objs[name].region
        return r.split(":", 1)[1] if r.startswith("bag:") else None

    def reachable(self, arm: str, name: str) -> bool:
        r = self.objs[name].region
        if r.startswith("held:") or r.startswith("bag:"):
            return False
        return r in self.zones[arm]

    def clone_fresh(self, deterministic=None) -> "World":
        objs = [replace(o, region=o.init_region, crushed=False, moved=False,
                        pack_index=-1, upright=True) for o in self.objs.values()]
        return World(objs, seed=self.seed, p_arm_noise=self.p_arm_noise,
                     deterministic=self.deterministic if deterministic is None else deterministic)

    # ---- skills ----
    def execute(self, arm: str, skill: str, obj=None, params=None) -> Result:
        params = params or {}
        self.t += 1
        fn = getattr(self, f"_sk_{skill}", None)
        if fn is None:
            res = Result(False, "unknown_skill")
        elif obj is not None and obj not in self.objs:
            res = Result(False, "not_found")
        else:
            res = fn(arm, obj, params)
        res.arm, res.skill, res.obj = arm, skill, obj or ""
        self.history.append(res)
        return res

    def _noise_fail(self, arm, skill, obj) -> bool:
        if self.deterministic:
            return False
        return self.u(f"{arm}|{skill}|{obj}") < self.p_arm_noise

    def _measure(self, arm, o):
        return round(o.mass * (1 + 0.05 * (2 * self.u(f"{arm}|meas|{o.name}") - 1)), 3)

    def _sk_reach(self, arm, obj, p):
        o = self.objs[obj]
        if o.region.startswith("held:"):
            return Result(False, "held_by_other")
        if not self.reachable(arm, obj):
            return Result(False, "out_of_reach")
        return Result(True)

    def _sk_grasp(self, arm, obj, p):
        o = self.objs[obj]
        if self.holding[arm] is not None:
            return Result(False, "hand_full")
        if o.region.startswith("held:"):
            return Result(False, "held_by_other")
        if not self.reachable(arm, obj):
            return Result(False, "out_of_reach")
        if o.mass > self.payload[arm]:
            return Result(False, "overload")
        force = min(float(p.get("force", 10.0)), self.force_limit[arm], FORCE_MAX)

        def touch(ok, slip):
            return {"obj": obj, "measured_weight": self._measure(arm, o), "slip": slip,
                    "grasp_ok": ok, "peak_force": force}

        if not self.deterministic:
            if force > crush_limit(o):
                o.crushed = True
                return Result(False, "crush", touch(False, False))
            if force < required_force(o):
                return Result(False, "slip", touch(False, True))
        if self._noise_fail(arm, "grasp", obj):
            return Result(False, "arm_noise", touch(False, False))
        self.holding[arm] = obj
        o.region = f"held:{arm}"
        o.moved = True
        return Result(True, "", touch(True, False))

    def _sk_place(self, arm, obj, p):
        comp = p.get("compartment", "main")
        if self.holding[arm] != obj:
            return Result(False, "not_holding")
        if comp not in COMPARTMENTS:
            return Result(False, "bad_compartment")
        if self.lock_holder not in (None, arm):
            return Result(False, "locked")
        o = self.objs[obj]
        if comp in SLOT_CLASSES and o.cls not in SLOT_CLASSES[comp]:
            return Result(False, "wrong_slot")
        used = sum(self.objs[n].size for n in self.bag[comp])
        if used + o.size > CAP[comp] + 1e-9:
            return Result(False, "no_space")
        if self._noise_fail(arm, "place", obj):
            return Result(False, "arm_noise")
        self.bag[comp].append(obj)
        o.region = f"bag:{comp}"
        o.upright = (p.get("orientation", "upright") == "upright")
        o.pack_index = self.pack_counter
        self.pack_counter += 1
        self.holding[arm] = None
        return Result(True)

    def _sk_release(self, arm, obj, p):
        if self.holding[arm] != obj:
            return Result(False, "not_holding")
        self.objs[obj].region = "table_center"
        self.holding[arm] = None
        return Result(True)

    def _sk_weigh(self, arm, obj, p):
        """Lift the object a few centimetres and put it back: a load-cell reading of its mass, no state change."""
        o = self.objs[obj]
        if self.holding[arm] is not None:
            return Result(False, "hand_full")
        if o.region.startswith("held:"):
            return Result(False, "held_by_other")
        if not self.reachable(arm, obj):
            return Result(False, "out_of_reach")
        if o.mass > self.payload[arm]:
            return Result(False, "overload")
        if self._noise_fail(arm, "weigh", obj):
            return Result(False, "arm_noise")
        mw = round(o.mass * (1 + 0.02 * (2 * self.u(f"{arm}|weigh|{obj}") - 1)), 3)
        return Result(True, "", {"obj": obj, "measured_weight": mw, "slip": False, "grasp_ok": True, "peak_force": 8.0})

    def _sk_retreat(self, arm, obj, p):
        return Result(True)

    def _sk_push_aside(self, arm, obj, p):
        if not self.reachable(arm, obj):
            return Result(False, "out_of_reach")
        self.objs[obj].moved = True
        return Result(True)

    def _sk_handover(self, arm, obj, p):
        to = p.get("to")
        if to not in self.ARMS or to == arm:
            return Result(False, "bad_target")
        if self.holding[arm] != obj:
            return Result(False, "not_holding")
        if self.holding[to] is not None:
            return Result(False, "hand_full")
        if self.objs[obj].mass > self.payload[to]:
            return Result(False, "overload")
        if self._noise_fail(arm, "handover", obj):
            return Result(False, "arm_noise")
        self.holding[arm] = None
        self.holding[to] = obj
        self.objs[obj].region = f"held:{to}"
        return Result(True)
