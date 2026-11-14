"""Simulator-agnostic bridge between raw simulator state and the symbolic world.

Isaac Lab (or any simulator) supplies per-object poses, masses, held flags and contact peaks; this module turns
them into (a) regions/compartments, (b) an `rtp.world.World` so `rules.check_final` defines success for Track B
and for the symbolic rewards with ONE implementation, and (c) outcome categories for the skill regression suite.
It is pure Python and fully tested on synthetic poses. The simulator-side code that fills `ObjState` is in
isaac/ and is untested.

Frame conventions (adjust `Layout` to the real scene): +z up; the table zones split along `zone_axis`; the bag is
an axis-aligned box in the world frame (yaw 0) with its opening on top.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

from .world import World, Obj, CAP

SIM_FORCE_MARGIN_N = 15.0   # simulator contact peaks exceed the commanded grip force; tune per scene


def crush_force_for(fragility: float) -> float:
    """Peak contact force (N) above which a simulated object counts as crushed (scales like world.crush_limit)."""
    return 30.0 - 22.0 * fragility + SIM_FORCE_MARGIN_N


@dataclass
class Layout:
    zone_axis: int = 0                       # 0 = x, 1 = y
    left_max: float = -0.20                  # coordinate below which an object is in table_left
    right_min: float = 0.20                  # above which it is table_right
    table_z: float = 0.0                     # table surface height
    floor_drop: float = 0.08                 # below table_z - floor_drop an object has fallen
    bag_center: tuple = (0.0, 0.55)          # (x, y) of the bag center
    bag_half: tuple = (0.18, 0.12)           # half extents (x, y)
    bag_floor_z: float = 0.0
    bag_height: float = 0.40
    # compartment split along bag local y (depth): laptop slot at the back, front pocket at the front
    laptop_slot_depth: float = 0.05          # thickness of the laptop slot strip at the back (local -y side)
    front_pocket_depth: float = 0.06         # thickness of the front pocket strip (local +y side)


@dataclass
class ObjState:
    name: str
    cls: str
    pos: tuple                                # world (x, y, z)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)        # order given by quat_order
    quat_order: str = "wxyz"                  # 'wxyz' for Isaac Lab 2.x, 'xyzw' for 3.0
    mass: float = 0.5
    fragility: float = 0.3
    size: float = 1.0
    friction: float = 1.0
    held_by: str | None = None                # "L" | "R" | None
    peak_contact_force: float = 0.0
    touched: bool = False                     # ever grasped or pushed
    init_region: str = ""                     # region at episode start (for the 'leave' rule)
    is_distractor: bool = False


def up_alignment(quat, order: str = "wxyz") -> float:
    """z-component of the object's local +z axis in the world frame (1 = upright, 0 = on its side, -1 = inverted).
    order: 'wxyz' (Isaac Lab 2.x) or 'xyzw' (Isaac Lab 3.0)."""
    if order == "xyzw":
        x, y, z, w = quat
    elif order == "wxyz":
        w, x, y, z = quat
    else:
        raise ValueError(f"unknown quaternion order {order!r}")
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return 1.0 - 2.0 * (x * x + y * y)


def is_upright(quat, tol_deg: float = 30.0, order: str = "wxyz") -> bool:
    return up_alignment(quat, order) >= math.cos(math.radians(tol_deg))


def in_bag(pos, lay: Layout) -> bool:
    x, y, z = pos
    return (abs(x - lay.bag_center[0]) <= lay.bag_half[0] and abs(y - lay.bag_center[1]) <= lay.bag_half[1]
            and lay.bag_floor_z - 0.02 <= z <= lay.bag_floor_z + lay.bag_height)


def compartment_of(pos, lay: Layout) -> str:
    """Compartment by depth (local y): back strip = laptop slot, front strip = front pocket, rest = main."""
    ly = pos[1] - lay.bag_center[1]
    if ly <= -lay.bag_half[1] + lay.laptop_slot_depth:
        return "laptop"
    if ly >= lay.bag_half[1] - lay.front_pocket_depth:
        return "front"
    return "main"


def region_of(o: ObjState, lay: Layout) -> str:
    if o.held_by:
        return f"held:{o.held_by}"
    if in_bag(o.pos, lay):
        return f"bag:{compartment_of(o.pos, lay)}"
    if o.pos[2] < lay.table_z - lay.floor_drop:
        return "floor"
    c = o.pos[lay.zone_axis]
    return "table_left" if c < lay.left_max else ("table_right" if c > lay.right_min else "table_center")


def world_from_sim(states: list, lay: Layout, pack_order: list | None = None, seed: int = 0) -> World:
    """Rebuild a symbolic World from a simulator snapshot so rules.check_final can score it.
    pack_order: object names in the order the simulator saw them settle in the bag (else ordered by height)."""
    objs = []
    for o in states:
        init = o.init_region or region_of(o, lay)
        objs.append(Obj(name=o.name, cls=o.cls, mass=o.mass, fragility=o.fragility, size=o.size,
                        region=region_of(o, lay), friction=o.friction, upright=is_upright(o.quat, order=o.quat_order),
                        is_distractor=o.is_distractor, crushed=o.peak_contact_force > crush_force_for(o.fragility),
                        moved=o.touched, init_region=init))
    w = World(objs, seed=seed, deterministic=True)
    by_comp = {c: [] for c in ("main", "front", "laptop")}
    for o in states:
        r = w.objs[o.name].region
        if r.startswith("bag:"):
            by_comp[r.split(":", 1)[1]].append(o)
    order = {n: i for i, n in enumerate(pack_order or [])}
    counter = 0
    for c, lst in by_comp.items():
        lst.sort(key=lambda s: s.pos[2])                       # physical stack: bottom -> top by height
        w.bag[c] = [s.name for s in lst]
    placed = sorted((s for lst in by_comp.values() for s in lst), key=lambda s: (order.get(s.name, 10 ** 6), s.pos[2]))
    for s in placed:
        w.objs[s.name].pack_index = counter
        counter += 1
    w.pack_counter = counter
    return w


def classify_skill_outcome(skill: str, *, reachable: bool = True, hand_empty: bool = True, holding_target: bool = False,
                           object_followed_lift: bool = True, peak_force: float = 0.0, crush_force: float = 45.0,
                           placed_in: str | None = None, wanted: str | None = None, accepts: bool = True,
                           space_ok: bool = True, lock_ok: bool = True, payload_ok: bool = True) -> str:
    """Map raw simulator observations of one skill execution to a regression-suite category."""
    if skill == "grasp":
        if not hand_empty:
            return "hand_full"
        if not reachable:
            return "out_of_reach"
        if not payload_ok:
            return "overload"
        if peak_force > crush_force:
            return "crush"
        if not object_followed_lift:
            return "slip"
        return "success"
    if skill == "place":
        if not holding_target:
            return "not_holding"
        if not lock_ok:
            return "locked"
        if not accepts:
            return "wrong_slot"
        if not space_ok:
            return "no_space"
        return "success" if placed_in == wanted else "other_failure"
    if skill == "handover":
        if not holding_target:
            return "not_holding"
        if not hand_empty:
            return "hand_full"
        return "success"
    return "success"


def report_from_states(arm: str, skill: str, obj, params: dict, pre: list, post: list, layout: Layout,
                       zones: dict, payload: dict, lock_holder=None):
    """Derive (category, touch) of ONE skill execution from simulator state before/after, with no symbolic cause
    strings. This is what a physics backend uses: it only needs poses, held flags and contact peaks.
    Returns (category, touch_dict). Random controller failures are indistinguishable from slips here (category 'slip')."""
    from .world import SLOT_CLASSES, CAP
    pre_by, post_by = {s.name: s for s in pre}, {s.name: s for s in post}
    holding_pre = {s.held_by: s.name for s in pre if s.held_by}
    hand_empty = arm not in holding_pre
    o0, o1 = pre_by.get(obj), post_by.get(obj)
    if skill in ("retreat", "release", "push_aside", "reach", "grasp", "place", "handover", "weigh") and obj is not None and o0 is None:
        return "other_failure", {}
    if skill == "grasp":
        reg = region_of(o0, layout)
        reachable = reg in zones[arm]
        peak = o1.peak_contact_force if o1.peak_contact_force > o0.peak_contact_force else 0.0
        cat = classify_skill_outcome("grasp", reachable=reachable, hand_empty=hand_empty, payload_ok=o0.mass <= payload[arm],
                                     peak_force=peak, crush_force=crush_force_for(o0.fragility),
                                     object_followed_lift=(o1.held_by == arm))
        touched_contact = cat in ("success", "slip", "crush")
        touch = ({"obj": obj, "measured_weight": o0.mass, "slip": cat == "slip", "grasp_ok": cat == "success",
                  "peak_force": peak} if touched_contact else {})
        return cat, touch
    if skill == "place":
        comp = params.get("compartment", "main")
        holding = holding_pre.get(arm) == obj
        accepts = comp not in SLOT_CLASSES or o0.cls in SLOT_CLASSES[comp]
        used = sum(s.size for s in pre if region_of(s, layout) == f"bag:{comp}")
        placed_in = region_of(o1, layout).split(":", 1)[1] if region_of(o1, layout).startswith("bag:") else None
        cat = classify_skill_outcome("place", holding_target=holding, lock_ok=lock_holder in (None, arm), accepts=accepts,
                                     space_ok=used + o0.size <= CAP.get(comp, 0) + 1e-9, placed_in=placed_in, wanted=comp)
        return cat, {}
    if skill == "handover":
        to = params.get("to")
        holding = holding_pre.get(arm) == obj
        to_empty = to not in holding_pre
        if holding and to_empty and o1.held_by == to:
            return "success", {}
        return classify_skill_outcome("handover", holding_target=holding, hand_empty=to_empty), {}
    if skill == "weigh":
        reachable = region_of(o0, layout) in zones[arm]
        cat = classify_skill_outcome("grasp", reachable=reachable, hand_empty=hand_empty, payload_ok=o0.mass <= payload[arm],
                                     object_followed_lift=True)
        return cat, ({"obj": obj, "measured_weight": o0.mass, "slip": False, "grasp_ok": True, "peak_force": 8.0}
                     if cat == "success" else {})
    if skill == "release":
        return ("success", {}) if (o1.held_by is None and holding_pre.get(arm) == obj) else ("not_holding", {})
    if skill == "push_aside":
        return ("success", {}) if region_of(o0, layout) in zones[arm] else ("out_of_reach", {})
    return "success", {}
