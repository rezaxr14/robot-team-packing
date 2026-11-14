"""Task specification, final-state rule checker, static plan feasibility, and dry run."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict

from .world import World, Obj, COMPARTMENTS, SLOT_CLASSES
from .plan import Plan, next_round

RULE_IDS = ("heavy_bottom", "laptop_slot", "bottle_upright", "lunchbox_last",
            "leave_distractors", "bag_exclusive")


@dataclass
class TaskSpec:
    items: list
    leave: list = field(default_factory=list)
    heavy_bottom: bool = True
    laptop_slot: str | None = "laptop"
    bottle_upright_away: bool = True
    last_item: str | None = None
    owner: str = "Maya"

    def text(self) -> str:
        parts = [f"Pack {self.owner}'s bag for school: " + ", ".join(self.items) + "."]
        if self.heavy_bottom:
            parts.append("Heavy items at the bottom.")
        if self.laptop_slot:
            parts.append(f"Laptop in the {self.laptop_slot} slot.")
        if self.bottle_upright_away:
            parts.append("Bottle upright and away from the laptop.")
        if self.last_item:
            parts.append(f"{self.last_item} last.")
        if self.leave:
            parts.append("Leave " + " and ".join(self.leave) + ".")
        return " ".join(parts)

    def rule_ids(self) -> list:
        ids = ["bag_exclusive", "leave_distractors"]
        if self.heavy_bottom:
            ids.append("heavy_bottom")
        if self.laptop_slot:
            ids.append("laptop_slot")
        if self.bottle_upright_away:
            ids.append("bottle_upright")
        if self.last_item:
            ids.append("lunchbox_last")
        return ids

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TaskSpec":
        return TaskSpec(**d)


def amend(spec: TaskSpec, **changes) -> TaskSpec:
    d = spec.to_dict()
    d.update(changes)
    return TaskSpec.from_dict(d)


def _cls_names(world, cls):
    return [n for n, o in world.objs.items() if o.cls == cls]


def check_final(world: World, spec: TaskSpec) -> dict:
    packed = set(world.packed())
    missing = [n for n in spec.items if n not in packed]
    crushed = [n for n, o in world.objs.items() if o.crushed]
    viol = []

    if spec.heavy_bottom:
        stack = [n for n in world.bag["main"] if n != spec.last_item]   # 'last' item is an explicit exception
        for i in range(len(stack)):
            for j in range(i + 1, len(stack)):
                if world.objs[stack[j]].mass > world.objs[stack[i]].mass + 0.15:
                    viol.append({"rule": "heavy_bottom", "detail": f"{stack[j]} above lighter {stack[i]}"})
    laptops = [n for n in _cls_names(world, "laptop") if n in packed]
    if spec.laptop_slot:
        for n in laptops:
            if world.compartment_of(n) != spec.laptop_slot:
                viol.append({"rule": "laptop_slot", "detail": f"{n} in {world.compartment_of(n)}"})
    if spec.bottle_upright_away:
        for n in [b for b in _cls_names(world, "bottle") if b in packed]:
            if not world.objs[n].upright:
                viol.append({"rule": "bottle_upright", "detail": f"{n} not upright"})
            if any(world.compartment_of(n) == world.compartment_of(l) for l in laptops):
                viol.append({"rule": "bottle_upright", "detail": f"{n} next to laptop"})
    if spec.last_item and spec.last_item in packed:
        last = world.objs[spec.last_item].pack_index
        if any(world.objs[n].pack_index > last for n in packed if n != spec.last_item):
            viol.append({"rule": "lunchbox_last", "detail": f"{spec.last_item} not last"})
    for n in spec.leave:
        if n in world.objs and (world.objs[n].moved or world.objs[n].region != world.objs[n].init_region):
            viol.append({"rule": "leave_distractors", "detail": f"{n} was moved"})

    n_valid = sum(1 for n in spec.items if n in packed and not world.objs[n].crushed)
    return {"success": not missing and not viol and not crushed,
            "missing": missing, "violations": viol, "crushed": crushed,
            "n_valid": n_valid, "n_items": len(spec.items)}


def static_check(plan: Plan, scene: dict, spec: TaskSpec):
    """Symbolic feasibility against the *reported* scene. Returns (ok, errors)."""
    zones = World([]).zones
    errors = []
    by_id = {t.id: t for t in plan.tasks}

    def anc(t, seen=None):
        seen = seen or set()
        for d in t.deps:
            if d not in seen:
                seen.add(d)
                anc(by_id[d], seen)
        return seen

    placed = set()
    for t in plan.tasks:
        if t.obj is not None and t.obj not in scene:
            errors.append(f"{t.id}: unknown object {t.obj}")
            continue
        if t.obj in spec.leave:
            errors.append(f"{t.id}: touches leave-item {t.obj}")
        if t.skill in ("grasp", "reach") and t.obj in scene:
            region = scene[t.obj].get("region", "")
            if region.startswith("table") and region not in zones[t.arm]:
                errors.append(f"{t.id}: {t.obj} in {region} not reachable by {t.arm}")
        if t.skill == "place":
            comp = t.params.get("compartment", "main")
            if comp not in COMPARTMENTS:
                errors.append(f"{t.id}: bad compartment {comp}")
            if t.obj in placed:
                errors.append(f"{t.id}: {t.obj} placed twice")
            placed.add(t.obj)
            prior = [by_id[d] for d in anc(t)]
            if not any(p.obj == t.obj and p.arm == t.arm and p.skill == "grasp" for p in prior) and \
               not any(p.obj == t.obj and p.skill == "handover" and p.params.get("to") == t.arm for p in prior):
                errors.append(f"{t.id}: place without prior grasp by {t.arm}")
    return (not errors), errors


def world_from_scene(scene: dict, spec: TaskSpec, seed: int = 0) -> World:
    objs = []
    for name, d in scene.items():
        objs.append(Obj(name=name, cls=d.get("cls", "unknown"), mass=float(d.get("mass_guess", 0.5)),
                        fragility=float(d.get("fragility_guess", 0.3)), size=float(d.get("size", 1.0)),
                        region=d.get("region", "table_center"), is_distractor=name in spec.leave))
    return World(objs, seed=seed, deterministic=True)


def run_plan_simple(world: World, plan: Plan, max_rounds: int = 200):
    """Execute a plan without recovery (dry run). Returns list of (task, result)."""
    status = {t.id: "pending" for t in plan.tasks}
    out = []
    for _ in range(max_rounds):
        chosen, _dead = next_round(plan.tasks, status)
        if not chosen:
            break
        for t in chosen:
            r = world.execute(t.arm, t.skill, t.obj, t.params)
            out.append((t, r))
            status[t.id] = "done" if r.ok else "dead"
    return out


def dry_run(plan: Plan, scene: dict, spec: TaskSpec) -> dict:
    """Deterministic rule-compliance check of a plan on a world rebuilt from the scene report."""
    w = world_from_scene(scene, spec)
    executed = run_plan_simple(w, plan)
    res = check_final(w, spec)
    res["executed"] = len(executed)
    res["failed_tasks"] = [t.id for t, r in executed if not r.ok]
    return res
