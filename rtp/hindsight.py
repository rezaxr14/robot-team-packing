"""Team hindsight relabeling.

Given a finished episode, rewrite the instruction to match what the team actually achieved
(items packed, constraints that held, the arm assignment that actually executed) and keep the
executed-successful plan as a positive example. Relabeled records are validated by replay.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

from .plan import Plan, Task
from .rules import TaskSpec, check_final
from .brain import BrainContext


@dataclass
class HindsightRecord:
    spec: dict
    instruction: str
    plan_json: str


def relabel(world, result, orig_spec: TaskSpec):
    packed = [n for n in sorted(world.packed(), key=lambda n: world.objs[n].pack_index)
              if n in orig_spec.items and not world.objs[n].crushed]
    if not packed:
        return None
    # last successful grasp/place/handover per item, in execution order
    ok = [e for e in result.executed if e["ok"]]
    tasks, prev = [], None
    for n in packed:
        g = [e["task"] for e in ok if e["task"]["obj"] == n and e["task"]["skill"] == "grasp"]
        p = [e["task"] for e in ok if e["task"]["obj"] == n and e["task"]["skill"] == "place"]
        if not p:
            return None
        pl = p[-1]
        deps = [prev] if prev else []
        if g:
            gt = g[-1]
            tasks.append(Task(f"g_{n}", gt["arm"], "grasp", n, dict(gt["params"]), []))
            deps = [f"g_{n}"] + deps
        tasks.append(Task(f"p_{n}", pl["arm"], "place", n, dict(pl["params"]), deps))
        prev = f"p_{n}"
    untouched = lambda n: n in world.objs and not world.objs[n].moved
    leave = [n for n in orig_spec.leave if untouched(n)] + \
            [n for n in orig_spec.items if n not in packed and untouched(n)]
    spec = TaskSpec(items=packed, leave=leave, heavy_bottom=orig_spec.heavy_bottom,
                    laptop_slot=orig_spec.laptop_slot, bottle_upright_away=orig_spec.bottle_upright_away,
                    last_item=orig_spec.last_item if orig_spec.last_item in packed else None,
                    owner=orig_spec.owner)
    laptops = [n for n in packed if world.objs[n].cls == "laptop"]
    where = world.compartment_of(laptops[0]) if laptops else None
    spec.laptop_slot = where if where in ("laptop", "front") else None        # only protected slots are a rule
    for _ in range(4):                       # drop constraints that did not hold, until consistent
        chk = check_final(world, spec)
        if chk["success"]:
            break
        for v in chk["violations"]:
            if v["rule"] == "heavy_bottom":
                spec.heavy_bottom = False
            elif v["rule"] == "bottle_upright":
                spec.bottle_upright_away = False
            elif v["rule"] == "lunchbox_last":
                spec.last_item = None
            elif v["rule"] == "leave_distractors":
                name = v["detail"].split()[0]
                if name in spec.leave:
                    spec.leave.remove(name)
    if not check_final(world, spec)["success"]:
        return None
    return HindsightRecord(spec.to_dict(), spec.text(), Plan(tasks, "hindsight").to_json())


class ReplayBrain:
    """Returns a fixed recorded plan (used to validate relabeled records by replay)."""
    def __init__(self, plan: Plan):
        self._plan = plan
        self.requests = []

    def plan(self, ctx: BrainContext, reason: str = ""):
        return self._plan

    def replan(self, ctx: BrainContext, reason: str):
        return self._plan
