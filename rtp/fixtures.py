"""Test fixtures that stand in for an imperfect LLM Brain.

FlakyBrain wraps a good planner and, with probability p_err, corrupts its plan in one of the ways real LLM plans
fail. If the context carries 'PLAN REJECTED' feedback it repairs the plan with probability p_fix. Used to test and
ablate the loop's verification stage; it says nothing about real LLM error rates.
Corruptions: ghost (unknown object, static-detectable), no_grasp (place without grasp, static), wrong_arm (arm
cannot reach, static), swap (ordering dependencies of two placements exchanged -> rule violation, dry-run only), drop_item (item never packed,
dry-run only), garbage (unparseable reply).
"""
from __future__ import annotations
import random
from dataclasses import replace

from .plan import Plan, PlanError, Task

KINDS = ("ghost", "no_grasp", "wrong_arm", "swap", "drop_item", "garbage")


class FlakyBrain:
    def __init__(self, inner, p_err: float = 0.5, p_fix: float = 0.9, seed: int = 0, kinds=KINDS):
        self.inner, self.p_err, self.p_fix, self.seed, self.kinds = inner, p_err, p_fix, seed, tuple(kinds)
        self.calls = 0
        self.requests: list = []
        self.injected: list = []

    def _rng(self):
        self.calls += 1
        return random.Random(f"flaky|{self.seed}|{self.calls}")

    def plan(self, ctx, reason: str = ""):
        rng = self._rng()
        plan = self.inner.plan(ctx, reason)
        self.requests = getattr(self.inner, "requests", [])
        rejected = any("PLAN REJECTED" in n for n in ctx.notes)
        p = (1 - self.p_fix) if rejected else self.p_err
        if rng.random() >= p:
            return plan
        kind = rng.choice(self.kinds)
        self.injected.append(kind)
        return self._corrupt(plan, kind, rng, ctx)

    def replan(self, ctx, reason: str):
        return self.plan(ctx, reason)

    def _corrupt(self, plan: Plan, kind: str, rng, ctx) -> Plan:
        ts = [replace(t, params=dict(t.params), deps=list(t.deps)) for t in plan.tasks]
        grasps = [t for t in ts if t.skill == "grasp"]
        places = [t for t in ts if t.skill == "place"]
        if kind == "garbage":
            raise PlanError("model returned prose instead of a plan")
        if not ts:
            return plan
        if kind == "ghost" and grasps:
            g = rng.choice(grasps)
            g.obj = "ghost_item"
        elif kind == "no_grasp" and places:
            p = rng.choice(places)
            ts = [t for t in ts if not (t.skill == "grasp" and t.obj == p.obj)]
            for t in ts:
                t.deps = [d for d in t.deps if d != f"g_{p.obj}"]
        elif kind == "wrong_arm" and grasps:
            g = rng.choice([t for t in grasps if ctx.scene.get(t.obj, {}).get("region") in ("table_left", "table_right")] or grasps)
            g.arm = "R" if g.arm == "L" else "L"
            for t in ts:
                if t.skill == "place" and t.obj == g.obj:
                    t.arm = g.arm
        elif kind == "swap" and len(places) >= 2:
            a, b = rng.sample(places, 2)
            # exchange only the ordering dependencies (waits on other placements); each place keeps its own grasp
            oa = [d for d in a.deps if d.startswith("p_")]
            ob = [d for d in b.deps if d.startswith("p_")]
            a.deps = [d for d in a.deps if not d.startswith("p_")] + [d for d in ob if d != a.id]
            b.deps = [d for d in b.deps if not d.startswith("p_")] + [d for d in oa if d != b.id]
        elif kind == "drop_item" and places:
            p = rng.choice(places)
            ts = [t for t in ts if t.obj != p.obj]
            ids = {t.id for t in ts}
            for t in ts:
                t.deps = [d for d in t.deps if d in ids]
        return Plan(ts, plan.notes + f";corrupted={kind}")
