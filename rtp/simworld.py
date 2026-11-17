"""Backend interface + adapter so `rtp.loop.run_episode` runs unchanged on any simulator.

A backend (Isaac Lab driver, or `rtp.fakesim.FakeSim`) executes one skill command and reports a CATEGORY plus a
touch dict; it also exposes simulator state as `ObjState`s. `SimWorld` mirrors that state into the symbolic World
after every skill via `simbridge.world_from_sim`, so the loop, rules, attribution and rewards see one interface.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Protocol

from .world import World, Result
from .simbridge import Layout, ObjState, world_from_sim


@dataclass
class SkillReport:
    category: str                      # one of rtp.skills.CATEGORIES (or 'other_failure')
    touch: dict = field(default_factory=dict)
    detail: str = ""                   # optional finer cause from simulator diagnostics (e.g. 'arm_noise')


_CAUSE_TOKEN = re.compile(r"^[a-z][a-z0-9_]*$")


def resolve_cause(category: str, detail: str = "") -> str:
    """The cause the loop dispatches on, from a backend's (category, detail) pair.

    A backend reports what it can prove: a coarse `category` from `rtp.skills.CATEGORIES` plus a
    free-form `detail`. When the category is the catch-all and the detail is a single-token
    identifier (FakeSim: `other_failure` + `arm_noise`), the token *is* the cause the recovery
    ladder and attribution key on, so it is promoted. Prose detail from a physics backend
    (`Crush: 92.0 N ...`) is kept out of `cause`, because the loop matches causes exactly.
    """
    if category in ("", "success"):
        return category
    if category == "other_failure" and detail and _CAUSE_TOKEN.match(detail):
        return detail
    return category


class SimBackend(Protocol):
    arm_zones: dict                    # {"L": {"table_left","table_center","bag"}, ...}
    payload: dict
    force_limit: dict

    def reset(self, seed: int) -> None: ...
    def run_skill(self, arm: str, skill: str, obj: str | None, params: dict) -> SkillReport: ...
    def object_states(self) -> list: ...


class SimWorld(World):
    def __init__(self, backend: SimBackend, layout: Layout | None = None, seed: int = 0):
        self.backend = backend
        self.layout = layout or Layout()
        backend.reset(seed)
        states = backend.object_states()
        w0 = world_from_sim(states, self.layout, [], seed)
        super().__init__(list(w0.objs.values()), seed=seed, deterministic=True)
        self.zones = {a: set(z) for a, z in backend.arm_zones.items()}
        self.payload = dict(backend.payload)
        self.force_limit = dict(backend.force_limit)
        self._init_region = {s.name: (s.init_region or w0.objs[s.name].region) for s in states}
        self.pack_order: list = []
        self._refresh(states)

    def _refresh(self, states=None):
        states = states if states is not None else self.backend.object_states()
        w = world_from_sim(states, self.layout, self.pack_order, self.seed)
        for n, o in w.objs.items():
            o.init_region = self._init_region[n]
            o.moved = o.moved or self.objs.get(n, o).moved
        self.objs, self.bag, self.pack_counter = w.objs, w.bag, w.pack_counter
        self.holding = {a: None for a in self.ARMS}
        for s in states:
            if s.held_by:
                self.holding[s.held_by] = s.name

    def execute(self, arm, skill, obj=None, params=None) -> Result:
        params = params or {}
        self.t += 1
        if obj is not None and obj not in self.objs:
            res = Result(False, "not_found")
        else:
            rep = self.backend.run_skill(arm, skill, obj, params)
            ok = rep.category == "success"
            if ok and skill == "place":
                self.pack_order.append(obj)
            self._refresh()
            cause = resolve_cause(rep.category, rep.detail)
            res = Result(ok, "" if ok else cause, rep.touch, detail=rep.detail)
        res.arm, res.skill, res.obj = arm, skill, obj or ""
        self.history.append(res)
        return res


class StateDiffBackend:
    """Wraps any backend and re-derives each skill's category from state before/after (simbridge.report_from_states),
    ignoring the inner backend's own category. Used to validate the state-diff logic a physics backend relies on."""

    def __init__(self, inner, layout: Layout | None = None):
        self.inner = inner
        self.layout = layout or getattr(inner, "layout", Layout())

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def reset(self, seed):
        self.inner.reset(seed)

    def object_states(self):
        return self.inner.object_states()

    def run_skill(self, arm, skill, obj, params):
        from .simbridge import report_from_states
        pre = self.inner.object_states()
        rep = self.inner.run_skill(arm, skill, obj, params)
        post = self.inner.object_states()
        lock = getattr(getattr(self.inner, "w", None), "lock_holder", None)
        cat, touch = report_from_states(arm, skill, obj, params, pre, post, self.layout, self.inner.arm_zones,
                                        self.inner.payload, lock)
        return SkillReport(cat, touch or rep.touch, detail="")
