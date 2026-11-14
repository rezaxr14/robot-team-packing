"""Arm skill specification and regression suite.

The suite is executor-agnostic: anything implementing `SkillExecutor` (the symbolic world here; the Isaac Lab
arm policies later) is run against the same cases. Run it after every arm fine-tune and compare to the previous
checkpoint's report; a case that passed before and fails now is a regression.

Outcomes are compared by CATEGORY, not by the symbolic cause string, so a physics executor only has to classify
what happened (success / slip / crush / out_of_reach / wrong_slot / no_space / locked / overload / not_holding /
hand_full). For stochastic executors, set n_trials > 1: a success case passes at >= pass_rate, an expected-failure
case passes when the expected failure category occurs at >= pass_rate.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol

from .world import World, Obj

CATEGORIES = ("success", "slip", "crush", "out_of_reach", "wrong_slot", "no_space", "locked", "overload",
              "not_holding", "hand_full", "other_failure")


@dataclass
class SkillSpec:
    name: str
    params: dict
    precondition: str
    success: str


SKILLS = [
    SkillSpec("grasp", {"force": "N, between slip threshold and crush limit"}, "hand empty; object reachable; payload ok",
              "object held by this arm"),
    SkillSpec("place", {"compartment": "main|front|laptop", "orientation": "upright|flat"},
              "holding the object; bag lock free or owned; compartment accepts class and has space",
              "object in the named compartment, hand empty"),
    SkillSpec("handover", {"to": "L|R"}, "holding the object; other hand empty; payload ok", "object held by the other arm"),
    SkillSpec("weigh", {}, "hand empty; object reachable", "measured mass returned; nothing moves"),
    SkillSpec("release", {}, "holding the object", "object on the table, hand empty"),
    SkillSpec("retreat", {}, "none", "arm in a safe pose"),
    SkillSpec("push_aside", {}, "object reachable", "object moved"),
]


@dataclass
class Step:
    arm: str
    skill: str
    obj: str | None
    params: dict = field(default_factory=dict)
    expect: str = "success"          # category of the outcome expected at this step


@dataclass
class Case:
    name: str
    objects: list                    # dicts: name, cls, mass, fragility, size, region, friction
    steps: list
    lock_to: str | None = None       # an arm that already holds the bag lock
    final: dict = field(default_factory=dict)   # e.g. {"bag": {"main": ["book"]}, "holding": {"L": None}}
    tags: tuple = ()


class SkillExecutor(Protocol):
    def reset(self, objects: list, seed: int, lock_to: str | None) -> None: ...
    def run(self, arm: str, skill: str, obj: str | None, params: dict) -> str: ...   # returns a CATEGORY
    def state(self) -> dict: ...


class SymbolicExecutor:
    def __init__(self, p_arm_noise: float = 0.0):
        self.p = p_arm_noise
        self.w = None

    def reset(self, objects, seed, lock_to=None):
        self.w = World([Obj(**o) for o in objects], seed=seed, p_arm_noise=self.p)
        self.w.lock_holder = lock_to

    def run(self, arm, skill, obj, params):
        r = self.w.execute(arm, skill, obj, params)
        return "success" if r.ok else (r.cause if r.cause in CATEGORIES else "other_failure")

    def state(self):
        return {"bag": {c: list(v) for c, v in self.w.bag.items()}, "holding": dict(self.w.holding)}


def _o(name, cls="book", mass=1.0, fragility=0.1, size=2.0, region="table_left", friction=1.0):
    return dict(name=name, cls=cls, mass=mass, fragility=fragility, size=size, region=region, friction=friction)


def regression_cases() -> list:
    """Force window for a 1.0 kg, fragility-0.5 object: slip below 10 N, crush above 19 N."""
    mid = _o("x", mass=1.0, fragility=0.5)
    return [
        Case("grasp_in_window", [mid], [Step("L", "grasp", "x", {"force": 14})], final={"holding": {"L": "x", "R": None}}, tags=("grasp",)),
        Case("grasp_slip_low_force", [mid], [Step("L", "grasp", "x", {"force": 4}, "slip")], tags=("grasp", "force")),
        Case("grasp_crush_high_force", [mid], [Step("L", "grasp", "x", {"force": 30}, "crush")], tags=("grasp", "force")),
        Case("grasp_out_of_reach", [_o("x", region="table_right")], [Step("L", "grasp", "x", {"force": 12}, "out_of_reach")], tags=("grasp", "reach")),
        Case("grasp_center_both_arms", [_o("x", region="table_center")], [Step("R", "grasp", "x", {"force": 12})], tags=("grasp", "reach")),
        Case("weigh_ok_reports_mass_and_changes_nothing", [mid], [Step("L", "weigh", "x", {})],
             final={"holding": {"L": None, "R": None}, "bag": {"main": [], "front": [], "laptop": []}}, tags=("weigh",)),
        Case("weigh_out_of_reach", [_o("x", region="table_right")], [Step("L", "weigh", "x", {}, "out_of_reach")], tags=("weigh", "reach")),
        Case("weigh_hand_full", [_o("x"), _o("y", region="table_left")],
             [Step("L", "grasp", "x", {"force": 12}), Step("L", "weigh", "y", {}, "hand_full")], tags=("weigh",)),
        Case("grasp_overload", [_o("x", mass=3.0)], [Step("L", "grasp", "x", {"force": 30}, "overload")], tags=("grasp",)),
        Case("grasp_hand_full", [_o("x"), _o("y", region="table_left")],
             [Step("L", "grasp", "x", {"force": 12}), Step("L", "grasp", "y", {"force": 12}, "hand_full")], tags=("grasp",)),
        Case("pick_place_main", [_o("x")], [Step("L", "grasp", "x", {"force": 12}), Step("L", "place", "x", {"compartment": "main"})],
             final={"bag": {"main": ["x"], "front": [], "laptop": []}, "holding": {"L": None, "R": None}}, tags=("place", "sequence")),
        Case("place_without_grasp", [_o("x")], [Step("L", "place", "x", {"compartment": "main"}, "not_holding")], tags=("place",)),
        Case("place_wrong_slot", [_o("b", cls="bottle", mass=0.9, size=1.5)],
             [Step("L", "grasp", "b", {"force": 12}), Step("L", "place", "b", {"compartment": "laptop"}, "wrong_slot")], tags=("place",)),
        Case("place_laptop_in_slot", [_o("l", cls="laptop", mass=1.8, fragility=0.4, size=3.0)],
             [Step("L", "grasp", "l", {"force": 17}), Step("L", "place", "l", {"compartment": "laptop"})],
             final={"bag": {"laptop": ["l"], "main": [], "front": []}}, tags=("place",)),
        Case("place_no_space", [_o("a", size=3.0), _o("c", size=3.0, region="table_center")],
             [Step("L", "grasp", "a", {"force": 12}), Step("L", "place", "a", {"compartment": "front"}),
              Step("R", "grasp", "c", {"force": 12}), Step("R", "place", "c", {"compartment": "front"}, "no_space")], tags=("place",)),
        Case("place_bag_locked_by_other", [_o("x")],
             [Step("L", "grasp", "x", {"force": 12}), Step("L", "place", "x", {"compartment": "main"}, "locked")], lock_to="R", tags=("place", "lock")),
        Case("handover_ok", [_o("x", region="table_center")],
             [Step("L", "grasp", "x", {"force": 12}), Step("L", "handover", "x", {"to": "R"})],
             final={"holding": {"L": None, "R": "x"}}, tags=("handover",)),
        Case("handover_hand_full", [_o("x", region="table_center"), _o("y", region="table_right")],
             [Step("L", "grasp", "x", {"force": 12}), Step("R", "grasp", "y", {"force": 12}), Step("L", "handover", "x", {"to": "R"}, "hand_full")], tags=("handover",)),
    ]


def run_case(ex: SkillExecutor, case: Case, seed: int) -> dict:
    ex.reset(case.objects, seed, case.lock_to)
    got = []
    ok = True
    for st in case.steps:
        cat = ex.run(st.arm, st.skill, st.obj, st.params)
        got.append(cat)
        if cat != st.expect:
            ok = False
            break                                     # later steps are meaningless after a deviation
    if ok and case.final:
        s = ex.state()
        for k, want in case.final.items():
            have = s.get(k, {})
            if isinstance(want, dict):
                for kk, vv in want.items():
                    if have.get(kk) != vv:
                        ok = False
    return {"ok": ok, "got": got}


def run_regression(ex: SkillExecutor, cases=None, n_trials: int = 1, pass_rate: float = 0.9, seed0: int = 0) -> dict:
    cases = cases or regression_cases()
    rep = {"cases": {}, "by_tag": {}}
    for c in cases:
        rs = [run_case(ex, c, seed0 + i) for i in range(n_trials)]
        rate = sum(r["ok"] for r in rs) / n_trials
        rep["cases"][c.name] = {"rate": rate, "passed": rate >= pass_rate, "example_outcomes": rs[0]["got"], "tags": list(c.tags)}
        for t in c.tags:
            b = rep["by_tag"].setdefault(t, [0, 0])
            b[0] += rep["cases"][c.name]["passed"]
            b[1] += 1
    rep["passed"] = sum(v["passed"] for v in rep["cases"].values())
    rep["total"] = len(cases)
    return rep


def compare(old: dict, new: dict) -> dict:
    """Regressions = cases that passed in `old` and fail in `new`; fixed = the reverse."""
    reg = [n for n, v in new["cases"].items() if not v["passed"] and old["cases"].get(n, {}).get("passed")]
    fixed = [n for n, v in new["cases"].items() if v["passed"] and not old["cases"].get(n, {"passed": True})["passed"]]
    return {"regressions": reg, "fixed": fixed, "ok": not reg}
