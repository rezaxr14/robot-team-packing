"""Brain interfaces.

ScriptedBrain: deterministic reference planner. It only applies a rule if the rule's id was
retrieved into its context, so harness retrieval quality changes behaviour (this is what
makes the harness-on/off ablation testable without an LLM). It is a test fixture and a
data-generation teacher, not a baseline to claim results with.

LLMBrain: adapter for the real Brain (text-only LLM + LoRA). `generate` is any callable
prompt:str -> completion:str (vLLM, HF generate, API...).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

from .plan import Plan, Task, PlanError, parse_plan
from .rules import TaskSpec


@dataclass
class BrainContext:
    spec: TaskSpec
    scene: dict
    rule_ids: set
    snippets: list = field(default_factory=list)      # harness Docs
    notes: list = field(default_factory=list)         # failure notes / replan reasons
    packed: set = field(default_factory=set)
    holding: dict = field(default_factory=dict)       # arm -> obj name or None
    abilities: dict = field(default_factory=dict)
    look_fn: object = None            # callable(list[str]) -> {name: scene entry}; set by the loop


def guess_force(o: dict) -> float:
    need = 4.0 + 6.0 * o["mass_guess"]
    crush = 30.0 - 22.0 * o["fragility_guess"]
    f = min(need * 1.25, crush - 2.0)
    if f < need:                       # window too tight: aim for the middle
        f = (need + crush) / 2.0
    return round(max(2.0, f), 2)


class ScriptedBrain:
    def __init__(self):
        self.requests: list = []       # objects the brain wants the Eye to look for

    def _arm_for(self, region: str, load: dict, holder=None) -> str:
        if holder:
            return holder
        if region == "table_left":
            return "L"
        if region == "table_right":
            return "R"
        return min(("L", "R"), key=lambda a: (load[a], a))

    def plan(self, ctx: BrainContext, reason: str = "") -> Plan:
        spec, scene, rules = ctx.spec, ctx.scene, ctx.rule_ids
        held = {n: a for a, n in ctx.holding.items() if n}
        scene = dict(scene)
        todo = [n for n in spec.items if n not in ctx.packed and (n in scene or n in held)]
        self.requests = [n for n in spec.items if n not in ctx.packed and n not in scene and n not in held]

        def mass(n):
            return scene[n]["mass_guess"] if n in scene else 0.5

        # An arm that holds an item cannot grasp another. Place held items first, unless that would put a light
        # item under a heavier pending one: then release it back to the table and re-grasp it in order.
        releases = {}
        for n, a in list(held.items()):
            heavier_pending = [m for m in todo if m not in held and m != spec.last_item and mass(m) > mass(n) + 0.15]
            if "heavy_bottom" in rules and n != spec.last_item and heavier_pending:
                releases[n] = Task(f"rel_{n}", a, "release", n, {}, [])
                scene[n] = {**scene.get(n, {"cls": "", "mass_guess": 0.5, "fragility_guess": 0.3, "size": 1.0}),
                            "region": "table_center"}
                del held[n]
        order = list(todo)
        if "heavy_bottom" in rules:
            order.sort(key=lambda n: -mass(n))
        if "lunchbox_last" in rules and spec.last_item in order:
            order.remove(spec.last_item)
            order.append(spec.last_item)
        order = [n for n in order if n in held] + [n for n in order if n not in held]   # held items first
        load = {"L": 0, "R": 0}
        tasks, prev = list(releases.values()), None
        for n in order:
            o = scene.get(n, {})
            cls = o.get("cls", "")
            arm = self._arm_for(o.get("region", ""), load, held.get(n))
            load[arm] += 1
            comp = "main"
            if cls == "laptop" and "laptop_slot" in rules and spec.laptop_slot:
                comp = spec.laptop_slot
            orient = "upright"
            if cls == "bottle" and "bottle_upright" not in rules:
                orient = "flat"
            deps = [prev] if prev else []
            if n in held:
                tasks.append(Task(f"p_{n}", held[n], "place", n, {"compartment": comp, "orientation": orient}, deps))
            else:
                gdeps = [f"rel_{n}"] if n in releases else []
                tasks.append(Task(f"g_{n}", arm, "grasp", n, {"force": guess_force(o)}, gdeps))
                tasks.append(Task(f"p_{n}", arm, "place", n, {"compartment": comp, "orientation": orient},
                                  [f"g_{n}"] + deps))
            prev = f"p_{n}"
        return Plan(tasks, notes="rules=" + ",".join(sorted(rules)))

    def replan(self, ctx: BrainContext, reason: str) -> Plan:
        return self.plan(ctx, reason)


PROMPT_HEADER = """You coordinate two blind robot arms (L covers table_left+table_center, R covers table_right+table_center; both reach the bag, one at a time).
Output ONLY a JSON object: {"tasks":[{"id","arm","skill","obj","params","deps"}],"notes"}.
Skills: grasp(params.force in N), place(params.compartment in main|front|laptop, params.orientation upright|flat), release, retreat, push_aside, handover(params.to), weigh (lifts an item and returns its measured mass; use it when the heavy-at-bottom order matters and an estimate looks doubtful).
Each place must depend on its grasp. An arm that is holding an item cannot grasp another: place or release it first. Never touch items you were told to leave."""


def build_prompt(ctx: BrainContext, reason: str = "") -> str:
    parts = [PROMPT_HEADER, "## Instruction", ctx.spec.text(), "## Scene (from the Eyes)",
             json.dumps(list(ctx.scene.values()), separators=(",", ":"))]
    if ctx.snippets:
        parts += ["## Retrieved notes"] + [f"[{d.id}] {d.body}" for d in ctx.snippets]
    if ctx.packed or any(ctx.holding.values()):
        parts += ["## Progress", json.dumps({"packed": sorted(ctx.packed), "holding": ctx.holding})]
    if ctx.notes or reason:
        parts += ["## Notes", "\n".join(ctx.notes + ([reason] if reason else []))]
    return "\n".join(parts)


class LLMBrain:
    def __init__(self, generate):
        self.generate = generate
        self.requests: list = []

    def plan(self, ctx: BrainContext, reason: str = "") -> Plan:
        return parse_plan(self.generate(build_prompt(ctx, reason)))   # may raise PlanError

    def replan(self, ctx: BrainContext, reason: str) -> Plan:
        return self.plan(ctx, reason)
