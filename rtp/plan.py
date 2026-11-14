"""Plan DAG: tasks assigned to arms, JSON schema, parsing, and the round scheduler."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field, asdict

SKILLS = ("reach", "grasp", "place", "release", "retreat", "push_aside", "handover", "weigh")
ARMS = ("L", "R")


class PlanError(ValueError):
    pass


@dataclass
class Task:
    id: str
    arm: str
    skill: str
    obj: str | None = None
    params: dict = field(default_factory=dict)
    deps: list = field(default_factory=list)


@dataclass
class Plan:
    tasks: list
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps({"tasks": [asdict(t) for t in self.tasks], "notes": self.notes}, separators=(",", ":"))   # compact: fewer tokens

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        if not isinstance(d, dict) or not isinstance(d.get("tasks"), list):
            raise PlanError("plan must be an object with a 'tasks' list")
        tasks, ids = [], set()
        for i, t in enumerate(d["tasks"]):
            if not isinstance(t, dict):
                raise PlanError(f"task {i} is not an object")
            for k in ("id", "arm", "skill"):
                if k not in t:
                    raise PlanError(f"task {i} missing '{k}'")
            if t["arm"] not in ARMS:
                raise PlanError(f"task {t['id']}: unknown arm {t['arm']}")
            if t["skill"] not in SKILLS:
                raise PlanError(f"task {t['id']}: unknown skill {t['skill']}")
            if t["id"] in ids:
                raise PlanError(f"duplicate task id {t['id']}")
            ids.add(t["id"])
            tasks.append(Task(id=str(t["id"]), arm=t["arm"], skill=t["skill"], obj=t.get("obj"),
                              params=dict(t.get("params") or {}), deps=list(t.get("deps") or [])))
        for t in tasks:
            for d_ in t.deps:
                if d_ not in ids:
                    raise PlanError(f"task {t.id}: unknown dependency {d_}")
        _check_acyclic(tasks)
        return Plan(tasks=tasks, notes=str(d.get("notes", "")))


def _check_acyclic(tasks):
    deps = {t.id: set(t.deps) for t in tasks}
    state = {}

    def visit(n):
        if state.get(n) == 1:
            raise PlanError("dependency cycle")
        if state.get(n) == 2:
            return
        state[n] = 1
        for m in deps[n]:
            visit(m)
        state[n] = 2

    for n in deps:
        visit(n)


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def strip_thinking(text: str) -> str:
    """Drop <think>...</think> reasoning (Qwen-style). An unclosed <think> means the reply was cut off while still
    reasoning: nothing after it is an answer, so it is dropped too."""
    t = _THINK.sub("", text)
    k = t.lower().find("<think>")
    return t[:k] if k >= 0 else t


def _lenient_loads(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(_TRAILING_COMMA.sub(r"\1", s))


def _balanced(text: str, open_ch: str, close_ch: str):
    """Yield top-level balanced {...} (or [...]) substrings, string-aware, in order of appearance."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def json_candidates(text: str, kind: str = "object") -> list:
    """Parsed JSON values found in model text, LAST first: fenced blocks, then bare balanced values. <think> is stripped."""
    t = strip_thinking(text)
    o, c = ("{", "}") if kind == "object" else ("[", "]")
    blobs = [b.strip() for b in _FENCE.findall(t)] + list(_balanced(t, o, c))
    out = []
    for b in reversed(blobs):
        try:
            v = _lenient_loads(b)
        except json.JSONDecodeError:
            continue
        if (kind == "object" and isinstance(v, dict)) or (kind == "list" and isinstance(v, list)):
            out.append(v)
        elif kind == "object" and isinstance(v, list):
            continue
    return out


def extract_json(text: str, prefer_keys=()) -> dict:
    """The model's answer object. Strips <think> blocks, accepts fenced or bare JSON and trailing commas, and takes the
    LAST object (models often draft before the final answer); with prefer_keys, the last object having one of those keys."""
    if not isinstance(text, str):
        raise PlanError("completion is not text")
    cands = json_candidates(text, "object")
    if not cands:
        if "{" not in strip_thinking(text):
            raise PlanError("no JSON object found")
        raise PlanError("invalid or unbalanced JSON")
    if prefer_keys:
        for c in cands:
            if any(k in c for k in prefer_keys):
                return c
    return cands[0]


def parse_plan(text: str) -> Plan:
    """The model's plan: the last JSON object with a 'tasks' key (else the last object), validated."""
    return Plan.from_dict(extract_json(text, prefer_keys=("tasks",)))


BAG_SKILLS = ("place",)


def _soft(t, dep) -> bool:
    """Ordering-only dependency: a place waiting on another object's place."""
    return t.skill == "place" and dep.skill == "place" and t.obj != dep.obj


def next_round(tasks, status: dict):
    """Pick this round's tasks. status: id -> pending|done|dead.
    Per arm: first pending task in plan order, runnable if every hard dep is done and every soft
    (ordering-only) dep is done or dead. A task with a dead hard dep becomes dead.
    Bag access is exclusive: at most one bag-access task per round (lock); extras wait."""
    by_id = {t.id: t for t in tasks}
    newly_dead = []
    for t in tasks:
        if status[t.id] == "pending" and any(
                status[d] == "dead" and not _soft(t, by_id[d]) for d in t.deps):
            status[t.id] = "dead"
            newly_dead.append(t.id)

    def ready(t):
        return all(status[d] == "done" or (status[d] == "dead" and _soft(t, by_id[d])) for d in t.deps)

    chosen, bag_used = [], False
    for arm in ARMS:
        for t in tasks:
            if t.arm != arm or status[t.id] != "pending":
                continue
            if ready(t) and not (t.skill in BAG_SKILLS and bag_used):
                chosen.append(t)
                bag_used = bag_used or t.skill in BAG_SKILLS
            break  # only the head pending task of this arm is considered
    return chosen, newly_dead
