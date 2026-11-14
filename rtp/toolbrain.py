"""ToolBrain: a multi-turn LLM Brain that can call tools before committing to a plan.

Tools (one JSON object per model turn: {"tool": name, "args": {...}}):
  check_rule {rule_id}        -> the harness text for a rule/skill/object note
  query_scene {cls?, region?} -> scene entries matching the filter
  look {objects: [names]}     -> a fresh Eye observation for objects missing from the scene (bounded by the loop)
  dry_run {plan}              -> static checks + deterministic dry-run (violations, failed tasks) of a DRAFT plan
  submit_plan {plan}          -> commit; replying with the plan object directly also commits
Only the committed plan reaches the arms; drafts cost model tokens, not robot time.
`chat(messages) -> str` is any chat function (e.g. ChatClient.chat). If the budget of turns runs out the last
dry-run draft is used, else PlanError (counted as a Brain failure by the loop).
"""
from __future__ import annotations
import json

from .brain import BrainContext, PROMPT_HEADER, build_prompt
from .plan import Plan, PlanError, extract_json
from .rules import dry_run, static_check

TOOL_HEADER = """You may call tools before answering. Reply with exactly ONE JSON object per turn.
Tool call: {"tool": "<name>", "args": {...}}
Tools: check_rule {"rule_id"}; query_scene {"cls"?, "region"?}; look {"objects": [names]}; dry_run {"plan"}; submit_plan {"plan"}.
dry_run executes your draft plan deterministically and reports rule violations and failed tasks: use it before submit_plan.
You can also commit by replying with the plan object itself."""

MAX_RESULT_CHARS = 2500


class ToolBrain:
    def __init__(self, chat, harness=None, max_turns: int = 6):
        self.chat, self.harness, self.max_turns = chat, harness, max_turns
        self.requests: list = []
        self.stats: dict = {}
        self.totals = {"turns": 0, "errors": 0, "plans": 0, "tool_calls": {}}   # accumulated over the episode

    # ---- tools ----
    def _check_rule(self, ctx, args):
        rid = str(args.get("rule_id", ""))
        for d in ctx.snippets:
            if d.id == rid:
                return d.body
        if self.harness is not None and rid in self.harness.docs:
            return self.harness.get(rid).body
        return f"unknown id {rid!r}"

    def _query_scene(self, ctx, args):
        rows = [o for o in ctx.scene.values()
                if all(o.get(k) == v for k, v in args.items() if k in ("cls", "region"))]
        return json.dumps(rows, separators=(",", ":"))

    def _look(self, ctx, args):
        if ctx.look_fn is None:
            return "look is not available"
        found = ctx.look_fn([str(n) for n in args.get("objects", [])])
        if found:
            ctx.scene.update(found)
        return json.dumps(list(found.values()), separators=(",", ":")) if found else "nothing new was seen"

    def _dry_run(self, ctx, args):
        plan = Plan.from_dict(args["plan"])
        ok, errs = static_check(plan, ctx.scene, ctx.spec)
        r = dry_run(plan, ctx.scene, ctx.spec)
        self.last_draft = plan
        return json.dumps({"static_ok": ok, "static_errors": errs[:5], "success": r["success"],
                           "violations": [f"{v['rule']}: {v['detail']}" for v in r["violations"]],
                           "missing": r["missing"], "failed_tasks": r["failed_tasks"]})

    # ---- loop ----
    def plan(self, ctx: BrainContext, reason: str = "") -> Plan:
        try:
            return self._plan(ctx, reason)
        finally:
            self.totals["plans"] += 1
            self.totals["turns"] += self.stats["turns"]
            self.totals["errors"] += self.stats["errors"]
            for k, v in self.stats["tool_calls"].items():
                self.totals["tool_calls"][k] = self.totals["tool_calls"].get(k, 0) + v

    def _plan(self, ctx: BrainContext, reason: str = "") -> Plan:
        self.stats = {"turns": 0, "tool_calls": {}, "errors": 0}
        self.last_draft = None
        msgs = [{"role": "system", "content": TOOL_HEADER + "\n\n" + PROMPT_HEADER},
                {"role": "user", "content": build_prompt(ctx, reason)}]
        for _ in range(self.max_turns):
            self.stats["turns"] += 1
            out = self.chat(msgs)
            msgs.append({"role": "assistant", "content": out})
            try:
                obj = extract_json(out, prefer_keys=("tasks", "tool"))
                if "tasks" in obj:
                    return Plan.from_dict(obj)
                name, args = obj.get("tool"), obj.get("args") or {}
                if name == "submit_plan":
                    return Plan.from_dict(args["plan"])
                fn = {"check_rule": self._check_rule, "query_scene": self._query_scene,
                      "look": self._look, "dry_run": self._dry_run}.get(name)
                if fn is None:
                    raise PlanError(f"unknown tool {name!r}")
                self.stats["tool_calls"][name] = self.stats["tool_calls"].get(name, 0) + 1
                result = fn(ctx, args)
            except (PlanError, KeyError, TypeError, ValueError) as e:
                self.stats["errors"] += 1
                result = f"ERROR: {type(e).__name__}: {e}"
            msgs.append({"role": "user", "content": "TOOL RESULT: " + str(result)[:MAX_RESULT_CHARS]})
        if self.last_draft is not None:
            return self.last_draft
        raise PlanError("no plan committed within the turn budget")

    def replan(self, ctx: BrainContext, reason: str) -> Plan:
        return self.plan(ctx, reason)
