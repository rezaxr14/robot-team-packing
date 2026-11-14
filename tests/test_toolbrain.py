import json
from pathlib import Path

import pytest

from rtp.harness import Harness
from rtp.datagen import sample_world, sample_spec, default_eyes
from rtp.brain import ScriptedBrain, BrainContext
from rtp.loop import LoopConfig, run_episode
from rtp.perception import Eye, EyeConfig
from rtp.plan import PlanError, extract_json, parse_plan
from rtp.toolbrain import ToolBrain

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")


def teacher_plans(seed):
    w = sample_world(seed)
    spec = sample_spec(w, seed)
    good = run_episode(sample_world(seed), default_eyes(seed), ScriptedBrain(), spec, H, LoopConfig()).plans[0]
    bad = run_episode(sample_world(seed), default_eyes(seed), ScriptedBrain(), spec, H,
                      LoopConfig(use_harness=False, static_rule_ids=())).plans[0]
    return spec, good, bad


class Script:
    """Fake chat model: replays canned replies and records what it was shown."""
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0) if self.replies else json.dumps({"tool": "query_scene", "args": {}})   # never commits


def call(tool, **args):
    return json.dumps({"tool": tool, "args": args})


def test_extract_json_helper():
    assert extract_json('x {"a": 1} y') == {"a": 1} and extract_json('```json\n{"b": 2}\n```') == {"b": 2}
    for bad in ("no", "{", "[1]", '{"a": }'):
        with pytest.raises(PlanError):
            extract_json(bad)


def test_draft_check_revise_then_submit_succeeds_and_sees_feedback():
    spec, good, bad = teacher_plans(1)
    chat = Script([call("check_rule", rule_id="heavy_bottom"), call("dry_run", plan=json.loads(bad)),
                   call("submit_plan", plan=json.loads(good))])
    brain = ToolBrain(chat, H)
    w = sample_world(1)
    res = run_episode(w, default_eyes(1), brain, spec, H, LoopConfig())
    assert res.success and brain.stats["turns"] == 3 and brain.stats["tool_calls"] == {"check_rule": 1, "dry_run": 1}
    third = chat.seen[2]                                         # what the model saw before its final turn
    fb = json.loads(third[-1]["content"].split("TOOL RESULT: ", 1)[1])
    assert fb["success"] is False and fb["violations"]           # the dry run told it the draft was bad
    assert "Heavy items" in third[-3]["content"] or "heavier" in third[-3]["content"].lower() or "TOOL RESULT" in third[-3]["content"]


def test_dry_run_of_good_plan_reports_success():
    spec, good, _ = teacher_plans(2)
    chat = Script([call("dry_run", plan=json.loads(good)), good])             # then commits by replying with the plan itself
    brain = ToolBrain(chat, H)
    res = run_episode(sample_world(2), default_eyes(2), brain, spec, H, LoopConfig())
    assert res.success and json.loads(chat.seen[1][-1]["content"].split("TOOL RESULT: ", 1)[1])["success"] is True


def test_errors_are_fed_back_not_fatal():
    spec, good, _ = teacher_plans(1)
    chat = Script(["not json at all", call("no_such_tool"), call("dry_run"), call("check_rule", rule_id="zzz"), good])
    brain = ToolBrain(chat, H, max_turns=6)
    res = run_episode(sample_world(1), default_eyes(1), brain, spec, H, LoopConfig())
    assert res.success and brain.stats["errors"] == 3
    assert "unknown id 'zzz'" in chat.seen[4][-1]["content"]


def test_turn_budget_falls_back_to_last_draft_else_fails():
    spec, good, bad = teacher_plans(1)
    chat = Script([call("dry_run", plan=json.loads(bad)), call("query_scene"), call("query_scene")])
    res = run_episode(sample_world(1), default_eyes(1), ToolBrain(chat, H, max_turns=3), spec, H, LoopConfig())
    assert res.plans and not res.success                            # committed the last (bad) draft
    chat2 = Script([call("query_scene")] * 3)
    res2 = run_episode(sample_world(1), default_eyes(1), ToolBrain(chat2, H, max_turns=3), spec, H, LoopConfig())
    assert not res2.plans and any(f["cause"] == "plan_parse" and f["agent"] == "Brain" for f in res2.failures)


def test_query_scene_filters():
    spec, good, _ = teacher_plans(1)
    seen = {}

    def chat(messages):
        if len(messages) == 2:
            return call("query_scene", cls="laptop")
        seen["result"] = messages[-1]["content"]
        return good

    run_episode(sample_world(1), default_eyes(1), ToolBrain(chat, H), spec, H, LoopConfig())
    rows = json.loads(seen["result"].split("TOOL RESULT: ", 1)[1])
    assert [r["cls"] for r in rows] == ["laptop"]


def test_look_tool_brings_back_missing_objects_through_the_loop():
    leaky = lambda s: [Eye(EyeConfig(name="A", blind_regions=("table_left", "table_right", "table_center"), miss_occluded_p=0.5), s)]
    found_any = 0
    for s in range(20):
        w = sample_world(s)
        spec = sample_spec(w, s)
        ctxs = {}

        class Peek(ScriptedBrain):
            pass
        brain = ToolBrain(None, H)                                  # chat set below once the spec is known
        names = list(spec.items)

        def chat(messages, names=names):
            if len(messages) == 2:
                return call("look", objects=names)
            return json.dumps({"tasks": []})

        brain.chat = chat
        res = run_episode(w, leaky(s), brain, spec, H, LoopConfig(allow_look=False))   # loop-initiated looks off: only the tool can look
        assert res.looks == 0                                       # disabled loop budget => tool refused
        w2 = sample_world(s)
        brain2 = ToolBrain(chat, H)
        res2 = run_episode(w2, leaky(s), brain2, spec, H, LoopConfig(allow_look=True, max_looks=2))
        found_any += res2.looks > 0
    assert found_any == 20


def test_experiments_runner_supports_tool_brain_and_records_totals():
    from rtp.experiments import ExpConfig, run_config
    spec, good, _ = teacher_plans(0)
    chat = Script([call("check_rule", rule_id="heavy_bottom"), good])
    recs = run_config(ExpConfig("tool", brain="tool"), [0], H, chat=chat)
    assert recs[0]["success"] and recs[0]["brain_totals"]["tool_calls"] == {"check_rule": 1}
    assert recs[0]["brain_totals"]["turns"] == 2 and recs[0]["brain_totals"]["plans"] == 1
