"""Multi-turn data-generation runner: no network, no torch (fake transport)."""
import json
import shutil
from pathlib import Path

import pytest

from rtp.benchmark import make_cases
from rtp.brain import ScriptedBrain
from rtp.datagen_memory import notes_from_records, write_memory_docs
from rtp.datagen_run import run_one, run_shard
from rtp.harness import Harness
from rtp.kaggle_chat import (JournaledChat, JournalMismatch, ProxyAuthError, ProxyQuotaError, ProxyRateLimit,
                             ProxyTransport, classify, usage_cost_usd)
from rtp.loop import run_episode

ROOT = Path(__file__).resolve().parents[1] / "harness_files"
SEED = 100000


@pytest.fixture()
def hroot(tmp_path):
    dst = tmp_path / "harness_files"
    shutil.copytree(ROOT, dst)
    return dst


def scripted_plan_json(seed=SEED):
    _s, w, spec, eyes, lc = make_cases([seed])[0]
    res = run_episode(w, eyes, ScriptedBrain(), spec, Harness(ROOT), lc)
    return res.plans[0], res


class FakeTransport:
    """Answers by turn structure so replans are deterministic: a first turn gets a tool call (if
    ``tool_first``), every later turn gets the plan. ``raise_at`` maps a 1-based call index to an exception."""
    model = "fake/model"

    def __init__(self, plan, tool_first=True, raise_at=None):
        self.plan, self.tool_first, self.raise_at = plan, tool_first, dict(raise_at or {})
        self.calls, self.params = 0, {"max_tokens": 100, "temperature": 0.0}

    def __call__(self, messages):
        self.calls += 1
        if self.calls in self.raise_at:
            raise self.raise_at[self.calls]
        first = len(messages) <= 2
        text = ('{"tool":"check_rule","args":{"rule_id":"heavy_bottom"}}' if (first and self.tool_first) else self.plan)
        return {"completion": text, "finish_reason": "stop", "response_model": "fake",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                          "cost": {"input_tokens_cost_nanodollars": 100, "output_tokens_cost_nanodollars": 50}},
                "latency_s": 0.01, "request_params": {"model": "fake/model", "max_tokens": 100}}


def test_classify_follows_the_skill_rules():
    assert classify(403, 'User location "RU" is not supported') == "geo"
    assert classify(401, "bad") == "auth"
    assert classify(403, "token expired") == "auth"
    assert classify(429, "daily quota exceeded") == "quota"
    assert classify(429, "slow down") == "rate_limit"
    assert classify(500, "boom") == "other"
    assert usage_cost_usd({"cost": {"input_tokens_cost_nanodollars": 3e8, "output_tokens_cost_nanodollars": 2e8}}) == pytest.approx(0.5)
    assert usage_cost_usd({"prompt_tokens": 1}) is None


def test_transport_learns_parameter_quirks_from_http_400():
    seen = []

    def post(url, headers, payload):
        seen.append(dict(payload))
        if "temperature" in payload:
            return 400, "temperature is not supported for this model", 0.01
        if "max_tokens" in payload:
            return 400, "use max_completion_tokens instead", 0.01
        return 200, json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}), 0.02
    t = ProxyTransport("x/y", url="http://h/openapi/chat/completions", api_key="k", post=post)
    out = t([{"role": "user", "content": "hi"}])
    assert out["completion"] == "ok"
    assert "temperature" not in seen[-1] and "max_completion_tokens" in seen[-1] and "max_tokens" not in seen[-1]
    assert out["request_params"]["temperature"] is None              # recorded, never assumed
    assert ProxyTransport("anthropic/claude-sonnet-5").params["temperature"] is None


def test_transport_raises_typed_errors():
    def mk(status, body):
        return ProxyTransport("m", url="u", api_key="k", post=lambda *a: (status, body, 0.0))
    with pytest.raises(ProxyQuotaError):
        mk(429, "monthly quota")([])
    with pytest.raises(ProxyRateLimit):
        mk(429, "slow")([])
    with pytest.raises(ProxyAuthError):
        mk(401, "expired")([])


def test_journal_replays_answered_turns_and_fails_closed(tmp_path):
    j = tmp_path / "j.jsonl"
    t1 = FakeTransport("PLAN", tool_first=False, raise_at={2: ProxyQuotaError("cap")})
    c1 = JournaledChat(t1, j)
    m1 = [{"role": "user", "content": "q"}]
    assert c1(m1) == "PLAN"
    with pytest.raises(ProxyQuotaError):
        c1(m1 + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "r"}])
    t2 = FakeTransport("PLAN2", tool_first=False)
    c2 = JournaledChat(t2, j)
    assert c2(m1) == "PLAN" and t2.calls == 0                       # replayed for free
    assert c2(m1 + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "r"}]) == "PLAN2" and t2.calls == 1
    c3 = JournaledChat(FakeTransport("x", tool_first=False), j)
    with pytest.raises(JournalMismatch):
        c3([{"role": "user", "content": "DIFFERENT"}])
    assert c2.totals()["replayed"] == 1 and c2.totals()["cost_usd"] == pytest.approx(150 / 1e9)


def test_harness_episode_records_everything(hroot, tmp_path):
    plan, ref = scripted_plan_json()
    t = FakeTransport(plan)
    rec = run_one(SEED, transport=t, harness_root=hroot, journal_dir=tmp_path / "j", shard="s0", iteration=2)
    assert rec["model"] == "fake/model" and rec["iteration"] == 2 and rec["mode"] == "datagen"
    assert [x["completion"][:6] for x in rec["turns"]][0] == '{"tool'
    assert rec["tool_calls"] == {"check_rule": 1 + rec["result"]["replans"]}   # one tool call per plan request
    assert t.calls == 2 * (1 + rec["result"]["replans"])
    assert rec["usage"]["prompt_tokens"] == 10 * t.calls and rec["plans"]
    assert rec["result"]["failures"] or rec["result"]["success"]              # outcome is recorded either way


def test_zero_shot_is_one_turn_without_the_harness(hroot):
    plan, ref = scripted_plan_json()
    t = FakeTransport(plan, tool_first=False)
    rec = run_one(SEED, transport=t, harness_root=hroot, condition="zero_shot")
    assert t.calls == 1 + rec["result"]["replans"] and rec["tool_calls"] == {} and rec["condition"] == "zero_shot"


def test_seed_namespaces_are_enforced(hroot):
    t = FakeTransport("{}", tool_first=False)
    with pytest.raises(AssertionError):
        run_one(5, transport=t, harness_root=hroot, mode="datagen")      # held-out seed never feeds data
    with pytest.raises(AssertionError):
        run_one(SEED, transport=t, harness_root=hroot, mode="eval")      # training seed never scored


def _calls_for(seed, plan, hroot):
    t = FakeTransport(plan)
    run_one(seed, transport=t, harness_root=hroot)
    return t.calls


def test_shard_pauses_on_quota_and_resumes_without_paying_twice(hroot, tmp_path):
    plan1, _ = scripted_plan_json(SEED)
    plan2, _ = scripted_plan_json(SEED + 1)
    n1, n2 = _calls_for(SEED, plan1, hroot), _calls_for(SEED + 1, plan2, hroot)   # deterministic calibration
    out = tmp_path / "out"
    t1 = FakeTransport(plan1, raise_at={n1 + 2: ProxyQuotaError("daily cap")})     # 2nd call of episode 2
    m1 = run_shard(seeds=[SEED, SEED + 1], transport=t1, harness_root=hroot, out_dir=out, shard="s0")
    assert m1["status"] == "paused" and "quota" in m1["reason"] and m1["recorded"] == 1
    t2 = FakeTransport(plan2)
    m2 = run_shard(seeds=[SEED, SEED + 1], transport=t2, harness_root=hroot, out_dir=out, shard="s0")
    assert m2["status"] == "ok" and m2["recorded"] == 2
    assert t2.calls == n2 - 1                                       # the answered first turn was replayed, not re-sent


def test_rate_limit_backs_off_then_continues(hroot, tmp_path):
    plan, _ = scripted_plan_json()
    sleeps = []
    t = FakeTransport(plan, tool_first=False, raise_at={1: ProxyRateLimit("slow")})
    m = run_shard(seeds=[SEED], transport=t, harness_root=hroot, out_dir=tmp_path / "o", sleep=sleeps.append)
    assert sleeps == [30] and m["status"] == "ok" and m["recorded"] == 1


def test_other_errors_are_kept_as_evidence_and_stop_after_n(hroot, tmp_path):
    from rtp.kaggle_chat import ProxyError
    t = FakeTransport("x", tool_first=False, raise_at={1: ProxyError("500"), 2: ProxyError("500"), 3: ProxyError("500")})
    m = run_shard(seeds=[SEED, SEED + 1, SEED + 2, SEED + 3], transport=t, harness_root=hroot, out_dir=tmp_path / "o")
    assert m["status"] == "error" and len(m["errors"]) == 3


def test_memory_notes_are_built_from_failures_and_retrievable(hroot, tmp_path):
    plan, _ = scripted_plan_json()
    recs = [{"model": "m1", "seed": 1, "instruction": "Pack Maya's bag", "plans": [plan],
             "result": {"success": True, "failures": []}},
            {"model": "m2", "seed": 2, "instruction": "Pack Maya's bag", "plans": ["{}"],
             "result": {"success": False, "failures": [{"agent": "Arm", "cause": "slip", "task": "t1", "obj": "bottle1"},
                                                       {"agent": "Arm", "cause": "slip", "task": "t4", "obj": "bottle1"}]}},
            {"model": "m3", "seed": 3, "outcome": "error"}]
    notes = notes_from_records(recs)
    kinds = sorted(n["kind"] for n in notes)
    assert kinds == ["failure", "success"]
    fail = next(n for n in notes if n["kind"] == "failure")
    assert "2x" in fail["text"] and "slip" in fail["text"] and fail["src"][0]["model"] == "m2"
    ids = write_memory_docs(notes, hroot, iteration=1)
    h = Harness(hroot)                                               # loads, no duplicate ids
    assert set(ids) <= set(h.docs)
    hits = [d.id for d in h.retrieve("bottle slip grasp force failure", k=6, types=("memory",))]
    assert fail["id"] in hits
    write_memory_docs([], hroot, iteration=2)                        # next iteration replaces the snapshot
    assert not [i for i in Harness(hroot).docs if i.startswith("mem_")]


def test_hosted_task_builder_generates_a_runnable_task(tmp_path, monkeypatch):
    """The generated hosted task runs end to end against a fake proxy (kaggle_benchmarks stubbed)."""
    import importlib.util
    import sys
    import types

    from ops.kaggle.multiturn.build_task import build
    plan, _ = scripted_plan_json()
    dest = tmp_path / "task.py"
    spec = build(seeds="100000:100002", condition="harness", mode="datagen", iteration=3, shard="unit",
                 memory_notes=[{"kind": "failure", "id": "mem_f_slip_bottle_arm", "keywords": "failure slip bottle",
                                "text": "Past failure (2x): Arm failed with 'slip' on bottle.", "src": []}], dest=dest)
    stub = types.ModuleType("kaggle_benchmarks")
    stub.task = lambda **kw: (lambda f: type("T", (), {"run": staticmethod(lambda llm: None), "fn": staticmethod(f)})())
    stub.llm = object()
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks", stub)
    monkeypatch.setenv("MODEL_PROXY_URL", "http://x")
    s = importlib.util.spec_from_file_location("gen_task", dest)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    assert mod.SPEC["spec_hash"] == spec["spec_hash"] and mod.SPEC["iteration"] == 3

    def post(url, headers, payload):
        first = len(payload["messages"]) <= 2
        text = '{"tool":"check_rule","args":{"rule_id":"heavy_bottom"}}' if first else plan
        body = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "model": "m",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                          "cost": {"input_tokens_cost_nanodollars": 100, "output_tokens_cost_nanodollars": 50}}}
        return 200, json.dumps(body), 0.01
    manifest = mod.execute("anthropic/claude-haiku-5-5", "http://x/openapi/chat/completions", "k",
                           repo_dir=Path(__file__).resolve().parents[1], work=tmp_path / "work", post=post,
                           harness_tmp=str(tmp_path / "h"))
    assert manifest["status"] == "ok" and manifest["recorded"] == 2 and manifest["memory_notes"] == 1
    assert manifest["spec_hash"] == spec["spec_hash"] and manifest["model"] == "anthropic/claude-haiku-5-5"
    assert (tmp_path / "h" / "memory" / "mem_f_slip_bottle_arm.md").exists()
    recs = [json.loads(line) for line in (tmp_path / "work" / "multiturn" / "unit" / "episodes.jsonl").read_text().splitlines()]
    assert {r["seed"] for r in recs} == {100000, 100001} and all(r["iteration"] == 3 for r in recs)


def _shard_records(hroot, tmp_path, seeds=(SEED, SEED + 1)):
    plan, _ = scripted_plan_json()
    out = tmp_path / "shard"
    run_shard(seeds=list(seeds), transport=FakeTransport(plan), harness_root=hroot, out_dir=out, shard="s", iteration=1)
    return out / "episodes.jsonl"


def test_replay_verification_accepts_genuine_records_and_quarantines_tampered_ones(hroot, tmp_path):
    from rtp.datagen_ingest import ingest, ledger_entry, next_memory, read_records
    path = _shard_records(hroot, tmp_path)
    recs = read_records(path)
    # tamper with: a flipped outcome, a held-out seed in datagen mode, an error record, a missing model
    recs[0]["result"]["success"] = not recs[0]["result"]["success"]
    bad_ns = {**recs[1], "seed": 7}
    err = {"schema": 1, "kind": "multiturn_episode", "outcome": "error", "seed": SEED + 9, "model": "m"}
    nomodel = {**recs[1], "model": ""}
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(json.dumps(r) for r in [recs[0], bad_ns, err, nomodel]) + "\n")
    got = ingest(tampered, hroot)
    reasons = sorted(q["reason"] for q in got["quarantined"])
    assert got["accepted"] == []
    assert set(reasons) == {"replay_mismatch", "seed_namespace", "error_record", "missing_identity_or_turns"}
    mm = next(q for q in got["quarantined"] if q["reason"] == "replay_mismatch")
    assert "success" in mm["diff"] and mm["diff"]["success"]["recorded"] != mm["diff"]["success"]["replayed"]
    clean = ingest(path, hroot)
    assert len(clean["accepted"]) == 2 and not clean["quarantined"]
    assert all(r["verification"]["replayed"] for r in clean["accepted"])
    led = ledger_entry(clean["accepted"], clean["quarantined"], shard="s", iteration=1)
    m = led["models"]["fake/model"]
    assert m["episodes"] == 2 and m["cost_usd"] is not None and led["quarantined"] == 0
    assert isinstance(next_memory(clean["accepted"]), list)


def test_eval_records_never_feed_memory_and_identity_pin_is_enforced(hroot, tmp_path):
    from rtp.datagen_ingest import ingest, next_memory
    plan, _ = scripted_plan_json(5)
    out = tmp_path / "ev"
    run_shard(seeds=[5], transport=FakeTransport(plan), harness_root=hroot, out_dir=out, mode="eval", shard="e")
    got = ingest(out / "episodes.jsonl", hroot)
    assert len(got["accepted"]) == 1 and next_memory(got["accepted"]) == []   # eval never feeds memory
    pinned = ingest(out / "episodes.jsonl", hroot, check_identity={"model": "other/model"})
    assert pinned["quarantined"][0]["reason"] == "identity_mismatch"


class LookFirst(FakeTransport):
    """Looks for objects first (changes what the world knows), then plans: replay must repeat the look."""

    def __call__(self, messages):
        self.calls += 1
        first = len(messages) <= 2
        txt = '{"tool":"look","args":{"objects":["book1","bottle1","laptop1"]}}' if first else self.plan
        return {"completion": txt, "finish_reason": "stop", "response_model": "f", "latency_s": 0.0,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "request_params": {"model": "x"}}


def test_replay_verification_repeats_tool_effects_such_as_look(hroot):
    from rtp.datagen_ingest import verify_record
    bad = []
    for seed in range(SEED, SEED + 12):
        p, _ = scripted_plan_json(seed)
        rec = run_one(seed, transport=LookFirst(p), harness_root=hroot)
        v = verify_record(rec, hroot)
        if not v["ok"]:
            bad.append((seed, v["reason"], list(v["diff"])))
    assert not bad, bad


def test_replay_verification_of_zero_shot_and_prompt_drift_is_reported_not_fatal(hroot, tmp_path):
    from rtp.datagen_ingest import verify_record
    p, _ = scripted_plan_json()
    rec = run_one(SEED, transport=FakeTransport(p, tool_first=False), harness_root=hroot, condition="zero_shot")
    assert verify_record(rec, hroot)["ok"]
    rec2 = run_one(SEED, transport=FakeTransport(p), harness_root=hroot)
    other = tmp_path / "other"
    shutil.copytree(hroot, other)
    write_memory_docs([{"kind": "failure", "id": "mem_f_x", "keywords": "failure slip", "text": "Past failure (1x).", "src": []}], other, 1)
    v = verify_record(rec2, other)                       # same recorded turns, different memory docs in the prompt
    assert v["ok"] and v["prompt_drift"] >= 0


def test_iteration_driver_end_to_end(hroot, tmp_path, monkeypatch):
    """build -> (fake hosted run) -> ingest -> memory for the next iteration -> status."""
    import argparse
    import ops.datagen_iterate as d
    monkeypatch.setattr(d, "STATE", tmp_path / "state")
    monkeypatch.setattr(d, "HARNESS", hroot)
    out = _shard_records(hroot, tmp_path)
    a = argparse.Namespace(shard="s", iteration=1, episodes=str(out))
    assert d.cmd_ingest(a) == 0
    assert (tmp_path / "state" / "accepted.jsonl").exists() and (tmp_path / "state" / "memory_it2.json").exists()
    assert d.cmd_ingest(a) == 0                                         # idempotent: no duplicate accepted records
    from rtp.datagen_ingest import read_records
    assert len(read_records(tmp_path / "state" / "accepted.jsonl")) == 2
    assert d.cmd_status(None) == 0
    wrong = argparse.Namespace(shard="s", iteration=7, episodes=str(out))  # identity pin: records say iteration 1
    d.cmd_ingest(wrong)
    q = read_records(tmp_path / "state" / "quarantine.jsonl")
    assert q and all(x["reason"] == "identity_mismatch" for x in q)


def test_arm_model_is_recorded_and_only_certified_records_reach_sft(hroot, tmp_path):
    from rtp.datagen_ingest import ingest, sft_eligible
    p, _ = scripted_plan_json()
    out = tmp_path / "o"
    run_shard(seeds=[SEED], transport=FakeTransport(p), harness_root=hroot, out_dir=out, shard="a")
    run_shard(seeds=[SEED + 1], transport=FakeTransport(scripted_plan_json(SEED + 1)[0]), harness_root=hroot,
              out_dir=out, shard="a", arm_model="newton_cert_v1")
    acc = ingest(out / "episodes.jsonl", hroot)["accepted"]
    assert {r["arm_model"] for r in acc} == {"sim_pose_v0", "newton_cert_v1"}
    assert [r["seed"] for r in sft_eligible(acc, {"newton_cert_v1"})] == [SEED + 1]
    assert sft_eligible(acc, set()) == []
