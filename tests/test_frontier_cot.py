"""Offline tests for the CoT trace harness: no endpoint is contacted."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtp import frontier_cot as fc  # noqa: E402
from rtp.brain import ScriptedBrain  # noqa: E402
from rtp.datagen import default_eyes, sample_spec, sample_world  # noqa: E402
from rtp.harness import Harness, tokenize  # noqa: E402
from rtp.perception import merge_scenes  # noqa: E402
from rtp.plan import parse_plan  # noqa: E402

HARNESS_ROOT = ROOT / "harness_files"


@pytest.fixture
def case():
    harness = Harness(str(HARNESS_ROOT))
    world = sample_world(0)
    spec = sample_spec(world, 0)
    scene = merge_scenes([e.report(world, 0) for e in default_eyes(0)])
    return harness, spec, scene


class FakeChatClient:
    """ChatClient-shaped fake: records calls, returns a fixed completion or raises."""

    def __init__(self, reply, raises=None):
        self.reply, self.raises, self.calls = reply, raises, []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        if self.raises is not None:
            raise self.raises
        return self.reply


def _teacher_reply(case):
    harness, spec, scene = case
    ctx = fc.cot_request(harness, spec, scene).ctx
    return "Reasoning: deterministic teacher draft.\nPlan: " + ScriptedBrain().plan(ctx).to_json()


def test_build_prompt_contains_retrieved_ids_and_cot_request(case):
    harness, spec, scene = case
    req = fc.cot_request(harness, spec, scene, k=6)
    assert "brain_rules" in req.doc_ids and "brain_skills" in req.doc_ids
    assert any(i.startswith("obj_") for i in req.doc_ids)
    assert any(i.startswith("heavy") or i.startswith("laptop") for i in req.doc_ids)
    for doc_id in req.doc_ids[:3]:                      # build_prompt renders "[id] body"
        assert f"[{doc_id}]" in req.prompt
    assert "## Chain-of-thought request" in req.prompt
    assert "Reasoning:" in req.prompt and "Plan:" in req.prompt
    assert fc.build_cot_prompt(harness, spec, scene, k=6) == req.prompt
    assert fc.build_cot_prompt(harness, spec, scene, style="full") != req.prompt


def test_build_prompt_accepts_optional_embed_fn(case):
    harness, spec, scene = case

    def embed(text):
        v = [0.0] * 32
        for tok in tokenize(text):
            h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
            v[int.from_bytes(h[:4], "big") % 32] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    req = fc.cot_request(harness, spec, scene, k=6, embed_fn=embed)
    assert req.doc_ids and "brain_rules" in req.doc_ids
    assert "## Retrieved notes" in req.prompt


def test_run_cot_fake_client_record_schema(case):
    harness, spec, scene = case
    reply = _teacher_reply(case)
    client = FakeChatClient(reply)
    rec = fc.run_cot(harness, spec, scene, client=client, seed=7)
    required = {"schema_version", "source", "is_frontier", "style", "seed", "model", "prompt",
                "prompt_tokens", "retrieved_doc_ids", "token_counter", "completion", "reasoning",
                "plan_json", "completion_tokens", "latency_ms", "error", "ts"}
    assert required <= set(rec)
    assert rec["source"] == "frontier" and rec["is_frontier"] is True
    assert rec["error"] is None
    assert rec["prompt"] == fc.build_cot_prompt(harness, spec, scene)
    assert rec["retrieved_doc_ids"] == fc.cot_request(harness, spec, scene).doc_ids
    assert rec["completion"] == reply
    assert rec["reasoning"].startswith("deterministic teacher draft")
    assert len(parse_plan(rec["plan_json"]).tasks) > 0
    assert rec["prompt_tokens"] > 0 and rec["completion_tokens"] > 0
    assert rec["latency_ms"] >= 0
    assert client.calls[0][1]["max_tokens"] == 1200


def test_run_cot_without_client_raises(case, monkeypatch):
    harness, spec, scene = case
    monkeypatch.delenv("RTP_SERVE_URL", raising=False)
    with pytest.raises(fc.NoFrontierClient, match="RTP_SERVE_URL"):
        fc.run_cot(harness, spec, scene)


def test_run_cot_records_client_error(case):
    harness, spec, scene = case
    client = FakeChatClient("", raises=RuntimeError("endpoint exploded"))
    rec = fc.run_cot(harness, spec, scene, client=client)
    assert rec["source"] == "frontier"
    assert "endpoint exploded" in rec["error"]
    assert rec["completion"] == "" and rec["plan_json"] == ""
    assert rec["completion_tokens"] == 0 and rec["latency_ms"] >= 0


def test_scripted_trace_is_marked_and_real(case):
    harness, spec, scene = case
    rec = fc.scripted_cot(harness, spec, scene, seed=3)
    assert rec["source"] == "scripted-teacher" and rec["is_frontier"] is False
    assert "ScriptedBrain" in rec["model"]
    assert rec["error"] is None
    assert rec["retrieved_doc_ids"] == fc.cot_request(harness, spec, scene).doc_ids
    assert rec["reasoning"] and "retrieved rules:" in rec["reasoning"]
    assert len(parse_plan(rec["plan_json"]).tasks) > 0
    assert rec["completion"].startswith("Reasoning: ") and "\nPlan: " in rec["completion"]
    assert rec["completion_tokens"] > 0 and rec["prompt_tokens"] > 0


def test_configured_client_from_env(case, monkeypatch):
    harness, spec, scene = case
    created = []

    class CapturingClient:
        def __init__(self, **kw):
            self.kw = kw
            created.append(self)

        def chat(self, messages, **kw):
            return _teacher_reply(case)

    monkeypatch.setenv("RTP_SERVE_URL", "http://localhost:9000/v1/chat/completions")
    monkeypatch.setenv("RTP_MODEL", "qwen-cot")
    monkeypatch.setattr(fc, "ChatClient", CapturingClient)
    rec = fc.run_cot(harness, spec, scene)
    assert rec["error"] is None and rec["model"] == "qwen-cot"
    assert created[0].kw["base_url"] == "http://localhost:9000/v1"
    assert created[0].kw["model"] == "qwen-cot"
    assert created[0].kw["api_key"] == "EMPTY"      # no frontier key is assumed


def test_write_read_traces_roundtrip(case, tmp_path):
    harness, spec, scene = case
    recs = [fc.scripted_cot(harness, spec, scene, seed=i) for i in range(2)]
    path = fc.write_traces(recs, tmp_path / "sub" / "cot_traces.jsonl")
    assert len(path.read_text().splitlines()) == 2
    loaded = fc.read_traces(path)
    assert loaded == recs
    assert fc.read_traces(tmp_path / "missing.jsonl") == []
