"""Offline tests for CoT prompt / frontier-task serialization (no model call, no Kaggle)."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops import cot_prompts as cp  # noqa: E402
from ops.kaggle.cot_frontier import build_task, run  # noqa: E402
from rtp import frontier_cot as fc  # noqa: E402
from rtp.datagen import default_eyes, sample_spec, sample_world  # noqa: E402
from rtp.harness import Harness  # noqa: E402
from rtp.perception import merge_scenes  # noqa: E402


@pytest.fixture(scope="module")
def rows():
    return cp.build_records(n=3, seed0=5)


def test_records_are_real_prompts_with_required_schema(rows):
    assert len(rows) == 3
    harness = Harness(str(ROOT / "harness_files"))
    for i, row in enumerate(rows):
        assert set(cp.REQUIRED_FIELDS) <= set(row)
        assert row["id"] == f"cot-{5 + i}"
        assert row["seed"] == 5 + i
        assert isinstance(row["prompt"], str) and len(row["prompt"]) > 500
        assert isinstance(row["retrieved_ids"], list) and "brain_rules" in row["retrieved_ids"]
        assert row["task_class"] == "pack_backpack"
        world = sample_world(5 + i)
        spec = sample_spec(world, 5 + i)
        scene = merge_scenes([e.report(world, 0) for e in default_eyes(5 + i)])
        assert row["prompt"] == fc.build_cot_prompt(harness, spec, scene, k=row["k"], style=row["style"])
        assert row["retrieved_ids"] == fc.cot_request(harness, spec, scene, k=row["k"]).doc_ids
        assert "## Chain-of-thought request" in row["prompt"]
    assert len({r["prompt_sha256"] for r in rows}) == 3   # distinct scenes, distinct prompts


def test_records_are_deterministic_and_style_aware():
    a = cp.build_records(n=2, seed0=1)
    b = cp.build_records(n=2, seed0=1)
    assert a == b
    full = cp.build_records(n=1, seed0=1, style="full")
    assert full[0]["prompt"] != a[0]["prompt"]
    assert full[0]["style"] == "full"


def test_jsonl_write_read_roundtrip_and_validation(rows, tmp_path):
    path = cp.write_records(rows, tmp_path / "sub" / "cot_prompts.jsonl")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    assert cp.read_records(path) == rows
    assert cp.read_records(tmp_path / "missing.jsonl") == []
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        cp.read_records(bad)


def test_cli_writes_prompts_without_any_model_call(rows, tmp_path, capsys):
    out = tmp_path / "prompts.jsonl"
    rc = cp.main(["--n", "2", "--seed0", "9", "--out", str(out)])
    assert rc == 0
    written = cp.read_records(out)
    assert [r["seed"] for r in written] == [9, 10]
    summary = json.loads(capsys.readouterr().out)
    assert summary["count"] == 2 and summary["sources"] == ["harness+datagen (no model call)"]


def test_hosted_task_builder_embeds_prompts_and_uses_model_proxy(rows):
    source = build_task.build_task_source(rows)
    assert "__SPEC_JSON__" not in source
    tree = ast.parse(source)                              # generated task is valid Python
    spec = None
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "SPEC" for t in node.targets)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "attr", None) == "loads"):
            spec = json.loads(ast.literal_eval(node.value.args[0]))
    assert spec is not None
    # kbench needs the real `dict` type object, not a postponed string annotation.
    assert not any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
                   and any(a.name == "annotations" for a in n.names) for n in tree.body)
    assert spec["task_name"] == build_task.TASK_NAME
    assert spec["source"] == "kaggle-benchmarks-proxy"
    assert spec["prompts"] == [{k: r[k] for k in cp.REQUIRED_FIELDS} for r in rows]
    for needle in ('kaggle_benchmarks', '@kbench.task', 'MODEL_PROXY_URL', 'MODEL_PROXY_API_KEY',
                   '/openapi/chat/completions', 'requests.post', 'llm.model',
                   'cot_frontier_traces.jsonl', 'cot_frontier_manifest.json', 'No invented answers',
                   'reasoning_effort', 'def rtp_cot_frontier_fullplan(llm) -> dict:'):
        assert needle in source, needle
    assert 'import kaggle_benchmarks' in source


def test_report_serialize_offline_names_model_and_api(rows):
    trace_rows = [{"id": r["id"], "seed": r["seed"], "prompt": r["prompt"], "completion": "Reasoning: x\nPlan: y",
                   "retrieved_ids": r["retrieved_ids"], "source": run.EXPECTED_SOURCE,
                   "model": "google/gemini-3-flash-preview", "latency_s": 0.5} for r in rows[:2]]
    manifest = {"model": "google/gemini-3-flash-preview",
                "endpoint": "https://mp-staging.kaggle.net/models/openapi/chat/completions",
                "api": "POST /openapi/chat/completions (OpenAI-compatible Model Proxy)",
                "errors": [], "total_usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    report = run._report(trace_rows, manifest, [], "completed", 0, "", Path("x.jsonl"),
                         "local_4050", [])
    assert report["model"] == "google/gemini-3-flash-preview"
    assert report["api_call"]["method"] == "POST" and "chat/completions" in report["api_call"]["url"]
    assert report["counts"]["total"] == 2 and report["counts"]["completed"] == 2
    assert report["source_labels"] == [run.EXPECTED_SOURCE]
    assert report["success"] is True and report["error_summary"]["count"] == 0

    failed = run._report([], {"model": "m", "errors": [{"id": "cot-0", "error": "HTTP 403: blocked"}]},
                         [], "failed", 0, "", Path("x.jsonl"), "local_4050", [])
    assert failed["counts"]["total"] == 0 and failed["success"] is False
    assert any("HTTP 403: blocked" in e for e in failed["error_summary"]["errors"])
