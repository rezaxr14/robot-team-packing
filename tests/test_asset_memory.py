"""Tests for success/failure memory retrieval on asset-backed episodes."""
import json
from pathlib import Path

from rtp.asset_memory import (augment_prompt, build_memory, read_memory, retrieve_memory,
                              write_memory)
from rtp.harness import Harness
from rtp.scene_assets import resolve_assets

ROOT = Path(__file__).resolve().parents[1]


def fake_catalog(tmp_path):
    entries = []
    for i, cls in enumerate(("book", "notebook", "lunchbox", "bottle", "chair")):
        f = tmp_path / f"m{i}.usd"
        f.write_text("fake")
        entries.append({"package": f"mem_{cls}_{i}", "repo": "nvidia/simready-assets",
                        "class": cls, "license": "CC-BY-4.0", "paths": [str(f)]})
    (tmp_path / "catalog.json").write_text(json.dumps({"assets": entries}))
    (tmp_path / "selection.json").write_text(json.dumps(
        [{"package": e["package"]} for e in entries]))
    return tmp_path


def test_memory_records_outcomes_and_roundtrips(tmp_path):
    d = fake_catalog(tmp_path)
    recs = resolve_assets(d / "catalog.json", d / "selection.json")
    rows, stats = build_memory(recs, 4, 300000, Harness(ROOT / "harness_files"))
    assert stats["episodes"] == 4
    assert stats["successes"] + stats["failures"] == 4
    assert all(r["evidence"] == "SIM-POSE" for r in rows)
    p = write_memory(rows, d / "memory.jsonl")
    assert read_memory(p) == rows


def test_retrieval_separates_success_from_failure(tmp_path):
    d = fake_catalog(tmp_path)
    recs = resolve_assets(d / "catalog.json", d / "selection.json")
    rows, _ = build_memory(recs, 6, 300000, Harness(ROOT / "harness_files"))
    if not any(r["outcome"] == "failure" for r in rows):
        import pytest
        pytest.skip("no failure occurred in this sample; nothing to retrieve")
    hits = retrieve_memory(rows, "pack the book", k=2, outcome="failure")
    assert all(r["outcome"] == "failure" for r, _score in hits)


def test_prompt_includes_retrieved_memories(tmp_path):
    d = fake_catalog(tmp_path)
    recs = resolve_assets(d / "catalog.json", d / "selection.json")
    rows, _ = build_memory(recs, 3, 300000, Harness(ROOT / "harness_files"))
    prompt = augment_prompt(recs, rows, 300000, Harness(ROOT / "harness_files"), k=2)
    assert "memory-success" in prompt or "memory-failure" in prompt
