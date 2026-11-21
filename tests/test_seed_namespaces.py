"""Tests for the seed namespace rules (RUNBOOK rule 8)."""
import json

import pytest

from rtp import seeds


def test_boundary_is_100000():
    assert seeds.TRAIN_MIN == 100_000
    assert seeds.is_train(100_000) and not seeds.is_test(100_000)
    assert seeds.is_test(99_999) and not seeds.is_train(99_999)


def test_assert_train_rejects_test_seeds():
    seeds.assert_train([100_000, 100_017])
    with pytest.raises(seeds.SeedNamespaceError):
        seeds.assert_train([100_000, 7])


def test_assert_test_rejects_training_seeds():
    seeds.assert_test([0, 99_999])
    with pytest.raises(seeds.SeedNamespaceError):
        seeds.assert_test([42, 100_001])


def test_split_bounds_stay_inside_namespace():
    assert seeds.split_bounds(seeds.TRAIN, 10) == (100_000, 100_010)
    assert seeds.split_bounds(seeds.TEST, 30) == (0, 30)
    assert seeds.split_bounds(seeds.TEST, 5, start=99_990) == (99_990, 99_995)
    with pytest.raises(seeds.SeedNamespaceError):
        seeds.split_bounds(seeds.TEST, 20, start=99_990)
    with pytest.raises(seeds.SeedNamespaceError):
        seeds.split_bounds(seeds.TRAIN, 5, start=5)
    with pytest.raises(ValueError):
        seeds.split_bounds("validation", 5)


def test_iter_seed_values_finds_nested_seeds_only():
    obj = {"meta": {"seed": 100_003}, "rows": [{"seed": 5}, {"nested": {"seed": 100_004}}, {"not_seed": 7}]}
    found = dict(seeds.iter_seed_values(obj))
    assert found["meta.seed"] == 100_003
    assert found["rows[0].seed"] == 5
    assert found["rows[1].nested.seed"] == 100_004
    assert len(found) == 3


def test_audit_paths_reports_but_does_not_modify(tmp_path):
    path = tmp_path / "corpus.jsonl"
    rows = [{"seed": 100_001, "x": 1}, {"seed": 11, "x": 2}, {"seed": 100_002, "x": 3}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    before = path.read_bytes()
    findings = seeds.audit_paths([path])
    assert [(f["record"], f["seed"]) for f in findings] == [(1, 11)]
    assert path.read_bytes() == before
