"""Tests for the frozen held-out test set."""
import json

import pytest

from rtp import seeds, testset


def test_build_is_deterministic_and_in_test_namespace():
    a = testset.build_testset(n_per_family=5)
    b = testset.build_testset(n_per_family=5)
    assert a == b
    assert len(a) == 5 * len(testset.FAMILIES)
    assert all(seeds.is_test(e["seed"]) for e in a)
    assert len({e["seed"] for e in a}) == len(a)


def test_freeze_and_verify_roundtrip(tmp_path):
    path = tmp_path / "heldout.json"
    episodes = testset.build_testset(n_per_family=2)
    manifest = testset.freeze(path, episodes)
    assert manifest["sha256"] == testset.digest(episodes)
    assert testset.verify(path, forbidden_train_seeds=[100000, 100123])["n"] == len(episodes)


def test_verify_rejects_tampering(tmp_path):
    path = tmp_path / "heldout.json"
    testset.freeze(path, testset.build_testset(n_per_family=2))
    data = json.loads(path.read_text())
    data["episodes"][0]["params"]["trial"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        testset.verify(path)


def test_verify_rejects_training_seed_in_test_set(tmp_path):
    path = tmp_path / "heldout.json"
    episodes = testset.build_testset(n_per_family=2)
    episodes[0]["seed"] = 100000
    testset.freeze(path, episodes)
    with pytest.raises(seeds.SeedNamespaceError):
        testset.verify(path)


def test_verify_rejects_train_seed_in_forbidden_list(tmp_path):
    path = tmp_path / "heldout.json"
    testset.freeze(path, testset.build_testset(n_per_family=2))
    with pytest.raises(seeds.SeedNamespaceError):
        testset.verify(path, forbidden_train_seeds=[3])
