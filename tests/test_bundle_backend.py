"""R2 tests: the skill-table sampler is deterministic and never invents un-certified outcomes."""
import json

import pytest

from rtp.bundle_backend import BundleBackend, BundleMissing
from rtp.simbridge import Layout
from rtp.simworld import SimWorld


def make_tables(tmp_path):
    root = tmp_path / "tables"
    (root / "skill_tables").mkdir(parents=True)
    (root / "skill_tables" / "manifest.json").write_text(json.dumps(
        {"arm_models": {"newton_cert_test": {"evidence_class": "SIM", "n_trials": 200}}}))
    (root / "skill_tables" / "grasp.json").write_text(json.dumps({
        "skill": "grasp", "evidence_class": "SIM", "arm_model": "newton_cert_test",
        "entries": [{"key": {}, "counts": {"success": 190, "slip": 8, "crush": 2}}]}))
    return root


def test_missing_table_raises_instead_of_inventing(tmp_path):
    b = BundleBackend(tmp_path)
    b.reset(1)
    with pytest.raises(BundleMissing):
        b.run_skill("L", "grasp", "obj_a", {})


def test_unknown_arm_model_raises(tmp_path):
    root = make_tables(tmp_path)
    table_path = root / "skill_tables" / "grasp.json"
    table = json.loads(table_path.read_text())
    table["arm_model"] = "uncertified"
    table_path.write_text(json.dumps(table))
    b = BundleBackend(root)
    b.reset(1)
    with pytest.raises(BundleMissing):
        b.run_skill("L", "grasp", "obj_a", {})


def test_sampling_is_seed_deterministic(tmp_path):
    root = make_tables(tmp_path)
    b = BundleBackend(root, arm_model="newton_cert_test")
    b.reset(7)
    first = [b.run_skill("L", "grasp", "obj_a", {"mass": 0.5}).category for _ in range(50)]
    b.reset(7)
    second = [b.run_skill("L", "grasp", "obj_a", {"mass": 0.5}).category for _ in range(50)]
    assert first == second
    assert set(first) <= {"success", "slip", "crush"}
    assert first.count("success") > 35                 # 95 % nominal rate, sampled 50 times


def test_backend_satisfies_simworld_contract(tmp_path):
    root = make_tables(tmp_path)
    b = BundleBackend(root, arm_model="newton_cert_test", layout=Layout())
    w = SimWorld(b, layout=b.layout, seed=7)
    assert len(w.objs) == 3
    rep = b.run_skill("L", "grasp", "obj_a", {})
    assert rep.category in ("success", "slip", "crush")
    assert rep.detail.startswith("SIM-ABSTRACT")
