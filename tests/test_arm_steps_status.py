"""The arms-first gate must not call stand-ins or failing/partial runs DONE (numpy-free, no sim)."""
import json
from pathlib import Path

from ops.arm_steps_status import DONE, FAIL, NOT_RUN, STAND_IN, STEPS, blocked, status

CERT = "thresholds:\n  nominal_success_min: 0.90\n  push_perturbed_success_min: 0.80\n  min_trials_a7: 200\n  min_trials_a10: 500\n"


def w(root: Path, name: str, obj):
    p = root / "reports" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def base(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "arm_cert.yaml").write_text(CERT)
    (tmp_path / "docs" / "findings").mkdir(parents=True)
    (tmp_path / "docs" / "findings" / "newton_arm_api.md").write_text("VERIFIED " * 5)
    w(tmp_path, "g5_reach.json", {"result": "PASS", "max_pos_err_mm": 0.0001})
    for n in ("a3_arm_smoke", "g7_wrench", "a6_scene"):
        w(tmp_path, n + ".json", {"result": "PASS", "evidence_class": "SIM"})
    return tmp_path


def test_empty_repo_is_all_not_run(tmp_path):
    st = status(tmp_path)
    assert all(st[k][0] == NOT_RUN for k in STEPS if k != "A2")


def test_failing_push_rows_and_not_run_rows_block_a5(tmp_path):
    root = base(tmp_path)
    w(root, "g6_newton_push.json", {"result": "FAIL", "evidence_class": "SIM",
                                    "checks": {"steady_deflection": True, "elbow_push": False, "payload": False},
                                    "rows": {"payload": {"status": "not run"}, "elbow_push": {"pass": False}}})
    st = status(root)
    assert st["A1"][0] == st["A3"][0] == st["A4"][0] == DONE
    assert st["A5"][0] == FAIL and "elbow_push" in st["A5"][1] and "payload" in st["A5"][1]
    assert [k for k, _ in blocked(st, "A6")] == ["A5"]          # A6 itself is DONE; A5 is what blocks it


def test_standin_and_short_runs_are_never_done(tmp_path):
    root = base(tmp_path)
    w(root, "skill_regression_scripted.json", {"evidence_class": "SIM-POSE", "kind": "sim_pose_standin", "trials": 10, "passed": 18, "total": 18})
    assert status(root)["A7"][0] == STAND_IN
    w(root, "skill_regression_scripted.json", {"evidence_class": "SIM", "kind": "physical", "trials": 10, "passed": 18, "total": 18})
    assert status(root)["A7"][0] == FAIL and "pre-registered" in status(root)["A7"][1]
    w(root, "skill_regression_scripted.json", {"evidence_class": "SIM", "kind": "physical", "trials": 200, "passed": 150, "total": 200})
    assert status(root)["A7"][0] == FAIL
    w(root, "skill_regression_scripted.json", {"evidence_class": "SIM", "kind": "physical", "trials": 200, "passed": 190, "total": 200})
    assert status(root)["A7"][0] == DONE


def test_a1_requires_the_millimetre_bound_and_a10_requires_all_cards(tmp_path):
    root = base(tmp_path)
    w(root, "g5_reach.json", {"result": "PASS", "max_pos_err_mm": 3.0})
    assert status(root)["A1"][0] == FAIL
    for k in ("grasp", "place"):
        w(root, f"arm_cert_{k}.json", {"result": "PASS", "evidence_class": "SIM"})
    assert status(root)["A10"][0] == FAIL and "missing" in status(root)["A10"][1]


def test_require_exit_codes(tmp_path):
    from ops.arm_steps_status import main
    import sys
    root = base(tmp_path)
    w(root, "g6_newton_push.json", {"result": "PASS", "evidence_class": "SIM", "checks": {}})
    old = sys.argv
    try:
        sys.argv = ["x", "--root", str(root), "--require", "A6"]
        assert main() == 0
        sys.argv = ["x", "--root", str(root), "--require", "A7"]
        assert main() == 1                                            # A7 has no report: cannot proceed to A8
    finally:
        sys.argv = old
