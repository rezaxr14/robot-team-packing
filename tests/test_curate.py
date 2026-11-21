"""R5 curation + SFT formatting: CPU-only, synthetic fixture records only.

No data, no GPU, no files outside the repo: every record below is built in memory. These tests
check the exclusion contract (certified arm, seed namespaces, replay verification, dedupe,
per-model cap), the failure->replan ranking, and that SFT rows are deterministic and tamper
evident. Nothing here builds or claims a dataset.
"""
import copy
import json

import pytest

from rtp import seeds
from rtp.curate import REASONS, curate
from rtp.sft_format import (UnverifiedRecordError, build_messages, canonical_hash,
                            provenance_hash, to_sft_rows, verify_row)

CERT = "newton_cert_test123"
UNCERT = "sim_pose_v0"
MODEL_A = "frontier/model-a"
MODEL_B = "frontier/model-b"
OK = {"ok": True, "replayed": True, "reason": None, "diff": {}}


def _plan(tasks=None):
    """A schema-1 plan string; key order is deliberately shuffled to test canonical hashing."""
    tasks = tasks if tasks is not None else [{"skill": "grasp", "obj": "bottle1", "arm": "L"}]
    return json.dumps({"tasks": tasks}, separators=(",", ":"))


def _turns(plan):
    return [
        {"completion": '{"tool":"check_rule","args":{"rule_id":"heavy_bottom"}}',
         "tool_result": {"ok": True, "violations": []}},
        {"completion": plan},
    ]


def _rec(*, seed=100001, model=MODEL_A, arm_model=CERT, success=True, failures=None,
         replans=0, verification=None, plans=None, turns=None, mode="datagen", outcome=None,
         instruction="Pack Maya's bag", image_refs=None):
    """A minimal accepted schema-1 record with replay verification attached."""
    plan = (plans[0] if plans else None) or _plan()
    rec = {
        "schema": 1, "kind": "multiturn_episode", "mode": mode, "condition": "harness",
        "shard": "unit", "iteration": 1, "arm_model": arm_model, "model": model,
        "seed": seed, "instruction": instruction,
        "plans": plans if plans is not None else [plan],
        "result": {"success": success, "plan_pass": True, "replans": replans,
                   "n_valid": 1, "n_items": 1, "violations": [], "missing": [],
                   "failures": failures if failures is not None else []},
        "verification": copy.deepcopy(OK) if verification is None else verification,
        "turns": turns if turns is not None else _turns(plan),
    }
    if image_refs is not None:
        rec["image_refs"] = image_refs
    if outcome is not None:
        rec["outcome"] = outcome
    return rec


def _cause(cause="slip", agent="Arm"):
    return {"agent": agent, "cause": cause, "task": "t1", "obj": "bottle1"}


def _curate(records, **kwargs):
    kw = {"arm_models": {CERT}, "test_seeds": (), "max_per_model": None, "dedupe": True}
    kw.update(kwargs)
    return curate(records, **kw)


def _audit_ok(manifest):
    """No silent dropping: counts add up and every drop has a closed-vocabulary reason."""
    assert manifest["n_input"] == manifest["n_kept"] + manifest["n_excluded"]
    assert len(manifest["excluded"]) == manifest["n_excluded"]
    assert all(e["reason"] in REASONS for e in manifest["excluded"])
    assert manifest["excluded_by_reason"] == manifest["reasons"]
    assert sum(manifest["reasons"].values()) == manifest["n_excluded"]


# --- exclusions -------------------------------------------------------------------------

def test_certified_arm_model_required():
    good, bad = _rec(seed=100001), _rec(seed=100002, arm_model=UNCERT)
    m = _curate([good, bad], arm_models={CERT})
    assert m["n_kept"] == 1 and m["n_kept"] + m["n_excluded"] == 2
    assert [r["seed"] for r in m["records"]] == [100001]
    assert m["excluded"][0]["reason"] == "uncertified_arm_model"
    assert m["excluded"][0]["arm_model"] == UNCERT
    _audit_ok(m)

    # Empty certified set fails closed: nothing is admitted.
    assert _curate([good], arm_models=set())["n_kept"] == 0


def test_test_namespace_and_explicit_heldout_seeds_excluded():
    recs = [
        _rec(seed=7),                         # below 100000
        _rec(seed=seeds.TRAIN_MIN - 1),       # exactly the boundary - 1
        _rec(seed=100500),                    # explicitly held out
        _rec(seed=100501),
    ]
    m = _curate(recs, test_seeds=(100500,))
    assert [r["seed"] for r in m["records"]] == [100501]
    assert m["reasons"]["test_seed"] == 3
    assert all(e["reason"] != "test_seed" or not seeds.is_train(e["seed"])
               or e["seed"] in (100500,) for e in m["excluded"])
    assert all(seeds.is_train(r["seed"]) and r["seed"] not in (100500,)
               for r in m["records"])
    _audit_ok(m)


def test_replay_failed_and_unverified_excluded():
    recs = [
        _rec(seed=100001),
        _rec(seed=100002, verification={"ok": False, "reason": "replay_mismatch",
                                        "diff": {"success": {"recorded": True, "replayed": False}}}),
        _rec(seed=100003, verification={}),
        _rec(seed=100004, verification={"ok": False, "reason": "replay_exhausted", "diff": {}},
             success=False, failures=[_cause()]),
    ]
    m = _curate(recs)
    assert [r["seed"] for r in m["records"]] == [100001]
    reasons = m["reasons"]
    assert reasons["replay_failed"] == 2 and reasons["unverified"] == 1
    mm = next(e for e in m["excluded"] if e["detail"].get("reason") == "replay_mismatch")
    assert "success" in mm["detail"]["diff_keys"]
    _audit_ok(m)


def test_error_eval_and_unattributed_failures_have_reasons():
    recs = [
        _rec(seed=100001, outcome="error"),
        _rec(seed=100002, mode="eval"),
        _rec(seed=100003, success=False, failures=[]),
        _rec(seed=100004, success=False, failures=[{"agent": "Arm", "cause": "unknown"}]),
        _rec(seed=100005, success=False, failures=[_cause()]),
    ]
    m = _curate(recs)
    assert [r["seed"] for r in m["records"]] == [100005]
    assert m["reasons"]["error_record"] == 1
    assert m["reasons"]["wrong_mode"] == 1
    assert m["reasons"]["unattributed_failure"] == 2
    _audit_ok(m)


# --- ranking, dedupe and caps ------------------------------------------------------------

def test_failure_to_replan_pairs_kept_first_labelled_most_valuable():
    recs = [
        _rec(seed=100001, plans=[_plan([{"arm": "L", "skill": "grasp", "obj": "a"}])]),
        _rec(seed=100002, success=False, failures=[_cause()],
             plans=[_plan([{"arm": "L", "skill": "grasp", "obj": "b"}])]),      # negative
        _rec(seed=100003, success=True, replans=2, failures=[_cause()],
             plans=[_plan([{"arm": "L", "skill": "grasp", "obj": "c"}])]),     # failure -> success
    ]
    m = _curate(recs)
    assert [r["seed"] for r in m["records"]] == [100003, 100001, 100002]
    assert [r["curation"]["category"] for r in m["records"]] == ["replan", "success", "failure"]
    assert m["kept_by_category"] == {"replan": 1, "success": 1, "failure": 1}
    assert m["records"][0]["curation"]["verified"] is True
    assert m["records"][0]["curation"]["arm_certified"] is True
    _audit_ok(m)


def test_duplicate_plan_patterns_dropped_canonically():
    # Same tasks, different JSON key order / seeds -> one pattern.
    recs = [
        _rec(seed=100001, plans=[_plan()]),
        _rec(seed=100002, plans=[_plan([{"arm": "L", "skill": "grasp", "obj": "bottle1"}])]),
        _rec(seed=100003, plans=[_plan([{"arm": "R", "skill": "grasp", "obj": "bottle1"}])]),
    ]
    m = _curate(recs)
    assert [r["seed"] for r in m["records"]] == [100001, 100003]
    assert m["reasons"]["duplicate_plan"] == 1
    assert m["excluded"][0]["detail"]["plan_pattern_hash"] == \
        m["records"][0]["curation"]["plan_pattern_hash"]
    _audit_ok(m)

    kept_all = _curate(recs, dedupe=False)
    assert kept_all["n_kept"] == 3 and not kept_all["reasons"]
    assert kept_all["sha256"] != m["sha256"]


def test_per_model_cap_is_deterministic():
    recs = [_rec(seed=100001 + i, model=MODEL_A,
                 plans=[_plan([{"arm": "L", "skill": "grasp", "obj": f"obj{i}"}])])
            for i in range(4)]
    recs += [_rec(seed=100101 + i, model=MODEL_B,
                  plans=[_plan([{"arm": "R", "skill": "grasp", "obj": f"obj{i}"}])])
             for i in range(2)]
    a = _curate(recs, max_per_model=2)
    b = _curate(list(reversed(recs)), max_per_model=2)
    assert a["kept_by_model"][MODEL_A]["total"] == 2
    assert a["kept_by_model"][MODEL_B]["total"] == 2
    assert a["reasons"]["per_model_cap"] == 2
    # Reversed input still honours the cap (equal-value ties are resolved by input order, so
    # determinism means the *same* input reproduces the same kept set, not every permutation).
    assert b["kept_by_model"][MODEL_A]["total"] == 2
    assert b["kept_by_model"][MODEL_B]["total"] == 2
    assert b["reasons"]["per_model_cap"] == 2
    assert _curate(recs, max_per_model=2)["sha256"] == a["sha256"]
    assert _curate(recs, max_per_model=0)["n_kept"] == 0
    with pytest.raises(ValueError):
        _curate(recs, max_per_model=-1)
    _audit_ok(a)


def test_cap_keeps_higher_value_records_of_the_same_model():
    recs = [_rec(seed=100001, plans=[_plan([{"arm": "L", "skill": "grasp", "obj": "a"}])]),
            _rec(seed=100002, replans=1, failures=[_cause()],
                 plans=[_plan([{"arm": "L", "skill": "grasp", "obj": "b"}])])]
    m = _curate(recs, max_per_model=1)
    assert [r["seed"] for r in m["records"]] == [100002]
    assert m["reasons"]["per_model_cap"] == 1


# --- SFT format --------------------------------------------------------------------------

def test_curated_records_format_to_deterministic_rows():
    recs = [_rec(seed=100001, image_refs=["scene_100001/views/s03_z1.jpg"],
                 plans=[_plan([{"arm": "L", "skill": "grasp", "obj": "a"}])]),
            _rec(seed=100002, success=False, failures=[_cause()],
                 plans=[_plan([{"arm": "L", "skill": "grasp", "obj": "b"}])])]
    m = _curate(recs)
    rows = to_sft_rows(m["records"])
    assert to_sft_rows(copy.deepcopy(m["records"])) == rows
    assert json.dumps(rows, sort_keys=True) == json.dumps(to_sft_rows(m["records"]), sort_keys=True)

    roles = [msg["role"] for msg in rows[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert "Images: scene_100001/views/s03_z1.jpg" in rows[0]["messages"][1]["content"]
    assert all(row["meta"]["schema"] == 1 and len(row["meta"]["provenance_hash"]) == 64
               for row in rows)
    assert [row["meta"]["seed"] for row in rows] == [100001, 100002]
    assert all(row["meta"]["arm_model"] == CERT for row in rows)
    assert all("thinking" not in msg for row in rows for msg in row["messages"])
    # The hash covers the conversation and identity, not the curation proof.
    assert all("curation" not in row for row in rows)


def test_provenance_hash_and_tamper_detection():
    msg = build_messages(_rec(seed=100001, image_refs=["x.jpg"]))
    h1 = provenance_hash(msg, 100001, CERT)
    assert len(h1) == 64
    assert provenance_hash(copy.deepcopy(msg), 100001, CERT) == h1
    tampered = copy.deepcopy(msg)
    tampered[1]["content"] += " EVIL"
    assert provenance_hash(tampered, 100001, CERT) != h1
    assert provenance_hash(msg, 100002, CERT) != h1
    assert provenance_hash(msg, 100001, UNCERT) != h1

    row = to_sft_rows(_curate([_rec(seed=100001)])["records"])[0]
    assert verify_row(row) == []
    bad = copy.deepcopy(row)
    bad["messages"][2]["content"] = '{"tool":"submit_plan","args":{}}'
    assert verify_row(bad) == ["meta.provenance_hash"]
    bad2 = copy.deepcopy(row)
    bad2["meta"]["seed"] = 100999
    assert verify_row(bad2) == ["meta.provenance_hash"]
    bad3 = copy.deepcopy(row)
    bad3["meta"]["schema"] = 99
    assert "meta.schema" in verify_row(bad3)

    # Canonical hashing ignores key order but not values.
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_formatter_refuses_unverified_or_uncertified_records():
    raw = _rec(seed=100001)
    with pytest.raises(UnverifiedRecordError, match="not curated"):
        to_sft_rows([raw])

    curated = _curate([raw])["records"][0]
    with pytest.raises(UnverifiedRecordError, match="certified arm model"):
        to_sft_rows([{**curated, "curation": {**curated["curation"], "arm_certified": False}}])
    with pytest.raises(UnverifiedRecordError, match="verification"):
        to_sft_rows([{**curated, "verification": {"ok": False, "reason": "replay_mismatch"}}])
    with pytest.raises(UnverifiedRecordError, match="arm_model"):
        to_sft_rows([{**curated, "arm_model": UNCERT}])
    with pytest.raises(UnverifiedRecordError, match="training namespace"):
        to_sft_rows([{**curated, "seed": 42}])
    with pytest.raises(UnverifiedRecordError, match="not a datagen"):
        to_sft_rows([{**curated, "mode": "eval"}])
