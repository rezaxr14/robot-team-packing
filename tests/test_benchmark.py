"""Offline tests for the held-out benchmark: no GPU, no network, no served model."""
import json
from pathlib import Path

import pytest

from rtp import benchmark as bm
from rtp.brain import ScriptedBrain
from rtp.experiments import ExpConfig
from rtp.harness import Harness
from rtp.stats import paired_bootstrap_diff, wilson

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")

VALID_PLAN = {"tasks": [{"id": "g_book1", "arm": "L", "skill": "grasp", "obj": "book1",
                         "params": {}, "deps": []}], "notes": ""}


def row(model="competitor", seed=0, success=True, plan=VALID_PLAN, **extra):
    r = {"model": model, "seed": seed, "success": success, "plan": plan}
    r.update(extra)
    return r


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) if not isinstance(r, str) else r for r in rows) + "\n")
    return path


def test_seed_disjointness_and_fixed_namespace():
    test = bm.test_seeds(8)
    train = bm.test_seeds(8, "train")
    assert test == tuple(range(8)) == bm.test_seeds(8)
    assert train == tuple(range(bm.TRAIN_SEED_BASE, bm.TRAIN_SEED_BASE + 8))
    assert not (set(test) & set(train))
    assert all(s < bm.TRAIN_SEED_BASE for s in test)
    # the other training namespaces used by the dataset builder are all >= 100000 too
    for offset in (0, 300000, 500000, 700000, 900000):
        assert not (set(test) & set(range(bm.TRAIN_SEED_BASE + offset,
                                          bm.TRAIN_SEED_BASE + offset + 1000)))
    with pytest.raises(AssertionError):
        bm.assert_no_train_overlap([bm.TRAIN_SEED_BASE])
    with pytest.raises(ValueError):
        bm.test_seeds(3, "validation")
    with pytest.raises(ValueError):
        bm.test_seeds(-1)


def test_deterministic_test_set_and_generated_cases():
    big = bm.test_seeds(30)
    small = bm.test_seeds(8)
    assert small == big[:len(small)]                       # adding seeds never shifts earlier ones
    cfg = ExpConfig("full", noisy_eyes=True)
    a, b = bm.make_cases(small, cfg), bm.make_cases(small, cfg)
    assert [c[0] for c in a] == list(small)
    for (s, w1, sp1, eyes1, _), (_, w2, sp2, eyes2, _) in zip(a, b):
        assert w1.seed == w2.seed == s
        assert {n: (o.mass, o.fragility, o.region) for n, o in w1.objs.items()} == \
               {n: (o.mass, o.fragility, o.region) for n, o in w2.objs.items()}
        assert sp1.to_dict() == sp2.to_dict()
        assert [e.cfg for e in eyes1] == [e.cfg for e in eyes2]
    assert bm.make_cases(bm.test_seeds(3), cfg)[0][1].objs["book1"].mass == \
           bm.make_cases(bm.test_seeds(3), cfg)[0][1].objs["book1"].mass


def test_score_brain_records_and_wilson():
    seeds = bm.test_seeds(3)
    out = bm.score_brain(ScriptedBrain(), seeds, H)
    assert out["seeds"] == list(seeds)
    assert len(out["records"]) == len(seeds)
    for r in out["records"]:
        assert isinstance(r["success"], bool) and isinstance(r["plan_pass"], bool)
        assert isinstance(r["msg_tokens"], int) and r["seed"] in seeds
        assert r["kind"] == "fixture" and r["attribution"]["available"] is True
        assert set(r["attribution"]["all_agents"]) <= {"Eye", "Brain", "Arm"}
    k, n = out["success"]["k"], out["success"]["n"]
    assert n == len(seeds)
    assert (out["success"]["lo"], out["success"]["hi"]) == wilson(k, n) == out["wilson"]["success"]
    pk, pn = out["plan_pass"]["k"], out["plan_pass"]["n"]
    assert (out["plan_pass"]["lo"], out["plan_pass"]["hi"]) == wilson(pk, pn) == out["wilson"]["plan_pass"]


def test_load_predictions_valid_and_strict(tmp_path):
    good = write_jsonl(tmp_path / "good.jsonl", [row(seed=0), row(seed=1, success=False, reasoning="why")])
    loaded = bm.load_predictions(good)
    assert [r["seed"] for r in loaded] == [0, 1]
    assert loaded[1]["reasoning"] == "why" and loaded[0]["kind"] == "prediction"

    bad_cases = {
        "missing field": [{k: v for k, v in row().items() if k != "success"}],
        "success as int": [row(success=1)],
        "seed as string": [row(seed="0")],
        "plan as invalid json": [row(plan="{not json")],
        "plan without tasks": [row(plan={"notes": ""})],
        "plan with bad arm": [row(plan={"tasks": [{"id": "x", "arm": "C", "skill": "grasp"}]})],
        "reasoning as int": [row(reasoning=7)],
        "duplicate seed": [row(seed=0), row(seed=0)],
        "not an object": ["[1, 2, 3]"],
        "garbage line": ["{oops"],
    }
    for label, rows in bad_cases.items():
        p = write_jsonl(tmp_path / f"bad_{label.replace(' ', '_')}.jsonl", rows)
        with pytest.raises(ValueError, match=rf"bad_{label.replace(' ', '_')}\.jsonl"):
            bm.load_predictions(p)

    with pytest.raises(FileNotFoundError):
        bm.load_predictions(tmp_path / "absent.jsonl")

    (tmp_path / "dir").mkdir()
    write_jsonl(tmp_path / "dir" / "a.jsonl", [row(model="a", seed=0)])
    write_jsonl(tmp_path / "dir" / "b.jsonl", [row(model="b", seed=1)])
    assert {r["model"] for r in bm.load_predictions(tmp_path / "dir")} == {"a", "b"}


def test_paired_comparison_on_shared_seeds():
    ours = [{"model": "ours", "seed": s, "success": s % 2 == 0,
             "plan_pass": True, "msg_tokens": 10 + s} for s in range(6)]
    comp = [{"model": "competitor", "seed": s, "success": s < 3} for s in range(4)]
    out = bm.compare(ours, comp, roster=[("competitor", "desc")])
    assert out["direction"].startswith("ours - competitor")
    ours_row = next(m for m in out["models"] if m["model"] == "ours")
    c = next(m for m in out["models"] if m["model"] == "competitor")
    assert c["status"] == "scored" and c["n"] == 4
    expected = paired_bootstrap_diff([ours[s]["success"] for s in range(4)],
                                     [comp[s]["success"] for s in range(4)])
    assert c["paired_vs_ours"] == {"n": 4, "mean_diff": expected[0], "lo": expected[1], "hi": expected[2]}
    assert (c["success"]["lo"], c["success"]["hi"]) == wilson(3, 4)
    assert c["plan_pass"] is None and c["mean_msg_tokens"] is None   # competitor has neither
    assert ours_row["success"]["k"] == 3 and ours_row["success"]["n"] == 6


def test_disjoint_seed_competitor_is_scored_but_not_paired():
    ours = [{"model": "ours", "seed": s, "success": True} for s in range(3)]
    comp = [{"model": "competitor", "seed": s, "success": False} for s in range(20, 23)]
    c = next(m for m in bm.compare(ours, comp, roster=[("competitor", "d")])["models"]
             if m["model"] == "competitor")
    assert c["status"] == "scored" and c["n"] == 3 and c["paired_vs_ours"] is None


def test_not_run_rows_have_no_numbers():
    ours = [{"model": "ours", "seed": s, "success": True} for s in range(3)]
    out = bm.compare(ours, [], roster=[("A model", "not run here", "no adapter")])
    assert len(out["models"]) == 2
    nr = next(m for m in out["models"] if m["model"] == "A model")
    assert nr["status"] == "not run"
    for key in ("n", "success", "plan_pass", "mean_msg_tokens", "paired_vs_ours"):
        assert nr[key] is None
    assert nr["note"] == "no adapter"
    # JSON round trip preserves the absence of numbers for unrun models
    encoded = json.dumps(out)
    assert '"status": "not run"' in encoded
    assert all(nr[k] is None for k in ("n", "success", "plan_pass", "mean_msg_tokens"))
