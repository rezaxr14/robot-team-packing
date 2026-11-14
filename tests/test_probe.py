from pathlib import Path

import pytest

from rtp.harness import Harness
from rtp.datagen import sample_world, sample_spec, default_eyes
from rtp.brain import ScriptedBrain
from rtp.loop import LoopConfig, run_episode
from rtp.perception import Eye, EyeConfig
from rtp.plan import parse_plan
from rtp.world import World, Obj

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
SEEDS = range(40)


def massy(s):
    return [Eye(EyeConfig(name="A", mass_sigma=0.45), s), Eye(EyeConfig(name="B", mass_sigma=0.45), s)]


def sweep(**cfg):
    out = []
    for s in SEEDS:
        w = sample_world(s)
        out.append(run_episode(w, massy(s), ScriptedBrain(), sample_spec(w, s), H, LoopConfig(**cfg)))
    return out


def test_weigh_skill_semantics():
    w = World([Obj("a", "book", 1.0, 0.1, 2.0, "table_left"), Obj("b", "book", 0.5, 0.1, 2.0, "table_right")], p_arm_noise=0.0)
    r = w.execute("L", "weigh", "a")
    assert r.ok and abs(r.touch["measured_weight"] - 1.0) <= 0.021 and w.holding["L"] is None and w.objs["a"].region == "table_left"
    assert w.execute("L", "weigh", "b").cause == "out_of_reach"
    w.holding["L"] = "a"
    assert w.execute("L", "weigh", "a").cause == "hand_full"
    assert w.execute("R", "weigh", "ghost").cause == "not_found"
    w2 = World([Obj("a", "book", 1.0, 0.1, 2.0, "table_left")], seed=3, p_arm_noise=0.0)
    assert w2.execute("L", "weigh", "a").touch == World([Obj("a", "book", 1.0, 0.1, 2.0, "table_left")], seed=3, p_arm_noise=0.0).execute("L", "weigh", "a").touch


def test_weigh_is_a_valid_plan_skill():
    p = parse_plan('{"tasks":[{"id":"w","arm":"L","skill":"weigh","obj":"book1"}]}')
    assert p.tasks[0].skill == "weigh"


def test_probe_removes_the_mass_error_failure_mode():
    none, probe = sweep(probe=False, allow_challenge=False), sweep(probe=True, allow_challenge=False)
    heavy = lambda rs: sum(v["rule"] == "heavy_bottom" for r in rs for v in r.violations)
    assert heavy(none) > 20 and heavy(probe) == 0
    assert sum(r.success for r in probe) > sum(r.success for r in none) + 15
    assert all(r.probes == 8 for r in probe) and sum(r.probes for r in none) == 0
    assert sum(r.rounds for r in probe) > sum(r.rounds for r in none)               # probing costs robot time


def test_probe_beats_challenge_only_under_large_mass_errors():
    chal, both = sweep(allow_challenge=True), sweep(allow_challenge=True, probe=True)
    assert sum(r.success for r in both) > sum(r.success for r in chal)
    assert all(r.challenges == 0 for r in both)                                     # nothing left to challenge after probing


def test_probe_is_skipped_without_the_ordering_rule_and_is_harmless_with_clean_eyes():
    w = sample_world(1)
    spec = sample_spec(w, 1)
    spec.heavy_bottom = False
    res = run_episode(w, default_eyes(1), ScriptedBrain(), spec, H, LoopConfig(probe=True))
    assert res.probes == 0 and res.success                                          # rule inactive: no probing cost
    w2 = sample_world(1)
    res2 = run_episode(w2, default_eyes(1), ScriptedBrain(), sample_spec(w2, 1), H, LoopConfig(probe=True))
    assert res2.success and res2.probes == 8


def test_probe_tolerates_arm_noise_and_records_failures_without_fatality():
    w = sample_world(2, p_arm_noise=0.3)
    res = run_episode(w, massy(2), ScriptedBrain(), sample_spec(w, 2), H, LoopConfig(probe=True))
    assert res.probes >= 6 and not any(f.get("fatal") and f["cause"] == "arm_noise" and f["task"].startswith("w_") for f in res.failures)


def test_experiment_grid_has_a_probe_config():
    from rtp.experiments import default_grid, run_config
    names = [c.name for c in default_grid()]
    assert "probe" in names
    rec = run_config([c for c in default_grid() if c.name == "probe"][0], [0], H)[0]
    assert rec["probes"] > 0
