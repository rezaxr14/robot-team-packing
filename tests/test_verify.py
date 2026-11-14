from pathlib import Path

import pytest

from rtp.harness import Harness
from rtp.datagen import sample_world, sample_spec, default_eyes
from rtp.brain import ScriptedBrain
from rtp.fixtures import FlakyBrain
from rtp.loop import LoopConfig, run_episode
from rtp.plan import PlanError

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
SEEDS = range(50)


def sweep(verify, p_err=0.6, p_fix=0.9, max_retries=2, kinds=None):
    out = []
    for s in SEEDS:
        w = sample_world(s)
        kw = {"kinds": kinds} if kinds else {}
        b = FlakyBrain(ScriptedBrain(), p_err=p_err, p_fix=p_fix, seed=s, **kw)
        out.append(run_episode(w, default_eyes(s), b, sample_spec(w, s), H,
                               LoopConfig(verify=verify, max_plan_retries=max_retries)))
    return out


def test_more_verification_means_fewer_bad_plans_reaching_the_arms():
    none, static, dry = sweep("none"), sweep("static"), sweep("dry_run")
    ok = lambda rs: sum(r.success for r in rs)
    assert ok(none) < ok(static) <= ok(dry) and ok(dry) >= 0.95 * len(SEEDS)
    assert sum(r.plan_retries for r in none) == 0 and sum(r.plan_retries for r in dry) > 0


def test_static_cannot_see_rule_violations_but_dry_run_can():
    only_rules = ("swap", "drop_item")                              # invisible to the static checker
    static, dry = sweep("static", kinds=only_rules, p_err=1.0), sweep("dry_run", kinds=only_rules, p_err=1.0)
    assert sum(r.plan_retries for r in static) == 0                 # nothing to reject statically
    assert sum(r.plan_retries for r in dry) > 0
    assert sum(r.success for r in dry) > sum(r.success for r in static)


def test_clean_plans_are_never_retried():
    rs = sweep("dry_run", p_err=0.0)
    assert sum(r.plan_retries for r in rs) == 0 and all(r.success for r in rs)


def test_retries_are_bounded():
    rs = sweep("dry_run", p_err=1.0, p_fix=0.0, max_retries=1)       # the Brain never repairs its plan
    for r in rs:
        assert r.plan_retries <= (1 + r.replans + r.looks) * 1


def test_unrepaired_plan_is_executed_and_flagged():
    rs = sweep("static", p_err=1.0, p_fix=0.0, kinds=("ghost",), max_retries=2)
    assert all(any(f["cause"] == "static_infeasible" for f in r.failures) for r in rs)
    assert all(r.plan_retries >= 2 for r in rs)


def test_feedback_text_reaches_the_brain():
    seen = []

    class Spy(FlakyBrain):
        def plan(self, ctx, reason=""):
            seen.append(list(ctx.notes))
            return super().plan(ctx, reason)

    w = sample_world(3)
    b = Spy(ScriptedBrain(), p_err=1.0, p_fix=1.0, seed=3, kinds=("ghost",))
    res = run_episode(w, default_eyes(3), b, sample_spec(w, 3), H, LoopConfig(verify="static"))
    assert res.success and res.plan_retries == 1
    assert not any("PLAN REJECTED" in n for n in seen[0]) and any("unknown object ghost_item" in n for n in seen[1])


def test_parse_failure_is_retried_only_when_verifying():
    class OnceBad:
        requests = []

        def __init__(self):
            self.inner, self.n = ScriptedBrain(), 0

        def plan(self, ctx, reason=""):
            self.n += 1
            if self.n == 1:
                raise PlanError("prose")
            return self.inner.plan(ctx, reason)

        replan = plan

    for verify, expect_ok in (("static", True), ("none", False)):
        w = sample_world(2)
        res = run_episode(w, default_eyes(2), OnceBad(), sample_spec(w, 2), H, LoopConfig(verify=verify))
        assert res.success == expect_ok, verify
        assert any(f["cause"] == "plan_parse" for f in res.failures)


def test_plan_pass_measures_the_first_attempt_not_the_repaired_one():
    rs = sweep("dry_run", p_err=1.0, p_fix=1.0, kinds=("ghost", "swap", "drop_item", "wrong_arm", "no_grasp"))
    assert sum(r.plan_pass for r in rs) < 0.2 * len(rs)              # one-shot plans were bad
    assert sum(r.success for r in rs) >= 0.9 * len(rs)               # after verification the team succeeds


def test_flaky_brain_is_deterministic_and_injects_every_kind():
    kinds = set()
    for s in range(60):
        w = sample_world(s)
        b = FlakyBrain(ScriptedBrain(), p_err=1.0, seed=s)
        try:
            run_episode(w, default_eyes(s), b, sample_spec(w, s), H, LoopConfig(verify="none"))
        except PlanError:
            pass
        kinds |= set(b.injected)
    assert kinds >= {"ghost", "no_grasp", "wrong_arm", "swap", "drop_item", "garbage"}
