from pathlib import Path

import pytest

from rtp.harness import Harness
from rtp.datagen import sample_world, sample_spec, default_eyes
from rtp.brain import ScriptedBrain
from rtp.loop import LoopConfig, run_episode
from rtp.simworld import SimWorld
from rtp.fakesim import FakeSim
from rtp.skills import run_regression
from rtp.skills import CATEGORIES

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")


def direct(seed, noisy, p=0.03):
    w = sample_world(seed, p_arm_noise=p)
    return w, run_episode(w, default_eyes(seed, noisy), ScriptedBrain(), sample_spec(w, seed), H, LoopConfig())


def via_sim(seed, noisy, p=0.03):
    w = SimWorld(FakeSim(lambda s: sample_world(s, p_arm_noise=p)), seed=seed)
    spec = sample_spec(sample_world(seed), seed)
    return w, run_episode(w, default_eyes(seed, noisy), ScriptedBrain(), spec, H, LoopConfig())


@pytest.mark.parametrize("noisy,p", [(False, 0.03), (True, 0.03), (True, 0.25)])
def test_sim_path_matches_direct_path(noisy, p):
    for s in range(12):
        _, a = direct(s, noisy, p)
        _, b = via_sim(s, noisy, p)
        assert (a.success, a.n_valid, a.rounds, a.replans, a.msg_tokens, a.missing, a.crushed) == \
               (b.success, b.n_valid, b.rounds, b.replans, b.msg_tokens, b.missing, b.crushed), (s, noisy, p)
        assert [(v["rule"]) for v in a.violations] == [(v["rule"]) for v in b.violations]
        assert a.score == pytest.approx(b.score)


def test_sim_world_mirrors_state_after_episode():
    w, res = via_sim(1, False)
    assert res.success and set(w.packed()) == {n for n, o in w.objs.items() if not o.is_distractor}
    assert all(o.region == o.init_region for o in w.objs.values() if o.is_distractor)
    assert w.holding == {"L": None, "R": None} and w.pack_counter == len(w.packed())
    assert w.backend.object_states()[0].peak_contact_force >= 0


def test_sim_world_not_found_and_pose_roundtrip():
    w = SimWorld(FakeSim(lambda s: sample_world(s)), seed=0)
    assert w.execute("L", "grasp", "ghost", {}).cause == "not_found"
    n = next(n for n, o in w.objs.items() if o.region == "table_center")
    assert w.objs[n].init_region == "table_center"


def test_crush_detected_through_peak_force_only():
    w = SimWorld(FakeSim(lambda s: sample_world(s, p_arm_noise=0.0)), seed=0)
    n = next(n for n, o in w.objs.items() if o.region == "table_left" and not o.is_distractor)
    r = w.execute("L", "grasp", n, {"force": 39.0})
    assert not r.ok and r.cause == "crush" and w.objs[n].crushed          # crushed flag came from the bridge, not the fake's world


class SimExecutor:
    """Regression-suite executor over SimWorld(FakeSim)."""
    def __init__(self):
        self.sw = None

    def reset(self, objects, seed, lock_to):
        from rtp.world import Obj, World
        self.sw = SimWorld(FakeSim(lambda s: World([Obj(**o) for o in objects], seed=s, p_arm_noise=0.0)), seed=seed)
        self.sw.backend.w.lock_holder = lock_to

    def run(self, arm, skill, obj, params):
        r = self.sw.execute(arm, skill, obj, params)
        return "success" if r.ok else (r.cause if r.cause in CATEGORIES else "other_failure")

    def state(self):
        return {"bag": {c: list(v) for c, v in self.sw.bag.items()}, "holding": dict(self.sw.holding)}


def test_regression_suite_passes_through_the_sim_path():
    rep = run_regression(SimExecutor())
    assert rep["passed"] == rep["total"], {k: v for k, v in rep["cases"].items() if not v["passed"]}


def test_state_diff_categories_pass_the_regression_suite():
    """Categories derived purely from pre/post states (what a physics backend must do) match every suite case."""
    from rtp.simworld import StateDiffBackend

    class Ex(SimExecutor):
        def reset(self, objects, seed, lock_to):
            from rtp.world import Obj, World
            self.sw = SimWorld(StateDiffBackend(FakeSim(lambda s: World([Obj(**o) for o in objects], seed=s, p_arm_noise=0.0))), seed=seed)
            self.sw.backend.inner.w.lock_holder = lock_to

    rep = run_regression(Ex())
    assert rep["passed"] == rep["total"], {k: v for k, v in rep["cases"].items() if not v["passed"]}


def test_state_diff_backend_runs_full_episodes_like_direct():
    from rtp.simworld import StateDiffBackend
    for s in range(10):
        w = SimWorld(StateDiffBackend(FakeSim(lambda sd: sample_world(sd, p_arm_noise=0.0))), seed=s)
        spec = sample_spec(sample_world(s), s)
        b = run_episode(w, default_eyes(s, False), ScriptedBrain(), spec, H, LoopConfig())
        _, a = direct(s, False, 0.0)
        assert (a.success, a.n_valid, a.rounds) == (b.success, b.n_valid, b.rounds), s


def test_resolve_cause_promotes_a_token_detail_but_not_prose():
    from rtp.simworld import resolve_cause
    # success is success, whatever the detail says
    assert resolve_cause("success", "gripped at 11.2 N") == "success"
    assert resolve_cause("", "") == ""
    # FakeSim reports a coarse category and the loop's own cause in `detail`
    assert resolve_cause("other_failure", "arm_noise") == "arm_noise"
    # prose from a physics backend never becomes a cause (the loop matches causes exactly)
    assert resolve_cause("other_failure", "Crush: 92.0 N on bottle1") == "other_failure"
    assert resolve_cause("other_failure", "unknown object") == "other_failure"
    # a specific category is never overwritten by a detail token
    assert resolve_cause("not_found", "arm_noise") == "not_found"
    assert resolve_cause("crush", "arm_noise") == "crush"
