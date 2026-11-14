from pathlib import Path

import pytest

from rtp.harness import Harness
from rtp.datagen import sample_world, sample_spec, default_eyes
from rtp.brain import ScriptedBrain, BrainContext, build_prompt
from rtp.loop import LoopConfig, run_episode
from rtp.perception import Eye, EyeConfig
from rtp.rules import TaskSpec, check_final
from rtp.plan import Plan
from rtp.world import World, Obj
from rtp.stats import summarize_binary

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
SEEDS = range(40)


def leaky(s):
    return [Eye(EyeConfig(name="A", blind_regions=("table_left", "table_right", "table_center"), miss_occluded_p=0.5), s)]


def massy(s):
    return [Eye(EyeConfig(name="A", mass_sigma=0.45), s), Eye(EyeConfig(name="B", mass_sigma=0.45), s)]


def sweep(eyes_fn, **cfg):
    out = []
    for s in SEEDS:
        w = sample_world(s)
        out.append(run_episode(w, eyes_fn(s), ScriptedBrain(), sample_spec(w, s), H, LoopConfig(**cfg)))
    return out


def test_look_recovers_items_missing_from_the_scene():
    on, off = sweep(leaky, allow_look=True), sweep(leaky, allow_look=False)
    assert sum(r.looks for r in on) > 0 and sum(r.looks for r in off) == 0
    assert sum(r.n_valid for r in on) > sum(r.n_valid for r in off) + 20
    assert sum(r.success for r in on) > sum(r.success for r in off)
    assert all(r.looks <= 2 for r in on)                                      # bounded by max_looks


def test_look_cannot_recover_what_no_eye_ever_sees():
    blind = lambda s: [Eye(EyeConfig(name="A", blind_regions=("table_left", "table_right", "table_center"),
                                     miss_occluded_p=1.0), s)]
    r = sweep(blind, allow_look=True)[0]
    assert r.looks == 2 and not r.success and r.n_valid == 0                  # two looks, still nothing: not an infinite loop


def test_challenge_corrects_mass_and_reduces_heavy_bottom_violations():
    on, off = sweep(massy, allow_challenge=True), sweep(massy, allow_challenge=False)
    assert sum(r.challenges for r in on) > 0 and sum(r.challenges for r in off) == 0
    heavy = lambda rs: sum(v["rule"] == "heavy_bottom" for r in rs for v in r.violations)
    assert heavy(on) < heavy(off)
    assert sum(r.success for r in on) > sum(r.success for r in off)
    assert all(not any(f["cause"] == "hand_full" for f in r.failures) for r in on)    # replans respect held items


def test_clean_eyes_never_challenge_or_look():
    rs = sweep(lambda s: default_eyes(s, False))
    assert sum(r.challenges for r in rs) == 0 and sum(r.looks for r in rs) == 0


def test_delta_mode_is_cheaper_than_push_but_still_observes_every_round():
    tok = lambda rs: sum(r.msg_tokens for r in rs)
    push = sweep(lambda s: default_eyes(s, True), comm_mode="push")
    delta = sweep(lambda s: default_eyes(s, True), comm_mode="delta")
    init_only = sweep(lambda s: default_eyes(s, True), comm_mode="init_only")
    # delta must keep observing (so it costs more than never looking again) yet send far less than a full push
    assert tok(init_only) < tok(delta) < 0.75 * tok(push)
    assert sum(r.success for r in delta) >= sum(r.success for r in push) - 2


def test_object_memory_keeps_previously_seen_objects():
    """An object seen at the start stays in the Brain's scene even if a later observation omits it."""
    class Flaky(Eye):
        def report(self, world, t=0):
            m = super().report(world, t)
            if t > 0:
                m.content["objects"] = [o for o in m.content["objects"] if o["name"] != "book1"]
            return m
    w = sample_world(1)
    res = run_episode(w, [Flaky(EyeConfig(name="A"), 1)], ScriptedBrain(), sample_spec(w, 1), H, LoopConfig())
    assert "book1" in res.scene_final and res.success


# ---------- Brain: held items ----------
def ctx_for(rules, holding):
    spec = TaskSpec(items=["a", "b"], leave=[], heavy_bottom="heavy_bottom" in rules, laptop_slot=None,
                    bottle_upright_away=False)
    scene = {"a": {"name": "a", "cls": "book", "region": "held:L", "size": 1.0, "mass_guess": 0.3, "fragility_guess": 0.1,
                   "occluded": False, "confidence": 0.9},
             "b": {"name": "b", "cls": "book", "region": "table_left", "size": 1.0, "mass_guess": 1.5, "fragility_guess": 0.1,
                   "occluded": False, "confidence": 0.9}}
    return BrainContext(spec=spec, scene=scene, rule_ids=set(rules), holding=holding)


def test_brain_releases_a_light_held_item_when_a_heavier_one_must_go_first():
    plan = ScriptedBrain().plan(ctx_for({"heavy_bottom"}, {"L": "a", "R": None}))
    ids = [t.id for t in plan.tasks]
    assert ids[0] == "rel_a" and "g_a" in ids and plan.tasks[ids.index("g_a")].deps == ["rel_a"]
    plan2 = ScriptedBrain().plan(ctx_for(set(), {"L": "a", "R": None}))
    assert not any(t.skill == "release" for t in plan2.tasks)                 # no ordering rule: just place it first
    assert plan2.tasks[0].id == "p_a" and plan2.tasks[0].skill == "place"


def test_release_plan_executes_and_satisfies_heavy_bottom():
    objs = [Obj("a", "book", 0.3, 0.1, 1.0, "held:L"), Obj("b", "book", 1.5, 0.1, 1.0, "table_left")]
    w = World(objs, p_arm_noise=0.0)
    w.holding["L"] = "a"
    plan = ScriptedBrain().plan(ctx_for({"heavy_bottom"}, {"L": "a", "R": None}))
    from rtp.rules import run_plan_simple
    run_plan_simple(w, plan)
    spec = TaskSpec(items=["a", "b"], leave=[], laptop_slot=None, bottle_upright_away=False)
    assert check_final(w, spec)["success"] and w.bag["main"] == ["b", "a"]


def test_prompt_mentions_held_item_constraint():
    assert "holding an item cannot grasp another" in build_prompt(ctx_for(set(), {"L": None, "R": None}))
