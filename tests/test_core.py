import json
import pytest
from pathlib import Path

from rtp.messages import Message, MessageBus, MessageError
from rtp.world import World, Obj, required_force, crush_limit
from rtp.plan import Plan, Task, PlanError, parse_plan, next_round
from rtp.rules import TaskSpec, check_final, static_check, dry_run, amend
from rtp.stats import wilson, paired_bootstrap_diff
from rtp.flowmath import interpolate, target_velocity, euler_integrate

HARNESS = Path(__file__).resolve().parents[1] / "harness_files"


def mk(**kw):
    base = dict(name="o", cls="book", mass=1.0, fragility=0.1, size=2.0, region="table_left")
    base.update(kw)
    return Obj(**base)


# ---------- messages ----------
def test_message_roundtrip_and_validation():
    m = Message("SCENE", "eye_A", "all", {"objects": [], "view": "A"})
    assert Message.from_json(m.to_json()) == m
    with pytest.raises(MessageError):
        Message("TOUCH", "arm_L", "brain", {"obj": "x"}).validate()
    with pytest.raises(MessageError):
        Message("NOPE", "a", "b").validate()


def test_bus_budget_drops_excess():
    bus = MessageBus(token_budget=20)
    m = Message("AMEND", "human", "brain", {"text": "x" * 200})
    assert bus.send(m) is False and bus.dropped == 1 and bus.tokens_used == 0


# ---------- world ----------
def test_grasp_force_window():
    o = mk(mass=1.0, fragility=0.5)                       # need 10N, crush 19N
    w = World([o], p_arm_noise=0)
    assert w.execute("L", "grasp", "o", {"force": o.mass and required_force(o) - 1}).cause == "slip"
    w = World([o], p_arm_noise=0)
    assert w.execute("L", "grasp", "o", {"force": crush_limit(o) + 1}).cause == "crush"
    assert w.objs["o"].crushed
    w = World([o], p_arm_noise=0)
    assert w.execute("L", "grasp", "o", {"force": 12}).ok and w.holding["L"] == "o"


def test_reach_zones_and_lock_and_slot():
    w = World([mk(region="table_right")], p_arm_noise=0)
    assert w.execute("L", "grasp", "o", {"force": 12}).cause == "out_of_reach"
    w = World([mk(cls="bottle", region="table_left")], p_arm_noise=0)
    w.execute("L", "grasp", "o", {"force": 12})
    assert w.execute("L", "place", "o", {"compartment": "laptop"}).cause == "wrong_slot"
    w.lock_holder = "R"
    assert w.execute("L", "place", "o", {"compartment": "main"}).cause == "locked"


def test_common_random_numbers_independent_of_other_draws():
    a = World([mk(), mk(name="p")], seed=7)
    b = World([mk(), mk(name="p")], seed=7)
    b.u("unrelated|key")                                  # extra draw must not shift keyed draws
    assert a.u("L|grasp|o") == b.u("L|grasp|o")
    assert a.u("L|grasp|o") == b.u("L|grasp|o")           # per-key counter advances identically


# ---------- plan ----------
def test_parse_plan_fenced_and_bare_and_errors():
    d = {"tasks": [{"id": "a", "arm": "L", "skill": "grasp", "obj": "x", "params": {"force": 5}},
                   {"id": "b", "arm": "L", "skill": "place", "obj": "x", "deps": ["a"]}]}
    assert len(parse_plan("sure!\n```json\n" + json.dumps(d) + "\n```").tasks) == 2
    assert len(parse_plan("plan: " + json.dumps(d) + " done").tasks) == 2
    for bad in ("no json", '{"tasks": [{"id":"a","arm":"X","skill":"grasp"}]}',
                '{"tasks": [{"id":"a","arm":"L","skill":"fly"}]}',
                '{"tasks": [{"id":"a","arm":"L","skill":"grasp","deps":["z"]}]}'):
        with pytest.raises(PlanError):
            parse_plan(bad)
    cyc = {"tasks": [{"id": "a", "arm": "L", "skill": "grasp", "deps": ["b"]},
                     {"id": "b", "arm": "R", "skill": "grasp", "deps": ["a"]}]}
    with pytest.raises(PlanError):
        Plan.from_dict(cyc)


def test_scheduler_bag_lock_and_soft_deps():
    t = [Task("g1", "L", "grasp", "x"), Task("g2", "R", "grasp", "y"),
         Task("p1", "L", "place", "x", deps=["g1"]), Task("p2", "R", "place", "y", deps=["g2"])]
    st = {x.id: "pending" for x in t}
    chosen, _ = next_round(t, st)
    assert {c.id for c in chosen} == {"g1", "g2"}
    st["g1"] = st["g2"] = "done"
    chosen, _ = next_round(t, st)
    assert [c.id for c in chosen] == ["p1"]               # only one bag access per round
    t2 = [Task("g1", "L", "grasp", "x"), Task("p1", "L", "place", "x", deps=["g1"]),
          Task("g2", "R", "grasp", "y"), Task("p2", "R", "place", "y", deps=["g2", "p1"])]
    st = {x.id: "pending" for x in t2}
    st.update(g1="dead", p1="dead", g2="done")
    chosen, dead = next_round(t2, st)
    assert "p2" in {c.id for c in chosen}                 # ordering-only dep on a dead place is ignored
    st = {x.id: "pending" for x in t2}
    st.update(g2="dead")
    _, dead = next_round(t2, st)
    assert "p2" in dead                                   # a dead hard dep kills the dependent


# ---------- rules ----------
def packed_world(order, comps, **over):
    objs = {"book": mk(name="book", mass=1.2, size=2.0), "pen": mk(name="pen", cls="pencilcase", mass=0.2, size=1.0),
            "lap": mk(name="lap", cls="laptop", mass=1.8, size=3.0), "bot": mk(name="bot", cls="bottle", mass=0.8, size=1.5),
            "lunch": mk(name="lunch", cls="lunchbox", mass=0.5, size=2.5)}
    w = World(list(objs.values()), p_arm_noise=0, deterministic=True)
    for n, c in zip(order, comps):
        w.objs[n].region = "table_center"
        w.holding["L"] = n
        assert w.execute("L", "place", n, {"compartment": c, **over}).ok, (n, c)
    return w


def test_check_final_detects_each_rule():
    spec = TaskSpec(items=["book", "pen", "lap", "bot", "lunch"], leave=[], last_item="lunch")
    good = packed_world(["book", "bot", "pen", "lap", "lunch"], ["main", "main", "main", "laptop", "main"])
    assert check_final(good, spec)["success"]
    bad_order = packed_world(["pen", "book", "bot", "lap", "lunch"], ["main", "main", "main", "laptop", "main"])
    assert {v["rule"] for v in check_final(bad_order, spec)["violations"]} == {"heavy_bottom"}
    bad_slot = packed_world(["book", "bot", "pen", "lap", "lunch"], ["main", "main", "main", "front", "main"])
    assert "laptop_slot" in {v["rule"] for v in check_final(bad_slot, spec)["violations"]}
    not_last = packed_world(["book", "lunch", "bot", "pen", "lap"], ["main", "main", "main", "main", "laptop"])
    assert "lunchbox_last" in {v["rule"] for v in check_final(not_last, spec)["violations"]}
    flat = packed_world(["book", "bot", "pen", "lap", "lunch"], ["main", "main", "main", "laptop", "main"], orientation="flat")
    assert "bottle_upright" in {v["rule"] for v in check_final(flat, spec)["violations"]}


def test_static_check_and_dry_run():
    scene = {"x": {"cls": "book", "region": "table_right", "mass_guess": 1.0, "fragility_guess": 0.1, "size": 2.0},
             "m": {"cls": "mug", "region": "table_left", "mass_guess": 0.3, "fragility_guess": 0.5, "size": 1.0}}
    spec = TaskSpec(items=["x"], leave=["m"], heavy_bottom=False, laptop_slot=None, bottle_upright_away=False)
    ok = Plan([Task("g", "R", "grasp", "x", {"force": 10}), Task("p", "R", "place", "x", {"compartment": "main"}, ["g"])])
    assert static_check(ok, scene, spec) == (True, [])
    assert dry_run(ok, scene, spec)["success"]
    wrong_arm = Plan([Task("g", "L", "grasp", "x"), Task("p", "L", "place", "x", {}, ["g"])])
    assert not static_check(wrong_arm, scene, spec)[0]
    touches = Plan(ok.tasks + [Task("t", "L", "push_aside", "m")])
    assert any("leave" in e for e in static_check(touches, scene, spec)[1])
    nograsp = Plan([Task("p", "R", "place", "x", {"compartment": "main"})])
    assert any("without prior grasp" in e for e in static_check(nograsp, scene, spec)[1])


def test_amend_changes_text():
    s = TaskSpec(items=["a"], laptop_slot="laptop")
    assert "front slot" in amend(s, laptop_slot="front").text()


# ---------- stats / flow math ----------
def test_wilson_known_value():
    lo, hi = wilson(8, 10)
    assert abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3
    assert wilson(0, 0) == (0.0, 1.0)


def test_paired_bootstrap_detects_shift():
    m, lo, hi = paired_bootstrap_diff([1, 1, 1, 0.9, 1], [0.5, 0.4, 0.6, 0.5, 0.5])
    assert lo > 0 and m > 0.4


def test_flow_matching_math_numpy():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    x0, x1 = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    assert np.allclose(interpolate(x0, x1, 0.0), x0) and np.allclose(interpolate(x0, x1, 1.0), x1)
    v = target_velocity(x0, x1)
    # along the straight path the true velocity field is constant, so Euler is exact for any step count
    out = euler_integrate(lambda x, t: (x1 - x) / (1.0 - t) if t < 1 else v, x0, steps=8)
    assert np.allclose(out, x1)
    assert np.allclose(euler_integrate(lambda x, t: v, x0, steps=3), x1)
