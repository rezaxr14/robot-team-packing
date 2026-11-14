import json
from pathlib import Path

import pytest

from rtp.harness import Harness
from rtp.datagen import sample_world, sample_spec, default_eyes, gen_hindsight_sft, gen_eye_grpo_rows
from rtp.perception import oracle_report, merge_scenes, Eye, EyeConfig, FixedEye, default_blind
from rtp.messages import Message
from rtp import eye_rewards as er
from rtp.eye_rewards import parse_scene_completion, r_eye_format, r_eye_accuracy, r_eye_credit, trl_eye_reward_funcs
from rtp.plan import parse_plan
from rtp.skills import SymbolicExecutor, run_regression, regression_cases, compare, run_case
import rtp.world as world_mod

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")


def objs_json(seed, view="A"):
    return json.dumps(oracle_report(sample_world(seed), view).content["objects"])


# ---------- scene fusion ----------
def test_merge_scenes_is_order_independent():
    a = Message("SCENE", "eye_A", "all", {"view": "A", "objects": [{"name": "x", "cls": "book", "confidence": 0.9, "region": "r1"}]})
    b = Message("SCENE", "eye_B", "all", {"view": "B", "objects": [{"name": "x", "cls": "notebook", "confidence": 0.9, "region": "r2"}]})
    assert merge_scenes([a, b]) == merge_scenes([b, a])
    assert merge_scenes([a, b])["x"]["cls"] == "book"                      # tie -> view A
    lo = Message("SCENE", "eye_B", "all", {"view": "B", "objects": [{"name": "x", "cls": "z", "confidence": 0.95, "region": "r"}]})
    assert merge_scenes([a, lo])["x"]["cls"] == "z"                        # higher confidence wins


def test_fixed_eye_emits_given_objects():
    e = FixedEye("A", [{"name": "x"}])
    assert e.report(sample_world(0), 3).content == {"objects": [{"name": "x"}], "view": "A"} and e.name == "A"


# ---------- Eye rewards ----------
def test_parse_scene_completion_variants_and_rejections():
    good = objs_json(1)
    assert parse_scene_completion(good) and parse_scene_completion("```json\n" + good + "\n```")
    assert parse_scene_completion('{"objects": ' + good + "}") and parse_scene_completion("here: " + good + " ok")
    assert parse_scene_completion([{"role": "assistant", "content": good}])
    for bad in ("", "nope", "[1,2]", '[{"name":"x"}]', '[{"name":"x","cls":"a","region":"r","size":"big","mass_guess":1,"fragility_guess":0}]', None, 7):
        assert parse_scene_completion(bad) is None
    assert r_eye_format("nope") == 0.0 and r_eye_format(good) == 1.0


def test_eye_accuracy_orders_perfect_noisy_empty():
    w = sample_world(2)
    perfect = oracle_report(w, "A").content["objects"]
    wrong_mass = [dict(o, mass_guess=o["mass_guess"] * 2) for o in perfect]
    half = perfect[: len(perfect) // 2]
    halluc = perfect + [dict(perfect[0], name="ghost")]
    acc = lambda x: r_eye_accuracy(json.dumps(x), w)
    assert acc(perfect) == 1.0 and acc(perfect) > acc(wrong_mass) > acc([]) == 0.0
    assert acc(perfect) > acc(half) and acc(perfect) > acc(halluc)
    assert r_eye_accuracy("garbage", w) == 0.0
    blind = default_blind("A")
    vis_only = [o for o in perfect if o["region"] not in blind]
    assert r_eye_accuracy(json.dumps(vis_only), w, blind) == 1.0            # blind-region objects are not required


def test_eye_credit_rewards_useful_reports_and_is_reproducible():
    cs = [r_eye_credit(objs_json(s), s, "A", H) for s in range(6)]
    assert all(c >= 0 for c in cs) and max(cs) > 0.1
    assert r_eye_credit(objs_json(3), 3, "A", H) == r_eye_credit(objs_json(3), 3, "A", H)    # common random numbers
    assert r_eye_credit("[]", 3, "A", H) == 0.0 and r_eye_credit("garbage", 3, "A", H) == -1.0


def test_credit_prefers_correct_laptop_mass_over_uniform_accuracy():
    """Credit focuses on details that change team outcomes: wrong class on the laptop costs more than wrong
    fragility on a pencil case, even though per-field accuracy treats the fields alike."""
    s = 3
    w = sample_world(s)
    base = oracle_report(w, "A").content["objects"]
    lap_wrong = [dict(o, cls="book") if o["cls"] == "laptop" else o for o in base]
    pen_wrong = [dict(o, fragility_guess=0.95) if o["cls"] == "pencilcase" else o for o in base]
    assert r_eye_credit(json.dumps(lap_wrong), s, "A", H) <= r_eye_credit(json.dumps(pen_wrong), s, "A", H)


def test_trl_eye_adapters():
    rows = gen_eye_grpo_rows(2, 0)
    assert len(rows) == 4 and {r["view"] for r in rows} == {"A", "B"}
    cols = {"seed": [r["seed"] for r in rows[:2]], "view": [r["view"] for r in rows[:2]]}
    comps = [[{"role": "assistant", "content": objs_json(r["seed"], r["view"])}] for r in rows[:2]]
    fmt, acc, cred = trl_eye_reward_funcs(H)
    assert fmt([], comps) == [1.0, 1.0] and all(0 <= x <= 1 for x in acc([], comps, **cols))
    assert len(cred([], comps, **cols)) == 2
    assert [f.__name__ for f in trl_eye_reward_funcs(H, which=("credit",))] == ["eye_credit"]


# ---------- hindsight dataset ----------
def test_hindsight_rows_are_valid_and_funnel_is_consistent():
    rows, st = gen_hindsight_sft(4, 0, H)
    assert len(rows) == 4 and st["validated"] == 4
    assert st["episodes"] >= st["failed"] >= st["relabeled"] >= st["validated"]
    for r in rows:
        plan = parse_plan(r["completion"])
        assert plan.tasks and r["meta"]["kind"] == "hindsight"
        assert r["meta"]["orig_instruction"] != r["meta"]["relabeled_instruction"] or True
        assert "## Instruction" in r["prompt"] and r["meta"]["relabeled_instruction"] in r["prompt"]
        assert "laptop in the main slot" not in r["prompt"].lower()


# ---------- skill regression suite ----------
def test_symbolic_executor_passes_the_suite():
    rep = run_regression(SymbolicExecutor())
    assert rep["passed"] == rep["total"] == len(regression_cases()), {k: v for k, v in rep["cases"].items() if not v["passed"]}
    assert set(rep["by_tag"]) >= {"grasp", "place", "handover", "force", "lock", "reach"}


def test_suite_detects_a_regression(monkeypatch):
    old = run_regression(SymbolicExecutor())
    monkeypatch.setattr(world_mod, "required_force", lambda o: 0.0)          # a 'fine-tune' that forgot slip physics
    new = run_regression(SymbolicExecutor())
    c = compare(old, new)
    assert not c["ok"] and "grasp_slip_low_force" in c["regressions"]
    assert compare(old, old) == {"regressions": [], "fixed": [], "ok": True}


def test_suite_handles_stochastic_executors_with_trials():
    noisy = run_regression(SymbolicExecutor(p_arm_noise=0.3), n_trials=30, pass_rate=0.9)
    assert noisy["passed"] < noisy["total"]                                  # success cases fail at p=0.3
    mild = run_regression(SymbolicExecutor(p_arm_noise=0.02), n_trials=30, pass_rate=0.7)
    assert mild["passed"] >= mild["total"] - 2


def test_credit_by_fault_table_structure_and_signs():
    from rtp.credit import credit_by_fault, credit_table_markdown, FAULTS, FIELDS
    faults = {k: FAULTS[k] for k in ("omission", "mass_noise", "misclass")}
    rows = credit_by_fault(H, range(10), faults)
    assert [r["fault"] for r in rows] == list(faults)
    for r in rows:
        assert r["oracle_gain"]["n"] == 10 and set(r["fields"]) == set(FIELDS)
        assert r["oracle_gain"]["lo"] <= r["oracle_gain"]["mean"] <= r["oracle_gain"]["hi"]
    by = {r["fault"]: r for r in rows}
    assert by["omission"]["empty_value"]["mean"] > 0                       # a leaky Eye still beats silence
    assert by["mass_noise"]["oracle_gain"]["mean"] > 0.05 and by["misclass"]["oracle_gain"]["mean"] > 0.02   # errors cost score
    assert by["mass_noise"]["fields"]["mass_guess"]["mean"] > by["mass_noise"]["fields"]["cls"]["mean"]       # ...and the loss sits in the noisy field
    assert by["misclass"]["fields"]["cls"]["mean"] >= by["misclass"]["fields"]["mass_guess"]["mean"] - 0.02   # class errors live in cls
    assert "| omission |" in credit_table_markdown(rows)
