import json
import os
import shutil
from pathlib import Path

import pytest

from rtp.harness import Harness, Monitor, parse_md
from rtp.datagen import sample_world, sample_spec, default_eyes, gen_brain_sft, gen_grpo_prompts, gen_eye_sft
from rtp.loop import LoopConfig, run_episode
from rtp.brain import ScriptedBrain, LLMBrain, build_prompt, BrainContext
from rtp.perception import Eye, EyeConfig
from rtp.rules import TaskSpec, dry_run, amend
from rtp.plan import parse_plan, PlanError
from rtp import rewards, credit, hindsight
from rtp.stats import wilson

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "harness_files"


@pytest.fixture(scope="module")
def harness():
    return Harness(HARNESS)


def run(seed, harness, noisy=False, cfg=None, p=0.03, amendment=None, eyes=None, hook=None):
    w = sample_world(seed, p_arm_noise=p)
    spec = sample_spec(w, seed)
    return w, spec, run_episode(w, eyes or default_eyes(seed, noisy), ScriptedBrain(), spec, harness,
                                cfg or LoopConfig(), amendment=amendment, eye_hook=hook)


# ---------- harness ----------
def test_harness_loads_and_retrieves_rules_for_instruction(harness):
    spec = TaskSpec(items=["a"], leave=["mug"], last_item="lunchbox1")
    ids = [d.id for d in harness.retrieve(spec.text(), 5, types=("rule",))]
    for needed in ("heavy_bottom", "laptop_slot", "bottle_upright", "lunchbox_last", "leave_distractors"):
        assert needed in ids


def test_harness_writeback(tmp_path):
    shutil.copytree(HARNESS, tmp_path / "h")
    h = Harness(tmp_path / "h")
    m = Monitor(h)
    for ok in (True, True, False):
        m.record("grasp", "book", ok, "" if ok else "slip")
    m.flush()
    m.flush()                                              # idempotent: stats block replaced, not duplicated
    txt = (tmp_path / "h/skills/grasp.md").read_text()
    assert txt.count("<!-- stats -->") == 1 and "success_rate: 0.667" in txt and "top_failure: slip" in txt
    h.add_note("grasp", "laptop slips below 17N")
    h.add_note("grasp", "laptop slips below 17N")
    assert (tmp_path / "h/skills/grasp.md").read_text().count("laptop slips below 17N") == 1
    h.reload()                                             # file still parses after write-back
    assert "grasp" in h.docs


def test_harness_rejects_bad_files(tmp_path):
    (tmp_path / "a.md").write_text("no front matter")
    with pytest.raises(ValueError):
        parse_md(tmp_path / "a.md")


# ---------- loop ----------
def test_oracle_team_succeeds_across_seeds(harness):
    ok = sum(run(s, harness)[2].success for s in range(40))
    assert ok >= 38


def test_episode_is_deterministic_given_seed(harness):
    a, b = run(5, harness, noisy=True)[2], run(5, harness, noisy=True)[2]
    assert (a.score, a.rounds, a.msg_tokens, len(a.executed)) == (b.score, b.rounds, b.msg_tokens, len(b.executed))


def test_amendment_before_laptop_is_placed_succeeds(harness):
    w = sample_world(3)
    spec = amend(sample_spec(w, 3), laptop_slot="laptop")
    res = run_episode(w, default_eyes(3), ScriptedBrain(), spec, harness, LoopConfig(),
                      amendment=(0, {"laptop_slot": "front"}))
    assert res.spec_final["laptop_slot"] == "front" and res.replans >= 1 and res.success
    assert w.compartment_of("laptop1") == "front"


def test_recovery_ladder_helps_under_arm_noise(harness):
    with_ladder = sum(run(s, harness, p=0.25)[2].success for s in range(30))
    # without recovery: no retries/replans possible
    w_ok = 0
    for s in range(30):
        w = sample_world(s, p_arm_noise=0.25)
        spec = sample_spec(w, s)
        cfg = LoopConfig(max_replans=0)
        res = run_episode(w, default_eyes(s), ScriptedBrain(), spec, harness, cfg)
        w_ok += res.success
    assert with_ladder >= 24 and with_ladder >= w_ok


def test_comm_modes_cost_ordering(harness):
    toks = {m: run(2, harness, cfg=LoopConfig(comm_mode=m))[2].msg_tokens for m in ("push", "pull", "init_only")}
    assert toks["push"] > toks["pull"] >= toks["init_only"]


def test_message_budget_drops_and_still_runs(harness):
    res = run(2, harness, cfg=LoopConfig(msg_budget=300))[2]
    assert res.msg_tokens <= 300


def test_missing_item_is_attributed_to_eye(harness):
    w = sample_world(4)
    spec = sample_spec(w, 4)
    blind = [Eye(EyeConfig(name="A", blind_regions=("table_left", "table_right", "table_center"),
                           miss_occluded_p=1.0), 4)]
    res = run_episode(w, blind, ScriptedBrain(), spec, harness, LoopConfig())
    assert not res.success and res.missing
    assert {f["agent"] for f in res.failures if f.get("fatal")} == {"Eye"}


def test_oracle_vs_noisy_failures_are_attributed(harness):
    agents = set()
    for s in range(60):
        res = run(s, harness, noisy=True)[2]
        agents |= {f["agent"] for f in res.failures if f.get("fatal")}
    assert "Eye" in agents


def test_retrieval_recall_reported(harness):
    assert run(1, harness)[2].retrieval_recall == 1.0


def test_no_harness_degrades_when_static_rules_missing(harness):
    cfg = LoopConfig(use_harness=False, static_rule_ids=("leave_distractors",))
    res = run(1, harness, cfg=cfg)[2]
    assert not res.success and res.violations


# ---------- LLM brain adapter ----------
def test_llm_brain_adapter_with_fake_generate(harness):
    w = sample_world(1)
    spec = sample_spec(w, 1)
    teacher = run_episode(sample_world(1), default_eyes(1), ScriptedBrain(), spec, harness, LoopConfig())
    plan_json = teacher.plans[0]
    prompts = []

    def gen(p):
        prompts.append(p)
        return "Here is the plan:\n```json\n" + plan_json + "\n```"

    res = run_episode(w, default_eyes(1), LLMBrain(gen), spec, harness, LoopConfig())
    assert res.success and "## Instruction" in prompts[0] and "Pack Maya" in prompts[0]
    bad = run_episode(sample_world(1), default_eyes(1), LLMBrain(lambda p: "I refuse"), spec, harness, LoopConfig())
    assert not bad.success and any(f["cause"] == "plan_parse" and f["agent"] == "Brain" for f in bad.failures)


# ---------- rewards ----------
def test_plan_rewards_order_good_over_bad(harness):
    w = sample_world(1)
    spec = sample_spec(w, 1)
    from rtp.perception import merge_scenes
    scene = merge_scenes([e.report(w, 0) for e in default_eyes(1)])
    good = run_episode(sample_world(1), default_eyes(1), ScriptedBrain(), spec, harness, LoopConfig()).plans[0]
    bad = run_episode(sample_world(1), default_eyes(1), ScriptedBrain(), spec, harness,
                      LoopConfig(use_harness=False, static_rule_ids=())).plans[0]
    assert rewards.combined(good, scene, spec) > rewards.combined(bad, scene, spec)
    assert rewards.r_format("garbage") == 0 and rewards.combined("garbage", scene, spec) == 0
    assert rewards.r_rules(good, scene, spec) == 1.0


def test_trl_adapters_match_direct_calls(harness):
    rows = gen_grpo_prompts(3, 10, harness)
    good = [run_episode(sample_world(10 + i), default_eyes(10 + i), ScriptedBrain(), sample_spec(sample_world(10 + i), 10 + i),
                        harness, LoopConfig()).plans[0] for i in range(3)]
    comps = [[{"role": "assistant", "content": g}] for g in good]       # TRL conversational format
    cols = {"scene_json": [r["scene_json"] for r in rows], "spec_json": [r["spec_json"] for r in rows]}
    fmt, feas, rul, ln = rewards.trl_reward_funcs()
    assert fmt([], comps) == [1.0] * 3 and all(0 <= x <= 1 for x in feas([], comps, **cols))
    assert len(rul([], comps, **cols)) == 3 and ln([], comps) == [1.0] * 3
    assert fmt([], ["not a plan"]) == [0.0]


def test_episode_reward_penalties():
    base = {"n_valid": 8, "n_items": 8, "violations": [], "crushed": [], "replans": 0, "msg_tokens": 0}
    assert rewards.episode_reward(base) == 1.0
    assert rewards.episode_reward({**base, "violations": [1]}) < 1.0
    assert rewards.episode_reward({**base, "msg_tokens": 5000}, lam_msg=0.1) < 1.0


# ---------- counterfactual credit ----------
def test_credit_single_eye_positive_and_crn_reproducible(harness):
    def make(seed):
        def run_fn(hook):
            w = sample_world(seed)
            return run_episode(w, [Eye(EyeConfig(name="A", mass_sigma=0.1), seed)], ScriptedBrain(),
                               sample_spec(w, seed), harness, LoopConfig(), eye_hook=hook)
        return run_fn
    f = make(3)
    assert f(None).score == f(None).score                                     # CRN: replay is identical
    c = credit.eye_credit(f, "A", "empty")
    assert c["credit"] > 0.5 and c["base_success"] and not c["cf_success"]
    r = credit.expected_credit(make, range(5), "A", "empty")
    assert r["mean"] > 0.5 and r["n"] == 5


def test_credit_redundant_eye_is_near_zero(harness):
    def run_fn(hook):
        w = sample_world(3)
        return run_episode(w, default_eyes(3), ScriptedBrain(), sample_spec(w, 3), harness, LoopConfig(), eye_hook=hook)
    m = credit.team_marginals(run_fn, ["A", "B"], "empty")
    assert abs(m["A"]) < 0.2 and abs(m["B"]) < 0.2                            # two perfect eyes are redundant


def test_oracle_credit_measures_perception_loss(harness):
    losses = []
    for s in range(12):
        def run_fn(hook, s=s):
            w = sample_world(s)
            return run_episode(w, default_eyes(s, True), ScriptedBrain(), sample_spec(w, s), harness, LoopConfig(), eye_hook=hook)
        losses.append(credit.eye_credit(run_fn, "A", "oracle")["credit"] + credit.eye_credit(run_fn, "B", "oracle")["credit"])
    assert max(losses) >= 0.0 and min(losses) > -1.0


# ---------- hindsight ----------
def test_hindsight_relabels_failed_episodes_and_replays(harness):
    made = valid = 0
    for s in range(60):
        w = sample_world(s, p_arm_noise=0.0)
        spec = sample_spec(w, s)
        res = run_episode(w, default_eyes(s, True), ScriptedBrain(), spec, harness, LoopConfig())
        if res.success:
            continue
        rec = hindsight.relabel(w, res, spec)
        if rec is None:
            continue
        made += 1
        w2 = sample_world(s, p_arm_noise=0.0)
        new_spec = TaskSpec.from_dict(rec.spec)
        r2 = run_episode(w2, default_eyes(s), hindsight.ReplayBrain(parse_plan(rec.plan_json)), new_spec, harness, LoopConfig())
        valid += r2.success
    assert made >= 3 and valid == made                                       # every relabel replays to success


# ---------- data generation ----------
def test_datagen_outputs_are_valid(harness, tmp_path):
    rows = gen_brain_sft(5, 100, harness)
    assert len(rows) == 5
    for r in rows:
        parse_plan(r["completion"])
        assert "## Instruction" in r["prompt"]
    g = gen_grpo_prompts(2, 0, harness)
    assert json.loads(g[0]["spec_json"])["items"] and json.loads(g[0]["scene_json"])
    e = gen_eye_sft(2, 0)
    assert len(e) == 4 and json.loads(e[0]["target"])
    from rtp.datagen import write_jsonl
    write_jsonl(rows, tmp_path / "x.jsonl")
    assert len((tmp_path / "x.jsonl").read_text().splitlines()) == 5


# ---------- Track A scorer (needs a VIKI-R clone) ----------
VIKI = os.environ.get("VIKI_R_REPO", "")


@pytest.mark.skipif(not VIKI or not Path(VIKI).exists(), reason="VIKI-R clone not available")
def test_viki_l2_scorer_position_randomness():
    import random
    from rtp.trackA import load_scorer, score_all_positions, score_seeded
    mod = load_scorer(VIKI)
    gt = {"robots": {"R1": "fetch"},
          "init_pos": {"R1": ["R1"], "cup_1": ["table", "shelf"], "tray_1": ["tray_pos"]},
          "goal_constraints": [],
          "temporal_constraints": [[[{"type": "asset", "name": "cup", "is_satisfied": True, "status": {"pos.name": "tray"}}]]]}
    robust = [{"R1": "<move, cup>"}, {"R1": "<reach, cup>"}, {"R1": "<grasp, cup>"}, {"R1": "<move, tray>"}, {"R1": "<place, tray>"}]
    fragile = [{"R1": "<move, table>"}] + robust[1:]
    assert score_all_positions(mod, robust, gt) == {"frac_success": 1.0, "n_combos": 2, "position_dependent": False}
    f = score_all_positions(mod, fragile, gt)
    assert f["position_dependent"] and f["frac_success"] == 0.5
    assert score_seeded(mod, fragile, gt, 1) == score_seeded(mod, fragile, gt, 1)      # seeded is repeatable


# ---------- closed-loop rollout reward ----------
def test_rollout_reward_scores_good_plans_high_and_garbage_zero(harness):
    from rtp.rewards import r_rollout, trl_reward_funcs, trl_reward_weights
    good = run_episode(sample_world(5), default_eyes(5, True), ScriptedBrain(), sample_spec(sample_world(5), 5), harness, LoopConfig()).plans[0]
    bad = run_episode(sample_world(5), default_eyes(5, True), ScriptedBrain(), sample_spec(sample_world(5), 5), harness,
                      LoopConfig(use_harness=False, static_rule_ids=())).plans[0]
    g, b = r_rollout(good, 5, harness), r_rollout(bad, 5, harness)
    assert g >= 0.85 and g > b and 0.0 <= b <= 1.0
    assert r_rollout("garbage", 5, harness) == 0.0 and r_rollout(good, 5, harness) == g          # deterministic
    assert len(trl_reward_funcs()) == 4 and len(trl_reward_funcs(harness=harness)) == 5
    assert len(trl_reward_weights(False)) == 4 and len(trl_reward_weights(True)) == 5
    rows = gen_grpo_prompts(2, 20, harness)
    out = trl_reward_funcs(harness=harness)[-1]([], [good, good], seed=[r["seed"] for r in rows])
    assert len(out) == 2 and all(0 <= x <= 1 for x in out)
