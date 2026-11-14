"""Collect REAL recorded data from the project code (rtp/, isaac/ constants) into generated/figdata.json.

Everything the data-bearing figures show (fig_mechanism, fig_loop, fig_messages, fig_harness, fig_tools, fig_signals,
fig_credit, fig_bridge) is read from this file, so no figure contains a hand-typed trace or number.
Run:  python scripts/collect_figdata.py --project /path/to/robot_team
All episodes here are SYMBOLIC-world episodes with a scripted Brain and parametric Eyes (evidence class FIXTURE):
they illustrate mechanisms; they are not evidence about physical robots or trained models.
"""
from __future__ import annotations
import argparse, ast, json, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--project", default=str(Path(__file__).resolve().parents[2] / "robot_team"))
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "generated" / "figdata.json"))
a = ap.parse_args()
PROJ = Path(a.project)
sys.path.insert(0, str(PROJ))

from rtp import messages as M, loop as L, world as W, rewards as RW  # noqa: E402
from rtp.datagen import sample_world, sample_spec  # noqa: E402
from rtp.loop import LoopConfig, run_episode  # noqa: E402
from rtp.brain import ScriptedBrain, BrainContext  # noqa: E402
from rtp.fixtures import FlakyBrain  # noqa: E402
from rtp.perception import Eye, EyeConfig, merge_scenes  # noqa: E402
from rtp.harness import Harness  # noqa: E402
from rtp import harness as HARN  # noqa: E402
from rtp.hindsight import relabel, ReplayBrain  # noqa: E402
from rtp.plan import parse_plan  # noqa: E402
from rtp.rules import TaskSpec  # noqa: E402
from rtp.credit import eye_credit  # noqa: E402
from rtp.toolbrain import ToolBrain  # noqa: E402

H = Harness(PROJ / "harness_files")
EVENTS: list = []


class LogBus(M.MessageBus):
    def send(self, msg):
        ok = super().send(msg)
        EVENTS.append({"ev": "msg", "type": msg.type, "from": msg.sender, "to": msg.receiver, "content": msg.content,
                       "note": msg.note, "ok": ok})
        return ok


class LogWorld(W.World):
    def execute(self, arm, skill, obj=None, params=None):
        res = super().execute(arm, skill, obj, params)
        EVENTS.append({"ev": "exec", "arm": arm, "skill": skill, "obj": obj, "ok": res.ok, "cause": res.cause,
                       "touch": res.touch})
        return res


L.MessageBus = LogBus


class RecBrain(ScriptedBrain):
    """ScriptedBrain that records the scene each plan() call was made from (what the planner actually saw)."""
    def __init__(self):
        super().__init__()
        self.scenes = []

    def plan(self, ctx, *args, **kw):
        self.scenes.append({k: dict(v) for k, v in ctx.scene.items()})
        return super().plan(ctx, *args, **kw)

    def replan(self, ctx, *args, **kw):
        self.scenes.append({k: dict(v) for k, v in ctx.scene.items()})
        return super().replan(ctx, *args, **kw)


def logged_world(seed):
    w = sample_world(seed)
    return LogWorld(list(w.objs.values()), seed=w.seed, p_arm_noise=w.p_arm_noise)


def noisy_eyes(seed, sigma):
    return [Eye(EyeConfig(name=n, mass_sigma=sigma), seed) for n in ("A", "B")]


def story(res):
    return {"success": res.success, "score": res.score, "violations": res.violations, "crushed": res.crushed,
            "missing": res.missing, "replans": res.replans, "rounds": res.rounds, "looks": res.looks,
            "challenges": res.challenges, "plan_retries": res.plan_retries, "plan_pass": res.plan_pass,
            "failures": [{k: f.get(k) for k in ("agent", "cause", "task", "obj", "detail", "fatal")} for f in res.failures],
            "executed": res.executed, "plans": res.plans, "scene_final": res.scene_final}


out = {}

# ---------------------------------------------------------------- fig_mechanism: one real episode (seed 24)
SEED_MECH, SIGMA = 24, 0.45
w0 = sample_world(SEED_MECH)
spec0 = sample_spec(w0, SEED_MECH)
EVENTS.clear()
rb_a = RecBrain()
a_res = run_episode(logged_world(SEED_MECH), noisy_eyes(SEED_MECH, SIGMA), rb_a, spec0, H,
                    LoopConfig(allow_challenge=False, verify="none", probe=False))
ev_a = list(EVENTS)
EVENTS.clear()
rb_b = RecBrain()
b_res = run_episode(logged_world(SEED_MECH), noisy_eyes(SEED_MECH, SIGMA), rb_b, spec0, H,
                    LoopConfig(allow_challenge=True, verify="static", probe=False))
ev_b = list(EVENTS)
EVENTS.clear()
c_res = run_episode(logged_world(SEED_MECH), noisy_eyes(SEED_MECH, SIGMA), ScriptedBrain(), spec0, H,
                    LoopConfig(allow_challenge=False, verify="static", probe=True))
items = []
for n, o in sorted(w0.objs.items()):
    rep = rb_a.scenes[0].get(n, {})
    items.append({"name": n, "cls": o.cls, "true_mass": o.mass, "eye_mass": rep.get("mass_guess"),
                  "true_frag": o.fragility, "eye_frag": rep.get("fragility_guess"), "eye_cls": rep.get("cls"),
                  "distractor": o.is_distractor})
chal = [e for e in ev_b if e["ev"] == "msg" and e["type"] == "CHALLENGE"]
out["mechanism"] = {
    "seed": SEED_MECH, "mass_sigma": SIGMA, "instruction": spec0.text(), "items": items,
    "no_fix": story(a_res), "challenge": story(b_res), "probe": story(c_res),
    "scene_at_plan": "first plan() call of the no-fix run", "planned_from_n_scenes": len(rb_a.scenes),
    "challenge_msgs": [{"from": e["from"], "to": e["to"], "content": e["content"]} for e in chal],
    "no_fix_exec": [{k: v for k, v in e.items() if k != "touch"} for e in ev_a if e["ev"] == "exec"],
    "challenge_exec": [{k: v for k, v in e.items() if k != "touch"} for e in ev_b if e["ev"] == "exec"],
}

# ---------------------------------------------------------------- fig_loop: one real, many-mechanism episode (seed 11)
SEED_LOOP, AMEND_ROUND = 11, 3


def leaky_eyes(s):
    kw = dict(blind_regions=("table_left", "table_right", "table_center"), miss_occluded_p=0.4, mass_sigma=0.35)
    return [Eye(EyeConfig(name="A", **kw), s), Eye(EyeConfig(name="B", **kw), s)]


wl = sample_world(SEED_LOOP)
specl = sample_spec(wl, SEED_LOOP)
EVENTS.clear()
brain = FlakyBrain(ScriptedBrain(), p_err=0.5, seed=SEED_LOOP)
l_res = run_episode(logged_world(SEED_LOOP), leaky_eyes(SEED_LOOP), brain, specl, H, LoopConfig(verify="dry_run"),
                    amendment=(AMEND_ROUND, {"laptop_slot": "front"}))
out["loop"] = {"seed": SEED_LOOP, "amend_round": AMEND_ROUND, "injected": brain.injected, "instruction": specl.text(),
               "story": story(l_res), "events": [{k: v for k, v in e.items() if k != "touch"} for e in EVENTS]}

# ---------------------------------------------------------------- fig_messages: schema + real messages
sc = [e for e in ev_b if e["ev"] == "msg" and e["type"] == "SCENE"][0]
touch = [e for e in ev_b if e["ev"] == "msg" and e["type"] == "TOUCH"][0]
out["messages"] = {
    "schema": {k: sorted(v) for k, v in M.SCHEMA.items()},
    "examples": {
        "SCENE": {"type": "SCENE", "sender": sc["from"], "receiver": sc["to"],
                  "content": {"view": sc["content"]["view"], "objects": [sc["content"]["objects"][0]]}},
        "TOUCH": {"type": "TOUCH", "sender": touch["from"], "receiver": touch["to"], "content": touch["content"]},
        "CHALLENGE": {"type": "CHALLENGE", "sender": chal[0]["from"], "receiver": chal[0]["to"], "content": chal[0]["content"]},
    },
    "chars_per_token_proxy": 4,
}
for k, m in out["messages"]["examples"].items():
    m["n_tokens_proxy"] = M.Message(m["type"], m["sender"], m["receiver"], m["content"]).n_tokens()

# ---------------------------------------------------------------- fig_harness: files, retrieval, tokenizer fix
docs = [{"id": d.id, "type": d.type, "keywords": d.keywords, "path": str(d.path.relative_to(PROJ)),
         "text": (d.path.read_text().split("---", 2)[2].strip() if d.path.read_text().startswith("---") else d.path.read_text().strip())}
        for d in H.docs.values()]
spec_text = spec0.text()
need = sorted(set(spec0.rule_ids()) - {"bag_exclusive"})
q2 = " ".join(o["cls"] for o in a_res.scene_final.values())
new_rules = [d.id for d in H.retrieve(spec_text, 5, types=("rule",))]
new_other = [d.id for d in H.retrieve(q2, 3, types=("object", "skill", "memory"))]
orig_tok = HARN.tokenize
HARN.tokenize = lambda s: re.findall(r"[a-z0-9]+", s.lower())  # reconstruction of the pre-fix tokenizer (see text)
H_old = Harness(PROJ / "harness_files")
old_rules = [d.id for d in H_old.retrieve(spec_text, 5, types=("rule",))]
old_tok_demo = HARN.tokenize("lunchbox1 last")
old_rank = [(i, round(sc, 3)) for i, sc in H_old.bm25.rank(spec_text) if H_old.docs[i].type == "rule"]
HARN.tokenize = orig_tok
new_rank = [(i, round(sc, 3)) for i, sc in H.bm25.rank(spec_text) if H.docs[i].type == "rule"]
out["harness"] = {
    "n_files": len(docs), "docs": docs, "instruction": spec_text, "required_rules": need,
    "retrieved_rules_new": new_rules, "retrieved_other_new": new_other, "retrieved_rules_old": old_rules,
    "tok_new": HARN.tokenize("lunchbox1 last"), "tok_old": old_tok_demo,
    "recall_new": len(set(need) & set(new_rules)) / len(need), "recall_old": len(set(need) & set(old_rules)) / len(need),
    "types": sorted({d["type"] for d in docs}),
    "rule_rank_old": old_rank, "rule_rank_new": new_rank,
}


# ---------------------------------------------------------------- fig_tools: a real draft + real dry_run feedback
def find_flawed(seed0=0):
    for s in range(seed0, seed0 + 200):
        w = sample_world(s)
        sp = sample_spec(w, s)
        scene = merge_scenes([Eye(EyeConfig(name=n), s).report(w, 0) for n in ("A", "B")])
        ctx = BrainContext(spec=sp, scene=scene, rule_ids=set(sp.rule_ids()), snippets=[])
        draft = FlakyBrain(ScriptedBrain(), p_err=1.0, seed=s, kinds=("swap",)).plan(ctx)
        tb = ToolBrain(None)
        try:
            res = json.loads(tb._dry_run(ctx, {"plan": json.loads(draft.to_json())}))
        except Exception:          # a swap can create a dependency cycle: such a draft is rejected at parse time
            continue
        if res["violations"]:
            good = ScriptedBrain().plan(ctx)
            return s, sp, draft, good, res, json.loads(tb._dry_run(ctx, {"plan": json.loads(good.to_json())}))
    raise RuntimeError("no flawed draft found")


ts, tsp, tdraft, tgood, tres, tres2 = find_flawed()
brief = lambda p: [{"id": t.id, "arm": t.arm, "skill": t.skill, "obj": t.obj, "deps": t.deps} for t in p.tasks]
out["tools"] = {"seed": ts, "instruction": tsp.text(), "tool_names": ["check_rule", "query_scene", "look", "dry_run", "submit_plan"],
                "draft_tasks": brief(tdraft), "good_tasks": brief(tgood), "dry_run_draft": tres, "dry_run_final": tres2,
                "corruption": "swap: ordering dependencies of two placements exchanged", "max_turns_default": 6}

# ---------------------------------------------------------------- fig_signals: reward weights read from code
out["signals"] = {"weights": RW.DEFAULT_WEIGHTS, "rollout_weight": RW.ROLLOUT_WEIGHT, "r_rules_penalty": 0.25,
                  "length_cap": 24, "episode_reward": {"w_viol": 0.15, "w_crush": 0.3, "w_replan": 0.01}}


# ---------------------------------------------------------------- fig_credit a: counterfactual credit, seed-24 episode
def run_fn(hook):
    w = sample_world(SEED_MECH)
    return run_episode(w, noisy_eyes(SEED_MECH, SIGMA), ScriptedBrain(), sample_spec(w, SEED_MECH), H,
                       LoopConfig(allow_challenge=False, verify="none"), eye_hook=hook)


base = run_fn(None)
rows = {}
for mode, fields in (("empty", None), ("degraded", None), ("oracle", None), ("oracle_fields", ["mass_guess"]),
                     ("oracle_fields", ["cls"]), ("oracle_fields", ["fragility_guess"]), ("oracle_fields", ["region"])):
    key = mode if not fields else f"oracle_{fields[0]}"
    r = eye_credit(run_fn, "A", mode, fields)
    rows[key] = {"credit": r["credit"], "cf_score": r["cf"], "cf_success": r["cf_success"]}
out["credit"] = {"seed": SEED_MECH, "eye": "A", "base_score": base.score, "base_success": base.success, "rows": rows,
                 "oracle_gain": -rows["oracle"]["credit"],
                 "field_gain": {f: -rows[f"oracle_{f}"]["credit"] for f in ("mass_guess", "cls", "fragility_guess", "region")},
                 "config": "scripted Brain, no challenge, verify none, mass noise 0.45 on both Eyes"}

# ---------------------------------------------------------------- fig_credit b: real hindsight relabeling
wh = logged_world(SEED_MECH)
h_res = run_episode(wh, noisy_eyes(SEED_MECH, SIGMA), ScriptedBrain(), spec0, H, LoopConfig(allow_challenge=False, verify="none"))
rec = relabel(wh, h_res, spec0)
new_spec = TaskSpec.from_dict(rec.spec)
r2 = run_episode(sample_world(SEED_MECH), noisy_eyes(SEED_MECH, SIGMA), ReplayBrain(parse_plan(rec.plan_json)), new_spec, H,
                 LoopConfig(allow_challenge=False, verify="none"))
out["hindsight"] = {"seed": SEED_MECH, "orig_instruction": spec0.text(), "orig_success": h_res.success,
                    "orig_violations": h_res.violations, "relabeled_instruction": rec.instruction,
                    "replay_success": r2.success, "replay_score": r2.score, "n_tasks": len(json.loads(rec.plan_json)["tasks"]),
                    "dropped": sorted(set(spec0.rule_ids()) - set(new_spec.rule_ids()))}

# ---------------------------------------------------------------- fig_bridge: Isaac layout constants read from source
src = (PROJ / "isaac" / "packing_sim.py").read_text()


def grab(name):
    return ast.literal_eval(re.search(rf"^{name}\s*=\s*(\{{.*?\}})\s*$", src, re.M | re.S).group(1)
                            if name == "SIM_DIMS" else re.search(rf"^{name}\s*=\s*(.+)$", src, re.M).group(1).split("#")[0].strip())


lay_txt = " ".join(re.search(r"LAYOUT = Layout\((.*?)\)\n", src, re.S).group(1).split())
lay = {k: ast.literal_eval(v) for k, v in re.findall(r"(\w+)=(\([^)]*\)|[-\d.]+|\w+)", lay_txt)}
eye_src = (PROJ / "isaac" / "cameras.py").read_text()
out["bridge"] = {"zone_x": grab("ZONE_X"), "arm_base": grab("ARM_BASE"), "row_y": grab("ROW_Y"), "layout": lay,
                 "eye_poses": ast.literal_eval(re.search(r"EYE_POSES = (\{.*\})", eye_src).group(1)),
                 "skill_outcomes": ["success", "slip", "crush", "out_of_reach", "wrong_slot", "no_space", "locked",
                                    "overload", "not_holding", "hand_full"]}

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out, indent=1, default=str))
print("wrote", a.out)
print("mechanism: no_fix", a_res.success, a_res.score, "| challenge", b_res.success, b_res.score, "| probe", c_res.success, c_res.score)
print("loop: events", len(out["loop"]["events"]), "looks", l_res.looks, "challenges", l_res.challenges, "retries", l_res.plan_retries,
      "replans", l_res.replans, "success", l_res.success)
print("credit:", out["credit"]["base_score"], out["credit"]["oracle_gain"], out["credit"]["field_gain"])
print("hindsight:", out["hindsight"]["relabeled_instruction"], out["hindsight"]["replay_success"], out["hindsight"]["dropped"])
print("rank old", old_rank, "new", new_rank)
print("harness:", out["harness"]["recall_new"], out["harness"]["recall_old"], out["harness"]["tok_old"], out["harness"]["tok_new"],
      out["harness"]["retrieved_rules_old"])
print("tools:", out["tools"]["dry_run_draft"]["violations"], out["tools"]["dry_run_final"]["success"])
print("bridge:", out["bridge"]["layout"])
