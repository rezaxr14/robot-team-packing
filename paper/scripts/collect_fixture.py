"""Recompute the FIXTURE mechanics checks (appendix F) with the project code; write generated/fixture.json.

Evidence class FIXTURE: symbolic world, scripted Brain, parametric Eyes, flaky-planner fixture. These numbers check that
mechanisms work; they are NEVER a claim about robots or trained models. Evaluation seeds 0..79 (< 100000).
Run:  python scripts/collect_fixture.py --project /path/to/robot_team
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--project", default=str(Path(__file__).resolve().parents[2] / "robot_team"))
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "generated" / "fixture.json"))
ap.add_argument("--n", type=int, default=80)
a = ap.parse_args()
PROJ = Path(a.project)
sys.path.insert(0, str(PROJ))

from rtp.harness import Harness  # noqa: E402
from rtp.experiments import verify_grid, run_config, ExpConfig  # noqa: E402
from rtp.datagen import sample_world, sample_spec  # noqa: E402
from rtp.loop import LoopConfig, run_episode  # noqa: E402
from rtp.brain import ScriptedBrain  # noqa: E402
from rtp.perception import Eye, EyeConfig  # noqa: E402
from rtp.credit import credit_by_fault  # noqa: E402
from rtp.stats import summarize_binary  # noqa: E402

H = Harness(PROJ / "harness_files")
SEEDS = list(range(a.n))
assert max(SEEDS) < 100000
out = {"n": a.n, "seeds": f"0..{a.n - 1}", "class": "FIXTURE"}


def wil(k, n):
    s = summarize_binary([True] * k + [False] * (n - k))
    return {"k": k, "n": n, "rate": s["rate"], "lo": s.get("lo"), "hi": s.get("hi")}


# 1. plan verification against the flaky-planner fixture (p_err 0.5, clean Eyes)
v = {}
for cfg in verify_grid():
    recs = run_config(cfg, SEEDS, H)
    v[cfg.name] = {**wil(sum(r["success"] for r in recs), len(recs)), "plan_retries": sum(r["plan_retries"] for r in recs),
                   "plan_pass_first": sum(r["plan_pass"] for r in recs)}
out["verify"] = v

# 2. communication modes (noisy Eyes)
c = {}
for mode in ("push", "delta", "pull", "init_only"):
    recs = run_config(ExpConfig(f"comm_{mode}", comm=mode, noisy_eyes=True), SEEDS, H)
    c[mode] = {**wil(sum(r["success"] for r in recs), len(recs)), "tokens_per_episode": sum(r["msg_tokens"] for r in recs) / len(recs),
               "messages_note": "tokenizer-free proxy, about 4 characters per token"}
out["comm"] = c


# 3. heavy Eye mass noise: probe vs challenge-only vs none
def run_mass(seed, sigma, **kw):
    w = sample_world(seed)
    eyes = [Eye(EyeConfig(name=n, mass_sigma=sigma), seed) for n in ("A", "B")]
    return run_episode(w, eyes, ScriptedBrain(), sample_spec(w, seed), H, LoopConfig(**kw))


m = {}
for name, kw in (("probe", dict(probe=True)), ("challenge_only", dict(allow_challenge=True)), ("none", dict(allow_challenge=False))):
    rs = [run_mass(s, 0.45, **kw) for s in SEEDS]
    m[name] = {**wil(sum(r.success for r in rs), len(rs)), "mean_rounds": sum(r.rounds for r in rs) / len(rs)}
out["mass_noise"] = {"sigma": 0.45, **m}

# 4. LOOK on a leaky Eye (omission fault); both Eyes leak
kw = dict(blind_regions=("table_left", "table_right", "table_center"), miss_occluded_p=0.4)
lk = {}
for n_eyes in (1, 2):
    for allow in (False, True):
        ok = 0
        for s in SEEDS:
            w = sample_world(s)
            eyes = [Eye(EyeConfig(name=n, **kw), s) for n in ("A", "B")[:n_eyes]]
            r = run_episode(w, eyes, ScriptedBrain(), sample_spec(w, s), H, LoopConfig(allow_look=allow))
            ok += r.success
        lk[f"eyes{n_eyes}_look{int(allow)}"] = wil(ok, len(SEEDS))
out["look"] = {"fault": "omission, miss probability 0.4 per Eye per item", **lk}

# 5. counterfactual credit by Eye fault type (Eye A; oracle gain G = R(oracle) - R(real))
rows = credit_by_fault(H, SEEDS)
out["credit"] = [{"fault": r["fault"], "base_score": r["base_score"], "oracle_gain": {k: r["oracle_gain"][k] for k in ("mean", "lo", "hi")},
                  "empty_value": {k: r["empty_value"][k] for k in ("mean", "lo", "hi")},
                  "fields": {f: r["fields"][f]["mean"] for f in r["fields"]}} for r in rows]

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("verify", "comm", "mass_noise", "look")}, indent=1)[:3500])
for r in out["credit"]:
    print(r["fault"], round(r["base_score"], 3), {k: round(x, 3) for k, x in r["oracle_gain"].items()}, {f: round(x, 3) for f, x in r["fields"].items()})
