"""Counterfactual Eye credit.

credit(eye, mode) = score(real reports) - score(episode where that Eye's reports are replaced).
Episodes are replayed under common random numbers (World.u keys draws by arm/skill/object, Eye
noise by seed/eye/time), so differences come from the replaced reports, not from luck.
Interpretation by mode:
  empty / degraded : positive = the Eye's real reports helped.
  oracle           : credit is NEGATIVE when the Eye's errors cost score (R(real) < R(oracle)); report -credit as
                     the 'oracle gain' = score a perfect Eye would recover (credit_by_fault does this).
  oracle_fields    : same, restricted to the named fields (which details matter).
`run_fn(eye_hook) -> EpisodeResult` must build a fresh world each call.
"""
from __future__ import annotations
import random

from .perception import replace_report
from .stats import bootstrap_ci


def make_hook(eye_name: str, mode: str, fields=None, only_t=None):
    def hook(name, msg, world):
        if name != eye_name or (only_t is not None and msg.t != only_t):
            return msg
        rng = random.Random(f"cf|{name}|{msg.t}|{mode}")
        return replace_report(msg, world, mode, rng, fields)
    return hook


def eye_credit(run_fn, eye_name: str, mode: str = "empty", fields=None, only_t=None) -> dict:
    base = run_fn(None)
    cf = run_fn(make_hook(eye_name, mode, fields, only_t))
    return {"credit": base.score - cf.score, "base": base.score, "cf": cf.score,
            "base_success": base.success, "cf_success": cf.success}


def team_marginals(run_fn, eye_names, mode: str = "empty") -> dict:
    return {n: eye_credit(run_fn, n, mode)["credit"] for n in eye_names}


def message_credits(run_fn, eye_name: str, times, mode: str = "empty") -> dict:
    """Per-report credit: replace only the report emitted at world time t."""
    return {t: eye_credit(run_fn, eye_name, mode, only_t=t)["credit"] for t in times}


def expected_credit(make_run_fn, seeds, eye_name: str, mode: str = "empty", fields=None) -> dict:
    cs = [eye_credit(make_run_fn(s), eye_name, mode, fields)["credit"] for s in seeds]
    lo, hi = bootstrap_ci(cs)
    return {"mean": sum(cs) / len(cs), "lo": lo, "hi": hi, "n": len(cs), "per_seed": cs}


# ---------------- analysis tables (paper Fig. 3) ----------------
FAULTS = {
    "omission":   dict(blind_regions=("table_left", "table_right", "table_center"), miss_occluded_p=0.4),
    "mass_noise": dict(mass_sigma=0.45),
    "frag_noise": dict(frag_sigma=0.3),
    "misclass":   dict(misclass_p=0.25),
    "region_err": dict(region_error_p=0.3),
    "stale":      dict(stale_p=0.5, mass_sigma=0.3),
}
FIELDS = ("cls", "region", "mass_guess", "fragility_guess")


def _runner(harness, fault_kwargs, n_eyes=2):
    from .datagen import sample_world, sample_spec
    from .loop import LoopConfig, run_episode
    from .brain import ScriptedBrain
    from .perception import Eye, EyeConfig

    def make(seed):
        def run_fn(hook):
            w = sample_world(seed)
            eyes = [Eye(EyeConfig(name=n, **fault_kwargs), seed) for n in ("A", "B")[:n_eyes]]
            return run_episode(w, eyes, ScriptedBrain(), sample_spec(w, seed), harness, LoopConfig(), eye_hook=hook)
        return run_fn
    return make


def _negate(d: dict) -> dict:
    return {**d, "mean": 0.0 - d["mean"], "lo": 0.0 - d["hi"], "hi": 0.0 - d["lo"], "per_seed": [0.0 - x for x in d["per_seed"]]}


def credit_by_fault(harness, seeds, faults=None, eye="A", n_eyes=2) -> list:
    """For each Eye fault type: `oracle_gain` = score recovered if that Eye were perfect (positive = its errors cost
    score), `fields[f]` = the gain from fixing only field f, and `empty_value` = R(real reports) - R(silent Eye)
    (positive = the Eye's reports help)."""
    rows = []
    for name, kw in (faults or FAULTS).items():
        make = _runner(harness, kw, n_eyes)
        base_scores = [make(s)(None).score for s in seeds]
        row = {"fault": name, "base_score": sum(base_scores) / len(base_scores),
               "oracle_gain": _negate(expected_credit(make, seeds, eye, "oracle")),
               "empty_value": expected_credit(make, seeds, eye, "empty"), "fields": {}}
        for f in FIELDS:
            row["fields"][f] = _negate(expected_credit(make, seeds, eye, "oracle_fields", fields=[f]))
        rows.append(row)
    return rows


def credit_table_markdown(rows) -> str:
    out = ["| Fault | mean score | Eye A recoverable score (oracle gain) [95% CI] | Eye A value (vs silent) | " + " | ".join(f"fix {f}" for f in FIELDS) + " |",
           "|---|---|---|---|" + "---|" * len(FIELDS)]
    for r in rows:
        ci = lambda d: f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
        out.append(f"| {r['fault']} | {r['base_score']:.3f} | {ci(r['oracle_gain'])} | {ci(r['empty_value'])} | "
                   + " | ".join(f"{r['fields'][f]['mean']:+.3f}" for f in FIELDS) + " |")
    return "\n".join(out)
