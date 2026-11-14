"""Experiment runner: config matrix -> per-episode JSONL records -> summaries.

One record schema serves the symbolic fixture now and the real Eye/Brain/Isaac runs later, so the table code
(rtp.tables) does not change. A record is one episode of one configuration under one seed.
"""
from __future__ import annotations
import json
from dataclasses import dataclass

from .datagen import sample_world, sample_spec, default_eyes
from .loop import LoopConfig, run_episode
from .brain import ScriptedBrain, LLMBrain
from .stats import summarize_binary, paired_bootstrap_diff, bootstrap_ci
from .rules import RULE_IDS


@dataclass
class ExpConfig:
    name: str
    brain: str = "scripted"            # scripted | llm (one-shot plan) | tool (multi-turn with tools)
    comm: str = "push"
    harness: bool = True
    static_rules: tuple = RULE_IDS
    n_eyes: int = 2
    noisy_eyes: bool = False
    p_arm_noise: float = 0.03
    amend: bool = False                # mid-episode amendment of the laptop slot at round 0
    max_replans: int = 2
    msg_budget: int = 10 ** 9
    k_rules: int = 5
    k_other: int = 3
    allow_look: bool = True
    allow_challenge: bool = True
    probe: bool = False
    verify: str = "static"
    max_plan_retries: int = 2
    flaky: float = 0.0                 # fixture: probability the scripted Brain emits a flawed plan


def record_from_result(name: str, seed: int, res) -> dict:
    """One episode result -> the record schema used by summaries/tables (symbolic or Isaac)."""
    fatal = [f for f in res.failures if f.get("fatal")]
    return {"config": name, "seed": seed, "success": res.success, "score": res.score, "n_valid": res.n_valid,
            "n_items": res.n_items, "plan_pass": res.plan_pass, "retrieval_recall": res.retrieval_recall,
            "replans": res.replans, "rounds": res.rounds, "msg_tokens": res.msg_tokens,
            "violations": [v["rule"] for v in res.violations], "crushed": len(res.crushed),
            "missing": len(res.missing), "fatal_agents": sorted({f["agent"] for f in fatal}),
            "fatal": [{"agent": f["agent"], "cause": f["cause"], "obj": f.get("obj")} for f in fatal],
            "n_failures_any": len(res.failures), "looks": res.looks, "challenges": res.challenges,
            "plan_retries": res.plan_retries, "probes": res.probes}


def run_config(cfg: ExpConfig, seeds, harness, generate=None, keep_story: bool = False, chat=None) -> list:
    records = []
    for s in seeds:
        w = sample_world(s, p_arm_noise=cfg.p_arm_noise)
        spec = sample_spec(w, s)
        eyes = default_eyes(s, cfg.noisy_eyes)[:cfg.n_eyes]
        if cfg.brain == "tool":
            from .toolbrain import ToolBrain
            brain = ToolBrain(chat, harness)
        else:
            brain = LLMBrain(generate) if cfg.brain == "llm" else ScriptedBrain()
            if cfg.flaky:
                from .fixtures import FlakyBrain
                brain = FlakyBrain(brain, p_err=cfg.flaky, seed=s)
        lc = LoopConfig(comm_mode=cfg.comm, use_harness=cfg.harness, static_rule_ids=tuple(cfg.static_rules),
                        max_replans=cfg.max_replans, msg_budget=cfg.msg_budget, k_rules=cfg.k_rules,
                        k_other=cfg.k_other, allow_look=cfg.allow_look, allow_challenge=cfg.allow_challenge,
                        verify=cfg.verify, max_plan_retries=cfg.max_plan_retries, probe=cfg.probe)
        amendment = (0, {"laptop_slot": "front"}) if cfg.amend else None
        res = run_episode(w, eyes, brain, spec, harness, lc, amendment=amendment)
        rec = record_from_result(cfg.name, s, res)
        if cfg.brain == "tool":
            rec["brain_totals"] = brain.totals
        if keep_story:
            from .attribution_eval import episode_story
            rec["story"] = episode_story(w, spec, res)
        records.append(rec)
    return records


def write_records(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_records(path) -> list:
    return [json.loads(l) for l in open(path) if l.strip()]


def summarize(records) -> dict:
    """Per-config summary: success (Wilson), valid-item fraction, plan-pass, gap, tokens, recall, fatal-agent shares."""
    out = {}
    for name in sorted({r["config"] for r in records}):
        rs = [r for r in records if r["config"] == name]
        succ = summarize_binary([r["success"] for r in rs])
        plan = summarize_binary([r["plan_pass"] for r in rs])
        valid = [r["n_valid"] / max(1, r["n_items"]) for r in rs]
        fatal = {}
        for r in rs:
            for a in r["fatal_agents"]:
                fatal[a] = fatal.get(a, 0) + 1
        out[name] = {"n": len(rs), "success": succ, "plan_pass": plan,
                     "gap_plan_minus_success": plan["rate"] - succ["rate"],
                     "valid_item_frac": sum(valid) / len(valid), "valid_ci": bootstrap_ci(valid),
                     "mean_msg_tokens": sum(r["msg_tokens"] for r in rs) / len(rs),
                     "mean_replans": sum(r["replans"] for r in rs) / len(rs),
                     "mean_retrieval_recall": sum(r["retrieval_recall"] for r in rs) / len(rs),
                     "episodes_with_fatal_agent": fatal}
    return out


def paired_compare(records, a: str, b: str, key: str = "success") -> dict:
    """mean(a - b) over seeds present in both configs, with a paired bootstrap interval."""
    ra = {r["seed"]: float(r[key]) for r in records if r["config"] == a}
    rb = {r["seed"]: float(r[key]) for r in records if r["config"] == b}
    seeds = sorted(set(ra) & set(rb))
    if not seeds:
        return {"n": 0}
    m, lo, hi = paired_bootstrap_diff([ra[s] for s in seeds], [rb[s] for s in seeds])
    return {"n": len(seeds), "mean_diff": m, "lo": lo, "hi": hi}


# Ablation grid following the project plan (confirm against the plan before reporting)
def default_grid(noisy: bool = True) -> list:
    base = dict(noisy_eyes=noisy)
    return [
        ExpConfig("full", **base),
        ExpConfig("static_prompt", harness=False, static_rules=("leave_distractors",), **base),
        ExpConfig("comm_pull", comm="pull", **base),
        ExpConfig("comm_init_only", comm="init_only", **base),
        ExpConfig("comm_delta", comm="delta", **base),
        ExpConfig("no_look", allow_look=False, **base),
        ExpConfig("no_challenge", allow_challenge=False, **base),
        ExpConfig("one_eye", n_eyes=1, **base),
        ExpConfig("no_replan", max_replans=0, **base),
        ExpConfig("probe", probe=True, **base),
        ExpConfig("amended", amend=True, **base),
        ExpConfig("arm_noise_high", p_arm_noise=0.2, **base),
    ]


def verify_grid(flaky: float = 0.5) -> list:
    """Plan-verification ablation against the FlakyBrain fixture (loop mechanics; not LLM evidence)."""
    return [ExpConfig(f"verify_{v}", verify=v, flaky=flaky) for v in ("none", "static", "dry_run")] + \
           [ExpConfig("verify_dry_run_0_retries", verify="dry_run", max_plan_retries=0, flaky=flaky)]
