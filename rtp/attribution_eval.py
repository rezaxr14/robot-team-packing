"""Validate the rule-based Eye/Brain/Arm failure attribution against human labels.

Workflow: sample_for_labeling -> a person fills `human_agent` (Eye|Brain|Arm|Mixed|Unclear) -> agreement().
The story shows what the Eyes reported versus ground truth, so a labeler can judge without reading code.
"""
from __future__ import annotations
import csv
import json
import random

LABELS = ("Eye", "Brain", "Arm", "Mixed", "Unclear")


def episode_story(world, spec, res) -> str:
    lines = [f"INSTRUCTION: {spec.text()}", f"OUTCOME: success={res.success} valid={res.n_valid}/{res.n_items} "
             f"violations={[v['rule'] + ':' + v['detail'] for v in res.violations]} crushed={res.crushed} missing={res.missing}"]
    diffs = []
    for n, o in sorted(world.objs.items()):
        rep = res.scene_final.get(n)
        if rep is None:
            diffs.append(f"{n}: never reported by any Eye (true class {o.cls})")
            continue
        bits = []
        if rep["cls"] != o.cls:
            bits.append(f"class reported {rep['cls']} vs true {o.cls}")
        if abs(rep["mass_guess"] - o.mass) / max(0.05, o.mass) > 0.15:
            bits.append(f"mass reported {rep['mass_guess']} vs true {o.mass:.2f}")
        if abs(rep["fragility_guess"] - o.fragility) > 0.15:
            bits.append(f"fragility reported {rep['fragility_guess']} vs true {o.fragility:.2f}")
        if bits:
            diffs.append(f"{n}: " + "; ".join(bits))
    lines.append("EYE ERRORS (final scene vs truth): " + (" | ".join(diffs) if diffs else "none"))
    lines.append("FAILURES (auto-attributed): " + json.dumps(res.failures, default=str))
    lines.append("EXECUTED: " + " ".join(f"{e['task']['id']}{'+' if e['ok'] else '-' + e['cause']}" for e in res.executed))
    return "\n".join(lines)


def sample_for_labeling(records: list, n: int, seed: int = 0) -> list:
    """Sample failed episodes that carry a story; one row per episode."""
    pool = [r for r in records if not r["success"] and r.get("story")]
    random.Random(seed).shuffle(pool)
    rows = []
    for r in pool[:n]:
        rows.append({"config": r["config"], "seed": r["seed"], "story": r["story"],
                     "auto_agents": "|".join(r["fatal_agents"]), "human_agent": ""})
    return rows


def write_label_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "seed", "auto_agents", "human_agent", "story"])
        w.writeheader()
        w.writerows(rows)


def _auto_label(auto: str) -> str:
    parts = [p for p in auto.split("|") if p]
    return parts[0] if len(parts) == 1 else ("Mixed" if parts else "Unclear")


def agreement(path) -> dict:
    rows = [r for r in csv.DictReader(open(path)) if r["human_agent"].strip()]
    if not rows:
        return {"n": 0}
    for r in rows:
        if r["human_agent"] not in LABELS:
            raise ValueError(f"human_agent must be one of {LABELS}, got {r['human_agent']!r}")
    a = [_auto_label(r["auto_agents"]) for r in rows]
    h = [r["human_agent"] for r in rows]
    n = len(rows)
    acc = sum(x == y for x, y in zip(a, h)) / n
    pe = sum((a.count(l) / n) * (h.count(l) / n) for l in LABELS)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 1.0
    conf = {}
    for x, y in zip(a, h):
        conf.setdefault(x, {})
        conf[x][y] = conf[x].get(y, 0) + 1
    return {"n": n, "accuracy": acc, "cohen_kappa": kappa, "confusion_auto_to_human": conf}
