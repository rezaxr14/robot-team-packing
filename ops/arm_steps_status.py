#!/usr/bin/env python3
"""Mechanical gate for the arms-first runbook (docs/AGENT_RUNBOOK.md): which of A1..A10 is really DONE.

Reads ONLY ``reports/`` and ``configs/arm_cert.yaml``; a step is DONE only when its report exists, says PASS, has the
required evidence class and (for A7/A8/A10) the pre-registered number of trials. Stand-ins and plumbing are reported as
STAND_IN, never DONE. Use it before starting a step:

    python ops/arm_steps_status.py                      # table
    python ops/arm_steps_status.py --require A5         # exit 1 unless A1..A5 are all DONE (do not start A6 otherwise)
    python ops/arm_steps_status.py --markdown           # block for docs/STATUS_HONEST.md (generated, never hand-typed)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [f"A{i}" for i in range(1, 11)]
DONE, FAIL, STAND_IN, NOT_RUN = "DONE", "FAIL", "STAND_IN", "NOT_RUN"


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def thresholds(root: Path) -> dict:
    t = {"nominal_success_min": 0.90, "push_perturbed_success_min": 0.80, "min_trials_a7": 200, "min_trials_a10": 500}
    p = root / "configs" / "arm_cert.yaml"
    if p.exists():
        for k in t:
            m = re.search(rf"^\s*{k}:\s*([0-9.]+)", p.read_text(), re.M)
            if m:
                t[k] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    return t


def approvals(root: Path) -> dict:
    """``configs/approvals.json``: {"D-79": "<who/when/quote>"}. A pre-registered criterion changed by a decision counts only
    once the user's approval is recorded here (rule 6); an agent's own statement that the user approved is not enough."""
    d = _load(root / "configs" / "approvals.json")
    return d if isinstance(d, dict) else {}


def _criterion_decisions(rep) -> set:
    return set(re.findall(r'"criterion": "(D-\d+)', json.dumps(rep)))


def _check_approvals(root: Path, rep, s, w):
    if rep is None:
        return s, w
    missing = sorted(d for d in _criterion_decisions(rep) if d not in approvals(root))
    if missing and s == DONE:
        return FAIL, f"pre-registered criterion changed by {missing} without an entry in configs/approvals.json"
    if missing:
        return s, w + f"; ALSO criterion changed by {missing} with no approval in configs/approvals.json"
    return s, w


def _result(rep, need_class=None, why=None):
    if rep is None:
        return NOT_RUN, "no report"
    if need_class and rep.get("evidence_class") != need_class:
        return STAND_IN, f"evidence_class is {rep.get('evidence_class')!r}, need {need_class!r}"
    if rep.get("result") != "PASS":
        bad = [k for k, v in (rep.get("checks") or {}).items() if v is not True]
        return FAIL, f"result={rep.get('result')!r}" + (f"; failing checks: {bad}" if bad else "")
    return DONE, why or "PASS"


def status(root: Path = ROOT) -> dict:
    R, T, out = root / "reports", thresholds(root), {}
    rep = _load(R / "g5_reach.json")
    s, w = _result(rep)
    if s == DONE and not (rep.get("max_pos_err_mm", 99) <= 1.0):
        s, w = FAIL, f"max_pos_err_mm={rep.get('max_pos_err_mm')} > 1.0"
    out["A1"] = (s, w)
    f = root / "docs" / "findings" / "newton_arm_api.md"
    n = f.read_text().count("VERIFIED") if f.exists() else 0
    out["A2"] = (DONE, f"{n} VERIFIED answers") if n >= 4 else (NOT_RUN if not f.exists() else FAIL, f"{n} VERIFIED answers (< 4)")
    out["A3"] = _result(_load(R / "a3_arm_smoke.json"), "SIM")
    out["A4"] = _result(_load(R / "g7_wrench.json"), "SIM")
    rep = _load(R / "g6_newton_push.json")
    s, w = _result(rep, "SIM")
    if rep and s != DONE:
        notrun = [k for k, v in (rep.get("rows") or {}).items() if isinstance(v, dict) and v.get("status") == "not run"]
        if notrun:
            w += f"; rows not run: {notrun}"
    out["A5"] = _check_approvals(root, rep, s, w)
    out["A6"] = _result(_load(R / "a6_scene.json"), "SIM")

    skills7 = ["grasp", "place", "release", "retreat", "weigh", "handover", "push_aside"]

    def regression(name, label):
        rep = _load(R / name)
        if rep is None:
            return NOT_RUN, "no report"
        if rep.get("evidence_class") != "SIM" or "stand" in str(rep.get("kind", "")):
            return STAND_IN, f"kind={rep.get('kind')!r}, evidence_class={rep.get('evidence_class')!r}: {str(rep.get('scope', ''))[:90]}"
        per = rep.get("per_skill")
        if isinstance(per, dict):                                   # authoritative: every skill must meet BOTH bars on its own
            bad = []
            for k in skills7:
                v = per.get(k) or {}
                n, p = v.get("n", 0), v.get("passed", 0)
                if not v.get("graded") or n < T["min_trials_a7"] or p > n or (p / max(n, 1)) < T["nominal_success_min"]:
                    bad.append(f"{k}({p}/{n})")
            if bad:
                return FAIL, f"skills not meeting >= {T['nominal_success_min']} at >= {T['min_trials_a7']} trials: {bad}"
            return _check_approvals(root, rep, DONE, f"all 7 skills >= {T['nominal_success_min']} at >= {T['min_trials_a7']} trials")
        # legacy aggregate: refuse implausible numbers instead of trusting them (a false DONE once came from passed=196,total=7)
        total, passed = rep.get("total", 0), rep.get("passed", 0)
        if not (isinstance(total, int) and isinstance(passed, int) and 0 <= passed <= total and total > 0):
            return FAIL, f"implausible aggregate passed={passed} total={total}"
        if rep.get("trials", 0) < T["min_trials_a7"]:
            return FAIL, f"trials={rep.get('trials')} < pre-registered {T['min_trials_a7']}"
        rate = passed / total
        if rate < T["nominal_success_min"]:
            return FAIL, f"pass rate {rate:.3f} < {T['nominal_success_min']}"
        return DONE, f"{passed}/{total} at {rep['trials']} trials"
    out["A7"] = regression("skill_regression_scripted.json", "scripted")
    out["A8"] = regression("skill_regression_learned.json", "learned")
    out["A9"] = _result(_load(R / "a9_push_robust.json"), "SIM")
    skills = ["grasp", "place", "release", "retreat", "weigh", "handover", "push_aside"]
    cards = {k: _load(R / f"arm_cert_{k}.json") for k in skills}
    miss = [k for k, v in cards.items() if v is None]
    bad = [k for k, v in cards.items() if v is not None and (v.get("result") != "PASS" or v.get("evidence_class") != "SIM")]
    if miss:
        out["A10"] = (NOT_RUN if len(miss) == len(skills) else FAIL, f"cards missing: {miss}")
    elif bad:
        out["A10"] = (FAIL, f"cards not PASS/SIM: {bad}")
    elif not (root / "skill_tables" / "manifest.json").exists():
        out["A10"] = (FAIL, "skill_tables/manifest.json missing")
    else:
        out["A10"] = (DONE, "7 cards PASS, tables pinned")
    return out


def blocked(st: dict, upto: str) -> list:
    return [(k, st[k]) for k in STEPS[: STEPS.index(upto) + 1] if st[k][0] != DONE]


def markdown(st: dict) -> str:
    rows = ["| step | state | detail |", "|---|---|---|"] + [f"| {k} | **{st[k][0]}** | {st[k][1]} |" for k in STEPS]
    return "### Arms-first steps (generated by `ops/arm_steps_status.py` from `reports/`)\n\n" + "\n".join(rows) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require", choices=STEPS, help="exit 1 unless every step up to and including this one is DONE")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    st = status(Path(a.root))
    if a.markdown:
        print(markdown(st))
        return 0
    for k in STEPS:
        print(f"{k:4s} {st[k][0]:9s} {st[k][1]}")
    first = next((k for k in STEPS if st[k][0] != DONE), None)
    print(f"\nnext step to finish: {first or 'none (A1-A10 all DONE)'}")
    if a.require:
        b = blocked(st, a.require)
        if b:
            print(f"BLOCKED: cannot proceed past {a.require}; not DONE: {[k for k, _ in b]}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
