#!/usr/bin/env python3
"""G-6b: push-yield-return behaviour of the compliance supervisor on an ABSTRACT plant (numpy).

Evidence class SIM-ABSTRACT: a 6-DoF Cartesian point mass, not the Newton articulation and not hardware.
It measures the rows of docs/IMPEDANCE_SPEC.md section 6 that do not need an articulated model and writes
``reports/g6b_compliance.json``. Not a safety certification.

    python ops/g6b_compliance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtp.control.compliance import PROFILES, ComplianceConfig                     # noqa: E402
from rtp.control.compliance_sim import hold, pulse, settle_time, simulate         # noqa: E402

K = PROFILES["free_motion"][0]


def main() -> int:
    rows, ok_all = {}, True

    defl = []
    for F in (5.0, 10.0, 20.0, 30.0):
        for ax, sg in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
            d = np.zeros(3); d[ax] = sg
            r = simulate(hold, pulse(d, F, 0.2, 2.0), 2.0)
            delta = float(np.linalg.norm(r["x"][int(1.9 / 0.001), :3]))
            defl.append({"F": F, "axis": ax, "sign": sg, "delta_m": delta, "F_over_K_m": F / K,
                         "rel_err": abs(delta - F / K) / (F / K)})
    rows["steady_deflection"] = {"max_rel_err": max(x["rel_err"] for x in defl), "limit": 0.10, "cases": len(defl),
                                 "pass": max(x["rel_err"] for x in defl) <= 0.10}

    r = simulate(hold, pulse([1, 0, 0], 20.0, 0.2, 1.2), 8.0)
    cfg = r["cfg"]
    after = slice(int(1.2 / 0.001), None)
    speed = np.linalg.norm(r["xd"][after, :3], axis=1)
    ts = settle_time(r, 1.2)
    rms = float(np.sqrt(np.mean(np.sum(r["x"][int(6.0 / 0.001):, :3] ** 2, axis=1))))
    over = float(max(0.0, -r["x"][after, 0].min()))
    rows["release_return"] = {"settle_s": ts, "limit_s": 2.5, "peak_return_speed_m_s": float(speed.max()),
                              "speed_limit_m_s": 1.1 * cfg.v_return, "overshoot_m": over, "overshoot_limit_m": 0.01,
                              "rms_after_6s_m": rms, "rms_limit_m": 0.002,
                              "pass": bool(ts is not None and ts <= 2.5 and speed.max() <= 1.1 * cfg.v_return
                                           and over <= 0.01 and rms <= 0.002)}

    plan = lambda s: np.array([min(0.1 * s, 0.3), 0, 0, 0, 0, 0])
    r = simulate(plan, pulse([-1, 0, 0], 25.0, 1.0, 2.0), 9.0)
    held = [i for i, st in enumerate(r["state"]) if st in ("YIELD", "RETURN")]
    frozen = float(np.ptp(r["s"][held])) if held else None
    end_err = float(np.linalg.norm(r["x"][-1, :3] - np.array([0.3, 0, 0])))
    rows["plan_pause_resume"] = {"plan_clock_range_while_held": frozen, "final_position_error_m": end_err,
                                 "events": [e for _t, e in r["events"]],
                                 "pass": bool(held and frozen == 0.0 and end_err <= 0.003)}

    r = simulate(lambda s: np.array([min(0.15 * s, 0.3), 0, 0, 0, 0, 0]), lambda t: np.zeros(6), 3.0)
    rows["moving_plan_no_false_yield"] = {"yield_count": r["yield_count"], "pass": r["yield_count"] == 0}

    rng = np.random.default_rng(3)
    segs, t = [], 0.0
    while t < 120.0:
        d = rng.uniform(0.3, 1.5)
        v = np.concatenate([rng.normal(size=3), rng.normal(size=3) * 0.1])
        v = v / np.linalg.norm(v[:3]) * rng.uniform(0, 35.0)
        v[3:] = np.clip(v[3:], -1.5, 1.5)
        segs.append((t, t + d, v)); t += d + rng.uniform(0.5, 3.0)
    f = lambda tt: next((v for a, b, v in segs if a <= tt < b), np.zeros(6))
    r = simulate(lambda s: np.array([0.05 * np.sin(0.5 * s), 0, 0, 0, 0, 0]), f, 120.0)
    c = r["cfg"]
    gap = np.linalg.norm(r["x"][:, :3] - r["x_ref"][:, :3], axis=1)
    held = np.array([s in ("YIELD", "RETURN") for s in r["state"]])
    slack = 2 * np.linalg.norm(r["xd"][:, :3], axis=1) * 0.001
    free = np.linalg.norm(r["f_ext"][:, :3], axis=1) < c.f_release
    ret = np.array([s == "RETURN" for s in r["state"]]) & free
    inv = {"ticks": int(len(r["t"])), "pushes": r["yield_count"],
           "max_force_N": float(np.linalg.norm(r["F"][:, :3], axis=1).max()), "force_limit_N": c.f_lin_max,
           "max_moment_Nm": float(np.linalg.norm(r["F"][:, 3:], axis=1).max()), "moment_limit_Nm": c.m_max,
           "leash_violations": int((gap[held] > c.delta_max + slack[held] + 1e-9).sum()),
           "max_return_speed_m_s": float(np.linalg.norm(r["xd"][ret, :3], axis=1).max()) if ret.any() else None,
           "faults": [e for _t, e in r["events"] if e == "FAULT"]}
    inv["pass"] = bool(inv["max_force_N"] <= c.f_lin_max + 1e-9 and inv["max_moment_Nm"] <= c.m_max + 1e-9
                       and inv["leash_violations"] == 0 and not inv["faults"]
                       and (inv["max_return_speed_m_s"] is None or inv["max_return_speed_m_s"] <= 1.15 * c.v_return))
    rows["invariants_random_pushes"] = inv

    pay = {}
    for m in (3.0, 4.5, 6.0):
        r = simulate(hold, pulse([1, 0, 0], 20.0, 0.2, 1.2), 10.0, mass=m)
        pay[str(m)] = {"settle_s": settle_time(r, 1.2), "final_state": str(r["state"][-1])}
    rows["payload_mass_kg_vs_3kg_identified"] = {"cases": pay, "pass": all(v["settle_s"] is not None and v["final_state"] == "TRACK" for v in pay.values())}

    r = simulate(hold, pulse([0, 1, 0], 15.0, 0.2, 1.2), 10.0, delay_ticks=3, est_noise=0.5)
    rows["latency_3ms_noise_0p5N"] = {"max_abs_x_m": float(np.abs(r["x"]).max()), "settle_s": settle_time(r, 1.2, tol=0.008),
                                      "pass": bool(np.isfinite(r["x"]).all() and np.abs(r["x"]).max() < 0.3 and settle_time(r, 1.2, tol=0.008) is not None)}

    r = simulate(hold, pulse([1, 0, 0], 120.0, 0.5, 0.52), 3.0)
    rows["impact_120N_20ms_controlled_stop"] = {"fault": "FAULT" in [e for _t, e in r["events"]],
                                                "final_speed_m_s": float(np.linalg.norm(r["xd"][-1, :3])),
                                                "pass": bool("FAULT" in [e for _t, e in r["events"]] and np.linalg.norm(r["xd"][-1, :3]) < 0.01)}
    for v in rows.values():
        ok_all &= bool(v["pass"])
    rep = {"gate": "G-6b", "evidence_class": "SIM-ABSTRACT (6-DoF point mass, numpy)", "pass": ok_all,
           "config": {k: getattr(ComplianceConfig(), k) for k in ComplianceConfig.__dataclass_fields__},
           "profile": "free_motion", "K_lin_N_per_m": K, "rows": rows,
           "not_covered": ["Newton articulation (null-space/elbow push, joint torque rate limit)", "hardware",
                           "ISO 10218-2:2025 contact limits (placeholders in config)", "safety certification"]}
    out = ROOT / "reports" / "g6b_compliance.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v["pass"] for k, v in rows.items()}, indent=2))
    print("G-6b:", "PASS (abstract plant)" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
