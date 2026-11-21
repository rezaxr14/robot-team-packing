#!/usr/bin/env python3
"""TEMP debug: log return-speed violations with controller limiter state."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtp.control import fr3_kinematics as fk
from rtp.control.compliance import ComplianceConfig, State
from ops.a5_newton_push import SEED, FLANGE_BODY, Harness


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    h = Harness()
    rng = np.random.default_rng(SEED)
    c = h.arm.cfg.compliance or ComplianceConfig()
    segs, t = [], 0.0
    while t < steps * h.dt + 10.0:
        d = rng.uniform(0.3, 1.5)
        v = np.concatenate([rng.normal(size=3), rng.normal(size=3) * 0.1])
        v = v / np.linalg.norm(v[:3]) * rng.uniform(0, 35.0)
        v[3:] = np.clip(v[3:], -1.5, 1.5)
        segs.append((t, t + d, v))
        t += d + rng.uniform(0.5, 3.0)

    def f_ext_at(tt):
        return next((v for a, b, v in segs if a <= tt < b), np.zeros(6))

    h.reset()
    n = 0
    logged = 0
    samples = []
    for k in range(steps):
        applied = f_ext_at(h.clock)
        rec = h.tick(applied, None, FLANGE_BODY)
        info = rec["info"]
        free = (np.linalg.norm(applied[:3]) < c.f_release
                and np.linalg.norm(applied[3:]) < c.m_release)
        v = float(np.linalg.norm(rec["vel"][:3]))
        if info["state"] == State.RETURN and free and v > 1.15 * c.v_return:
            n += 1
            if logged < 30:
                samples.append({
                    "step": k, "v_post": round(v, 4),
                    "v_pre": round(float(np.linalg.norm(info["xdot"][:3])), 4),
                    "tau_speed_norm": round(float(np.linalg.norm(info["tau_speed"])), 2),
                    "tau_norm": round(float(np.linalg.norm(rec["tau"])), 2),
                    "torque_clamped": info["torque_clamped"],
                    "rate_limited": info["rate_limited"],
                    "ee_err": round(float(np.linalg.norm(info["x"][:3] - info["x_ref"][:3])), 4),
                    "f_ext_applied": round(float(np.linalg.norm(applied[:3])), 2),
                    "f_sup": round(float(np.linalg.norm(info["force"][:3])), 2),
                    "flags": info["flags"],
                })
                logged += 1
    print("STEPS", steps, "VIOLATIONS", n)
    for s in samples:
        print(s)


if __name__ == "__main__":
    main()
