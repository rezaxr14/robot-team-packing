"""Abstract-plant simulation of the compliance supervisor (numpy; evidence class SIM-ABSTRACT).

A 6-DoF Cartesian point mass (mass ``m``, rotational inertia ``I``, gravity compensated) with an optional
actuation delay and a scripted external wrench (the "human"). It proves the *behaviour* of
``ComplianceSupervisor`` (push -> yield -> release -> bounded return -> resume) and its invariants. It is NOT
the Newton articulation and NOT hardware; those are separate evidence classes.
"""
from __future__ import annotations

import numpy as np

from .compliance import ComplianceConfig, ComplianceSupervisor, State


def simulate(plan_fn, f_ext_fn, duration: float, *, dt: float = 0.001, mass: float = 3.0, inertia: float = 0.15,
             profile: str = "free_motion", cfg: ComplianceConfig | None = None, delay_ticks: int = 0,
             x0=None, est_noise: float = 0.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sup = ComplianceSupervisor(cfg, profile)
    x = np.asarray(plan_fn(0.0) if x0 is None else x0, float).copy()
    xd = np.zeros(6)
    inv_m = np.array([1 / mass] * 3 + [1 / inertia] * 3)
    n = int(round(duration / dt))
    pend = [np.zeros(6)] * delay_ticks
    log = {k: [] for k in ("t", "x", "xd", "f_ext", "F", "state", "s", "x_ref", "force_clamp", "leash", "speed")}
    events: list = []
    for k in range(n):
        t = k * dt
        fe = np.asarray(f_ext_fn(t), float)
        est = fe + (rng.normal(0.0, est_noise, 6) if est_noise else 0.0)
        out = sup.step(dt, x, xd, est, plan_fn)
        for e in out.events:
            events.append((t, e))
        pend.append(out.force)
        applied = pend.pop(0) if delay_ticks else out.force
        xd = xd + (applied + fe) * inv_m * dt
        x = x + xd * dt
        log["t"].append(t)
        log["x"].append(x.copy()); log["xd"].append(xd.copy()); log["f_ext"].append(fe.copy())
        log["F"].append(out.force.copy()); log["state"].append(out.state.value); log["s"].append(out.s)
        log["x_ref"].append(out.x_ref.copy())
        log["force_clamp"].append(out.flags["force"]); log["leash"].append(out.flags["leash"]); log["speed"].append(out.flags["speed"])
    res = {k: (np.array(v) if k not in ("state",) else np.array(v)) for k, v in log.items()}
    res["events"] = events
    res["yield_count"] = sup.yield_count
    res["cfg"] = sup.cfg
    return res


def hold(_s):
    return np.zeros(6)


def pulse(direction, force: float, t0: float, t1: float, axis_len: int = 6):
    d = np.zeros(axis_len)
    d[:len(direction)] = direction

    def f(t):
        return d * force if t0 <= t < t1 else np.zeros(axis_len)
    return f


def settle_time(res, t_from: float, tol: float = 0.005, target=None) -> float | None:
    tgt = np.zeros(3) if target is None else np.asarray(target)[:3]
    err = np.linalg.norm(res["x"][:, :3] - tgt, axis=1)
    idx = np.where((res["t"] >= t_from) & (err < tol))[0]
    for i in idx:                                            # first time it enters and stays inside
        if np.all(err[i:i + 100] < tol):
            return float(res["t"][i] - t_from)
    return None
