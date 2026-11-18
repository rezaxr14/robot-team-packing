"""G-6: sim impedance vs analytic mass-spring-damper (SIMULATION CROSS-CHECK ONLY).

  python ops/g6_impedance.py --out reports/g6_impedance.json

A measured gate, NOT a safety certification: it checks that the torch L1 law in
``rtp/control/impedance.py`` reproduces the analytic step response of the
1-DoF mass-spring-damper (error <= 5%), is stable at the chosen dt, obeys the
torque-rate limit, and yields/returns under a scripted human push.  Per D-22,
impedance research/redesign and all hardware / L0 / L1 sign-off belong to
Claude; this gate says nothing about real-robot safety.  A value is only
reported if it was actually computed here; the gate FAILS honestly if the
error exceeds 5%.

Exit 0 = PASS, 1 = FAIL.  Always writes the JSON report (with pass/fail).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python ops/g6_impedance.py`

from rtp.control.impedance import (  # noqa: E402
    CartesianImpedance,
    ImpedanceConfig,
    analytic_step_response,
    jacobian_fd,
    push_response,
    rate_limit,
    simulate_step_response,
    validate_dt,
)


def versions() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def step_response_check(cfg: ImpedanceConfig, x_step: float, duration: float) -> dict:
    t, x, _ = simulate_step_response(cfg, x_step, duration)
    x_ref = analytic_step_response(t, x_step, cfg.k_lin, cfg.zeta, cfg.mass)
    abs_err = torch.abs(x - x_ref)
    max_err = float(abs_err.max())
    rms_err = float(torch.sqrt(torch.mean((x - x_ref) ** 2)))
    # Normalise by the step amplitude (the system's natural response scale).
    max_rel = max_err / abs(x_step) * 100.0
    rms_rel = rms_err / abs(x_step) * 100.0
    finite = bool(torch.isfinite(x).all())
    bounded = bool(torch.max(torch.abs(x)) <= 1.5 * abs(x_step) + 1e-9)
    return {
        "x_step_m": x_step,
        "duration_s": duration,
        "dt_s": cfg.dt,
        "samples": int(t.numel()),
        "max_abs_error_m": max_err,
        "rms_abs_error_m": rms_err,
        "max_relative_error_pct": max_rel,
        "rms_relative_error_pct": rms_rel,
        "error_limit_pct": 5.0,
        "all_finite": finite,
        "bounded": bounded,
        "stable": finite and bounded,
        "pass": max_rel <= 5.0 and finite and bounded,
        "x_sim_final": float(x[-1]),
        "x_analytic_final": float(x_ref[-1]),
    }


def dt_stability_check(cfg: ImpedanceConfig) -> dict:
    dt_stable = validate_dt(cfg.k_lin, cfg.zeta, cfg.mass, cfg.dt)
    too_large = None
    try:
        validate_dt(cfg.k_lin, cfg.zeta, cfg.mass, dt_stable)
    except ValueError:
        too_large = True
    else:
        too_large = False
    return {
        "dt_s": cfg.dt,
        "omega_n_radps": (cfg.k_lin / cfg.mass) ** 0.5,
        "hard_stability_limit_s": dt_stable,
        "chosen_dt_accepted": True,
        "unstable_dt_rejected": too_large,
        "pass": too_large,
    }


def torque_rate_check(cfg: ImpedanceConfig) -> dict:
    """Feed a deliberately large target jump and verify the rate limiter binds.

    The low-pass is disabled here so the target step reaches the controller in
    full; otherwise a 0.005 coefficient would mask the rate limiter entirely.
    """
    from dataclasses import replace

    cfg = replace(cfg, lowpass_alpha=1.0)
    ctrl = CartesianImpedance(cfg)
    ctrl.reset(1)
    J = jacobian_fd(torch.zeros(1, 7, dtype=cfg.dtype))
    base = torch.zeros(1, 6, dtype=cfg.dtype)
    t0 = ctrl.compute(torch.zeros(1, 7, dtype=cfg.dtype), torch.zeros(1, 7, dtype=cfg.dtype), base, J=J)
    jump = torch.ones(1, 6, dtype=cfg.dtype) * 0.5
    t1 = ctrl.compute(torch.zeros(1, 7, dtype=cfg.dtype), torch.zeros(1, 7, dtype=cfg.dtype), jump, J=J)
    delta = torch.abs(t1 - t0)
    max_delta_nm = cfg.torque_rate_nm_per_ms * cfg.dt_ms()
    max_seen = float(delta.max())
    return {
        "torque_rate_nm_per_ms": cfg.torque_rate_nm_per_ms,
        "max_delta_per_step_nm": max_delta_nm,
        "max_delta_observed_nm": max_seen,
        "pass": max_seen <= max_delta_nm + 1e-9,
    }


def push_check(cfg: ImpedanceConfig) -> dict:
    m = push_response(cfg, force=10.0, push_duration=1.0, release_duration=2.5)
    m["pass"] = bool(
        m["within_tolerance"]
        and m["settle_time_s"] is not None
        and m["settle_time_s"] <= 2.5
        and m["overshoot_fraction"] <= 0.10
    )
    return m


def gains(cfg: ImpedanceConfig) -> dict:
    return {
        "k_lin_N_per_m": cfg.k_lin,
        "k_rot_Nm_per_rad": cfg.k_rot,
        "k_null_Nm_per_rad": cfg.k_null,
        "zeta": cfg.zeta,
        "mass_kg_toy": cfg.mass,
        "torque_rate_nm_per_ms": cfg.torque_rate_nm_per_ms,
        "lowpass_alpha": cfg.lowpass_alpha,
        "dt_s": cfg.dt,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/g6_impedance.json")
    ap.add_argument("--x-step", type=float, default=0.05, help="step amplitude (m)")
    ap.add_argument("--duration", type=float, default=2.0, help="step-response horizon (s)")
    a = ap.parse_args(argv)

    cfg = ImpedanceConfig()
    step = step_response_check(cfg, a.x_step, a.duration)
    dtchk = dt_stability_check(cfg)
    rate = torque_rate_check(cfg)
    push = push_check(cfg)

    checks = {
        "step_response_error_le_5pct": step["pass"],
        "stable_at_dt": step["stable"],
        "dt_validated": dtchk["pass"],
        "torque_rate_limited": rate["pass"],
        "push_and_return": push["pass"],
    }
    result = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "gate": "G-6",
        "scope": "SIMULATION CROSS-CHECK ONLY -- NOT a safety certification; hardware/L0/L1 sign-off is Claude's (D-22)",
        "model": "1-DoF mass-spring-damper equivalent + batched torch Cartesian law (docs/IMPEDANCE_SPEC.md)",
        "versions": versions(),
        "gains": gains(cfg),
        "step_response": step,
        "dt_stability": dtchk,
        "torque_rate": rate,
        "push_return": push,
        "checks": checks,
        "result": result,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("G-6 impedance (SIMULATION CROSS-CHECK ONLY -- not a safety guarantee)")
    print(f"  versions: torch={report['versions']['torch']} numpy={report['versions']['numpy']} py={report['versions']['python']}")
    print(f"  gains: K_lin={cfg.k_lin} N/m, K_rot={cfg.k_rot} Nm/rad, zeta={cfg.zeta}, mass={cfg.mass} kg, dt={cfg.dt}s")
    print(f"  step response: max rel error={step['max_relative_error_pct']:.4f}% "
          f"(limit 5%), rms={step['rms_relative_error_pct']:.4f}%, stable={step['stable']}")
    print(f"  dt: omega_n={dtchk['omega_n_radps']:.4f} rad/s, stability limit={dtchk['hard_stability_limit_s']:.6f}s, "
          f"unstable dt rejected={dtchk['unstable_dt_rejected']}")
    print(f"  torque rate: max {rate['max_delta_observed_nm']:.4f} Nm/step <= {rate['max_delta_per_step_nm']:.4f} Nm/step")
    print(f"  push: F={push['force_N']} N, deflection={push['steady_deflection_measured_m']:.4f} m "
          f"(expected {push['steady_deflection_expected_m']:.4f}), "
          f"return error={push['return_error_m']:.5f} m, settle={push['settle_time_s']}s, "
          f"overshoot={push['overshoot_fraction']:.4f}")
    for k, v in checks.items():
        print(f"    [{'x' if v else ' '}] {k}")
    print(f"  -> {result}  ({out})")
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
