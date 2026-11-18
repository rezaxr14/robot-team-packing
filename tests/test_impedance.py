"""Torch tests for the impedance SIMULATION CROSS-CHECK (not a safety gate).

These verify the authored math in ``rtp/control/impedance.py``: damping formula,
torque-rate clamp, analytic step-response error, push-and-return, batched
shapes, and honest rejection of an unstable dt.  They make no claim about real
robot safety; hardware/L0/L1 sign-off is Claude's (D-22).
"""
import numpy as np
import pytest
import torch

from rtp.control.impedance import (
    CartesianImpedance,
    ImpedanceConfig,
    analytic_step_response,
    damping,
    forward_kinematics,
    jacobian_autograd,
    jacobian_fd,
    push_response,
    rate_limit,
    simulate_step_response,
    validate_dt,
)


def test_damping_formula():
    # D = 2 zeta sqrt(K); with zeta=1 this is exactly 2 sqrt(K) [spec fallback].
    d = damping(150.0, 1.0)
    assert torch.allclose(d, torch.tensor(2.0 * np.sqrt(150.0)), atol=1e-9)
    # zeta scales linearly.
    assert torch.allclose(damping(150.0, 0.9), 0.9 * damping(150.0, 1.0), atol=1e-9)
    # Inertia-aware form D = 2 zeta sqrt(K * Lambda).
    d_lam = damping(150.0, 1.0, inertia=2.0)
    assert torch.allclose(d_lam, torch.tensor(2.0 * np.sqrt(150.0 * 2.0)), atol=1e-9)
    # Vector K (diagonal) stays element-wise.
    K = torch.tensor([150.0, 150.0, 150.0, 10.0, 10.0, 10.0], dtype=torch.float64)
    dv = damping(K, 1.0)
    assert dv.shape == (6,)
    assert torch.allclose(dv[:3], torch.tensor(2.0 * np.sqrt(150.0)), atol=1e-9)
    with pytest.raises(ValueError):
        damping(150.0, -0.1)


def test_torque_rate_clamp():
    from dataclasses import replace

    # Disable the target low-pass so the step fully reaches the rate limiter.
    cfg = replace(ImpedanceConfig(), lowpass_alpha=1.0)
    ctrl = CartesianImpedance(cfg)
    ctrl.reset(1)
    q = torch.zeros(1, 7, dtype=cfg.dtype)
    dq = torch.zeros(1, 7, dtype=cfg.dtype)
    J = jacobian_fd(q)
    tau0 = ctrl.compute(q, dq, torch.zeros(1, 6, dtype=cfg.dtype), J=J)
    # A large target jump would command a big torque change; rate limit must bind.
    tau1 = ctrl.compute(q, dq, torch.ones(1, 6, dtype=cfg.dtype) * 0.5, J=J)
    max_delta = cfg.torque_rate_nm_per_ms * cfg.dt_ms()
    assert torch.all(torch.abs(tau1 - tau0) <= max_delta + 1e-9)
    # Direct primitive: first value passes; subsequent jump is clipped to +/- max_delta.
    first = rate_limit(torch.tensor(10.0), None, cfg)
    second = rate_limit(torch.tensor(10.0), first, cfg)
    third = rate_limit(torch.tensor(-10.0), second, cfg)
    assert float(first) == 10.0
    assert abs(float(second - first)) <= max_delta + 1e-9
    assert abs(float(third - second)) <= max_delta + 1e-9


def test_step_response_error_within_5pct():
    cfg = ImpedanceConfig()
    x_step = 0.05
    t, x, _ = simulate_step_response(cfg, x_step, 2.0)
    analytic = analytic_step_response(t, x_step, cfg.k_lin, cfg.zeta, cfg.mass)
    rel = float(torch.max(torch.abs(x - analytic))) / abs(x_step) * 100.0
    assert torch.isfinite(x).all()
    assert rel <= 5.0, f"step response error {rel:.4f}% exceeds 5%"
    assert torch.allclose(x[-1], analytic[-1], atol=1e-4)


def test_step_response_handles_all_damping_regimes():
    # Analytic closed form must be finite and settle to x_step in each regime.
    t = torch.linspace(0, 3.0, 3001, dtype=torch.float64)
    for zeta in (0.7, 1.0, 1.3):
        x = analytic_step_response(t, 0.05, 150.0, zeta, 2.0)
        assert torch.isfinite(x).all()
        assert abs(float(x[-1]) - 0.05) < 1e-4


def test_push_returns_within_tolerance():
    cfg = ImpedanceConfig()
    m = push_response(cfg, force=10.0, push_duration=1.0, release_duration=2.5)
    assert m["within_tolerance"], m
    assert m["return_error_m"] <= cfg.return_tol
    assert m["settle_time_s"] is not None and m["settle_time_s"] <= 2.5
    assert m["overshoot_fraction"] <= 0.10
    # Steady deflection ~= F/K (within the spec's +/-10% style band).
    assert abs(m["deflection_error_pct"]) < 10.0, m


def test_batch_shape_and_finite():
    cfg = ImpedanceConfig()
    B = 4
    torch.manual_seed(0)
    q = torch.randn(B, 7, dtype=cfg.dtype) * 0.2
    dq = torch.randn(B, 7, dtype=cfg.dtype) * 0.1
    target = torch.zeros(B, 6, dtype=cfg.dtype)
    p, R = forward_kinematics(q)
    assert p.shape == (B, 3) and R.shape == (B, 3, 3)
    J = jacobian_fd(q)
    assert J.shape == (B, 6, 7)
    ctrl = CartesianImpedance(cfg)
    ctrl.reset(B)
    tau = ctrl.compute(q, dq, target, J=J)
    assert tau.shape == (B, 7)
    assert torch.isfinite(tau).all()
    # Coriolis/gravity hooks are additive and keep shape.
    tau2 = ctrl.compute(q, dq, target, J=J,
                        coriolis=torch.ones(B, 7, dtype=cfg.dtype),
                        gravity=torch.ones(B, 7, dtype=cfg.dtype) * 2.0)
    assert tau2.shape == (B, 7) and torch.isfinite(tau2).all()


def test_jacobian_fd_matches_autograd():
    torch.manual_seed(1)
    q = torch.randn(7, dtype=torch.float64) * 0.2
    Jfd = jacobian_fd(q.unsqueeze(0)).squeeze(0)
    Jag = jacobian_autograd(q)
    assert torch.allclose(Jfd, Jag, atol=1e-5), float((Jfd - Jag).abs().max())


def test_unstable_dt_rejected():
    cfg = ImpedanceConfig()
    wn = (cfg.k_lin / cfg.mass) ** 0.5
    # Chosen dt is accepted.
    validate_dt(cfg.k_lin, cfg.zeta, cfg.mass, cfg.dt)
    limit = 2.0 / wn
    # A dt beyond the hard stability limit must raise, not silently pass.
    with pytest.raises(ValueError, match="unstable"):
        validate_dt(cfg.k_lin, cfg.zeta, cfg.mass, limit * 1.5)
    # And the step simulator refuses it too.
    with pytest.raises(ValueError, match="unstable"):
        simulate_step_response(
            ImpedanceConfig(dt=limit * 1.5), 0.05, 0.1
        )
