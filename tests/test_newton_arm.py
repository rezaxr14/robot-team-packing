"""CPU-lightweight tests for :mod:`rtp.control.newton_arm` (no GPU, no Newton).

The controller's math and wiring are exercised with an injected fake FK/Jacobian
provider; the Newton/Warp bridges are covered by the A5 harness
``ops/a5_newton_push.py`` (evidence class SIM).  These tests are FIXTURE/MODEL:
they verify the law, not any hardware or safety property.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rtp.control.compliance import State
from rtp.control.newton_arm import (
    FLANGE_BODY,
    FOREARM_BODY,
    ELBOW_BODY,
    Q_HOME,
    NewtonArmConfig,
    NewtonCompliantArm,
    dynamic_nullspace,
    matrix_from_rotvec,
    rate_limit_numpy,
    rotvec_from_matrix,
)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Fake kinematics provider (numpy only): p = p0 + 0.1*q[:3], R = rotvec(q[3:6]).
# --------------------------------------------------------------------------- #
class FakeKinematics:
    def __init__(self, jac: np.ndarray | None = None) -> None:
        self.p0 = np.array([0.3, 0.0, 0.6])
        self.J = np.asarray(jac if jac is not None else np.eye(6, 7), dtype=float)

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        return self.p0 + 0.1 * q[:3], matrix_from_rotvec(0.1 * q[3:6])

    def jacobian(self, q):
        return self.J


def _arm(jac=None, **cfgkw):
    kin = FakeKinematics(jac)
    cfg = NewtonArmConfig(**cfgkw)
    return NewtonCompliantArm(cfg, kinematics=kin), kin


# --------------------------------------------------------------------------- #
def test_module_imports_without_torch():
    """Import must succeed with torch blocked (mirrors rtp/control/__init__.py)."""
    code = """
import sys
class BlockTorch:
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError("No module named 'torch'", name='torch')
        return None
sys.meta_path.insert(0, BlockTorch())
import rtp.control.newton_arm as m
print('OK', m.Q_HOME, m.FLANGE_BODY)
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_constants_match_calibrated_kinematics():
    from rtp.control.fr3_kinematics import Q_HOME as FK_Q_HOME
    assert Q_HOME == FK_Q_HOME
    assert (FLANGE_BODY, ELBOW_BODY, FOREARM_BODY) == (9, 5, 7)


def test_rotation_helpers_roundtrip():
    rng = np.random.default_rng(123)
    for _ in range(20):
        v = rng.normal(size=3) * 0.4
        R = matrix_from_rotvec(v)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(rotvec_from_matrix(R), v, atol=1e-12)
    assert np.allclose(rotvec_from_matrix(np.eye(3)), np.zeros(3), atol=1e-15)


def test_rotation_helper_half_turn_is_not_identity():
    """A pi rotation must not resolve to zero.

    The FR3 home flange orientation is exactly a pi rotation; the old helper returned 0 for
    it, so the controller's orientation reference silently collapsed to identity and a
    relative plan was applied as an absolute one (the A7 hand came out upside down).
    """
    from rtp.control.fr3_kinematics import Q_HOME as FK_Q_HOME
    from rtp.control.fr3_kinematics import fk_numpy
    _, R_home = fk_numpy(np.asarray(FK_Q_HOME, float))
    v = rotvec_from_matrix(R_home)
    assert np.linalg.norm(v) == pytest.approx(np.pi, abs=1e-9)
    assert np.allclose(matrix_from_rotvec(v), R_home, atol=1e-12)
    # Every axis/sign choice must round-trip; the recovered axis comes from R, so compare
    # the matrices rather than the vector sign.
    for axis in np.eye(3):
        R = matrix_from_rotvec(np.pi * axis)
        assert np.allclose(matrix_from_rotvec(rotvec_from_matrix(R)), R, atol=1e-12)
        R = matrix_from_rotvec(-np.pi * axis)
        assert np.allclose(matrix_from_rotvec(rotvec_from_matrix(R)), R, atol=1e-12)


def test_rate_limit_numpy_matches_impedance_torch():
    torch = pytest.importorskip("torch")
    from rtp.control.impedance import ImpedanceConfig, rate_limit
    cfg = ImpedanceConfig()
    md = cfg.torque_rate_nm_per_ms * cfg.dt_ms()
    rng = np.random.default_rng(7)
    for _ in range(50):
        prev, val = rng.uniform(-5, 5, size=2)
        got = rate_limit_numpy(np.array([val]), np.array([prev]), md)
        want = rate_limit(torch.tensor([val], dtype=torch.float64),
                          torch.tensor([prev], dtype=torch.float64), cfg)
        assert np.allclose(got, want.numpy(), atol=1e-12)
    # first command passes through unclamped
    assert rate_limit_numpy(np.array([99.0]), None, 1.0)[0] == 99.0


def test_k_null_is_bounded():
    with pytest.raises(ValueError):
        NewtonArmConfig(k_null=20.1)
    with pytest.raises(ValueError):
        NewtonArmConfig(k_null=-1.0)
    assert NewtonArmConfig(k_null=20.0).k_null == 20.0


def test_real_fr3_kinematics_wiring_on_cpu():
    """Controller runs with the A1-calibrated torch FK/Jacobian on CPU."""
    torch = pytest.importorskip("torch")
    del torch
    from rtp.control import fr3_kinematics
    arm = NewtonCompliantArm(kinematics=fr3_kinematics)
    q = np.asarray(Q_HOME, dtype=float)
    tau, info = arm.step(q, np.zeros(7), np.zeros(6))
    assert tau.shape == (7,) and np.all(np.isfinite(tau))
    assert info["J"].shape == (6, 7)
    pos, _ = fr3_kinematics.fk_numpy(q)
    assert np.allclose(info["x"][:3], pos, atol=1e-9)
    assert info["state"] == State.TRACK


def test_rotation_error_is_relative_rrT():
    """x[3:] must be rotvec(R @ R_ref^T), never an absolute/difference-of-quat."""
    arm, kin = _arm()
    qh = np.asarray(Q_HOME, dtype=float)
    ref = arm._reference()
    q = qh.copy()
    q[3] += 0.4                       # rotate about the fake's local x axis
    x, xdot, J = arm.task_state(q, np.zeros(7))
    _, R_ref = kin.fk(qh)
    _, R = kin.fk(q)
    expected = rotvec_from_matrix(np.asarray(R) @ np.asarray(R_ref).T)
    assert np.allclose(x[3:], expected, atol=1e-12)
    assert ref.shape == (6,)
    assert xdot.shape == (6,) and J.shape == (6, 7)


def test_nullspace_projector_annihilates_task_torques():
    rng = np.random.default_rng(11)
    J = rng.normal(size=(6, 7))
    pinv = np.linalg.pinv(J, rcond=1e-3)
    N = np.eye(7) - J.T @ pinv.T
    assert np.allclose(N @ J.T, np.zeros((7, 6)), atol=1e-10)


def _random_spd(rng, n=7):
    A = rng.normal(size=(n, n))
    return A @ A.T + 0.5 * np.eye(n)


def test_dynamic_nullspace_random_spd_full_rank_j():
    """(a) N@N ~ N and J Minv N ~ 0 (NOT J Minv N.T) for random SPD M, full-rank J."""
    rng = np.random.default_rng(20261120)
    M = _random_spd(rng)
    J = rng.normal(size=(6, 7))
    assert np.linalg.matrix_rank(J) == 6
    N = dynamic_nullspace(J, M)
    Minv = np.linalg.inv(M)
    assert np.max(np.abs(N @ N - N)) < 1e-8
    assert np.max(np.abs(J @ Minv @ N)) < 1e-8
    # NOTE: J Minv N.T is deliberately NOT tested: the identity holds for N only.


def test_dynamic_nullspace_rejects_non_spd():
    """(b) a non-SPD M (and a wrong-shape M) raises ValueError."""
    J = np.eye(6, 7)
    M_indef = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="positive definite"):
        dynamic_nullspace(J, M_indef)
    M_singular = np.ones((7, 7))
    with pytest.raises(ValueError, match="positive definite"):
        dynamic_nullspace(J, M_singular)
    with pytest.raises(ValueError, match="expected M 7x7"):
        dynamic_nullspace(J, np.eye(6))


def test_step_mass_matrix_dynamic_and_plain_fallback():
    """(c) valid M selects the dynamic projector; None / invalid M fall back to plain."""
    rng = np.random.default_rng(77)
    J = rng.normal(size=(6, 7))
    arm, _ = _arm(jac=J, friction_comp=False, speed_limiter_gain=0.0)
    q = np.asarray(Q_HOME, dtype=float) + 0.01
    dq = rng.normal(size=7) * 0.01
    f_ext = rng.normal(size=6)
    M = _random_spd(rng)

    _, plain = arm.step(q, dq, f_ext)
    assert plain["nullspace"] == "plain"
    Pinv = np.linalg.pinv(J, rcond=1e-3)
    N_plain = np.eye(7) - J.T @ Pinv.T
    assert np.allclose(plain["tau_null"],
                       N_plain @ (arm.cfg.k_null * (arm.q_home - q)
                                  - 2.0 * np.sqrt(arm.cfg.k_null) * dq), atol=1e-10)

    _, dyn = arm.step(q, dq, f_ext, mass_matrix=M)
    assert dyn["nullspace"] == "dynamic"
    N_dyn = dynamic_nullspace(J, M)
    assert np.allclose(dyn["tau_null"],
                       N_dyn @ (arm.cfg.k_null * (arm.q_home - q)
                                - 2.0 * np.sqrt(arm.cfg.k_null) * dq), atol=1e-10)
    # an invalid mass matrix is ignored, not raised, so legacy callers keep working
    _, bad = arm.step(q, dq, f_ext, mass_matrix=np.diag([-1.0] + [1.0] * 6))
    assert bad["nullspace"] == "plain"


def test_torque_is_jtf_plus_nullspace_plus_gravity_with_clamps():
    rng = np.random.default_rng(5)
    J = rng.normal(size=(6, 7))
    arm, kin = _arm(jac=J, tau_max=8.0, torque_rate_nm_per_ms=1000.0,
                    friction_comp=False, speed_limiter_gain=0.0)
    q = np.asarray(Q_HOME, dtype=float) + 0.01
    dq = rng.normal(size=7) * 0.01
    g = rng.normal(size=7)
    fext = rng.normal(size=6)
    tau, info = arm.step(q, dq, fext, gravity_comp=g)
    assert np.allclose(info["tau_task"], J.T @ info["force"], atol=1e-12)
    kn = arm.cfg.k_null
    pinv = np.linalg.pinv(J, rcond=1e-3)
    N = np.eye(7) - J.T @ pinv.T
    want_null = N @ (kn * (arm.q_home - q) - 2.0 * np.sqrt(kn) * dq)
    assert np.allclose(info["tau_null"], want_null, atol=1e-10)
    assert np.allclose(info["tau_raw"], info["tau_task"] + info["tau_null"] + g, atol=1e-12)
    assert np.all(np.abs(tau) <= 8.0 + 1e-12)
    assert info["torque_clamped"] == bool(np.any(np.abs(info["tau_raw"]) > 8.0))


def test_torque_rate_limit_is_applied_per_tick():
    arm, _ = _arm(tau_max=1e6, torque_rate_nm_per_ms=1.0)
    q = np.asarray(Q_HOME, dtype=float)
    # first command unclamped (previous is None), then at most 1 Nm per tick.
    # Move the (fake) plant 1 rad on joint 0 -> task error 0.1 m -> large torque.
    t0, _ = arm.step(q, np.zeros(7), np.zeros(6))
    q2 = q.copy()
    q2[0] += 1.0
    t1, info1 = arm.step(q2, np.zeros(7), np.zeros(6))
    assert info1["rate_limited"] is True
    assert np.all(np.abs(t1 - t0) <= 1.0 + 1e-12)
    t2, _ = arm.step(q2, np.zeros(7), np.zeros(6))
    assert np.all(np.abs(t2 - t1) <= 1.0 + 1e-12)


def test_residual_wrench_compensation_is_bounded_and_configurable():
    """Residual feedback is off by default and never exceeds the force caps."""
    arm, _ = _arm()
    assert arm.cfg.residual_comp_gain == 0.0
    cfg = NewtonArmConfig(residual_comp_gain=4.0, residual_beta=0.1)
    arm2 = NewtonCompliantArm(cfg, kinematics=FakeKinematics())
    q = np.asarray(Q_HOME, dtype=float)
    _, info = arm2.step(q, np.zeros(7), np.array([35.0, 0, 0, 1.5, 0, 0]))
    from rtp.control.compliance import ComplianceConfig
    caps = ComplianceConfig()
    assert np.all(np.abs(info["force_used"][:3]) <= caps.f_lin_max + 1e-9)
    assert np.all(np.abs(info["force_used"][3:]) <= caps.m_max + 1e-9)
    with pytest.raises(ValueError):
        NewtonArmConfig(residual_comp_gain=-1.0)


def test_speed_limiter_only_acts_when_returning_or_released():
    """The secondary limiter is inactive in TRACK with no push, active in RETURN."""
    arm, _ = _arm()
    assert arm.cfg.speed_limiter_gain > 0
    q = np.asarray(Q_HOME, dtype=float)
    _, info = arm.step(q, np.zeros(7), np.zeros(6))
    assert np.all(info["tau_speed"] == 0.0)
    # force a RETURN state by pushing then releasing on the fake plant
    for _ in range(300):
        arm.step(q, np.zeros(7), np.array([20.0, 0, 0, 0, 0, 0]))
    for _ in range(400):
        _, info = arm.step(q, np.zeros(7), np.zeros(6))
        if info["state"] == State.RETURN:
            break
    assert info["state"] == State.RETURN
    # with zero velocity there is no excess speed, so still no torque
    assert np.all(info["tau_speed"] == 0.0)


def test_plan_clock_freezes_on_push_and_resumes():
    """The supervisor's plan clock and state machine are wired into the arm."""
    arm, _ = _arm()
    q = np.asarray(Q_HOME, dtype=float)
    home_pos = arm._reference()[:3]

    def plan(s):
        return np.concatenate([home_pos + np.array([0.05 * s, 0, 0]), np.zeros(3)])

    states = []
    s_push = []
    for k in range(500):
        _, info = arm.step(q, np.zeros(7), np.array([20.0, 0, 0, 0, 0, 0]), plan)
        states.append(info["state"])
        s_push.append(info["s"])
    assert all(s == State.YIELD for s in states)
    assert max(s_push) - min(s_push) < 1e-12      # exactly frozen in YIELD

    seen = set()
    for k in range(4000):
        _, info = arm.step(q, np.zeros(7), np.zeros(6), plan)
        seen.add(info["state"])
        if k > 2500:
            break
    assert State.RETURN in seen
    assert State.TRACK in seen   # TRACK resumes after the return settles
