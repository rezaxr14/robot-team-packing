"""Batched torch Cartesian impedance -- SIMULATION CROSS-CHECK ONLY.

.. warning::
   **SIMULATION CROSS-CHECK ONLY; NOT A SAFETY CERTIFICATION.**

   This module re-implements the already-authored control law in
   ``docs/IMPEDANCE_SPEC.md`` (Frankta reference structure [V:S-20]) so the
   torch math can be checked against an analytic mass-spring-damper and a
   scripted human push *in simulation*.  It is deliberately self-contained and
   CPU-testable; it is NOT the deployed L1 controller, does not implement the
   full clamp/leash/state-machine stack, and makes NO claim that a real arm is
   safe around people.  Per decision D-22, impedance research/redesign and all
   hardware / L0 / L1 sign-off are owned by **Claude**.  The numeric defaults
   below are starting values from the spec, not validated safety limits.

What is implemented (the authored spec, verbatim math):

* ``tau = J^T (-K e - D J dq) + (I - J^T J^+^T) tau_null + coriolis (+ gravity)``
* ``tau_null = k_n (q_n - q) - 2 sqrt(k_n) dq``  (weak nullspace PD)
* ``D = 2 zeta sqrt(K)`` (inertia-free fallback) or ``2 zeta sqrt(K Lambda)``
  when an inertia is supplied, matching spec section 3.2.
* torque-rate saturation (1 Nm/ms default) and target low-pass (0.005 default).
* configurable K (150 N/m / 10 Nm/rad defaults), zeta, dt.

Supporting pieces for the cross-check (not part of the deployed law):

* forward kinematics and a Jacobian for the calibrated FR3 flange
  (``ImpedanceConfig.kinematics = "fr3"``, the default after A1; ``"toy"`` keeps
  the original generic link chain), with torch autograd or central differences
  as independent cross-checks;
* a 1-DoF mass-spring-damper equivalent with its closed-form step response, for
  the G-6 analytic-vs-sim comparison;
* ``push_response`` that scripted-pushes that equivalent and reports
  displacement / return-to-pose metrics.

All tensors are float64 by default (physics, not learning); the controller
accepts any float dtype as long as inputs and buffers agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

__all__ = [
    "ImpedanceConfig",
    "CartesianImpedance",
    "damping",
    "rate_limit",
    "forward_kinematics",
    "jacobian_fd",
    "jacobian_autograd",
    "analytic_step_response",
    "simulate_step_response",
    "push_response",
    "validate_dt",
]

# Number of joints of the self-contained toy arm (a 7-DoF serial chain, like
# an FR3, but with generic link lengths -- NOT a calibrated FR3 model).
NQ = 7
# Cartesian dimension of the task (position + rotation vector).
NTASK = 6
# Link length of each toy-arm segment (metres) -- arbitrary, for kinematics only.
_LINK_LENGTH = 0.15
# Alternating joint axes (0=x, 1=y, 2=z) of the toy arm.
_AXES = (0, 1, 2, 0, 1, 2, 0)


@dataclass
class ImpedanceConfig:
    """Starting parameters from ``docs/IMPEDANCE_SPEC.md`` section 4 (defaults).

    These are **[D] starting values**, not measured/validated safety limits.
    ``mass`` is a toy stand-in for the Cartesian inertia Lambda; real hardware
    identification and sign-off belong to Claude.
    """

    #: ``"fr3"`` uses the calibrated FR3 flange model from
    #: :mod:`rtp.control.fr3_kinematics` (verified against Newton in A1);
    #: ``"toy"`` keeps the original generic chain for legacy regression tests.
    kinematics: str = "fr3"
    k_lin: float = 150.0            # N/m, translation (spec example [V:S-20])
    k_rot: float = 10.0             # Nm/rad, rotation (spec example [V:S-20])
    k_null: float = 5.0             # Nm/rad, nullspace PD (spec: <= 20 so elbow yields)
    zeta: float = 1.0               # damping ratio (critical; window 0.9-1.2)
    dt: float = 0.001               # s, 1 kHz control loop
    mass: float = 2.0               # kg, toy Cartesian inertia Lambda (placeholder)
    torque_rate_nm_per_ms: float = 1.0   # Nm/ms torque-rate saturation [V:S-20]
    lowpass_alpha: Optional[float] = 0.005  # target low-pass coefficient (None/1 => off)
    f_lin_max: float = 40.0         # N, translational force cap (placeholder)
    m_max: float = 4.0              # Nm, rotational torque cap (placeholder)
    tau_max: float = 50.0           # Nm, per-joint torque saturation (placeholder)
    return_tol: float = 0.005       # m, return-to-pose tolerance (spec epsilon_e = 3mm-ish)
    dtype: torch.dtype = torch.float64
    q_rest: torch.Tensor = field(default=None)  # type: ignore[assignment]  # nullspace rest posture

    def k_diag(self) -> torch.Tensor:
        """6-D diagonal stiffness ``[K_lin x3, K_rot x3]``."""
        return torch.tensor(
            [self.k_lin, self.k_lin, self.k_lin, self.k_rot, self.k_rot, self.k_rot],
            dtype=self.dtype,
        )

    def dt_ms(self) -> float:
        return self.dt * 1000.0

    def __post_init__(self) -> None:
        if self.kinematics not in ("fr3", "toy"):
            raise ValueError(f"kinematics must be 'fr3' or 'toy', got {self.kinematics!r}")


# --------------------------------------------------------------------------- #
# Damping and rate limiting (shared primitives used by the gate and tests).
# --------------------------------------------------------------------------- #
def damping(K, zeta: float, inertia=None) -> torch.Tensor:
    """Diagonal damping ``D = 2 zeta sqrt(K)`` (or ``2 zeta sqrt(K Lambda)``).

    With ``inertia=None`` this is the reference's inertia-free fallback
    ``D = 2 zeta sqrt(K)``; with ``zeta=1`` that is exactly the ``D = 2 sqrt(K)``
    in the spec.  Supplying a positive ``inertia`` gives the inertia-aware form
    ``D = 2 zeta sqrt(K * inertia)`` (spec section 3.2, Lambda).  K may be a
    scalar or a diagonal vector; the return matches its shape.
    """
    if not torch.is_tensor(K):
        K = torch.as_tensor(K, dtype=torch.float64)
    K = torch.as_tensor(K)
    if zeta < 0:
        raise ValueError("zeta must be non-negative")
    scale = 2.0 * zeta
    if inertia is None:
        return scale * torch.sqrt(K)
    inertia_t = torch.as_tensor(inertia, dtype=K.dtype)
    if torch.any(inertia_t <= 0):
        raise ValueError("inertia must be positive")
    return scale * torch.sqrt(K * inertia_t)


def rate_limit(value: torch.Tensor, previous: Optional[torch.Tensor], cfg: ImpedanceConfig) -> torch.Tensor:
    """Clamp the per-step change of a torque/force command.

    At 1 kHz, 1 Nm/ms == 1 Nm per step (max_delta = rate * dt_ms).  ``previous``
    of ``None`` means no prior command exists, so the first value passes
    through unclamped (the caller then stores it as the new previous value).
    """
    max_delta = cfg.torque_rate_nm_per_ms * cfg.dt_ms()
    if max_delta <= 0:
        raise ValueError("torque_rate_nm_per_ms must be positive")
    if previous is None:
        return value
    return previous + torch.clamp(value - previous, -max_delta, max_delta)


def validate_dt(K: float, zeta: float, mass: float, dt: float, safety: float = 1.0) -> float:
    """Reject an integrator step that is too large to be stable.

    Semi-implicit (symplectic) Euler on ``m x'' + D x' + K x = 0`` is stable for
    ``dt < 2 / omega_n`` with ``omega_n = sqrt(K/m)``.  We require
    ``dt <= safety / omega_n`` (default safety = 1.0, a 2x margin) and raise
    ``ValueError`` otherwise, so an unstable dt is never silently accepted.
    Returns the stability limit ``2/omega_n`` for logging.
    """
    if mass <= 0 or K <= 0 or dt <= 0 or zeta < 0:
        raise ValueError("require mass>0, K>0, dt>0, zeta>=0")
    wn = (K / mass) ** 0.5
    dt_stable = 2.0 / wn
    if dt > safety / wn:
        raise ValueError(
            f"dt={dt:g}s is unstable for K={K:g}, mass={mass:g}: "
            f"omega_n={wn:.4f} rad/s requires dt <= {safety / wn:.6f}s "
            f"(hard stability limit {dt_stable:.6f}s)"
        )
    return dt_stable


# --------------------------------------------------------------------------- #
# Self-contained 7-DoF toy arm: forward kinematics and analytic Jacobian.
# --------------------------------------------------------------------------- #
def _eye(n: int, dtype: torch.dtype, device: torch.device, batch: int = 1) -> torch.Tensor:
    return torch.eye(n, dtype=dtype, device=device).expand(batch, n, n).clone()


def _rot_about(axis: int, angle: torch.Tensor) -> torch.Tensor:
    """Batched 3x3 rotation about a coordinate axis. ``angle``: (B,)."""
    b = angle.shape[0]
    c, s = torch.cos(angle), torch.sin(angle)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    R = _eye(3, angle.dtype, angle.device, b)
    if axis == 0:
        R[:, 1, 1], R[:, 1, 2] = c, -s
        R[:, 2, 1], R[:, 2, 2] = s, c
    elif axis == 1:
        R[:, 0, 0], R[:, 0, 2] = c, s
        R[:, 2, 0], R[:, 2, 2] = -s, c
    else:
        R[:, 0, 0], R[:, 0, 1] = c, -s
        R[:, 1, 0], R[:, 1, 1] = s, c
    del z, o
    return R


def _rotvec(R: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix -> rotation vector (axis * angle), shape (B,3)."""
    b = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cosang = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    vee = torch.stack(
        (R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]), dim=1
    )
    # Numerically stable angle: sin(theta) = ||vee||/2, cos(theta) = (trace-1)/2.
    # atan2 avoids the infinite derivative of acos near theta=0, which broke the
    # central-difference vs autograd Jacobian check for the calibrated FR3 chain.
    sinang = 0.5 * torch.linalg.norm(vee, dim=1)
    ang = torch.atan2(sinang, cosang)
    small = sinang < 1e-12
    safe = torch.where(small, torch.ones_like(sinang), 2.0 * sinang)
    scale = torch.where(small, torch.ones_like(sinang) * 0.5, ang / safe)
    return scale.unsqueeze(1) * vee


def forward_kinematics(q: torch.Tensor, kinematics: str = "fr3") -> tuple[torch.Tensor, torch.Tensor]:
    """7-DoF arm FK. ``q``: (B,7) -> position (B,3), rotation (B,3,3).

    ``kinematics="fr3"`` (default) uses the calibrated Menagerie FR3 v2 flange
    chain from :mod:`rtp.control.fr3_kinematics` (A1, reports/g5_reach.json).
    ``"toy"`` keeps the original generic link chain for legacy tests.
    """
    q = torch.as_tensor(q)
    if q.shape[-1] != NQ:
        raise ValueError(f"expected {NQ} joints, got {tuple(q.shape)}")
    if kinematics == "fr3":
        from rtp.control import fr3_kinematics
        return fr3_kinematics.fk(q)
    if kinematics != "toy":
        raise ValueError(f"kinematics must be 'fr3' or 'toy', got {kinematics!r}")
    b = q.shape[0]
    T = _eye(4, q.dtype, q.device, b)
    L = q.new_tensor(_LINK_LENGTH)
    for i, axis in enumerate(_AXES):
        Ri = _rot_about(axis, q[:, i])
        Aj = _eye(4, q.dtype, q.device, b)
        Aj[:, :3, :3] = Ri
        Aj[:, :3, 3] = torch.stack((L, q.new_tensor(0.0), q.new_tensor(0.0))).expand(b, 3)
        T = T @ Aj
    return T[:, :3, 3], T[:, :3, :3]


def _task_from_q(q: torch.Tensor, kinematics: str = "fr3") -> torch.Tensor:
    p, R = forward_kinematics(q, kinematics)
    return torch.cat((p, _rotvec(R)), dim=1)  # (B,6)


def jacobian_fd(q: torch.Tensor, h: float = 1e-6, kinematics: str = "fr3") -> torch.Tensor:
    """Central-difference task Jacobian ``d[p; rotvec]/dq``, shape (B,6,7)."""
    q = torch.as_tensor(q)
    b, nq = q.shape
    J = torch.zeros(b, NTASK, nq, dtype=q.dtype, device=q.device)
    for j in range(nq):
        dq = torch.zeros_like(q)
        dq[:, j] = h
        J[:, :, j] = (_task_from_q(q + dq, kinematics) - _task_from_q(q - dq, kinematics)) / (2.0 * h)
    return J


def jacobian_autograd(q_row: torch.Tensor, kinematics: str = "fr3") -> torch.Tensor:
    """Autograd Jacobian for a single sample (q: (7,)) -> (6,7).

    Provided as an independent cross-check of :func:`jacobian_fd`; not used by
    the controller itself.
    """
    x = q_row.detach().clone().requires_grad_(True)
    y = _task_from_q(x.unsqueeze(0), kinematics).squeeze(0)
    rows = []
    for i in range(y.numel()):
        grad = torch.autograd.grad(y[i], x, retain_graph=True, allow_unused=True)[0]
        rows.append(torch.zeros_like(x) if grad is None else grad)
    return torch.stack(rows, dim=0)


# --------------------------------------------------------------------------- #
# Batched Cartesian impedance controller.
# --------------------------------------------------------------------------- #
class CartesianImpedance:
    """Batched L1 Cartesian impedance law (simulation cross-check).

    Usage::

        cfg = ImpedanceConfig()
        ctrl = CartesianImpedance(cfg)
        ctrl.reset(batch)
        q, dq = ...            # (B,7)
        x_ref = ...            # (B,6) Cartesian target
        J = jacobian_fd(q)     # (B,6,7); caller may pass the real Jacobian
        tau = ctrl.compute(q, dq, x_ref, J=J)

    Internal state (filtered target and previous torque for the rate limiter)
    is per-batch and (re)initialised by :meth:`reset`.  ``compute`` is pure
    with respect to inputs apart from updating that state.
    """

    def __init__(self, cfg: ImpedanceConfig):
        self.cfg = cfg
        self.K = cfg.k_diag()
        # Inertia-free fallback D = 2 zeta sqrt(K); see module docstring.
        self.D = damping(self.K, cfg.zeta)
        self._filt_target: Optional[torch.Tensor] = None
        self._prev_tau: Optional[torch.Tensor] = None

    def reset(self, batch: int, *, q_rest: Optional[torch.Tensor] = None) -> None:
        self._filt_target = None
        self._prev_tau = None
        if q_rest is None:
            q_rest = torch.zeros(NQ, dtype=self.cfg.dtype)
        self.q_rest = torch.as_tensor(q_rest, dtype=self.cfg.dtype).expand(batch, NQ).clone()

    def _rest(self, q: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "q_rest") or self.q_rest.shape[0] != q.shape[0]:
            self.reset(q.shape[0])
        return self.q_rest

    def compute(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        target: torch.Tensor,
        *,
        J: Optional[torch.Tensor] = None,
        coriolis: Optional[torch.Tensor] = None,
        gravity: Optional[torch.Tensor] = None,
        apply_rate_limit: bool = True,
    ) -> torch.Tensor:
        """Return joint torques ``tau`` of shape (B,7).

        ``target`` is a Cartesian pose (B,6) = [p_ref; rotvec_ref].  ``J`` is
        the (B,6,7) task Jacobian; if omitted it is computed by finite
        differences.  ``coriolis``/``gravity`` are optional (B,7) hooks; gravity
        is added explicitly in sim as the spec notes.
        """
        q = torch.as_tensor(q, dtype=self.cfg.dtype)
        dq = torch.as_tensor(dq, dtype=self.cfg.dtype)
        target = torch.as_tensor(target, dtype=self.cfg.dtype)
        b = q.shape[0]
        if J is None:
            J = jacobian_fd(q, kinematics=self.cfg.kinematics).to(self.cfg.dtype)
        J = torch.as_tensor(J, dtype=self.cfg.dtype)
        if J.shape != (b, NTASK, NQ):
            raise ValueError(f"J must have shape {(b, NTASK, NQ)}, got {tuple(J.shape)}")

        # Target low-pass (disabled with alpha=None or 1.0).  Held targets are
        # unaffected once converged; this is the spec's simple reference filter.
        alpha = self.cfg.lowpass_alpha
        if alpha is None or alpha >= 1.0:
            filt = target
        else:
            if self._filt_target is None or self._filt_target.shape[0] != b:
                self._filt_target = target.clone()
            self._filt_target = (1.0 - alpha) * self._filt_target + alpha * target
            filt = self._filt_target

        # Cartesian error e and Cartesian velocity v = J dq.
        err = _task_from_q(q, self.cfg.kinematics) - filt  # (B,6)
        v = torch.bmm(J, dq.unsqueeze(2)).squeeze(2)      # (B,6)
        F = -self.K * err - self.D * v                    # (B,6)

        # Clamp order (i): translational force and rotational torque caps.
        Ft, Fr = F[:, :3], F[:, 3:]
        Ft = _cap_norm(Ft, self.cfg.f_lin_max)
        Fr = _cap_norm(Fr, self.cfg.m_max)
        F = torch.cat((Ft, Fr), dim=1)

        tau_task = torch.bmm(J.transpose(1, 2), F.unsqueeze(2)).squeeze(2)  # (B,7)

        # Nullspace PD with the dynamically consistent projector N = I - J^T J^+^T.
        Jpinv = torch.linalg.pinv(J, rcond=1e-3)          # (B,7,6)
        N = torch.eye(NQ, dtype=self.cfg.dtype) - torch.bmm(J.transpose(1, 2), Jpinv.transpose(1, 2))
        kn = self.cfg.k_null
        tau_null_raw = kn * (self._rest(q) - q) - 2.0 * (kn ** 0.5) * dq   # (B,7)
        tau_null = torch.bmm(N, tau_null_raw.unsqueeze(2)).squeeze(2)

        tau = tau_task + tau_null
        if coriolis is not None:
            tau = tau + torch.as_tensor(coriolis, dtype=self.cfg.dtype)
        if gravity is not None:
            tau = tau + torch.as_tensor(gravity, dtype=self.cfg.dtype)

        # Clamp order (iii)/(iv): joint torque saturation then torque-rate limit.
        tau = torch.clamp(tau, -self.cfg.tau_max, self.cfg.tau_max)
        if apply_rate_limit:
            tau = rate_limit(tau, self._prev_tau, self.cfg)
        self._prev_tau = tau
        return tau


def _cap_norm(x: torch.Tensor, cap: float) -> torch.Tensor:
    if cap <= 0:
        raise ValueError("force/torque cap must be positive")
    n = torch.linalg.norm(x, dim=1, keepdim=True)
    scale = torch.clamp(cap / torch.clamp(n, min=1e-12), max=1.0)
    return x * scale


# --------------------------------------------------------------------------- #
# 1-DoF mass-spring-damper equivalent: analytic solution and discrete sim.
# --------------------------------------------------------------------------- #
def analytic_step_response(t, x_step: float, K: float, zeta: float, mass: float) -> torch.Tensor:
    """Closed-form position for a step of ``x_step`` in the target, starting at rest.

    Solves ``m x'' + D x' + K x = K x_ref`` with ``D = 2 zeta sqrt(K m)`` and a
    unit step in ``x_ref`` at t=0.  Handles under-, critically- and over-damped
    cases.  ``t`` may be a scalar, list, or tensor; returns a float64 tensor.
    """
    t = torch.as_tensor(t, dtype=torch.float64)
    wn = (K / mass) ** 0.5
    x0 = float(x_step)
    if zeta < 1.0 - 1e-12:
        wd = wn * (1.0 - zeta ** 2) ** 0.5
        env = torch.exp(-zeta * wn * t)
        x = x0 * (1.0 - env * (torch.cos(wd * t) + (zeta / (1.0 - zeta ** 2) ** 0.5) * torch.sin(wd * t)))
    elif abs(zeta - 1.0) <= 1e-12:
        x = x0 * (1.0 - (1.0 + wn * t) * torch.exp(-wn * t))
    else:
        root = (zeta ** 2 - 1.0) ** 0.5
        r1 = -wn * (zeta - root)
        r2 = -wn * (zeta + root)
        A = -x0 * r2 / (r2 - r1)
        B = x0 * r1 / (r2 - r1)
        x = x0 + A * torch.exp(r1 * t) + B * torch.exp(r2 * t)
    return x


def _integrate_1d(cfg: ImpedanceConfig, force_fn, n_steps: int,
                  apply_rate_limit: bool = False, x0: float = 0.0, v0: float = 0.0):
    """Semi-implicit Euler on m x'' = F_imp + F_ext. Returns (t, x, v) tensors."""
    K = cfg.k_lin
    D = damping(K, cfg.zeta, cfg.mass).item()
    dt = cfg.dt
    x = float(x0)
    v = float(v0)
    prev_F: Optional[torch.Tensor] = None
    ts = [0.0]
    xs = [x]
    vs = [v]
    for i in range(1, n_steps + 1):
        F_imp = -K * x - D * v
        if apply_rate_limit:
            F_imp_t = rate_limit(torch.tensor(F_imp, dtype=cfg.dtype), prev_F, cfg)
            F_imp = float(F_imp_t)
            prev_F = F_imp_t
        F_ext = force_fn(i * dt, x, v)
        v = v + (F_imp + F_ext) / cfg.mass * dt
        x = x + v * dt                     # semi-implicit (use updated v)
        ts.append(i * dt)
        xs.append(x)
        vs.append(v)
    return (
        torch.tensor(ts, dtype=torch.float64),
        torch.tensor(xs, dtype=torch.float64),
        torch.tensor(vs, dtype=torch.float64),
    )


def simulate_step_response(cfg: ImpedanceConfig, x_step: float, duration: float,
                           apply_rate_limit: bool = False):
    """Simulate a target step ``0 -> x_step``; return ``(t, x, v)``.

    The target is stepped at t=0 so the initial error is ``-x_step``.  By
    default no rate limit is applied: the analytic mass-spring-damper has none,
    and the torque-rate saturation is verified independently.
    """
    validate_dt(cfg.k_lin, cfg.zeta, cfg.mass, cfg.dt)
    n = int(round(duration / cfg.dt))
    return _integrate_1d(cfg, lambda t, x, v: cfg.k_lin * x_step, n, apply_rate_limit)


def push_response(cfg: ImpedanceConfig, force: float = 10.0,
                  push_duration: float = 1.0, release_duration: float = 2.5,
                  direction: float = 1.0) -> dict:
    """Scripted human push on the simulated mass-spring-damper; return metrics.

    A constant external force ``force`` is applied for ``push_duration``, then
    removed for ``release_duration``.  Reported metrics: peak displacement,
    measured vs ``F/K`` steady deflection, return error vs the planned pose,
    settle time, overshoot and peak speed.  This is a compliance *behaviour*
    measurement in simulation only -- not a contact-force safety result.
    """
    validate_dt(cfg.k_lin, cfg.zeta, cfg.mass, cfg.dt)
    npush = int(round(push_duration / cfg.dt))
    nrel = int(round(release_duration / cfg.dt))
    Fext = direction * force

    def f_push(t, x, v):
        return Fext

    tp, xp, vp = _integrate_1d(cfg, f_push, npush)
    x_release = float(xp[-1])
    v_release = float(vp[-1])
    steady_expected = Fext / cfg.k_lin

    tr, xr, vr = _integrate_1d(cfg, lambda t, x, v: 0.0, nrel, x0=x_release, v0=v_release)

    tol = cfg.return_tol
    within = torch.abs(xr) <= tol
    if within.any():
        first = int(torch.nonzero(within)[0, 0])
        settle_time = float(tr[first])
    else:
        settle_time = None

    peak_displacement = float(torch.max(torch.abs(xp))) if npush else abs(x_release)
    return_abs = float(torch.abs(xr[-1]))
    overshoot = float(torch.clamp(-torch.min(xr) / abs(steady_expected), min=0.0)) if steady_expected != 0 else 0.0
    peak_speed = float(torch.max(torch.abs(vr)))
    return {
        "force_N": float(force),
        "push_duration_s": float(push_duration),
        "release_duration_s": float(release_duration),
        "peak_displacement_m": peak_displacement,
        "steady_deflection_measured_m": x_release,
        "steady_deflection_expected_m": steady_expected,
        "deflection_error_pct": abs(x_release - steady_expected) / abs(steady_expected) * 100.0 if steady_expected else 0.0,
        "return_error_m": return_abs,
        "return_tolerance_m": tol,
        "settle_time_s": settle_time,
        "overshoot_fraction": overshoot,
        "peak_return_speed_mps": peak_speed,
        "within_tolerance": return_abs <= tol,
    }
