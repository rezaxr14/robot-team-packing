"""Compliant FR3 control on a native Newton articulation (SIM only).

.. warning::
   **SIMULATION EVIDENCE ONLY — NOT A SAFETY CERTIFICATION.** This module maps the
   push/yield/return supervisor (:mod:`rtp.control.compliance`) onto the FR3 v2
   articulation that A3 steps natively in Newton.  Force/speed/torque limits are
   placeholders (D-22, D-58); hardware/L0/L1 sign-off is Claude's.  No claim about
   the arms being safe around people is made here.

What it does
------------
One control tick (nominally 1 kHz):

1. reads ``q, dq`` from Newton ``state.joint_q`` / ``state.joint_qd`` (A2 answer 3);
2. computes the FR3 flange pose ``p, R`` and the geometric Jacobian ``J`` from
   :mod:`rtp.control.fr3_kinematics` (A1, verified against Newton FK to < 1 mm);
3. forms the 6-vector task state ``x = [p; rotvec(R @ R_ref^T)]``.  The
   orientation error is a **rotation-vector error**, never a difference of
   quaternions, exactly as the runbook requires;
4. calls :class:`~rtp.control.compliance.ComplianceSupervisor.step` to obtain the
   clamped Cartesian wrench ``F`` and the plan-clock gate / state machine;
5. maps it to joint torque ``tau_task = J^T F``;
6. adds a weak null-space PD ``(I - J^T J^+^T)(k_null (q_home - q) - 2 sqrt(k_null) dq)``
   with ``k_null <= 20 Nm/rad`` so the elbow yields;
7. adds gravity/bias compensation ``g(q, dq)`` (A2 answer 4; native Newton
   ``eval_inverse_dynamics_passive``, or the Isaac Lab
   ``gravity_compensation_forces.torch`` accessor on the wrapper);
8. applies joint torque saturation and the 1 Nm/ms torque-rate limit from
   :func:`rtp.control.impedance.rate_limit`;
9. writes the torque in **effort mode** — ``control.joint_f`` on the native
   Newton solver, or ``actuators.target_command.set_effort_index`` followed by
   ``write_data_to_sim()`` on the A2-verified Isaac Lab wrapper.

Torch-free import
-----------------
Like :mod:`rtp.control.__init__`, importing this module does **not** require
torch: the supervisor (numpy) and the rotation helpers are available with a bare
numpy install.  Torch/fr3_kinematics, Newton and Warp are imported lazily, only
when the corresponding helper is actually used.  Pass ``kinematics=<fake>`` to the
constructor for CPU-only tests with an injected FK/Jacobian provider.

Evidence class: SIM when driven by the Newton harness ``ops/a5_newton_push.py``.
Pure-math unit tests in ``tests/test_newton_arm.py`` are FIXTURE/MODEL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .compliance import ComplianceConfig, ComplianceSupervisor, State

__all__ = [
    "NewtonArmConfig",
    "Q_HOME",
    "FLANGE_BODY",
    "ELBOW_BODY",
    "FOREARM_BODY",
    "rotvec_from_matrix",
    "matrix_from_rotvec",
    "rate_limit_numpy",
    "dynamic_nullspace",
    "NewtonCompliantArm",
]

#: Mirrors :data:`rtp.control.fr3_kinematics.Q_HOME` (kept here as a plain tuple so
#: this module imports without torch).  The unit test asserts they are equal.
Q_HOME = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

#: Body labels / indices in the Menagerie FR3 v2 MJCF loaded by ``add_mjcf``:
#: body 9 = fr3v2_link8 (flange), 5 = link4 (elbow), 7 = link6 (forearm).
FLANGE_BODY = 9
ELBOW_BODY = 5
FOREARM_BODY = 7


@dataclass
class NewtonArmConfig:
    """Gains and clamps for :class:`NewtonCompliantArm`.

    All numeric limits are **[D] starting values** from ``docs/IMPEDANCE_SPEC.md``
    and are placeholders until ISO 10218-2:2025 is read (D-22), not safety limits.
    """

    dt: float = 0.001                    # s, 1 kHz control loop (A2 answer 7)
    profile: str = "free_motion"         # stiffness profile from PROFILES (allow-list)
    k_null: float = 15.0                 # Nm/rad null-space PD; must be <= 20 (spec: 15-20)
    tau_max: float = 50.0                # Nm per-joint saturation (placeholder)
    torque_rate_nm_per_ms: float = 1.0   # Nm/ms torque-rate saturation [V:S-20]
    #: Coulomb friction feed-forward.  The Menagerie FR3 v2 MJCF declares
    #: ``frictionloss="0.2"`` Nm on every actuated joint; that bias is the main
    #: reason the measured steady deflection exceeded F/K in the first A5 run.
    #: Value is read from the model, not identified here; set to 0 to disable.
    friction_nm: float = 0.2
    friction_comp: bool = True
    #: Residual-wrench compensation (runbook A5: injected wrench + controller
    #: contact estimate).  The unmodelled joint friction shifts the equilibrium
    #: by ``(J^+)^T tau_fric``; feeding back the filtered residual
    #: ``F_imp + f_ext`` drives it to zero.  Default off; enabled by the harness
    #: after measuring that a fixed frictionloss feed-forward is not valid.
    residual_comp_gain: float = 0.0
    residual_beta: float = 0.05
    #: Additional joint-space damping on excess return speed.  The supervisor's
    #: soft limiter (D-66) is Cartesian and its force is clipped at F_lin_max;
    #: this term reinforces the same cap using the full joint-torque budget.
    speed_limiter_gain: float = 4000.0   # N s / m (mapped through J[:3].T)
    compliance: Optional[ComplianceConfig] = None
    q_home: tuple = Q_HOME
    #: torque-rate ramp at start-up: ``None`` means the first command is unclamped,
    #: matching :func:`rtp.control.impedance.rate_limit` (previous = None).
    prev_tau: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.k_null < 0 or self.k_null > 20.0:
            raise ValueError(f"k_null must be in [0, 20] Nm/rad (elbow must yield), got {self.k_null}")
        if self.tau_max <= 0 or self.torque_rate_nm_per_ms <= 0 or self.dt <= 0:
            raise ValueError("dt, tau_max and torque_rate_nm_per_ms must be positive")
        if self.friction_nm < 0 or self.speed_limiter_gain < 0 or self.residual_comp_gain < 0:
            raise ValueError("friction_nm, speed_limiter_gain and residual_comp_gain must be non-negative")
        if not 0.0 < self.residual_beta <= 1.0:
            raise ValueError("residual_beta must be in (0, 1]")
        if len(self.q_home) != 7:
            raise ValueError("q_home must have 7 joints")

    @property
    def max_delta(self) -> float:
        """Per-step torque change: same law as ``ImpedanceConfig.dt_ms() * rate``."""
        return self.torque_rate_nm_per_ms * self.dt * 1000.0


# --------------------------------------------------------------------------- #
# Small numpy geometry helpers (no torch, no scipy).
# --------------------------------------------------------------------------- #
def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle), numerically stable.

    Uses ``atan2(||vee||/2, (trace-1)/2)`` rather than ``acos`` so the derivative
    stays finite near zero angle (same choice as ``impedance._rotvec`` after A1).
    """
    R = np.asarray(R, dtype=float)
    cosang = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    sinang = 0.5 * float(np.linalg.norm(vee))
    ang = float(np.arctan2(sinang, cosang))
    if sinang < 1e-12:
        return 0.5 * vee
    return (ang / (2.0 * sinang)) * vee


def matrix_from_rotvec(v: np.ndarray) -> np.ndarray:
    """Rotation vector -> rotation matrix (inverse of :func:`rotvec_from_matrix`)."""
    v = np.asarray(v, dtype=float)
    theta = float(np.linalg.norm(v))
    K = _skew(v)
    if theta < 1e-12:
        return np.eye(3) + K + 0.5 * (K @ K)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * K + b * (K @ K)


def rate_limit_numpy(value: np.ndarray, previous: Optional[np.ndarray], max_delta: float) -> np.ndarray:
    """Numpy twin of :func:`rtp.control.impedance.rate_limit`.

    ``max_delta = torque_rate_nm_per_ms * dt_ms`` is exactly the bound used there;
    ``previous is None`` lets the first command through unclamped.  Kept in numpy
    so this module imports without torch; ``tests/test_newton_arm.py`` checks
    parity against the torch implementation.
    """
    if max_delta <= 0:
        raise ValueError("max_delta must be positive")
    v = np.asarray(value, dtype=float)
    if previous is None:
        return v.copy()
    return np.asarray(previous, dtype=float) + np.clip(v - previous, -max_delta, max_delta)


def _to_numpy(x: Any) -> np.ndarray:
    """Accept a numpy array or a (CPU/CUDA) torch tensor; return float64 numpy."""
    if hasattr(x, "detach"):          # torch tensor
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):          # warp array
        x = x.numpy()
    return np.asarray(x, dtype=float)


def dynamic_nullspace(J: np.ndarray, M: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    """Mass-weighted (dynamically consistent) null-space projector.

    With ``Minv = inv(M)``, ``Lam = inv(J Minv Jᵀ)``, ``Jbar = Minv Jᵀ Lam``:

        N = I - Jᵀ Jbarᵀ

    The exact property this buys is::

        J @ Minv @ N ~ 0        # a torque in this null space produces ZERO task acceleration
        N @ N ~ N               # N is a projector

    i.e. ``tau_null = N @ tau_null_raw`` cannot accelerate the task
    ``xdd = J Minv (tau_task + tau_null + ...)``.  That is why an elbow/link push
    can be absorbed by the null space without being transmitted to the flange.

    .. important::
       The identity holds for ``N`` itself, **not** for ``N.T``: with this
       convention ``J @ Minv @ N.T`` is *not* zero.  Do not test ``J Minv N.T``.

    ``M`` is symmetrised and must be a finite, symmetric positive-definite 7x7
    matrix; ``rcond`` guards the inversion of ``J Minv Jᵀ`` against rank
    deficiency.  Raises ``ValueError`` for any violation.
    """
    J = np.asarray(J, dtype=float).reshape(6, 7)
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)                       # symmetrise (ID/FD noise is float32)
    if M.shape != (7, 7) or J.shape != (6, 7):
        raise ValueError(f"expected M 7x7 and J 6x7, got M{M.shape} and J{J.shape}")
    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(J)):
        raise ValueError("M and J must be finite")
    try:
        ev = np.linalg.eigvalsh(M)
    except np.linalg.LinAlgError as exc:      # pragma: no cover - defensive
        raise ValueError("M is not symmetric positive definite") from exc
    if ev[0] <= 0.0:
        raise ValueError(f"M is not positive definite (min eigenvalue {ev[0]:.3e})")
    Minv = np.linalg.inv(M)
    A = J @ Minv @ J.T                        # 6x6 operational-space inertia inverse
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[-1] <= float(rcond) * sv[0]:
        raise ValueError(f"J Minv J^T is rank-deficient (sv_min/sv_max={sv[-1] / sv[0]:.3e})")
    Lam = np.linalg.inv(A)
    Jbar = Minv @ J.T @ Lam
    return np.eye(7) - J.T @ Jbar.T


class NewtonCompliantArm:
    """One-arm compliant controller for the native Newton FR3 articulation.

    The class is plant-agnostic about *state*: ``step()`` takes ``q, dq`` as numpy
    arrays (or torch tensors) plus the externally estimated wrench ``f_ext`` and
    returns a 7-vector of joint torques.  Newton-specific helpers
    (:meth:`set_effort_mode_native`, :meth:`write_effort_native`,
    :meth:`gravity_comp_native`, :meth:`write_effort_isaaclab`) bridge to the
    verified APIs of A2; the harness ``ops/a5_newton_push.py`` shows the wiring.
    """

    def __init__(
        self,
        cfg: Optional[NewtonArmConfig] = None,
        *,
        kinematics: Any = None,
        q_home: Optional[np.ndarray] = None,
        profile: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or NewtonArmConfig()
        if profile is not None:
            self.cfg.profile = profile
        self.q_home = _to_numpy(self.cfg.q_home if q_home is None else q_home)
        self._kin = kinematics           # injected fake for CPU tests; None => fr3_kinematics
        self.prev_tau = None if self.cfg.prev_tau is None else _to_numpy(self.cfg.prev_tau).copy()
        self._ref_rotvec: Optional[np.ndarray] = None
        self._ref_pos: Optional[np.ndarray] = None
        self._fric_dir = np.zeros(7)      # remembered stiction direction per joint
        self._res_filt = np.zeros(6)      # filtered wrench residual
        self.sup = ComplianceSupervisor(self.cfg.compliance, self.cfg.profile)

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self, profile: Optional[str] = None, x0: Optional[np.ndarray] = None) -> None:
        """Reset the supervisor, plan clock and rate-limiter state."""
        if profile is not None:
            self.cfg.profile = profile
        self.prev_tau = None
        self._ref_rotvec = None
        self._ref_pos = None
        self._fric_dir = np.zeros(7)
        self._res_filt = np.zeros(6)
        self.sup.reset(self.cfg.profile, x0=x0)

    @property
    def state(self) -> State:
        return self.sup.state

    def _kinematics(self) -> Any:
        if self._kin is not None:
            return self._kin
        from rtp.control import fr3_kinematics  # lazy: torch only needed here
        return fr3_kinematics

    def _reference(self) -> np.ndarray:
        """Reference pose ``[p_home; rotvec(R_home)]`` for the rotation error.

        The supervisor's ``x_ref`` uses zero rotation vector (the plan carries no
        orientation change in A5), so the rotational part of ``x`` is the error
        ``rotvec(R @ R_ref^T)`` directly.
        """
        if self._ref_rotvec is None:
            p, R = self._kinematics().fk(self.q_home)
            self._ref_pos = _to_numpy(p)
            self._ref_rotvec = rotvec_from_matrix(_to_numpy(R))
        return np.concatenate([self._ref_pos, self._ref_rotvec])

    # -- core control tick -------------------------------------------------- #
    def task_state(self, q, dq, ref_rotvec: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(x, xdot, J)`` of the flange, ``x = [p; rotvec(R R_ref^T)]``.

        ``xdot = J @ dq`` is the world-frame flange twist (linear velocity and
        angular velocity); it is the damping input the supervisor expects.
        """
        q = _to_numpy(q)
        dq = _to_numpy(dq)
        p, R = self._kinematics().fk(q)
        J = _to_numpy(self._kinematics().jacobian(q)).reshape(6, 7)
        Rerr = _to_numpy(R) @ matrix_from_rotvec(
            self._ref_rotvec if ref_rotvec is None else ref_rotvec
        ).T
        x = np.concatenate([_to_numpy(p), rotvec_from_matrix(Rerr)])
        xdot = J @ dq
        return x, xdot, J

    def step(
        self,
        q,
        dq,
        f_ext,
        plan_fn: Optional[Callable[[float], np.ndarray]] = None,
        *,
        gravity_comp: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        scale: float = 1.0,
        ref_rotvec: Optional[np.ndarray] = None,
        mass_matrix: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict]:
        """One control tick. Returns ``(tau, info)`` with ``tau`` shape ``(7,)``.

        ``f_ext`` is the 6-vector external wrench estimate (in simulation the
        injected wrench, optionally cross-checked against an inverse-dynamics
        estimate; see the harness).  ``plan_fn(s) -> [px,py,pz,rx,ry,rz]`` is the
        planned pose **in the task frame of** :meth:`task_state`: world position
        and a rotation vector *relative to the home orientation*.  A5's plans
        hold the home orientation, so the last three entries are zero; the
        default plan holds the home position with zero rotation error.

        ``mass_matrix`` is the 7x7 joint-space inertia ``M(q)`` from
        :meth:`mass_matrix_native`.  When it is a valid SPD 7x7 matrix the null
        space is the dynamically consistent projector :func:`dynamic_nullspace`
        (``J Minv N ~ 0``); otherwise (including ``None``, for legacy CPU tests)
        the plain ``I - Jᵀ(J⁺)ᵀ`` projector is used.  ``info['nullspace']``
        records which one was applied.
        """
        dt = self.cfg.dt if dt is None else float(dt)
        if ref_rotvec is None:
            ref_rotvec = self._reference()[3:]
        q = _to_numpy(q)
        dq = _to_numpy(dq)
        f_ext = _to_numpy(f_ext)
        fn, mn = np.linalg.norm(f_ext[:3]), np.linalg.norm(f_ext[3:])
        if plan_fn is None:
            home_pos = self._reference()[:3]
            plan_fn = lambda _s: np.concatenate([home_pos, np.zeros(3)])  # noqa: E731

        x, xdot, J = self.task_state(q, dq, ref_rotvec)
        out = self.sup.step(dt, x, xdot, f_ext, plan_fn, scale=scale)
        c = self.cfg.compliance or ComplianceConfig()

        # Residual-wrench compensation: (F_imp + f_ext) is nonzero at steady
        # state only because of unmodelled friction/bias.  Feeding a filtered
        # copy back drives that residual to zero, so the equilibrium is
        # F_imp = -f_ext, i.e. |delta| = F/K as specified.  The corrected wrench
        # is re-clamped to the supervisor's force/moment caps.
        force_used = out.force.copy()
        wrench_applied = bool(fn > 1e-9 or mn > 1e-9)
        # Experiment: also estimate the free-space tracking bias in TRACK so the
        # static friction floor does not leave a multi-mm error at the plan.
        # RETURN is excluded (F_imp is the whole return spring there; amplifying
        # it saturates F_lin_max and breaks the return-speed bound).
        if self.cfg.residual_comp_gain > 0.0 and (wrench_applied or out.state == State.TRACK):
            residual = out.force + f_ext
            self._res_filt += self.cfg.residual_beta * (residual - self._res_filt)
            force_used = force_used + self.cfg.residual_comp_gain * self._res_filt
            nl, nm = float(np.linalg.norm(force_used[:3])), float(np.linalg.norm(force_used[3:]))
            if nl > c.f_lin_max:
                force_used[:3] *= c.f_lin_max / nl
            if nm > c.m_max:
                force_used[3:] *= c.m_max / nm

        # Cartesian wrench -> joint torque.
        tau_task = J.T @ force_used

        # Weak null-space PD (k_null <= 20 so elbows yield).  Prefer the
        # dynamically consistent projector when a valid mass matrix is supplied;
        # fall back to the plain projector for legacy callers/tests.
        N, ns_kind = None, "plain"
        if mass_matrix is not None:
            try:
                N = dynamic_nullspace(J, mass_matrix)
                ns_kind = "dynamic"
            except (ValueError, np.linalg.LinAlgError):
                N = None
        if N is None:
            Jpinv = np.linalg.pinv(J, rcond=1e-3)
            N = np.eye(7) - J.T @ Jpinv.T
        kn = self.cfg.k_null
        tau_null_raw = kn * (self.q_home - q) - 2.0 * np.sqrt(kn) * dq
        tau_null = N @ tau_null_raw

        g = np.zeros(7) if gravity_comp is None else _to_numpy(gravity_comp)
        tau_raw = tau_task + tau_null + g

        # Coulomb friction feed-forward.  MJCF frictionloss is 0.2 Nm per joint
        # and opposes motion; during a sustained push it biases the equilibrium
        # by roughly friction/K.  Use the joint torques that would create the
        # measured external wrench to pick the stiction direction while pushing,
        # and the last significant joint velocity otherwise.  Without a push and
        # with no motion the memory is cleared so it cannot create a rest bias.
        tau_fric = np.zeros(7)
        if self.cfg.friction_comp and self.cfg.friction_nm > 0.0:
            if fn > c.f_push or mn > c.m_push:
                push_torques = J.T @ f_ext
                cand = np.where(np.abs(push_torques) > 1e-3, np.sign(push_torques), 0.0)
                self._fric_dir = np.where(cand != 0.0, cand, self._fric_dir)
            else:
                moving = np.abs(dq) > 0.02
                self._fric_dir = np.where(moving, np.sign(dq), self._fric_dir)
                if not np.any(moving) and np.max(np.abs(dq)) < 0.01:
                    self._fric_dir = np.zeros(7)
            tau_fric = self.cfg.friction_nm * self._fric_dir
            tau_raw = tau_raw + tau_fric

        # Reinforce the supervisor's soft speed limiter on EVERY return tick.
        # The supervisor's Cartesian limiter is clipped at F_lin_max; this term
        # acts on the excess *measured* flange speed through J[:3].T using the
        # remaining joint-torque budget, still subject to the clamp/rate limit.
        tau_speed = np.zeros(7)
        released = fn < c.f_release and mn < c.m_release
        letting_go = out.state == State.RETURN or (out.state == State.YIELD and released)
        if letting_go and self.cfg.speed_limiter_gain > 0.0:
            # Damp the velocity at the END of the step, not the one measured at
            # its start: a reactive damper on xdot=J@dq is one tick late and the
            # spring can push the flange past v_return inside that tick.  With
            # M(q) available, predict xdot_next = xdot + J M^-1 tau_raw dt.
            v = xdot[:3]
            if mass_matrix is not None and ns_kind == "dynamic":
                try:
                    Msym = 0.5 * (np.asarray(mass_matrix, dtype=float)
                                  + np.asarray(mass_matrix, dtype=float).T)
                    dq_pred = dq + dt * (np.linalg.solve(Msym, tau_raw))
                    v = J[:3] @ dq_pred
                except np.linalg.LinAlgError:
                    pass
            speed = float(np.linalg.norm(v))
            cap = c.v_return
            if speed > cap:
                F_lim = -self.cfg.speed_limiter_gain * (speed - cap) * v / speed
                tau_speed = J[:3].T @ F_lim
                tau_raw = tau_raw + tau_speed

        torque_clamped = bool(np.any(np.abs(tau_raw) > self.cfg.tau_max))
        tau = np.clip(tau_raw, -self.cfg.tau_max, self.cfg.tau_max)
        limited = rate_limit_numpy(tau, self.prev_tau, self.cfg.max_delta)
        rate_limited = self.prev_tau is not None and bool(np.any(limited != tau))
        self.prev_tau = limited.copy()

        info = {
            "state": out.state,
            "events": list(out.events),
            "flags": dict(out.flags),
            "s": float(out.s),
            "k_lin": float(out.k_lin),
            "force": out.force.copy(),
            "force_used": force_used.copy(),
            "x": x,
            "xdot": xdot,
            "x_ref": out.x_ref.copy(),
            "J": J,
            "tau_task": tau_task,
            "tau_null": tau_null,
            "nullspace": ns_kind,
            "gravity_comp": g,
            "tau_friction": tau_fric,
            "tau_speed": tau_speed,
            "tau_raw": tau_raw,
            "tau": limited.copy(),
            "integral_active": bool(out.integral_active),
            "integ_norm": float(out.integ_norm),
            "gate_t_hold": float(self.sup._t_hold),
            "torque_clamped": torque_clamped,
            "rate_limited": rate_limited,
        }
        return limited, info

    # -- verified Newton / Isaac Lab bridges (A2 answers 2, 3, 4) ----------- #
    @staticmethod
    def set_effort_mode_native(model: Any) -> None:
        """Disable the MJCF position actuators: zero ``joint_target_ke/kd``.

        With zero position gains, ``SolverMuJoCo`` integrates only the external
        joint force ``control.joint_f`` that :meth:`write_effort_native` writes —
        the same notion of "effort mode" that A2 verified on the wrapper via an
        explicit ``IdealPDActuatorCfg(stiffness=0, damping=0)``.
        """
        import warp as wp  # lazy
        model.joint_target_ke = wp.zeros_like(model.joint_target_ke)
        model.joint_target_kd = wp.zeros_like(model.joint_target_kd)

    @staticmethod
    def write_effort_native(control: Any, tau: np.ndarray) -> None:
        """Publish joint torques to the native Newton ``control.joint_f`` view."""
        import warp as wp  # lazy
        arr = np.asarray(tau, dtype=np.float32).reshape(-1)
        control.joint_f = wp.array(arr, dtype=control.joint_f.dtype, device=control.joint_f.device)

    @staticmethod
    def mass_matrix_native(model: Any, state: Any) -> np.ndarray:
        """Joint-space inertia ``M(q)`` from Newton's passive inverse dynamics.

        A2 answer 4: ``newton.eval_inverse_dynamics_passive(..., mass_matrix=M)``
        fills ``M`` with the 7x7 joint-space mass matrix.  ``eval_fk`` is run
        first so ``state.body_q`` is consistent with ``state.joint_q``.  Uses
        ``na = model.joint_dof_count`` (``model.joint_q.numpy()`` is a warp
        array whose item indexing raised on this Newton version), asserts that
        it matches the joint-coordinate length, and returns a float64 numpy
        matrix.  The A5 probe measured it symmetric (err ~6e-8) and SPD at
        ``Q_HOME`` (eigenvalues 7.3e-4 ... 1.87).
        """
        import newton  # lazy
        import warp as wp  # lazy
        na = int(model.joint_dof_count)
        jq = np.asarray(model.joint_q.numpy()).reshape(-1)
        assert na == len(jq), f"joint_dof_count {na} != len(joint_q) {len(jq)}"
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        M = wp.zeros((1, na, na), device=model.device)   # warp float32 (ID kernel requires f32)
        newton.eval_inverse_dynamics_passive(model, state, mass_matrix=M)
        return np.asarray(M.numpy()).reshape(na, na).astype(np.float64)

    @staticmethod
    def gravity_comp_native(model: Any, state: Any) -> np.ndarray:
        """Gravity + bias torque ``g(q,dq)+C(q,dq)dq`` from Newton's RNEA.

        A2 answer 4: ``newton.eval_inverse_dynamics_passive`` writes the joint
        force that holds the articulation static under gravity (plus Coriolis at
        the current velocity).  ``eval_fk`` is run first so ``state.body_q`` is
        consistent with ``state.joint_q`` as that API requires.
        """
        import newton  # lazy
        import warp as wp  # lazy
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        g = wp.zeros_like(state.joint_q)
        newton.eval_inverse_dynamics_passive(model, state, gravity_force=g)
        return _to_numpy(g).reshape(-1).astype(float)

    @staticmethod
    def external_wrench_estimate(
        model: Any,
        state: Any,
        tau_cmd: np.ndarray,
        J: np.ndarray,
        dq_prev: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Model-based external wrench ``F_hat = (J^+)^T (M qdd + C + g - tau)``.

        The sim analogue of ``tau_measured - tau_model`` (there is no torque
        sensor in Newton): ``qdd`` comes from a finite difference of ``dq``.
        Reported next to the injected wrench in the harness so the two can be
        compared; stick-slip friction in the MJCF makes it an estimate, not a
        measurement.
        """
        import newton  # lazy
        import warp as wp  # lazy
        n = int(np.asarray(state.joint_q.numpy()).size)
        na = int(np.asarray(model.joint_q.numpy()).size)
        M = wp.zeros((1, na, na), dtype=wp.float32, device=model.device)
        C = wp.zeros((n,), dtype=wp.float32, device=model.device)
        g = wp.zeros((n,), dtype=wp.float32, device=model.device)
        newton.eval_inverse_dynamics_passive(model, state, mass_matrix=M, coriolis_force=C, gravity_force=g)
        dq = _to_numpy(state.joint_qd.numpy()).reshape(-1)
        qdd = (dq - np.asarray(dq_prev, dtype=float).reshape(-1)) / dt
        tau_ext = np.asarray(M.numpy()).reshape(na, na) @ qdd + _to_numpy(C) + _to_numpy(g) - np.asarray(tau_cmd, float)
        return np.linalg.pinv(np.asarray(J, float), rcond=1e-3).T @ tau_ext

    @staticmethod
    def write_effort_isaaclab(robot: Any, tau: np.ndarray, device: Any = None) -> None:
        """A2-verified wrapper route (not used by the native-Newton harness).

        ``robot.actuators.target_command.set_effort_index(value=..., full_data=True)``
        then an explicit ``robot.write_data_to_sim()`` — the latter is required
        because ``Articulation.update(dt)`` only refreshes read-only data.
        """
        import torch  # lazy
        value = torch.as_tensor(np.asarray(tau, dtype=np.float32))
        if value.ndim == 1:
            value = value.unsqueeze(0)
        dev = device if device is not None else robot.data.joint_pos.torch.device
        robot.actuators.target_command.set_effort_index(value=value.to(dev), full_data=True)
        robot.write_data_to_sim()
