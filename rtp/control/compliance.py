"""Compliance supervisor: push-yield-return state machine on top of a Cartesian impedance law.

Implements the L1 behaviour of ``docs/IMPEDANCE_SPEC.md`` (D-26): a person can push any part of an arm,
the arm acts as a spring, the planned trajectory pauses (plan-clock gate), and on release the arm
returns to the pose it was planned to be in at bounded speed, then the trajectory resumes from where it
paused. Pure numpy and plant-agnostic: poses/errors are 6-vectors (3 linear, 3 rotational tangent-space);
the caller (torch ``CartesianImpedance`` or the C++ reference) maps the returned Cartesian wrench F to joint
torques with ``Jᵀ F`` and adds null-space/coriolis terms.

States: TRACK -> YIELD (push or |e| > e_pause) -> RETURN (released for t_rel) -> TRACK (settled);
FAULT on a clamp storm or a yield longer than T_yield_max. Invariants enforced every tick:
|F_lin| <= F_lin_max, |M| <= M_max, |x_ref - x| (linear) <= delta_max (leash), speed damper at the
speed cap. NOT a safety certification: see docs/IMPEDANCE_SPEC.md section 8.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

PROFILES = {  # (K_lin N/m, K_rot Nm/rad); allow-list, the Brain never sets these (spec section 5)
    "free_motion": (200.0, 15.0), "approach": (300.0, 20.0), "grasp": (500.0, 25.0),
    "transport": (250.0, 15.0), "place": (400.0, 20.0), "retreat": (200.0, 15.0), "handover": (150.0, 10.0),
}


class State(str, Enum):
    TRACK = "TRACK"
    YIELD = "YIELD"
    RETURN = "RETURN"
    FAULT = "FAULT"


@dataclass
class ComplianceConfig:
    zeta: float = 1.0
    mass_lin: float = 3.0          # effective Cartesian mass (kg); identified in Phase 12
    inertia_rot: float = 0.15      # effective rotational inertia (kg m^2)
    f_push: float = 8.0            # N
    f_release: float = 4.0         # N
    m_push: float = 1.0            # Nm (rotational push detection)
    m_release: float = 0.5
    e_pause: float = 0.03          # m
    t_rel: float = 0.2             # s
    delta_max: float = 0.20        # m, leash
    rot_delta_max: float = 0.6     # rad, rotational leash
    f_lin_max: float = 40.0        # N   PLACEHOLDER until ISO 10218-2:2025 limits are read
    f_fault: float = 80.0          # N   external force this large is a collision/abuse, not a push: controlled stop
    m_max: float = 4.0             # Nm  PLACEHOLDER
    v_lin_max: float = 0.25        # m/s PLACEHOLDER
    v_return: float = 0.10         # m/s
    w_return: float = 0.30         # rad/s
    d_lim: float = 3000.0          # speed-saturating damper gain
    t_yield_max: float = 30.0      # s
    eps_e: float = 0.003           # m, resume linear tolerance
    eps_rot: float = 0.02          # rad
    t_settle: float = 0.15         # s
    v_settled: float = 0.01        # m/s
    stiffness_tau: float = 0.1     # s, first-order filter on K (gain changes never step)
    ref_tau: float = 0.02          # s, first-order filter on the reference in TRACK
    fault_clamp_ticks: int = 2000  # consecutive force-clamp ticks that trip FAULT


@dataclass
class Output:
    force: np.ndarray              # 6-vector wrench to map through Jᵀ
    state: State
    x_ref: np.ndarray
    s: float                       # plan clock
    k_lin: float
    events: list = field(default_factory=list)
    flags: dict = field(default_factory=dict)


def _n(v) -> float:
    return float(np.linalg.norm(v))


class ComplianceSupervisor:
    def __init__(self, cfg: ComplianceConfig | None = None, profile: str = "free_motion"):
        self.cfg = cfg or ComplianceConfig()
        self.reset(profile)

    def reset(self, profile: str = "free_motion", x0=None):
        if profile not in PROFILES:
            raise KeyError(f"unknown stiffness profile {profile!r}; allowed: {sorted(PROFILES)}")
        self.profile = profile
        self.state = State.TRACK
        self.s = 0.0
        self.x_ref = None if x0 is None else np.asarray(x0, float).copy()
        self.x_hold = None
        self.k_lin, self.k_rot = PROFILES[profile]
        self._t_rel = self._t_settle = self._t_yield = 0.0
        self._clamp_run = 0
        self.yield_count = 0
        self.xref_dot = np.zeros(6)

    def set_profile(self, profile: str):
        if profile not in PROFILES:
            raise KeyError(f"unknown stiffness profile {profile!r}; allowed: {sorted(PROFILES)}")
        self.profile = profile

    # ------------------------------------------------------------------
    def step(self, dt: float, x, xdot, f_ext, plan_fn, scale: float = 1.0) -> Output:
        """One control tick. ``plan_fn(s) -> 6-vector`` is the planned pose at plan-clock time ``s``;
        ``f_ext`` is the estimated external wrench (6-vector); ``scale`` in [0.5, 1.5] is the learned-policy
        stiffness modulation (clipped here, never trusted)."""
        c = self.cfg
        x, xdot, f_ext = (np.asarray(a, float) for a in (x, xdot, f_ext))
        events: list = []
        flags = {"force": False, "leash": False, "speed": False}

        k_t, k_r = PROFILES[self.profile]
        scale = float(np.clip(scale, 0.5, 1.5))
        a = dt / (c.stiffness_tau + dt)                      # K follows the profile through a first-order filter
        self.k_lin += a * (k_t * scale - self.k_lin)
        self.k_rot += a * (k_r * scale - self.k_rot)

        if self.x_ref is None:
            self.x_ref = np.asarray(plan_fn(self.s), float).copy()
        f_n, m_n = _n(f_ext[:3]), _n(f_ext[3:])
        if f_n > c.f_fault and self.state != State.FAULT:
            self.state = State.FAULT
            events.append("FAULT")
        pushing = f_n > c.f_push or m_n > c.m_push
        released = f_n < c.f_release and m_n < c.m_release

        self.xref_dot = np.zeros(6)
        if self.state == State.TRACK:
            self.s += dt
            target = np.asarray(plan_fn(self.s), float)
            prev = self.x_ref.copy()
            self.x_ref += (dt / (c.ref_tau + dt)) * (target - self.x_ref)
            self.xref_dot = (self.x_ref - prev) / dt            # velocity feed-forward: no steady lag on a moving plan
            e = x - self.x_ref
            if pushing or _n(e[:3]) > c.e_pause:
                self.s -= dt                                  # plan clock gate: this tick does not count
                self.state, self.x_hold = State.YIELD, self.x_ref.copy()
                self._t_rel = self._t_yield = 0.0
                self.yield_count += 1
                events.append("PUSHED")
        elif self.state == State.YIELD:
            self._t_yield += dt
            self.x_ref = self.x_hold
            # leash: the held reference follows the person. It is stiffness-aware, so the spring force K*delta can
            # never exceed the force/moment caps by construction (clamps are then rare transients, not steady state).
            lash_lin = min(c.delta_max, 0.95 * c.f_lin_max / self.k_lin)
            lash_rot = min(c.rot_delta_max, 0.95 * c.m_max / self.k_rot)
            lin = x[:3] - self.x_hold[:3]
            if _n(lin) > lash_lin:
                self.x_hold[:3] += lin - lin / _n(lin) * lash_lin
                flags["leash"] = True
            rot = x[3:] - self.x_hold[3:]
            if _n(rot) > lash_rot:
                self.x_hold[3:] += rot - rot / _n(rot) * lash_rot
                flags["leash"] = True
            self._t_rel = self._t_rel + dt if released else 0.0
            if self._t_rel >= c.t_rel:
                self.state, self._t_settle = State.RETURN, 0.0
                events.append("YIELDED")
            elif self._t_yield > c.t_yield_max:
                self.state = State.FAULT
                events.append("FAULT")
        elif self.state == State.RETURN:
            self.x_ref = self.x_hold
            e = x - self.x_ref
            if pushing:
                self.state, self._t_rel, self._t_yield = State.YIELD, 0.0, 0.0
                self.yield_count += 1
                events.append("PUSHED")
            else:
                settled = (_n(e[:3]) < c.eps_e and _n(e[3:]) < c.eps_rot and _n(xdot[:3]) < c.v_settled)
                self._t_settle = self._t_settle + dt if settled else 0.0
                if self._t_settle >= c.t_settle:
                    self.state = State.TRACK
                    events.append(f"RESUMED err={_n(e[:3]):.6f}")

        e = x - self.x_ref
        # critically damped per axis group: D = 2 zeta sqrt(K * inertia)
        d_lin = 2.0 * c.zeta * np.sqrt(self.k_lin * c.mass_lin)
        d_rot = 2.0 * c.zeta * np.sqrt(self.k_rot * c.inertia_rot)
        ev = xdot - self.xref_dot
        force = np.concatenate([-self.k_lin * e[:3] - d_lin * ev[:3], -self.k_rot * e[3:] - d_rot * ev[3:]])

        # speed saturation (soft limiter, bounded so it can never reverse the velocity in one tick)
        letting_go = self.state == State.RETURN or (self.state == State.YIELD and released)
        cap = c.v_return if letting_go else c.v_lin_max      # never whip back: cap applies the moment the person lets go
        v = _n(xdot[:3])
        if v > cap:
            mag = min(c.d_lim * (v - cap), 0.5 * c.mass_lin * (v - cap) / dt)
            force[:3] -= mag * xdot[:3] / v
            flags["speed"] = True
        w = _n(xdot[3:])
        wcap = c.w_return if letting_go else 3.0 * c.w_return
        if w > wcap:
            mag = min(c.d_lim * 0.05 * (w - wcap), 0.5 * c.inertia_rot * (w - wcap) / dt)
            force[3:] -= mag * xdot[3:] / w
            flags["speed"] = True

        fl, mm = _n(force[:3]), _n(force[3:])
        if fl > c.f_lin_max:
            force[:3] *= c.f_lin_max / fl
            flags["force"] = True
        if mm > c.m_max:
            force[3:] *= c.m_max / mm
            flags["force"] = True
        self._clamp_run = self._clamp_run + 1 if flags["force"] else 0
        if self._clamp_run >= c.fault_clamp_ticks and self.state != State.FAULT:
            self.state = State.FAULT
            events.append("FAULT")
        if self.state == State.FAULT:                          # controlled stop: clamped pure damping, no spring
            force = np.concatenate([-d_lin * xdot[:3], -d_rot * xdot[3:]])
            fl, mm = _n(force[:3]), _n(force[3:])
            if fl > c.f_lin_max:
                force[:3] *= c.f_lin_max / fl
            if mm > c.m_max:
                force[3:] *= c.m_max / mm
        return Output(force=force, state=self.state, x_ref=self.x_ref.copy(), s=self.s, k_lin=float(self.k_lin),
                      events=events, flags=flags)
