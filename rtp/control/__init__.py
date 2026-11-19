"""L1 Cartesian impedance + compliance supervisor -- simulation only.

SIMULATION CROSS-CHECK ONLY -- NOT A SAFETY CERTIFICATION.
This package implements the already-authored math in ``docs/IMPEDANCE_SPEC.md``
so the torch control law can be cross-checked against an analytic
mass-spring-damper in simulation.  ``compliance`` adds the push-yield-return state
machine, plan-clock gate, leash and clamps (numpy, abstract plant).  Neither
validates the L0 stack, force/pressure limits from ISO 10218-2:2025, or any
real-robot behaviour.  Hardware and L0/L1 sign-off is owned by Claude
(decision D-22); nothing here may be used to claim the arms are safe around
people.
"""

try:  # the torch law needs torch; the numpy compliance supervisor must stay importable without it
    from .impedance import (
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
except ModuleNotFoundError as _e:  # pragma: no cover - only without torch
    if _e.name != "torch":
        raise

from .compliance import ComplianceConfig, ComplianceSupervisor, PROFILES, State

__all__ = [
    "ComplianceConfig",
    "ComplianceSupervisor",
    "PROFILES",
    "State",
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
