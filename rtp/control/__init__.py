"""L1 Cartesian impedance -- simulation cross-check only.

SIMULATION CROSS-CHECK ONLY -- NOT A SAFETY CERTIFICATION.
This package implements the already-authored math in ``docs/IMPEDANCE_SPEC.md``
so the torch control law can be cross-checked against an analytic
mass-spring-damper in simulation.  It does NOT implement or validate the full
L0/L1 safety stack, the leash, force/pressure limits from ISO 10218-2:2025, or
any real-robot behaviour.  Hardware and L0/L1 sign-off is owned by Claude
(decision D-22); nothing here may be used to claim the arms are safe around
people.
"""

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
