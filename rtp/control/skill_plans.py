"""Scripted Cartesian plans for arm skills (geometry only; no physics claim).

A skill needs two things from a plan: a joint path to execute and a task-frame reference for the
compliance supervisor. :func:`plan_from_joint_path` turns the output of
:func:`rtp.control.scripted_ik.solve_ik` into both, so the same path drives a position-controlled
backend and the bounded-delta impedance controller without duplicating geometry.

The task-frame convention matches :meth:`rtp.control.newton_arm.NewtonCompliantArm.task_state`:
``[world position; rotvec(R(q) R_home^T)]``. Timings are in seconds; the caller chooses the control
tick. No stiffness, force or speed limit is set here (D-31); those belong to the controller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fr3_kinematics as fk
from .scripted_ik import rotation_error

__all__ = ["ScriptedPlan", "plan_from_joint_path", "plan_to_pose"]


@dataclass
class ScriptedPlan:
    """A joint path and its task-frame reference, parameterised by elapsed seconds."""

    path: np.ndarray                  # (N, 7) joint configurations, endpoints included
    duration_s: float
    q_home: np.ndarray

    def joint_at(self, s: float) -> np.ndarray:
        """Linearly interpolated joint configuration at elapsed time ``s``."""
        u = min(max(float(s) / self.duration_s, 0.0), 1.0)
        idx = u * (len(self.path) - 1)
        i = min(int(idx), len(self.path) - 2)
        return self.path[i] + (idx - i) * (self.path[i + 1] - self.path[i])

    def pose_at(self, s: float) -> np.ndarray:
        """Task-frame reference ``[px,py,pz,rx,ry,rz]`` at elapsed time ``s``."""
        q = self.joint_at(s)
        p, R = fk.fk_numpy(q)
        _, R_home = fk.fk_numpy(self.q_home)
        return np.concatenate([np.asarray(p, float), rotation_error(R_home, R)])

    @property
    def end_pose(self) -> np.ndarray:
        return self.pose_at(self.duration_s)


def plan_from_joint_path(path, duration_s: float, q_home=None) -> ScriptedPlan:
    """Wrap a joint path (e.g. ``IKResult.path``) as a timed :class:`ScriptedPlan`."""
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 7 or len(arr) < 2:
        raise ValueError(f"path must have shape (N>=2, 7), got {arr.shape}")
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    home = np.asarray(fk.Q_HOME if q_home is None else q_home, dtype=float)
    return ScriptedPlan(path=arr, duration_s=float(duration_s), q_home=home)


def plan_to_pose(q0, target_pos, target_rot=None, duration_s: float = 2.0, **ik_kwargs) -> ScriptedPlan:
    """Solve IK from ``q0`` to a flange pose and return the timed plan."""
    from .scripted_ik import solve_ik

    result = solve_ik(q0, target_pos, target_rot, **ik_kwargs)
    if not result.converged:
        raise ValueError(f"IK did not converge: pos_error={result.pos_error_m:.4g} m, "
                         f"rot_error={result.rot_error_rad:.4g} rad")
    return plan_from_joint_path(result.path, duration_s, q_home=q0)
