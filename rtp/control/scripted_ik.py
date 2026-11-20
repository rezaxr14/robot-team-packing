"""Damped least-squares Cartesian IK for the calibrated FR3 flange (scripted skills).

This is the deterministic "expert" used by the scripted arm skills before any learning:
given a start joint configuration and a target flange pose, it returns a joint configuration
and the per-step joint path. It never sets stiffness, force or speed limits — those belong to
the impedance/compliance layer (`rtp/control/compliance.py`), per D-31.

Evidence class: pure geometry (verified against the calibrated FR3 model in A1). The physics
outcome of executing the path belongs to the Newton skill regression (A7), not to this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fr3_kinematics import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER, fk_numpy, jacobian_numpy

__all__ = ["IKResult", "solve_ik", "rotation_error"]


@dataclass
class IKResult:
    q: np.ndarray                 # final joint configuration (7,)
    path: np.ndarray              # joint path including q0 (N,7)
    pos_error_m: float
    rot_error_rad: float
    converged: bool


def rotation_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Rotation vector of ``R_des @ R_cur.T`` (world-frame correction)."""
    R = R_des @ R_cur.T
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    s = math.sin(ang)
    if abs(s) < 1e-8:
        return np.zeros(3)
    scale = ang / (2.0 * s)
    return scale * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def solve_ik(q0, target_pos, target_rot=None, *, max_iters: int = 300,
             step: float = 0.35, damping: float = 1e-3,
             pos_tol: float = 1e-4, rot_tol: float = 1e-3,
             clamp_joints: bool = True) -> IKResult:
    """Damped least-squares IK to a flange position (and optional 3x3 rotation).

    ``step`` caps the Cartesian correction per iteration (metres / radians mixed vector);
    the joint update uses the Moore-Penrose pseudoinverse with ``damping``. Joint limits from
    the Menagerie FR3 model are enforced by clamping when ``clamp_joints`` is true.
    """
    q = np.asarray(q0, dtype=float).copy()
    if q.shape != (7,):
        raise ValueError(f"q0 must have shape (7,), got {q.shape}")
    p_des = np.asarray(target_pos, dtype=float).reshape(3)
    R_des = None if target_rot is None else np.asarray(target_rot, dtype=float).reshape(3, 3)
    lo, hi = np.asarray(JOINT_LIMITS_LOWER), np.asarray(JOINT_LIMITS_UPPER)
    path = [q.copy()]
    for _ in range(max_iters):
        p, R = fk_numpy(q)
        e = p_des - p
        if R_des is not None:
            e = np.concatenate((e, rotation_error(R, R_des)))
        norm = float(np.linalg.norm(e))
        limit = pos_tol if R_des is None else math.hypot(pos_tol, rot_tol)
        if norm <= limit:
            break
        if norm > step:
            e = e * (step / norm)
        J = jacobian_numpy(q)[:len(e)]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(len(e)), e)
        q = q + dq
        if clamp_joints:
            q = np.clip(q, lo + 1e-3, hi - 1e-3)
        path.append(q.copy())
    p, R = fk_numpy(q)
    perr = float(np.linalg.norm(p_des - p))
    rerr = 0.0 if R_des is None else float(np.linalg.norm(rotation_error(R, R_des)))
    ok = perr <= pos_tol and (R_des is None or rerr <= rot_tol)
    return IKResult(q=q, path=np.asarray(path), pos_error_m=perr, rot_error_rad=rerr, converged=ok)
