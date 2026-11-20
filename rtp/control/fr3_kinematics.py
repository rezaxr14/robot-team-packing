"""Calibrated Franka Research 3 (FR3 v2) forward kinematics and Jacobian.

Provenance
----------
The constants below are extracted **verbatim** from the MuJoCo Menagerie model
``franka_fr3_v2/fr3v2.xml`` (Apache-2.0, derived from ``frankarobotics/franka_description``
v2.9.0), which is the model we load into Newton for simulation (G-5).  They are
therefore not a hand-typed DH hypothesis: each entry is the ``pos``/``quat`` of a
``<body>`` in that MJCF, and every joint is a revolute ``+Z`` joint at its body
origin.

    link   pos (m)            quat (w, x, y, z)
    ------------------------------------------------------------
    link1  (0,      0,    0.333)   (1, 0, 0, 0)
    link2  (0,      0,    0    )   (0.707107, -0.707107, 0, 0)
    link3  (0,     -0.316,0    )   (0.707107,  0.707107, 0, 0)
    link4  (0.0825, 0,    0    )   (0.707107,  0.707107, 0, 0)
    link5  (-0.0825,0.384,0    )   (0.707107, -0.707107, 0, 0)
    link6  (0,      0,    0    )   (0.707107,  0.707107, 0, 0)
    link7  (0.088,  0,    0    )   (0.707107,  0.707107, 0, 0)
    flange (0,      0,    0.107)   (1, 0, 0, 0)   (fr3v2_link8, no joint)

The frame returned by :func:`fk` is the **flange** frame (``fr3v2_link8``), i.e. the
mounting face of the Franka Hand, which is what ``rtp/control/newton_arm.py`` and the
push test control.  Verified against Newton's own ``eval_fk`` to < 1 mm over 100 random
joint configurations by ``ops/g5_reach.py`` (report: ``reports/g5_reach.json``).

Evidence class: this module is pure geometry.  The *reach test* that validates it
against Newton is ``SIM-POSE``; nothing here is a hardware calibration.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "NQ",
    "Q_HOME",
    "JOINT_LIMITS_LOWER",
    "JOINT_LIMITS_UPPER",
    "FLANGE_OFFSET",
    "fk",
    "jacobian",
    "fk_numpy",
    "jacobian_numpy",
]

NQ = 7

#: Straight-up "ready" joint configuration used by the runbook (A3).
Q_HOME = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

#: Joint limits from the MJCF ``<joint range=...>`` (radians).
JOINT_LIMITS_LOWER = (-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508)
JOINT_LIMITS_UPPER = (2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508)

#: Distance from ``fr3v2_link7`` origin to the flange (``fr3v2_link8``), metres.
FLANGE_OFFSET = 0.107

_SQRT_HALF = math.sqrt(0.5)

#: (translation, quaternion wxyz) of each link frame relative to its parent.
_CHAIN = (
    ((0.0, 0.0, 0.333), (1.0, 0.0, 0.0, 0.0)),          # link1
    ((0.0, 0.0, 0.0), (_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0)),   # link2
    ((0.0, -0.316, 0.0), (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0)),  # link3
    ((0.0825, 0.0, 0.0), (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0)),  # link4
    ((-0.0825, 0.384, 0.0), (_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0)),  # link5
    ((0.0, 0.0, 0.0), (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0)),     # link6
    ((0.088, 0.0, 0.0), (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0)),   # link7
)


def _quat_to_matrix(w: float, x: float, y: float, z: float, dtype, device) -> torch.Tensor:
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=dtype,
        device=device,
    )


def _link_transform(index: int, dtype, device) -> torch.Tensor:
    """Constant 4x4 transform of link ``index`` (0-based) relative to its parent."""
    pos, quat = _CHAIN[index]
    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = _quat_to_matrix(*quat, dtype=dtype, device=device)
    T[:3, 3] = torch.tensor(pos, dtype=dtype, device=device)
    return T


_FLANGE = torch.eye(4)
_FLANGE[2, 3] = FLANGE_OFFSET


def _rot_z(angle: torch.Tensor) -> torch.Tensor:
    """Batched 4x4 rotation about local +Z. ``angle`` shape (B,)."""
    b = angle.shape[0]
    R = torch.zeros((b, 4, 4), dtype=angle.dtype, device=angle.device)
    c, s = torch.cos(angle), torch.sin(angle)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    R[:, 3, 3] = 1.0
    return R


def _prepare(q) -> torch.Tensor:
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=torch.float64)
    if q.shape[-1] != NQ:
        raise ValueError(f"expected the last dimension to be {NQ} joint angles, got {tuple(q.shape)}")
    return q.reshape(-1, NQ)


def _chain_batch(Q: torch.Tensor, dtype, device):
    """Run the chain once, returning the flange pose and per-joint world axes/origins."""
    b = Q.shape[0]
    T = torch.eye(4, dtype=dtype, device=device).expand(b, 4, 4).clone()
    axes, origins = [], []
    for i in range(NQ):
        T = T @ _link_transform(i, dtype, device)
        axes.append(T[:, :3, 2].clone())
        origins.append(T[:, :3, 3].clone())
        T = T @ _rot_z(Q[:, i])
    T = T @ torch.as_tensor(_FLANGE, dtype=dtype, device=device)
    return T, axes, origins


def fk(q, dtype: torch.dtype | None = None):
    """Forward kinematics of the FR3 flange.

    ``q`` is ``(7,)`` or ``(B, 7)`` and may be a list/tuple.  Returns
    ``(position (3,), rotation (3,3))`` for a single configuration, or
    ``(position (B,3), rotation (B,3,3))`` for a batch.
    """
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=dtype or torch.float64)
    dtype = dtype or q.dtype
    dtype = dtype if dtype.is_floating_point else torch.float64
    Q = _prepare(q).to(dtype=dtype)
    T, _, _ = _chain_batch(Q, dtype, Q.device)
    pos, rot = T[:, :3, 3], T[:, :3, :3]
    if q.dim() == 1:
        return pos[0], rot[0]
    return pos, rot


def jacobian(q, dtype: torch.dtype | None = None):
    """Geometric Jacobian of the flange in world coordinates.

    ``q`` is ``(7,)`` or ``(B, 7)``.  Returns ``(6, 7)`` / ``(B, 6, 7)`` with the
    top three rows translational and the bottom three rotational.
    """
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=dtype or torch.float64)
    dtype = dtype or q.dtype
    dtype = dtype if dtype.is_floating_point else torch.float64
    Q = _prepare(q).to(dtype=dtype)
    T, axes, origins = _chain_batch(Q, dtype, Q.device)
    p_e = T[:, :3, 3]
    J = torch.zeros((Q.shape[0], 6, NQ), dtype=dtype, device=Q.device)
    for i in range(NQ):
        z = axes[i]
        r = p_e - origins[i]
        J[:, :3, i] = torch.linalg.cross(z, r)
        J[:, 3:, i] = z
    if q.dim() == 1:
        return J[0]
    return J


def fk_numpy(q):
    """Convenience numpy wrapper (float64): ``(pos (3,), rot (3,3))``."""
    pos, rot = fk(torch.as_tensor(q, dtype=torch.float64))
    return pos.numpy(), rot.numpy()


def jacobian_numpy(q):
    """Convenience numpy wrapper (float64): ``(6, 7)``."""
    return jacobian(torch.as_tensor(q, dtype=torch.float64)).numpy()
