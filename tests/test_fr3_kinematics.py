"""A1 tests: analytic checks for the calibrated FR3 flange kinematics.

The Newton cross-check itself is ``ops/g5_reach.py`` and writes ``reports/g5_reach.json``;
these tests are the fast, dependency-light checks that guard the embedded constants.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from rtp.control import fr3_kinematics as fk


def test_zero_pose_reach_matches_newton_constant():
    # Newton eval_fk on franka_fr3_v2/fr3v2.xml, recorded in reports/g5_reach.json.
    NEWTON_ZERO = np.array([0.088001, 0.0, 0.926001])
    pos, rot = fk.fk(np.zeros(7))
    assert np.allclose(pos.numpy(), NEWTON_ZERO, atol=2e-3)
    assert abs(float(np.linalg.norm(pos.numpy())) - 0.930173) < 2e-3
    assert torch.allclose(rot @ rot.T, torch.eye(3, dtype=torch.float64), atol=1e-10)


def test_home_pose_matches_newton_constant():
    NEWTON_HOME = np.array([0.307021, 0.0, 0.590270])
    pos, _ = fk.fk(fk.Q_HOME)
    assert np.allclose(pos.numpy(), NEWTON_HOME, atol=2e-3)


def test_jacobian_matches_finite_difference():
    rng = np.random.default_rng(1234)
    for _ in range(5):
        q = rng.uniform(fk.JOINT_LIMITS_LOWER, fk.JOINT_LIMITS_UPPER)
        J = fk.jacobian_numpy(q)
        assert J.shape == (6, 7)
        eps = 1e-7
        Jfd = np.zeros_like(J)
        for i in range(7):
            qp, qm = q.copy(), q.copy()
            qp[i] += eps
            qm[i] -= eps
            pp, Rp = fk.fk_numpy(qp)
            pm, Rm = fk.fk_numpy(qm)
            Jfd[:3, i] = (pp - pm) / (2 * eps)
            # Angular columns: world-frame increment is Rp @ Rm^T (R^T dR would
            # give the axis in the flange frame, not in world).
            dR = Rp @ Rm.T
            cos = (np.trace(dR) - 1.0) / 2.0
            ang = math.acos(float(np.clip(cos, -1.0, 1.0)))
            if ang < 1e-12:
                w = np.zeros(3)
            else:
                w = (1.0 / (2.0 * math.sin(ang))) * np.array(
                    [dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]
                ) * ang
            Jfd[3:, i] = w / (2 * eps)
        np.testing.assert_allclose(J, Jfd, atol=1e-5, rtol=1e-5)


def test_batched_matches_single():
    q = np.linspace(fk.JOINT_LIMITS_LOWER, fk.JOINT_LIMITS_UPPER, 3)
    pos_b, rot_b = fk.fk(q)
    J_b = fk.jacobian(q)
    assert pos_b.shape == (3, 3) and rot_b.shape == (3, 3, 3) and J_b.shape == (3, 6, 7)
    for i in range(3):
        p, R = fk.fk(q[i])
        np.testing.assert_allclose(pos_b[i].numpy(), p.numpy(), atol=1e-12)
        np.testing.assert_allclose(rot_b[i].numpy(), R.numpy(), atol=1e-12)
        np.testing.assert_allclose(J_b[i].numpy(), fk.jacobian(q[i]).numpy(), atol=1e-12)


def test_embedded_limits_are_valid_and_narrower_than_reach():
    lo = np.asarray(fk.JOINT_LIMITS_LOWER)
    hi = np.asarray(fk.JOINT_LIMITS_UPPER)
    assert lo.shape == hi.shape == (7,)
    assert np.all(lo < hi)
    # Joint 4 is a half-range elbow joint on the Menagerie model.
    assert hi[3] < 0.0 and lo[3] < hi[3]
    assert fk.FLANGE_OFFSET == pytest.approx(0.107)


def test_rejects_wrong_joint_count():
    with pytest.raises(ValueError):
        fk.fk([0.0] * 6)
    with pytest.raises(ValueError):
        fk.jacobian([0.0] * 8)
