"""Tests for the scripted FR3 IK used by arm skills (pure geometry)."""
import numpy as np

from rtp.control import fr3_kinematics as fk
from rtp.control.scripted_ik import solve_ik, rotation_error


def test_position_only_ik_converges_within_1mm():
    q0 = np.asarray(fk.Q_HOME, dtype=float)
    p0, _ = fk.fk_numpy(q0)
    target = p0 + np.array([0.04, 0.0, -0.03])
    r = solve_ik(q0, target)
    assert r.converged and r.pos_error_m < 1e-3
    assert r.path.shape[1] == 7 and len(r.path) > 1


def test_pose_ik_converges():
    q0 = np.asarray(fk.Q_HOME, dtype=float)
    p0, R0 = fk.fk_numpy(q0)
    r = solve_ik(q0, p0 + np.array([0.0, 0.03, 0.0]), R0)
    assert r.pos_error_m < 1e-3 and r.rot_error_rad < 1e-2


def test_rotation_error_is_zero_for_identical_rotations():
    R = np.eye(3)
    assert np.allclose(rotation_error(R, R), 0.0)


def test_joint_limits_respected():
    q0 = np.asarray(fk.Q_HOME, dtype=float)
    target = np.array([0.9, 0.2, 0.6])
    r = solve_ik(q0, target)
    assert np.all(r.q >= np.asarray(fk.JOINT_LIMITS_LOWER) - 1e-6)
    assert np.all(r.q <= np.asarray(fk.JOINT_LIMITS_UPPER) + 1e-6)
