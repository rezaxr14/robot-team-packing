"""Tests for scripted skill plans (pure geometry, no physics)."""
import numpy as np
import pytest

from rtp.control import fr3_kinematics as fk
from rtp.control.scripted_ik import solve_ik
from rtp.control.skill_plans import plan_from_joint_path, plan_to_pose


def test_plan_endpoints_match_start_and_target():
    q0 = np.asarray(fk.Q_HOME, dtype=float)
    p0, R0 = fk.fk_numpy(q0)
    target = p0 + np.array([0.04, 0.0, -0.03])
    plan = plan_to_pose(q0, target, duration_s=2.0)
    start = plan.pose_at(0.0)
    end = plan.pose_at(plan.duration_s)
    assert np.linalg.norm(start[:3] - p0) < 1e-3
    assert np.linalg.norm(end[:3] - target) < 1e-3
    assert np.linalg.norm(end[3:]) < 0.1              # position-only IK does not preserve orientation


def test_joint_and_pose_interpolation_stay_bounded():
    q0 = np.asarray(fk.Q_HOME, dtype=float)
    plan = plan_to_pose(q0, fk.fk_numpy(q0)[0] + np.array([0.0, 0.03, -0.02]), duration_s=1.0)
    poses = np.asarray([plan.pose_at(i / 50) for i in range(51)])
    steps = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    assert steps.max() < 0.02                      # no Cartesian jump along the scripted path
    joints = np.asarray([plan.joint_at(i / 50) for i in range(51)])
    assert np.all(joints >= np.asarray(fk.JOINT_LIMITS_LOWER) - 1e-6)
    assert np.all(joints <= np.asarray(fk.JOINT_LIMITS_UPPER) + 1e-6)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        plan_from_joint_path([[0.0] * 7], 1.0)
    with pytest.raises(ValueError):
        plan_from_joint_path([[0.0] * 7, [0.1] * 7], 0.0)
    with pytest.raises(ValueError):
        plan_to_pose(fk.Q_HOME, [10.0, 10.0, 10.0], max_iters=5)


def test_plan_accepts_existing_ik_path():
    q0 = np.asarray(fk.Q_HOME, dtype=float)
    res = solve_ik(q0, fk.fk_numpy(q0)[0] + np.array([0.02, 0.0, 0.0]))
    plan = plan_from_joint_path(res.path, 0.5, q_home=q0)
    assert plan.path.shape == res.path.shape
    assert np.allclose(plan.joint_at(plan.duration_s), res.q, atol=1e-9)
