"""CPU-only tests for the pure helpers of the A7 S4-S6 ladder fixtures.

No torch, no newton, no GPU: the three ops scripts import only numpy at module level, so these
helpers can be imported and tested on the training/default venv without the Newton environment.
These tests verify helper logic only -- they are NOT physics evidence for any A7 rung.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ops.a7_s4_place import (
    box_bag_penetration,
    compartment_contains_center,
    make_bag,
)
from ops.a7_s5_weigh import (
    estimate_mass_from_gravity_torque,
    recover_flange_wrench,
    relative_error,
)
from ops.a7_s6_handover import (
    HandoverStateMachine,
    box_below_clearance,
    tilt_angle_deg,
)


# --- S4: compartment containment and penetration --------------------------------------------
def test_make_bag_builds_floor_dividers_and_three_compartments():
    walls, comps = make_bag((0.0, 0.0, 0.90), (0.12, 0.07), n=3)
    # floor + (n-1) dividers + 4 outer walls
    assert len(walls) == 1 + (3 - 1) + 4
    assert len(comps) == 3
    assert [c.name for c in comps] == ["compartment_0", "compartment_1", "compartment_2"]
    # compartment centres are distinct and ordered in x
    xs = [float(c.center[0]) for c in comps]
    assert xs == sorted(xs) and len(set(xs)) == 3


def test_compartment_contains_center_strict_and_margin():
    _walls, comps = make_bag((0.0, 0.0, 0.90), (0.12, 0.07), n=3)
    c = comps[1]
    assert compartment_contains_center(c.center, c) is True
    assert compartment_contains_center(c.center + np.array([0.0, 0.5, 0.0]), c) is False
    # a margin shrinks the volume: a point near the wall falls outside
    near_wall = c.center + c.half * 0.99
    assert compartment_contains_center(near_wall, c) is True
    assert compartment_contains_center(near_wall, c, margin=0.02 * max(c.half)) is False


def test_only_one_compartment_contains_a_given_center():
    _walls, comps = make_bag((0.0, 0.0, 0.90), (0.12, 0.07), n=3)
    point = comps[1].center
    hits = [i for i, c in enumerate(comps) if compartment_contains_center(point, c)]
    assert hits == [1]


def test_box_bag_penetration_is_zero_on_floor_and_grows_below_it():
    walls, _comps = make_bag((0.0, 0.0, 0.90), (0.12, 0.07), n=3)
    bh = 0.025
    resting = np.array([0.0, 0.0, 0.90 + bh])          # bottom exactly on floor top
    sunk = np.array([0.0, 0.0, 0.90 + bh - 0.005])
    assert box_bag_penetration(resting, np.array([bh, bh, bh]), walls) == pytest.approx(0.0, abs=1e-9)
    assert box_bag_penetration(sunk, np.array([bh, bh, bh]), walls) == pytest.approx(0.005, abs=1e-9)


# --- S5: mass estimate math ------------------------------------------------------------------
def _jacobian():
    rng = np.random.default_rng(0)
    return rng.standard_normal((6, 7))                 # full row rank generically


def test_recover_flange_wrench_returns_downward_load():
    J = _jacobian()
    m, g = 0.7, 9.81
    F = np.array([0.0, 0.0, -m * g, 0.0, 0.0, 0.0])
    dtau = J.T @ F
    F_hat = recover_flange_wrench(J, dtau, np.zeros(7))
    np.testing.assert_allclose(F_hat, F, atol=1e-9)


def test_mass_estimate_matches_true_mass_over_range():
    J = _jacobian()
    for m in (0.1, 0.3, 1.0, 2.0, 3.0):
        dtau = J.T @ np.array([0.0, 0.0, -m * 9.81, 0.0, 0.0, 0.0])
        est = estimate_mass_from_gravity_torque(J, dtau, np.zeros(7))
        assert est == pytest.approx(m, rel=1e-9, abs=1e-9)
        assert relative_error(est, m) < 1e-9


def test_mass_estimate_uses_the_difference_not_absolute_torque():
    J = _jacobian()
    m = 1.2
    F = np.array([0.0, 0.0, -m * 9.81, 0.0, 0.0, 0.0])
    # A large bare-arm gravity torque plus the payload difference must cancel to the same mass.
    tau_bare = J.T @ np.array([3.0, -2.0, 5.0, 0.5, -0.5, 1.0])
    tau_with = tau_bare + J.T @ F
    est = estimate_mass_from_gravity_torque(J, tau_with, tau_bare)
    assert est == pytest.approx(m, rel=1e-9, abs=1e-9)


def test_mass_estimate_without_difference_is_zero():
    J = _jacobian()
    est = estimate_mass_from_gravity_torque(J, np.zeros(7), np.zeros(7))
    assert est == pytest.approx(0.0, abs=1e-12)


# --- S6: handover state transitions and geometry helpers -------------------------------------
def test_handover_happy_path_transitions():
    sm = HandoverStateMachine()
    assert sm.state == "L_HOLD"
    assert sm.step("R_GRASPED") == "BOTH"
    assert sm.step("L_RELEASED") == "R_HOLD"
    assert sm.step("R_LIFTED") == "DONE"
    assert sm.history == ["L_HOLD", "BOTH", "R_HOLD", "DONE"]
    assert sm.transferred_without_ground_contact() is True


@pytest.mark.parametrize("state,event", [
    ("L_HOLD", "L_RELEASED"),   # cannot release before R grasps
    ("L_HOLD", "R_LIFTED"),     # cannot lift before release
    ("BOTH", "R_GRASPED"),      # duplicate grasp
    ("DONE", "R_LIFTED"),       # terminal state
])
def test_handover_rejects_invalid_transitions(state, event):
    sm = HandoverStateMachine()
    for ev in ("R_GRASPED", "L_RELEASED", "R_LIFTED"):
        if sm.state == state:
            break
        sm.step(ev)
    assert sm.state == state
    with pytest.raises(ValueError):
        sm.step(event)


def test_handover_ground_contact_invalidates_transfer():
    sm = HandoverStateMachine()
    sm.step("R_GRASPED")
    sm.step("L_RELEASED")
    sm.step("R_LIFTED")
    assert sm.state == "DONE"
    sm.note_ground_contact()
    assert sm.transferred_without_ground_contact() is False


def test_tilt_angle_deg():
    assert tilt_angle_deg([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
    # 90 deg about world x: local +z tips into -y, tilt from +z is 90 deg
    s = math.sqrt(0.5)
    assert tilt_angle_deg([s, s, 0.0, 0.0]) == pytest.approx(90.0, abs=1e-9)
    # 45 deg about world x
    c45, s45 = math.cos(math.radians(22.5)), math.sin(math.radians(22.5))
    assert tilt_angle_deg([c45, s45, 0.0, 0.0]) == pytest.approx(45.0, abs=1e-9)


def test_box_below_clearance():
    table_top, bh = 0.90, 0.025
    clear = np.array([0.0, 0.0, table_top + bh + 0.01])
    touching = np.array([0.0, 0.0, table_top + bh - 0.001])
    assert box_below_clearance(clear, table_top, bh) is False
    assert box_below_clearance(touching, table_top, bh) is True
