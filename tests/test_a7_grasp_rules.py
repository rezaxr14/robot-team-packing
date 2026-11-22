"""CPU tests for the A7 grasp rung rules (no simulator)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))

from rtp.simbridge import slip_bound_holds  # noqa: E402


def test_lift_rule_is_deterministic_monotone_and_capped():
    from a7_s2_lift_v2 import lift_delta_for_mass

    assert lift_delta_for_mass(0.1) < lift_delta_for_mass(0.5) < lift_delta_for_mass(1.0)
    assert lift_delta_for_mass(1.0) == lift_delta_for_mass(2.0) == lift_delta_for_mass(9.0)
    assert lift_delta_for_mass(0.5) >= 0.12
    assert lift_delta_for_mass(1.0) >= 0.14
    assert lift_delta_for_mass(-1.0) == lift_delta_for_mass(0.0) == 0.10  # clip at the lower end


def test_slip_bound_two_pads():
    # 2*0.3*8.2 = 4.92 N >= 0.5*9.81 = 4.905: holds just above the boundary
    assert slip_bound_holds(0.5, 0.3, 8.2)
    assert not slip_bound_holds(0.5, 0.3, 8.1)
    # heavier needs proportionally more force
    assert not slip_bound_holds(3.0, 0.3, 15.0)
    assert slip_bound_holds(3.0, 0.9, 16.4)
