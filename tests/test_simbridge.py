import math

import pytest

from rtp.simbridge import (Layout, ObjState, up_alignment, is_upright, in_bag, compartment_of, region_of,
                           world_from_sim, classify_skill_outcome, crush_force_for)
from rtp.rules import TaskSpec, check_final
from rtp.skills import CATEGORIES

LAY = Layout()
BX, BY = LAY.bag_center
UP = (1.0, 0.0, 0.0, 0.0)
SIDE = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)       # 90 deg about x: lying on its side


def st(name, cls, pos, **kw):
    return ObjState(name=name, cls=cls, pos=pos, **kw)


def test_upright_detection():
    assert up_alignment(UP) == pytest.approx(1.0) and is_upright(UP)
    assert up_alignment(SIDE) == pytest.approx(0.0, abs=1e-9) and not is_upright(SIDE)
    assert not is_upright((0.0, 1.0, 0.0, 0.0))                         # inverted
    assert is_upright((2.0, 0.0, 0.0, 0.0))                             # unnormalised quaternion


def test_table_zones_and_floor_and_held():
    assert region_of(st("a", "book", (-0.5, 0.0, 0.05)), LAY) == "table_left"
    assert region_of(st("a", "book", (0.0, 0.0, 0.05)), LAY) == "table_center"
    assert region_of(st("a", "book", (0.4, 0.0, 0.05)), LAY) == "table_right"
    assert region_of(st("a", "book", (0.0, 0.0, -0.5)), LAY) == "floor"
    assert region_of(st("a", "book", (-0.5, 0, 0.05), held_by="L"), LAY) == "held:L"
    lay_y = Layout(zone_axis=1, left_max=-0.1, right_min=0.1)
    assert region_of(st("a", "book", (5.0, 0.3, 0.0)), lay_y) == "table_right"


def test_compartments_by_depth():
    back, mid, front = BY - LAY.bag_half[1] + 0.02, BY, BY + LAY.bag_half[1] - 0.02
    assert compartment_of((BX, back, 0.1), LAY) == "laptop"
    assert compartment_of((BX, mid, 0.1), LAY) == "main"
    assert compartment_of((BX, front, 0.1), LAY) == "front"
    assert in_bag((BX, BY, 0.1), LAY) and not in_bag((BX + 1.0, BY, 0.1), LAY) and not in_bag((BX, BY, 1.0), LAY)
    assert region_of(st("l", "laptop", (BX, back, 0.1)), LAY) == "bag:laptop"


def packed_states():
    return [st("book1", "book", (BX, BY, 0.05), mass=1.2, size=2.0, init_region="table_left"),
            st("pen1", "pencilcase", (BX, BY, 0.20), mass=0.2, size=1.0, init_region="table_right"),
            st("bot1", "bottle", (BX + 0.05, BY, 0.12), mass=0.9, size=1.5, init_region="table_left"),
            st("lap1", "laptop", (BX, BY - LAY.bag_half[1] + 0.02, 0.15), mass=1.8, size=3.0, fragility=0.4, init_region="table_center"),
            st("lunch1", "lunchbox", (BX, BY, 0.30), mass=0.5, size=2.5, fragility=0.6, init_region="table_right"),
            st("mug", "mug", (0.1, 0.0, 0.05), is_distractor=True, init_region="table_center")]


def test_world_from_sim_scores_a_correct_pack_as_success():
    w = world_from_sim(packed_states(), LAY, pack_order=["book1", "bot1", "pen1", "lap1", "lunch1"])
    spec = TaskSpec(items=["book1", "pen1", "bot1", "lap1", "lunch1"], leave=["mug"], last_item="lunch1")
    chk = check_final(w, spec)
    assert chk["success"], chk
    assert w.bag["laptop"] == ["lap1"] and w.bag["main"][0] == "book1"   # stack ordered by height


def test_world_from_sim_flags_each_failure_mode():
    spec = TaskSpec(items=["book1", "pen1", "bot1", "lap1", "lunch1"], leave=["mug"], last_item="lunch1")
    order = ["book1", "bot1", "pen1", "lap1", "lunch1"]

    def rules_for(mut):
        s = packed_states()
        mut(s)
        return {v["rule"] for v in check_final(world_from_sim(s, LAY, order), spec)["violations"]}, check_final(world_from_sim(s, LAY, order), spec)

    r, _ = rules_for(lambda s: s[2].__setattr__("quat", SIDE))
    assert r == {"bottle_upright"}
    r, _ = rules_for(lambda s: s[3].__setattr__("pos", (BX, BY, 0.15)))                       # laptop in main
    assert "laptop_slot" in r
    r, _ = rules_for(lambda s: s[5].__setattr__("touched", True))                              # distractor moved
    assert r == {"leave_distractors"}
    _, c = rules_for(lambda s: s[4].__setattr__("peak_contact_force", 200.0))                  # lunchbox crushed
    assert c["crushed"] == ["lunch1"] and not c["success"]
    _, c = rules_for(lambda s: s[1].__setattr__("pos", (-0.5, 0.0, 0.05)))                     # item left on the table
    assert c["missing"] == ["pen1"]
    r, _ = rules_for(lambda s: s[0].__setattr__("pos", (BX, BY, 0.38)))                        # heavy book on top
    assert "heavy_bottom" in r


def test_pack_order_overrides_height_for_last_item():
    s = packed_states()
    s[4].pos = (BX, BY, 0.02)                      # lunchbox lowest in the bag but placed last
    w = world_from_sim(s, LAY, pack_order=["book1", "bot1", "pen1", "lap1", "lunch1"])
    assert w.objs["lunch1"].pack_index == max(o.pack_index for o in w.objs.values())


def test_crush_force_scales_with_fragility():
    assert crush_force_for(0.0) > crush_force_for(0.5) > crush_force_for(1.0)


def test_classify_skill_outcome_covers_categories():
    c = classify_skill_outcome
    assert c("grasp") == "success"
    assert c("grasp", hand_empty=False) == "hand_full" and c("grasp", reachable=False) == "out_of_reach"
    assert c("grasp", payload_ok=False) == "overload" and c("grasp", peak_force=99.0, crush_force=45.0) == "crush"
    assert c("grasp", object_followed_lift=False) == "slip"
    assert c("place", holding_target=False) == "not_holding" and c("place", holding_target=True, lock_ok=False) == "locked"
    assert c("place", holding_target=True, accepts=False) == "wrong_slot" and c("place", holding_target=True, space_ok=False) == "no_space"
    assert c("place", holding_target=True, placed_in="main", wanted="main") == "success"
    assert c("place", holding_target=True, placed_in="front", wanted="main") == "other_failure"
    assert c("handover", holding_target=True, hand_empty=False) == "hand_full"
    for cat in ("success", "slip", "crush", "out_of_reach", "wrong_slot", "no_space", "locked", "overload", "not_holding", "hand_full"):
        assert cat in CATEGORIES


def test_quaternion_order_wxyz_vs_xyzw():
    import math as m
    side_xyzw = (m.sin(m.pi / 4), 0.0, 0.0, m.cos(m.pi / 4))              # 90 deg about x, (x, y, z, w)
    assert up_alignment(side_xyzw, "xyzw") == pytest.approx(0.0, abs=1e-9)
    assert is_upright((0.0, 0.0, 0.0, 1.0), order="xyzw") and not is_upright(side_xyzw, order="xyzw")
    assert up_alignment((1.0, 0.0, 0.0, 0.0), "wxyz") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        up_alignment((1, 0, 0, 0), "bogus")
    flat = st("b", "bottle", (BX, BY, 0.1), quat=side_xyzw, quat_order="xyzw")
    up = st("b", "bottle", (BX, BY, 0.1), quat=(0.0, 0.0, 0.0, 1.0), quat_order="xyzw")
    assert not world_from_sim([flat], LAY).objs["b"].upright and world_from_sim([up], LAY).objs["b"].upright
