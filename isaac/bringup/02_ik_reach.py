"""Bring-up 2: can each arm reach above each item in its zone (and above the bag compartments)?
  python isaac/bringup/02_ik_reach.py --headless"""
from _launch import launch, check
args, app = launch("IK reach bring-up")

from isaac.packing_sim import build, ITEM_NAMES, LAYOUT, HOVER, _region

sim, scene, ps = build()
ps.reset(args.seed)
tot = ok = 0
for s in ps.object_states():
    reg = _region(s, LAYOUT)
    for a in "LR":
        if reg in ps.arm_zones[a]:
            reached, err = ps.goto(a, (s.pos[0], s.pos[1], s.pos[2] + HOVER))
            tot += 1; ok += reached
            print(f"  arm {a} -> above {s.name:12s} ({reg}): reached={reached} err={err:.4f}")
            ps._sk_retreat(a, None, {}, [])
cx, cy = LAYOUT.bag_center
for a in "LR":
    for dy in (-0.10, 0.0, 0.10):
        reached, err = ps.goto(a, (cx, cy + dy, 0.30))
        tot += 1; ok += reached
        print(f"  arm {a} -> bag y{dy:+.2f}: reached={reached} err={err:.4f}")
    ps._sk_retreat(a, None, {}, [])
check("reach success rate >= 90%", ok / max(1, tot) >= 0.9, f"({ok}/{tot})")
app.close()
