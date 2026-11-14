"""Bring-up 1: build the scene, reset a seed, check that every item is where the bridge says it is.
  python isaac/bringup/01_scene.py --headless --seed 0
Paste back the whole output (it is short). Failures here mean spawner kwargs or asset paths need fixing."""
from _launch import launch, check
args, app = launch("scene bring-up")

import torch
from isaac.packing_sim import build, ITEM_NAMES, LAYOUT, _region

sim, scene, ps = build(device=args.device if hasattr(args, "device") and args.device else "cuda:0")
ps.reset(args.seed)
states = ps.object_states()
ok = True
for s in states:
    reg = _region(s, LAYOUT)
    print(f"  {s.name:12s} pos=({s.pos[0]:+.3f},{s.pos[1]:+.3f},{s.pos[2]:+.3f}) region={reg} init={ps.init_regions[s.name]}")
ok &= check("all items above the floor", all(s.pos[2] > -0.05 for s in states))
ok &= check("all items start on the table", all(ps.init_regions[s.name].startswith("table") for s in states),
            str({s.name: ps.init_regions[s.name] for s in states if not ps.init_regions[s.name].startswith("table")}))
ok &= check("symbolic truth matches sim classes", all(ps.truth.objs[s.name].cls == s.cls for s in states))
for a in "LR":
    pos = ps._ee_pose_b(a)[0][0].tolist()
    ok &= check(f"arm {a} ee finite", all(abs(v) < 2 for v in pos), f"{[round(v, 3) for v in pos]}")
print("log:", ps.log[:6])
print("RESULT", "PASS" if ok else "FAIL")
app.close()
