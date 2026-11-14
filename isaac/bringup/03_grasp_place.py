"""Bring-up 3: one pick-and-place per class with a force from the Brain's own formula.
  python isaac/bringup/03_grasp_place.py --headless
Prints the outcome category of each skill and where the item ended up."""
from _launch import launch, check
args, app = launch("grasp/place bring-up")

from isaac.packing_sim import build, LAYOUT, _region
from rtp.brain import guess_force
from rtp.perception import oracle_report
from rtp.simbridge import region_of

sim, scene, ps = build()
ps.reset(args.seed)
scene_json = {o["name"]: o for o in oracle_report(ps.truth, "A").content["objects"]}
results = []
for cls in ("book", "notebook", "pencilcase", "bottle", "lunchbox", "laptop"):
    name = next(s.name for s in ps.object_states() if s.cls == cls)
    st = {s.name: s for s in ps.object_states()}[name]
    arm = "L" if _region(st, LAYOUT) in ("table_left", "table_center") else "R"
    f = guess_force(scene_json[name])
    g = ps.run_skill(arm, "grasp", name, {"force": f})
    comp = "laptop" if cls == "laptop" else "main"
    p = ps.run_skill(arm, "place", name, {"compartment": comp, "orientation": "upright"}) if g.category == "success" else None
    ps._sk_retreat(arm, None, {}, [])
    final = _region({s.name: s for s in ps.object_states()}[name], LAYOUT)
    print(f"  {name:12s} arm={arm} force={f:5.1f}  grasp={g.category:12s} place={(p.category if p else '-'):12s} final={final}")
    results.append(g.category == "success" and p is not None and p.category == "success" and final == f"bag:{comp}")
check("pick-and-place works for all six classes", all(results), str(results))
print("log:", ps.log[:8])
app.close()
