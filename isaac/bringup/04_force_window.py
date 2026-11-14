"""Bring-up 4: measure the slip and crush thresholds in simulation and compare with the symbolic model.
  python isaac/bringup/04_force_window.py --headless --out reports/isaac
Writes force_window.csv. Use it to calibrate rtp.world.required_force / FINGER_STIFFNESS / TCP_OFFSET so the Brain's
force formula lands inside the simulated window."""
import csv
from pathlib import Path
from _launch import launch, check
args, app = launch("force window sweep")

from isaac.packing_sim import build
from rtp.world import required_force, crush_limit

sim, scene, ps = build()
FORCES = [2, 4, 6, 8, 10, 12, 15, 18, 22, 26, 30, 35]
rows = []
for cls in ("book", "notebook", "pencilcase", "bottle", "lunchbox", "laptop"):
    for f in FORCES:
        ps.reset(args.seed)
        name = next(s.name for s in ps.object_states() if s.cls == cls)
        o = ps.truth.objs[name]
        arm = "L" if o.region in ("table_left", "table_center") else "R"
        rep = ps.run_skill(arm, "grasp", name, {"force": float(f)})
        rows.append({"cls": cls, "mass": round(o.mass, 3), "fragility": round(o.fragility, 2), "force": f,
                     "category": rep.category, "sym_need": round(required_force(o), 1), "sym_crush": round(crush_limit(o), 1)})
        print(rows[-1], flush=True)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
with open(out / "force_window.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for cls in ("book", "notebook", "pencilcase", "bottle", "lunchbox", "laptop"):
    ok = [r["force"] for r in rows if r["cls"] == cls and r["category"] == "success"]
    print(f"{cls:11s} sim success window: {min(ok) if ok else None}..{max(ok) if ok else None} N")
check("every class has at least one successful force", all(any(r["category"] == "success" for r in rows if r["cls"] == c)
                                                       for c in ("book", "notebook", "pencilcase", "bottle", "lunchbox", "laptop")))
app.close()
