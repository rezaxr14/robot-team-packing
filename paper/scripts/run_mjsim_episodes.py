"""Closed-loop episodes in the laptop-scale MuJoCo scene.

This is the same `rtp.loop.run_episode` that runs against Isaac Lab (`isaac/bringup/05_episode.py`) and the symbolic
world, with `mjsim.GantryBackend` as the executor. It is *not* the paper's ISAAC evidence class: the scene is a
two-gantry MuJoCo approximation built to catch controller/geometry regressions on a laptop GPU. Records are written
in the shared schema (`rtp.experiments.record_from_result`) so they can be compared with the symbolic grid.

  python scripts/run_mjsim_episodes.py --episodes 3
  python scripts/run_mjsim_episodes.py --episodes 10 --tag mjsim --out reports/mjsim
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtp.brain import ScriptedBrain                       # noqa: E402
from rtp.datagen import default_eyes, sample_spec, sample_world  # noqa: E402
from rtp.experiments import record_from_result            # noqa: E402
from rtp.harness import Harness                           # noqa: E402
from rtp.loop import LoopConfig, run_episode              # noqa: E402
from mjsim.scene import physical_layout                   # noqa: E402
from rtp.simworld import SimWorld                         # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eyes", choices=("oracle", "noisy"), default="oracle")
    ap.add_argument("--probe", action="store_true", help="weigh every item before planning")
    ap.add_argument("--tag", default="mjsim")
    ap.add_argument("--out", default=str(ROOT / "reports" / "mjsim"))
    ap.add_argument("--render", default=None, help="directory for a PNG after each episode")
    args = ap.parse_args(argv)

    from mjsim.backend import GantryBackend
    from mjsim.scene import items_from_world

    layout = physical_layout()
    H = Harness(ROOT / "harness_files")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []

    for seed in range(args.seed, args.seed + args.episodes):
        world = sample_world(seed)
        backend = GantryBackend(items_from_world(world, layout), layout=layout, seed=seed)
        sw = SimWorld(backend, layout=layout, seed=seed)
        spec = sample_spec(world, seed)
        res = run_episode(sw, default_eyes(seed, args.eyes == "noisy"), ScriptedBrain(), spec, H,
                          LoopConfig(probe=args.probe))
        rec = record_from_result(args.tag, seed, res)
        recs.append(rec)
        print(f"seed {seed}: success={res.success} valid={res.n_valid}/{res.n_items} rounds={res.rounds} "
              f"crashed={res.crushed} missing={res.missing} failures={[(f['agent'], f['cause']) for f in res.failures][:4]}",
              flush=True)
        with open(out / "episodes.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if args.render:
            png = Path(args.render)
            png.mkdir(parents=True, exist_ok=True)
            try:
                from PIL import Image
                Image.fromarray(backend.render("A")).save(png / f"seed{seed:04d}.png")
            except Exception as e:                              # noqa: BLE001 - rendering needs a GL context
                print(f"  (render skipped: {e})")

    ok = sum(bool(r.get("success")) for r in recs)
    print(f"{ok}/{len(recs)} episodes succeeded; records in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
