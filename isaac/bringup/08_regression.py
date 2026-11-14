"""Bring-up 8: the skill regression suite on the simulator (scripted skills, or a learned policy with --policy).
  python isaac/bringup/08_regression.py --headless --trials 3
  python isaac/bringup/08_regression.py --headless --policy $RUNS/arm_flow --demos $DATA/arm_demos.npz --trials 3
Writes <out>/regression_<tag>.json and compares with a previous report for the same tag if one exists."""
import json
from pathlib import Path
from _launch import launch, check
args, app = launch("regression suite", extra=[(["--trials"], dict(type=int, default=3)), (["--policy"], dict(default=None)),
                                              (["--demos"], dict(default=None)), (["--tag"], dict(default=None))])

from isaac.packing_sim import build
from isaac.regression import IsaacExecutor
from rtp.skills import run_regression, compare

sim, scene, ps = build()
tag = args.tag or ("learned" if args.policy else "scripted")
if args.policy:
    from isaac.learned_sim import LearnedPackingSim, load_policy
    ps = LearnedPackingSim(sim, scene, load_policy(args.policy), args.demos)
rep = run_regression(IsaacExecutor(ps), n_trials=args.trials, pass_rate=0.66, seed0=args.seed)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
path = out / f"regression_{tag}.json"
if path.exists():
    print("compare with previous:", compare(json.loads(path.read_text()), rep))
path.write_text(json.dumps(rep, indent=1))
for n, v in rep["cases"].items():
    print(f"  {'ok  ' if v['passed'] else 'FAIL'} {n:28s} rate={v['rate']:.2f} outcomes={v['example_outcomes']}")
check("regression suite", rep["passed"] == rep["total"], f"({rep['passed']}/{rep['total']})")
app.close()
