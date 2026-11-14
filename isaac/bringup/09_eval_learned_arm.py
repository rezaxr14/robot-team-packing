"""Bring-up 9: closed-loop episodes with the LEARNED arm policy, same seeds as the scripted arm for a paired comparison.
  python isaac/bringup/09_eval_learned_arm.py --headless --episodes 50 --policy $RUNS/arm_flow --demos $DATA/arm_demos.npz --tag isaac_learned_flow
Writes <out>/episodes.jsonl (record schema of the symbolic grid) so rtp.tables / paired_compare work unchanged."""
import json
from pathlib import Path
from _launch import launch, check
args, app = launch("learned arm episodes", extra=[(["--episodes"], dict(type=int, default=20)), (["--policy"], dict(required=True)),
                                                  (["--demos"], dict(required=True)), (["--tag"], dict(default="isaac_learned"))])

from isaac.packing_sim import build, LAYOUT
from isaac.learned_sim import LearnedPackingSim, load_policy
from rtp.datagen import sample_world, sample_spec, default_eyes
from rtp.experiments import record_from_result
from rtp.harness import Harness
from rtp.loop import LoopConfig, run_episode
from rtp.brain import ScriptedBrain
from rtp.simworld import SimWorld

H = Harness(Path(__file__).resolve().parents[2] / "harness_files")
sim, scene, _ = build()
ps = LearnedPackingSim(sim, scene, load_policy(args.policy), args.demos)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
ok = 0
for seed in range(args.seed, args.seed + args.episodes):
    sw = SimWorld(ps, layout=LAYOUT, seed=seed)
    res = run_episode(sw, default_eyes(seed, False), ScriptedBrain(), sample_spec(sample_world(seed), seed), H, LoopConfig())
    ok += res.success
    print(f"seed {seed}: success={res.success} valid={res.n_valid}/{res.n_items} fatal={[(f['agent'], f['cause']) for f in res.failures if f.get('fatal')][:3]}", flush=True)
    with open(out / "episodes.jsonl", "a") as fh:
        fh.write(json.dumps(record_from_result(args.tag, seed, res)) + "\n")
check("learned policy completes episodes", ok >= 1, f"({ok}/{args.episodes} succeeded)")
app.close()
