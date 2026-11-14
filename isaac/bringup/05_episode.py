"""Bring-up 5: full closed-loop episodes in Isaac (scripted Brain by default, optional LLM Brain).
  python isaac/bringup/05_episode.py --headless --episodes 5
  python isaac/bringup/05_episode.py --headless --episodes 50 --brain-endpoint http://localhost:8001/v1 --brain-model brain --tool-brain
Writes <out>/episodes.jsonl in the same record schema as the symbolic grid (rtp.experiments.record_from_result)."""
import json
from pathlib import Path
from _launch import launch, check
args, app = launch("closed-loop episodes", extra=[
    (["--episodes"], dict(type=int, default=5)), (["--eyes"], dict(choices=("oracle", "noisy"), default="oracle")),
    (["--brain-endpoint"], dict(default=None)), (["--brain-model"], dict(default=None)),
    (["--tool-brain"], dict(action="store_true")), (["--tag"], dict(default="isaac")),
    (["--eye-endpoint"], dict(default=None)), (["--eye-model"], dict(default=None)),
    (["--probe"], dict(action="store_true", help="weigh every item with the arms before planning"))])

from isaac.packing_sim import build, LAYOUT
from rtp.datagen import sample_world, sample_spec, default_eyes
from rtp.experiments import record_from_result
from rtp.harness import Harness
from rtp.loop import LoopConfig, run_episode
from rtp.brain import ScriptedBrain, LLMBrain
from rtp.simworld import SimWorld

H = Harness(Path(__file__).resolve().parents[2] / "harness_files")
brain_factory = lambda: ScriptedBrain()
if args.brain_endpoint:
    from rtp.llm_client import ChatClient
    client = ChatClient(args.brain_endpoint, args.brain_model)
    if args.tool_brain:
        from rtp.toolbrain import ToolBrain
        brain_factory = lambda: ToolBrain(lambda m: client.chat(m, max_tokens=1500), H)
    else:
        brain_factory = lambda: LLMBrain(client.generate(max_tokens=1500))

sim, scene, ps = build(with_cameras=bool(args.eye_endpoint))
recs = []
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
for seed in range(args.seed, args.seed + args.episodes):
    sw = SimWorld(ps, layout=LAYOUT, seed=seed)
    spec = sample_spec(sample_world(seed), seed)
    if args.eye_endpoint:                                 # real VLM Eyes reading rendered images (needs --enable_cameras)
        from rtp.vlm_eye import VLMEye
        eye_client = ChatClient(args.eye_endpoint, args.eye_model)
        eyes = [VLMEye(v, lambda m: eye_client.chat(m, max_tokens=1024), ps.render_eye) for v in ("A", "B")]
    else:
        eyes = default_eyes(seed, args.eyes == "noisy")
    res = run_episode(sw, eyes, brain_factory(), spec, H, LoopConfig(probe=args.probe))
    rec = record_from_result(args.tag, seed, res)
    if args.eye_endpoint:
        rec["eye_stats"] = {v.name: v.stats for v in eyes}
    recs.append(rec)
    print(f"seed {seed}: success={res.success} valid={res.n_valid}/{res.n_items} rounds={res.rounds} "
          f"violations={[v['rule'] for v in res.violations]} fatal={[(f['agent'], f['cause']) for f in res.failures if f.get('fatal')][:3]}", flush=True)
    with open(out / "episodes.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
ok = sum(r["success"] for r in recs)
check("at least one episode succeeds end to end", ok >= 1, f"({ok}/{len(recs)})")
print("sim log tail:", ps.log[-6:])
app.close()
