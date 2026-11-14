"""Collect arm demonstrations from the scripted IK skills for distillation (train/distill_arm.py).
  python isaac/bringup/07_collect_demos.py --headless --episodes 300 --out data/arm_demos.npz
Records only SUCCESSFUL skills. Each control step becomes one sample: blind observation (proprio 15, contact 8,
skill id, target id, params 4) -> next HORIZON joint+gripper targets (8 dims)."""
import numpy as np
from pathlib import Path
from _launch import launch, check
args, app = launch("collect arm demos", extra=[(["--episodes"], dict(type=int, default=100)), (["--seed0"], dict(type=int, default=200000)),
                                               (["--horizon"], dict(type=int, default=16))])

from isaac.packing_sim import build, PackingSim, ITEM_NAMES, LAYOUT
from rtp.datagen import sample_world, sample_spec, default_eyes
from rtp.harness import Harness
from rtp.loop import LoopConfig, run_episode
from rtp.brain import ScriptedBrain
from rtp.simworld import SimWorld

SKILL_ID = {"grasp": 0, "place": 1, "release": 2, "retreat": 3, "reach": 4}
COMP_ID = {"main": 0, "front": 1, "laptop": 2}
H = Harness(Path(__file__).resolve().parents[2] / "harness_files")
buf = {k: [] for k in ("proprio", "contact", "skill", "target", "params", "actions")}
lens = {"skill": [], "steps": []}          # demonstration length per successful skill (for learned-policy rollouts)


class Recording:
    """Wraps PackingSim.run_skill to turn each successful skill's control trace into samples."""
    def __init__(self, ps, horizon):
        self.ps, self.h = ps, horizon

    def __getattr__(self, k):
        return getattr(self.ps, k)

    def run_skill(self, arm, skill, obj, params):
        self.ps.trace = []
        rep = self.ps.run_skill(arm, skill, obj, params)
        tr, self.ps.trace = self.ps.trace, None
        if rep.category == "success" and skill in SKILL_ID and tr:
            tgt = ITEM_NAMES.index(obj) if obj in ITEM_NAMES else 0
            if skill == "place":
                tgt = 16 + COMP_ID.get(params.get("compartment", "main"), 0)
            prm = [float(params.get("force", 0.0)) / 40.0, COMP_ID.get(params.get("compartment", "main"), 0) / 2.0,
                   1.0 if params.get("orientation", "upright") == "upright" else 0.0, 0.0]
            lens["skill"].append(SKILL_ID[skill]); lens["steps"].append(len(tr))
            acts = np.array([t["action"] for t in tr], dtype="f4")
            for i, t in enumerate(tr):
                chunk = acts[i:i + self.h]
                if len(chunk) < self.h:
                    chunk = np.concatenate([chunk, np.repeat(chunk[-1:], self.h - len(chunk), 0)])
                buf["proprio"].append(t["proprio"]); buf["contact"].append(t["contact"])
                buf["skill"].append(SKILL_ID[skill]); buf["target"].append(tgt); buf["params"].append(prm)
                buf["actions"].append(chunk)
        return rep


sim, scene, ps = build()
rec = Recording(ps, args.horizon)
for seed in range(args.seed0, args.seed0 + args.episodes):
    sw = SimWorld(rec, layout=LAYOUT, seed=seed)
    spec = sample_spec(sample_world(seed), seed)
    res = run_episode(sw, default_eyes(seed, False), ScriptedBrain(), spec, H, LoopConfig())
    print(f"seed {seed}: success={res.success} samples so far={len(buf['skill'])}", flush=True)
out = Path(args.out if args.out.endswith(".npz") else "data/arm_demos.npz"); out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, proprio=np.array(buf["proprio"], "f4"), contact=np.array(buf["contact"], "f4"),
                    skill=np.array(buf["skill"], "i8"), target=np.array(buf["target"], "i8"),
                    params=np.array(buf["params"], "f4"), actions=np.array(buf["actions"], "f4"),
                    len_skill=np.array(lens["skill"], "i8"), len_steps=np.array(lens["steps"], "i8"))
check("collected demonstrations", len(buf["skill"]) > 1000, f"({len(buf['skill'])} samples) -> {out}")
app.close()
