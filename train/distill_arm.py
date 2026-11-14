"""Distill skill demonstrations into the blind ArmPolicy (flow-matching or MLP head).

STATUS: UNTESTED (needs torch). Use --synthetic first to verify the pipeline on your GPU before Isaac Lab data
exists, then point --data at an .npz of expert rollouts with keys:
  proprio[N,15] contact[N,8] skill[N] target[N] params[N,4] actions[N,H,A]
Reports held-out MAE of sampled action chunks, per skill id, so regression across skills can be checked.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from rtp.flowhead import ArmPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None)
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--head", choices=("flow", "mlp"), default="flow")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--out", default="runs/arm_distill")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()


def synthetic(n=20000, H=16, A=8, seed=0):
    g = np.random.default_rng(seed)
    skill, target = g.integers(0, 8, n), g.integers(0, 32, n)
    base = (skill[:, None] / 8.0 + target[:, None] / 32.0 + np.linspace(0, 1, H)[None]) / 3.0
    act = np.repeat(base[:, :, None], A, axis=2) + 0.01 * g.normal(size=(n, H, A))
    return {"proprio": g.normal(size=(n, 15)).astype("f4"), "contact": np.zeros((n, 8), "f4"), "skill": skill,
            "target": target, "params": np.zeros((n, 4), "f4"), "actions": act.astype("f4")}


d = synthetic() if a.synthetic else dict(np.load(a.data))
N = len(d["skill"])
idx = np.random.default_rng(0).permutation(N)
tr, te = idx[: int(0.9 * N)], idx[int(0.9 * N):]
H, A = d["actions"].shape[1:]
model = ArmPolicy(horizon=H, act_dim=A, head=a.head).to(a.device)
opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)


def batch(ix):
    t = lambda k, dt=None: torch.as_tensor(d[k][ix], dtype=dt).to(a.device)
    return {"proprio": t("proprio", torch.float32), "contact": t("contact", torch.float32), "skill": t("skill", torch.long),
            "target": t("target", torch.long), "params": t("params", torch.float32), "actions": t("actions", torch.float32)}


rng = np.random.default_rng(1)
for step in range(a.steps):
    loss = model.loss(batch(rng.choice(tr, a.batch)))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 200 == 0:
        print(f"step {step} loss {loss.item():.4f}", flush=True)

model.eval()
per_skill = {}
for s in np.unique(d["skill"][te]):
    ix = te[d["skill"][te] == s][:512]
    b = batch(ix)
    per_skill[int(s)] = float((model.sample(b) - b["actions"]).abs().mean())
report = {"head": a.head, "horizon": int(H), "act_dim": int(A), "heldout_mae_mean": float(np.mean(list(per_skill.values()))), "per_skill_mae": per_skill}
Path(a.out).mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), Path(a.out) / "arm.pt")
json.dump(report, open(Path(a.out) / "report.json", "w"), indent=2)
print(json.dumps(report, indent=2))
