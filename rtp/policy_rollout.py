"""Receding-horizon execution of a learned ArmPolicy (flow or MLP head) against any environment.

obs_fn()  -> dict(proprio: list[15], contact: list[8])
apply_fn(action: list[8]) -> None      (applies joint+gripper targets and advances the simulation one control step)
The policy samples an action chunk of length H; the first `exec_steps` actions are applied, then it re-observes and
re-samples. The skill command (skill id, target id, params) is fixed for the whole skill. No termination signal is
learned: the caller supplies n_steps (e.g. the median demonstration length for that skill).
"""
from __future__ import annotations
import torch


@torch.no_grad()
def rollout_skill(policy, obs_fn, apply_fn, n_steps: int, skill_id: int, target_id: int, params, exec_steps: int = 4,
                  device: str = "cpu") -> dict:
    policy.eval()
    done, log = 0, {"replans": 0, "actions": []}
    while done < n_steps:
        o = obs_fn()
        batch = {"proprio": torch.tensor([o["proprio"]], dtype=torch.float32, device=device),
                 "contact": torch.tensor([o["contact"]], dtype=torch.float32, device=device),
                 "skill": torch.tensor([skill_id], dtype=torch.long, device=device),
                 "target": torch.tensor([target_id], dtype=torch.long, device=device),
                 "params": torch.tensor([list(params)], dtype=torch.float32, device=device)}
        chunk = policy.sample(batch)[0].cpu()                       # (H, A)
        log["replans"] += 1
        for k in range(min(exec_steps, chunk.shape[0], n_steps - done)):
            a = chunk[k].tolist()
            apply_fn(a)
            log["actions"].append(a)
            done += 1
    return log


def median_lengths(skill_ids, lengths, default: int = 120) -> dict:
    """{skill_id: median demonstration length in control steps} from collected demos."""
    out = {}
    for s in set(int(x) for x in skill_ids):
        v = sorted(int(l) for sid, l in zip(skill_ids, lengths) if int(sid) == s)
        out[s] = v[len(v) // 2] if v else default
    return out
