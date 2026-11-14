"""LearnedPackingSim: PackingSim whose arm skills are executed by a distilled blind ArmPolicy instead of scripted IK.

The policy sees only proprioception, contact proxies and the skill command (skill id, target id, params): no images, no
instruction, no object poses. Bookkeeping (held, peak force) uses the same rules as the scripted sim, so outcomes are
classified by the same state-diff code and episodes are directly comparable.
STATUS: written, never run (needs Isaac Sim + a trained arm policy). Depends on rtp.policy_rollout (tested).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

from isaac.packing_sim import (PackingSim, ITEM_NAMES, ITEM_CLASS, SIM_DIMS, FINGER_OPEN, LIFT_OK, _region, _crushed,
                               TCP_OFFSET, HOVER)
from rtp.flowhead import ArmPolicy
from rtp.policy_rollout import rollout_skill, median_lengths
from rtp.simbridge import SIM_FORCE_MARGIN_N

SKILL_ID = {"grasp": 0, "place": 1, "release": 2, "retreat": 3, "reach": 4}
COMP_ID = {"main": 0, "front": 1, "laptop": 2}


def load_policy(run_dir: str, device="cuda:0"):
    rep = json.loads((Path(run_dir) / "report.json").read_text())
    m = ArmPolicy(horizon=rep["horizon"], act_dim=rep["act_dim"], head=rep["head"])
    m.load_state_dict(torch.load(Path(run_dir) / "arm.pt", map_location=device))
    return m.to(device).eval()


class LearnedPackingSim(PackingSim):
    def __init__(self, sim, scene, policy, demos_npz: str, exec_steps: int = 4):
        super().__init__(sim, scene)
        self.policy, self.exec_steps = policy, exec_steps
        d = np.load(demos_npz)
        self.lengths = median_lengths(d["len_skill"].tolist(), d["len_steps"].tolist())

    def _apply(self, arm):
        r, e = self.arms[arm], self.ent[arm]

        def apply(a):
            q = torch.tensor([a[:7]], dtype=torch.float32, device=self.sim.device)
            r.set_joint_position_target(q, joint_ids=e.joint_ids)
            self.last_q[arm] = q
            self.grip_open[arm] = float(min(max(a[7], 0.0), FINGER_OPEN))
            self._set_fingers(arm, self.grip_open[arm])
            self._step()
        return apply

    def _run_policy(self, arm, skill, target_id, params):
        return rollout_skill(self.policy, lambda: {"proprio": self._proprio(arm), "contact": self._contact(arm)}, self._apply(arm),
                             n_steps=self.lengths.get(SKILL_ID[skill], 120), skill_id=SKILL_ID[skill], target_id=target_id,
                             params=params, exec_steps=self.exec_steps, device=self.sim.device)

    def _sk_grasp(self, arm, obj, p, pre):
        o = {s.name: s for s in pre}[obj]
        if not self._feasible_grasp(arm, o):
            return
        force = min(float(p.get("force", 10.0)), self.force_limit[arm])
        z0 = o.pos[2]
        self._run_policy(arm, "grasp", ITEM_NAMES.index(obj), [force / 40.0, 0.0, 1.0, 0.0])
        self.touched.add(obj)
        self.peaks[obj] = max(self.peaks.get(obj, 0.0), force + SIM_FORCE_MARGIN_N)
        if self._obj_pos(obj)[2] > z0 + LIFT_OK and not _crushed(self, obj):
            self.held[obj] = arm
        else:
            self.grip_open[arm] = FINGER_OPEN
            self._hold(arm, 30)

    def _sk_place(self, arm, obj, p, pre):
        if self.held.get(obj) != arm:
            return
        comp = p.get("compartment", "main")
        self._run_policy(arm, "place", 16 + COMP_ID.get(comp, 0),
                         [0.0, COMP_ID.get(comp, 0) / 2.0, 1.0 if p.get("orientation", "upright") == "upright" else 0.0, 0.0])
        self.held.pop(obj, None)

    def _sk_release(self, arm, obj, p, pre):
        if self.held.get(obj) == arm:
            self._run_policy(arm, "release", 0, [0, 0, 0, 0])
            self.held.pop(obj, None)

    def _sk_retreat(self, arm, obj, p, pre):
        self._run_policy(arm, "retreat", 0, [0, 0, 0, 0])
