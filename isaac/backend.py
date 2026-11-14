"""IsaacBackend: implements rtp.simworld.SimBackend on top of TeamPackEnv.  >>> SKELETON, UNTESTED <<<

The logic that turns simulator state into skill outcomes is NOT here: it is `rtp.simbridge.report_from_states`
(tested through StateDiffBackend against the regression suite and full episodes). This class only has to
(1) start a skill on an arm, (2) step the simulation until the skill ends, and (3) hand back object states
before and after. Contact peaks and held flags come from the env; see isaac/DESIGN.md for the open items.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtp.simbridge import Layout, report_from_states
from rtp.simworld import SkillReport


class IsaacBackend:
    def __init__(self, env, env_id: int = 0, max_skill_steps: int = 400):
        self.env, self.env_id, self.max_skill_steps = env, env_id, max_skill_steps
        self.layout = env.cfg.layout if hasattr(env, "cfg") else Layout()
        self.arm_zones = {"L": {"table_left", "table_center", "bag"}, "R": {"table_right", "table_center", "bag"}}
        self.payload = {"L": 2.5, "R": 2.5}
        self.force_limit = {"L": 35.0, "R": 35.0}
        self.init_regions, self.touched, self.peaks, self.held = {}, set(), {}, {}

    def reset(self, seed: int):
        raise NotImplementedError("env.reset(seed=...) in your Isaac Lab version; then record init_regions from object_states()")

    def object_states(self):
        return self.env.object_states(self.env_id, init_regions=self.init_regions, touched=self.touched,
                                      peaks=self.peaks, held=self.held)

    def _step_until_skill_done(self, arm: str):
        raise NotImplementedError("step env with the arm policy until the skill terminates or max_skill_steps; "
                                  "accumulate self.peaks[obj] = max contact force; update self.held on grasp/release")

    def run_skill(self, arm, skill, obj, params):
        pre = self.object_states()
        self.env.set_skill(f"arm_{arm}", skill, obj, params)
        self._step_until_skill_done(arm)
        post = self.object_states()
        cat, touch = report_from_states(arm, skill, obj, params, pre, post, self.layout, self.arm_zones,
                                        self.payload, lock_holder=None)
        if skill in ("grasp", "push_aside") and obj is not None and cat in ("success", "slip", "crush"):
            self.touched.add(obj)
        return SkillReport(cat, touch)
