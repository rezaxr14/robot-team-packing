"""R2: a `SimBackend` whose skill outcomes are sampled from certified skill tables.

This is a **sampling backend**, not a physics simulator. It exists so multi-turn datagen can run
fast while remaining replay-deterministic, and so the Brain only sees outcomes that the certified
arm actually produced. Every sampled outcome must be backed by a table generated from A10 evidence
(`skill_tables/manifest.json`); if the table for a skill is absent or its `arm_model` is not in
the manifest, the backend raises instead of inventing an outcome.

Evidence class: SIM-ABSTRACT (sampling from measured SIM tables); never promoted to SIM.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from .simbridge import Layout, ObjState
from .simworld import SkillReport

class BundleMissing(RuntimeError):
    """No certified table for the requested skill / arm model."""


class BundleBackend:
    kind = "skill_table_sampler"
    evidence_class = "SIM-ABSTRACT"

    def __init__(self, root: Path | str, layout: Layout | None = None, *, arm_model: str | None = None):
        self.root = Path(root)
        self.layout = layout or Layout()
        self.arm_zones = {"L": {"table_left", "table_center", "bag"},
                          "R": {"table_right", "table_center", "bag"}}
        self.payload = {"L": 2.5, "R": 2.5}
        self.force_limit = {"L": 40.0, "R": 40.0}
        self.arm_model = arm_model
        self.manifest = self._load_manifest()
        self.tables: dict[str, dict] = {}
        self._states: list[ObjState] = []
        self._seed = 0
        self._draws: list[tuple] = []

    def _load_manifest(self) -> dict:
        path = self.root / "skill_tables" / "manifest.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def load(self, skill: str) -> dict:
        """Load and validate the table for ``skill`` once, then cache it."""
        if skill in self.tables:
            return self.tables[skill]
        path = self.root / "skill_tables" / f"{skill}.json"
        if not path.exists():
            raise BundleMissing(f"missing certified skill table: {path}")
        table = json.loads(path.read_text())
        model = table.get("arm_model")
        known = self.manifest.get("arm_models", {})
        if not model or (known and model not in known):
            raise BundleMissing(f"arm_model {model!r} is not in skill_tables/manifest.json")
        if not table.get("entries") and not table.get("counts"):
            raise BundleMissing(f"skill table {path} has no measured entries")
        self.tables[skill] = table
        return table

    # -- SimBackend -------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        self._seed = int(seed)
        self._draws = []
        self._states = []
        layout = self.layout
        for name in ("obj_a", "obj_b", "obj_c"):
            zone = "table_left" if name != "obj_c" else "table_center"
            x = -0.25 if zone == "table_left" else 0.0
            self._states.append(ObjState(name=name, cls="object", pos=(x, 0.0, layout.table_z + 0.025),
                                         mass=0.5, fragility=0.3, size=0.05, friction=0.6,
                                         init_region=zone))
        self._held = {"L": None, "R": None}

    def object_states(self) -> list:
        return [ObjState(**s.__dict__) for s in self._states]

    def _choose_entry(self, table: dict, params: dict) -> dict:
        entries = table.get("entries", [])
        if not entries:
            return table
        for entry in entries:
            cond = entry.get("key", {})
            if all(params.get(k) == v for k, v in cond.items()):
                return entry
        return max(entries, key=lambda e: int(e.get("counts", {}).get("success", 0)))

    def run_skill(self, arm: str, skill: str, obj: str | None, params: dict) -> SkillReport:
        table = self.load(skill)
        entry = self._choose_entry(table, dict(params or {}))
        counts = {k: int(v) for k, v in entry.get("counts", {}).items() if int(v) > 0}
        if not counts:
            raise BundleMissing(f"no measured outcomes in the selected entry of skill {skill}")
        cats = sorted(counts)
        weights = [counts[c] for c in cats]
        rnd = random.Random(f"bundle|{self.arm_model}|{self._seed}|{arm}|{skill}|{obj}|"
                            f"{json.dumps(params or {}, sort_keys=True)}|{len(self._draws)}")
        category = rnd.choices(cats, weights=weights, k=1)[0]
        self._draws.append((arm, skill, obj, category))
        return SkillReport(category=category, touch={"sampled_from": skill, "arm_model": self.arm_model},
                           detail="SIM-ABSTRACT: sampled from a certified skill table, not a physics run")

    def draw_log(self) -> list[tuple]:
        return list(self._draws)
