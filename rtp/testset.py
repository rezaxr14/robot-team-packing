"""Frozen held-out evaluation set (RUNBOOK rule 8 / M3).

Every evaluation seed is < 100000; training seeds are >= 100000 and must never appear here. The
manifest is hash-frozen so a later run can prove it used the same episodes. This module only
builds/verifies the *index* of episodes (family, seed, parameters); the episodes themselves are
generated on demand by the harness from those seeds.

Evidence class: FIXTURE. No model score is stored or implied.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from . import seeds as seed_rules

FAMILIES = (
    "packing", "heavy_fragile", "occlusion", "distractors", "handover",
    "push_interruption", "kitchen_objects", "unseen_layouts",
)


def digest(episodes: list[dict]) -> str:
    """Stable sha256 over the canonical episode index."""
    canonical = json.dumps(episodes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_testset(n_per_family: int = 30, seed0: int = 0) -> list[dict]:
    """Return ``n_per_family`` deterministic episodes for each family, all in the test namespace."""
    if n_per_family <= 0:
        raise ValueError(f"n_per_family must be positive, got {n_per_family}")
    first, last = seed_rules.split_bounds(seed_rules.TEST, n_per_family * len(FAMILIES), seed0)
    out = []
    for fi, family in enumerate(FAMILIES):
        for i in range(n_per_family):
            out.append({
                "family": family,
                "seed": first + fi * n_per_family + i,
                "params": {"family": family, "trial": i},
            })
    assert len(out) == last - first
    assert all(seed_rules.is_test(e["seed"]) for e in out)
    return out


def freeze(path: Path | str, episodes: list[dict]) -> dict:
    """Write a hash-frozen manifest and return it."""
    path = Path(path)
    manifest = {"schema": 1, "evidence_class": "FIXTURE", "train_min": seed_rules.TRAIN_MIN,
                "families": list(FAMILIES), "n": len(episodes), "sha256": digest(episodes),
                "episodes": episodes}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify(path: Path | str, *, forbidden_train_seeds: Iterable[int] = ()) -> dict:
    """Read a manifest and check its hash and namespaces. Raises on any violation."""
    manifest = json.loads(Path(path).read_text())
    episodes = manifest["episodes"]
    if manifest.get("sha256") != digest(episodes):
        raise ValueError("held-out set hash mismatch (manifest was edited)")
    got_seeds = [e["seed"] for e in episodes]
    seed_rules.assert_test(got_seeds, where="held-out test set")
    overlap = sorted(set(got_seeds) & {int(s) for s in forbidden_train_seeds})
    if overlap:
        raise seed_rules.SeedNamespaceError(
            f"training seeds {overlap[:8]} appear in the held-out set"
        )
    if len(set(got_seeds)) != len(got_seeds):
        raise ValueError("duplicate seeds in held-out set")
    families = {e["family"] for e in episodes}
    if families != set(FAMILIES):
        raise ValueError(f"families mismatch: {sorted(families)}")
    return manifest
