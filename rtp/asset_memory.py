"""Success/failure memory RAG for the asset-backed environment.

Notebook 70 demonstrates this on the symbolic fixture. This module connects the same idea to the
real SimReady packages: episodes run in asset-backed pose-level worlds, and their outcomes are
stored as two BM25 indexes (successes and failures). A new task retrieves from both so the Brain
can see what worked and what failed on scenes built from the same asset library.

Evidence class: SIM-POSE (real assets, pose-level dynamics). The retrieval index is BM25; a Qdrant
vector side can be fused with ``rtp.vectordb.HybridRetriever`` when an embedding model is supplied.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rtp.brain import BrainContext, ScriptedBrain, build_prompt
from rtp.datagen import default_eyes, retrieve_docs
from rtp.harness import BM25, Doc, Harness
from rtp.loop import LoopConfig, run_episode
from rtp.perception import merge_scenes
from rtp.scene_assets import AssetRecord, to_world_objects
from rtp.world import World

from rtp.asset_datagen import spec_for_assets


def _story(row: dict) -> str:
    return (f"task: {row['instruction']} | outcome: {row['outcome']} | "
            f"assets: {', '.join(row['assets'])} | cause: {row.get('cause', '')} | "
            f"plan: {row.get('plan', '')[:400]}")


def build_memory(records: Iterable[AssetRecord], n: int, seed0: int, harness: Harness,
                 *, manifest: str = "") -> tuple[list[dict], dict]:
    """Run n asset-backed episodes and store their outcomes as retrievable memory rows."""
    records = list(records)
    cfg = LoopConfig()
    rows = []
    for i in range(n):
        seed = seed0 + i
        objs = to_world_objects(records, seed=seed)
        world = World(objs, seed=seed, p_arm_noise=0.03)
        spec = spec_for_assets(objs, seed)
        res = run_episode(world, default_eyes(seed, True), ScriptedBrain(), spec, harness, cfg)
        rows.append({"seed": seed, "outcome": "success" if res.success else "failure",
                     "instruction": spec.text(), "assets": [r.package for r in records],
                     "cause": getattr(res, "cause", "") or "",
                     "plan": res.plans[0] if res.plans else "",
                     "manifest": manifest, "evidence": "SIM-POSE"})
    stats = {"episodes": len(rows), "successes": sum(r["outcome"] == "success" for r in rows),
             "failures": sum(r["outcome"] == "failure" for r in rows),
             "source": "asset-backed pose-level environment"}
    return rows, stats


def write_memory(rows: Iterable[dict], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def read_memory(path: Path | str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def retrieve_memory(rows: Iterable[dict], query: str, *, k: int = 3,
                    outcome: str | None = None) -> list[tuple[dict, float]]:
    """BM25 over memory stories, optionally restricted to successes or failures."""
    subset = [r for r in rows if outcome is None or r["outcome"] == outcome]
    if not subset:
        return []
    bm25 = BM25([_story(r) for r in subset], list(range(len(subset))))
    return [(subset[int(idx)], float(score)) for idx, score in bm25.rank(query)[:k]]


def augment_prompt(records: Iterable[AssetRecord], memory: Iterable[dict], seed: int,
                   harness: Harness, *, k: int = 3) -> str:
    """Build a Brain prompt that includes retrieved success and failure memories."""
    objs = to_world_objects(records, seed=seed)
    spec = spec_for_assets(objs, seed)
    scene = merge_scenes([e.report(World(objs, seed=seed), 0)
                          for e in default_eyes(seed, True)])
    docs = retrieve_docs(harness, LoopConfig(), spec, scene, [])
    rows = list(memory)
    for outcome in ("success", "failure"):
        for row, score in retrieve_memory(rows, spec.text(), k=k, outcome=outcome):
            docs.append(Doc(id=f"memory-{outcome}-{row['seed']}", type="note",
                            keywords=outcome, body=_story(row)))
    ctx = BrainContext(spec=spec, scene=scene,
                       rule_ids={d.id for d in docs if getattr(d, "type", "") == "rule"},
                       snippets=docs)
    return build_prompt(ctx)
