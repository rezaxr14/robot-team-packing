"""Generate SFT/GRPO rows from episodes run in the asset-backed pose-level environment.

This differs from ``rtp.datagen.gen_brain_sft`` in one important way: the objects come from real
SimReady packages resolved through ``rtp.scene_assets`` (paths, licences, measured mass where the
package provides it). The environment is still pose-level/symbolic, so the evidence class is
``SIM-POSE``, never ``Newton physics``. Each row carries the asset manifest path so a plan can be
traced back to the packages that produced the scene.
"""
from __future__ import annotations

from typing import Iterable

from rtp.brain import BrainContext, ScriptedBrain, build_prompt
from rtp.datagen import default_eyes, retrieve_docs
from rtp.harness import Harness
from rtp.loop import LoopConfig, run_episode
from rtp.perception import merge_scenes
from rtp.rules import TaskSpec
from rtp.scene_assets import AssetRecord, to_world_objects
from rtp.world import World


def spec_for_assets(objs, seed: int, *, laptop_slot: str = "laptop") -> TaskSpec:
    """Build a task spec whose last item and slot come from the asset classes actually present."""
    import random
    rng = random.Random(f"asset-spec|{seed}")
    items = [o.name for o in objs if not o.is_distractor]
    lunch = [o.name for o in objs if o.cls in ("lunchbox",) and not o.is_distractor]
    slot = laptop_slot if laptop_slot in ("laptop", "front") else "laptop"
    return TaskSpec(items=sorted(items), leave=sorted(o.name for o in objs if o.is_distractor),
                    laptop_slot=slot, last_item=sorted(lunch)[0] if lunch else None,
                    **({"order": rng.sample(items, len(items))} if hasattr(TaskSpec, "__dataclass_fields__")
                       and "order" in TaskSpec.__dataclass_fields__ else {}))


def gen_asset_sft(records: Iterable[AssetRecord], n: int, seed0: int, harness: Harness, *,
                  cfg: LoopConfig | None = None, noisy_eyes: bool = True,
                  manifest: str = "") -> tuple[list[dict], dict]:
    """Run the closed loop on asset-backed worlds and keep teacher-validated plans.

    Returns ``(rows, funnel)`` where funnel records attempts / successes / kept, so the sample
    efficiency is measured rather than assumed.
    """
    records = list(records)
    cfg = cfg or LoopConfig()
    rows, attempts, successes = [], 0, 0
    tries = 0
    while len(rows) < n and tries < 20 * n:
        s = seed0 + tries
        tries += 1
        objs = to_world_objects(records, seed=s)
        world = World(objs, seed=s, p_arm_noise=0.03)
        spec = spec_for_assets(objs, s)
        attempts += 1
        res = run_episode(world, default_eyes(s, noisy_eyes), ScriptedBrain(), spec, harness, cfg)
        if not res.success:
            continue
        successes += 1
        if not res.plans:
            continue
        scene = merge_scenes([e.report(World(objs, seed=s), 0) for e in default_eyes(s, noisy_eyes)])
        docs = retrieve_docs(harness, cfg, spec, scene, [])
        ctx = BrainContext(spec=spec, scene=scene,
                           rule_ids={d.id for d in docs if d.type == "rule"}, snippets=docs)
        rows.append({"prompt": build_prompt(ctx), "completion": res.plans[0],
                     "meta": {"seed": s, "teacher": "scripted", "evidence": "SIM-POSE",
                              "asset_manifest": manifest, "n_assets": len(records),
                              "msg_tokens": res.msg_tokens}})
    return rows, {"attempts": attempts, "successes": successes, "kept": len(rows),
                  "seed0": seed0, "evidence": "SIM-POSE", "asset_manifest": manifest}


def gen_asset_hindsight(records, n: int, seed0: int, harness: Harness, *,
                        noisy_eyes: bool = True, max_tries: int | None = None,
                        manifest: str = "") -> tuple[list[dict], dict]:
    """Hindsight rows from FAILED episodes in the asset-backed world, each replay-validated.

    Same funnel semantics as ``rtp.datagen.gen_hindsight_sft`` but the objects come from real
    SimReady packages. Evidence class remains SIM-POSE.
    """
    from rtp.hindsight import ReplayBrain, relabel
    from rtp.plan import parse_plan
    from rtp.rules import TaskSpec

    cfg = LoopConfig()
    rows = []
    stats = {"episodes": 0, "failed": 0, "relabeled": 0, "validated": 0,
             "evidence": "SIM-POSE", "asset_manifest": manifest}
    seed_i = 0
    limit = max_tries or 50 * n
    while len(rows) < n and seed_i < limit:
        seed = seed0 + seed_i
        seed_i += 1
        objs = to_world_objects(records, seed=seed)
        world = World(objs, seed=seed, p_arm_noise=0.03)
        spec = spec_for_assets(objs, seed)
        stats["episodes"] += 1
        res = run_episode(world, default_eyes(seed, noisy_eyes), ScriptedBrain(), spec, harness, cfg)
        if res.success:
            continue
        stats["failed"] += 1
        rec = relabel(world, res, spec)
        if rec is None:
            continue
        stats["relabeled"] += 1
        new_spec = TaskSpec.from_dict(rec.spec)
        objs2 = to_world_objects(records, seed=seed)
        replay = run_episode(World(objs2, seed=seed), default_eyes(seed, noisy_eyes),
                             ReplayBrain(parse_plan(rec.plan_json)), new_spec, harness, cfg)
        if not replay.success:
            continue
        stats["validated"] += 1
        scene = merge_scenes([e.report(World(objs, seed=seed), 0)
                              for e in default_eyes(seed, noisy_eyes)])
        docs = retrieve_docs(harness, cfg, new_spec, scene, [])
        ctx = BrainContext(spec=new_spec, scene=scene,
                           rule_ids={d.id for d in docs if d.type == "rule"}, snippets=docs)
        rows.append({"prompt": build_prompt(ctx), "completion": rec.plan_json,
                     "meta": {"seed": seed, "kind": "hindsight",
                              "orig_instruction": spec.text(),
                              "relabeled_instruction": rec.instruction,
                              "evidence": "SIM-POSE", "asset_manifest": manifest}})
    return rows, stats
