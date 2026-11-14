"""Procedural scene / spec / dataset generation for the symbolic world.

Produces: random packing scenes, task specs, Brain SFT pairs validated by closed-loop success,
GRPO prompt rows (scene_json + spec_json for rewards.trl_reward_funcs), and Eye SFT targets
(SCENE JSON; image paths are placeholders to be filled by the Isaac Lab renderer).
"""
from __future__ import annotations
import json
import random

from .world import World, Obj
from .rules import TaskSpec
from .perception import Eye, EyeConfig, oracle_report
from .brain import ScriptedBrain, build_prompt, BrainContext
from .loop import LoopConfig, run_episode, retrieve_docs

# class -> (mass range, fragility range, size)
CLASSES = {
    "book": ((0.8, 1.4), (0.05, 0.2), 2.0),
    "notebook": ((0.2, 0.5), (0.05, 0.2), 1.0),
    "pencilcase": ((0.1, 0.3), (0.2, 0.4), 1.0),
    "bottle": ((0.6, 1.0), (0.15, 0.3), 1.5),
    "lunchbox": ((0.3, 0.7), (0.5, 0.8), 2.5),
    "laptop": ((1.5, 1.9), (0.35, 0.5), 3.0),
}
RECIPE = ["book", "book", "notebook", "notebook", "pencilcase", "bottle", "lunchbox", "laptop"]
DISTRACTORS = ["mug", "plate"]
REGIONS = ("table_left", "table_right", "table_center")


def sample_world(seed: int, p_arm_noise: float = 0.03) -> World:
    rng = random.Random(f"world|{seed}")
    objs, counts = [], {}
    for cls in RECIPE:
        counts[cls] = counts.get(cls, 0) + 1
        (m0, m1), (f0, f1), size = CLASSES[cls]
        objs.append(Obj(name=f"{cls}{counts[cls]}", cls=cls, mass=round(rng.uniform(m0, m1), 3),
                        fragility=round(rng.uniform(f0, f1), 3), size=size, region=rng.choice(REGIONS),
                        friction=round(rng.uniform(0.9, 1.1), 3)))
    for d in DISTRACTORS:
        objs.append(Obj(name=d, cls=d, mass=0.3, fragility=0.5, size=1.0, region=rng.choice(REGIONS),
                        is_distractor=True))
    return World(objs, seed=seed, p_arm_noise=p_arm_noise)


def sample_spec(world: World, seed: int) -> TaskSpec:
    rng = random.Random(f"spec|{seed}")
    items = [n for n, o in world.objs.items() if not o.is_distractor]
    rng.shuffle(items)
    lunch = [n for n in items if n.startswith("lunchbox")]
    return TaskSpec(items=sorted(items), leave=sorted(d for d in DISTRACTORS if d in world.objs),
                    laptop_slot=rng.choice(["laptop", "laptop", "front"]),
                    last_item=lunch[0] if lunch else None)


def default_eyes(seed: int, noisy: bool = False):
    if not noisy:
        return [Eye(EyeConfig(name="A"), seed), Eye(EyeConfig(name="B"), seed)]
    return [Eye(EyeConfig(name="A", blind_regions=("table_right",), miss_occluded_p=0.8, mass_sigma=0.1,
                          frag_sigma=0.05, misclass_p=0.02), seed),
            Eye(EyeConfig(name="B", blind_regions=("table_left",), miss_occluded_p=0.8, mass_sigma=0.1,
                          frag_sigma=0.05, misclass_p=0.02), seed)]


def gen_brain_sft(n: int, seed0: int, harness, noisy_eyes: bool = False, cfg: LoopConfig = None):
    """Rows {prompt, completion, meta} from episodes the scripted teacher solved in the closed loop."""
    rows, tries = [], 0
    cfg = cfg or LoopConfig()
    while len(rows) < n and tries < 20 * n:
        s = seed0 + tries
        tries += 1
        w = sample_world(s)
        spec = sample_spec(w, s)
        teacher = ScriptedBrain()
        res = run_episode(w, default_eyes(s, noisy_eyes), teacher, spec, harness, cfg)
        if not res.success or not res.plans:
            continue
        w0 = sample_world(s)
        eyes = default_eyes(s, noisy_eyes)
        from .perception import merge_scenes
        scene = merge_scenes([e.report(w0, 0) for e in eyes])
        docs = retrieve_docs(harness, cfg, spec, scene, [])
        ctx = BrainContext(spec=spec, scene=scene, rule_ids={d.id for d in docs if d.type == "rule"}, snippets=docs)
        rows.append({"prompt": build_prompt(ctx), "completion": res.plans[0],
                     "meta": {"seed": s, "msg_tokens": res.msg_tokens, "teacher": "scripted"}})
    return rows


def gen_grpo_prompts(n: int, seed0: int, harness, noisy_eyes: bool = True):
    rows = []
    for i in range(n):
        s = seed0 + i
        w = sample_world(s)
        spec = sample_spec(w, s)
        from .perception import merge_scenes
        scene = merge_scenes([e.report(w, 0) for e in default_eyes(s, noisy_eyes)])
        docs = retrieve_docs(harness, LoopConfig(), spec, scene, [])
        ctx = BrainContext(spec=spec, scene=scene, rule_ids={d.id for d in docs if d.type == "rule"}, snippets=docs)
        rows.append({"prompt": build_prompt(ctx), "scene_json": json.dumps(scene),
                     "spec_json": json.dumps(spec.to_dict()), "seed": s})
    return rows


def gen_eye_sft(n: int, seed0: int):
    """SCENE-JSON targets for Eye SFT. `image` is a placeholder; render it from the same state in Isaac Lab."""
    rows = []
    for i in range(n):
        s = seed0 + i
        w = sample_world(s)
        for view in ("A", "B"):
            m = oracle_report(w, view, 0)
            rows.append({"image": f"renders/seed{s}_view{view}.png", "target": json.dumps(m.content["objects"]),
                         "meta": {"seed": s, "view": view}})
    return rows


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def gen_hindsight_sft(n: int, seed0: int, harness, noisy_eyes: bool = True, max_tries: int = None,
                      cfg: LoopConfig = None):
    """Hindsight-relabeled Brain SFT rows from FAILED episodes, each validated by replay.
    Returns (rows, stats). stats gives the yield funnel: failures seen -> relabeled -> replay-validated."""
    from .hindsight import relabel, ReplayBrain
    from .plan import parse_plan
    from .rules import TaskSpec
    from .perception import merge_scenes
    cfg = cfg or LoopConfig()
    rows, stats, s = [], {"episodes": 0, "failed": 0, "relabeled": 0, "validated": 0}, 0
    while len(rows) < n and s < (max_tries or 50 * n):
        seed, s = seed0 + s, s + 1
        w = sample_world(seed)
        spec = sample_spec(w, seed)
        stats["episodes"] += 1
        res = run_episode(w, default_eyes(seed, noisy_eyes), ScriptedBrain(), spec, harness, cfg)
        if res.success:
            continue
        stats["failed"] += 1
        rec = relabel(w, res, spec)
        if rec is None:
            continue
        stats["relabeled"] += 1
        new_spec = TaskSpec.from_dict(rec.spec)
        w2 = sample_world(seed)
        eyes2 = default_eyes(seed, noisy_eyes)
        r2 = run_episode(w2, eyes2, ReplayBrain(parse_plan(rec.plan_json)), new_spec, harness, cfg)
        if not r2.success:
            continue
        stats["validated"] += 1
        w0 = sample_world(seed)
        scene = merge_scenes([e.report(w0, 0) for e in default_eyes(seed, noisy_eyes)])
        docs = retrieve_docs(harness, cfg, new_spec, scene, [])
        ctx = BrainContext(spec=new_spec, scene=scene, rule_ids={d.id for d in docs if d.type == "rule"}, snippets=docs)
        rows.append({"prompt": build_prompt(ctx), "completion": rec.plan_json,
                     "meta": {"seed": seed, "kind": "hindsight", "orig_instruction": spec.text(),
                              "relabeled_instruction": rec.instruction}})
    return rows, stats


def gen_eye_grpo_rows(n: int, seed0: int):
    """Rows for Eye GRPO: image placeholder + (seed, view). Render the image from the same seed in Isaac Lab."""
    rows = []
    for i in range(n):
        s = seed0 + i
        for view in ("A", "B"):
            rows.append({"prompt": f"[image renders/seed{s}_view{view}.png] List every object you can see as JSON "
                                   f"(name, cls, region, size, mass_guess, fragility_guess, occluded, confidence).",
                         "seed": s, "view": view})
    return rows
