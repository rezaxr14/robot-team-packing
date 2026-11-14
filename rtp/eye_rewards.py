"""Rewards for training the Eye (SCENE JSON from a completion).

Three signals, so the paper can compare them head-to-head:
  r_eye_format   : valid SCENE JSON with the required fields
  r_eye_accuracy : field-level agreement with ground truth (the usual perception reward)
  r_eye_credit   : counterfactual team credit -- score of the whole episode when THIS report is what the Eye
                   emitted, relative to the same episode with the Eye's report empty (common random numbers).
                   It rewards the details that change team outcomes (e.g. mass of the laptop) rather than
                   every field equally.
"""
from __future__ import annotations
import json
import random

from .perception import FixedEye, default_blind
from .rules import TaskSpec

REQUIRED = ("name", "cls", "region", "size", "mass_guess", "fragility_guess")


def parse_scene_completion(text):
    """Return a list of object dicts or None. Accepts a JSON list or {"objects": [...]}, fenced or bare, with <think>
    blocks and trailing commas tolerated; takes the LAST valid scene found."""
    from .plan import json_candidates
    if isinstance(text, list) and text and isinstance(text[-1], dict):     # TRL conversational format
        text = text[-1].get("content", "")
    if not isinstance(text, str):
        return None
    for obj in json_candidates(text, "list") + [d.get("objects") for d in json_candidates(text, "object")]:
        if not isinstance(obj, list) or not all(isinstance(o, dict) and all(k in o for k in REQUIRED) for o in obj):
            continue
        out = []
        try:
            for o in obj:
                out.append({"name": str(o["name"]), "cls": str(o["cls"]), "region": str(o["region"]),
                            "size": float(o["size"]), "mass_guess": float(o["mass_guess"]),
                            "fragility_guess": float(o["fragility_guess"]),
                            "occluded": bool(o.get("occluded", False)), "confidence": float(o.get("confidence", 0.9))})
        except (TypeError, ValueError):
            continue
        return out
    return None


def r_eye_format(completion) -> float:
    return 1.0 if parse_scene_completion(completion) is not None else 0.0


def r_eye_accuracy(completion, world, blind_regions=()) -> float:
    """F1-style score over objects visible from this view: class and region exact, mass within 15%,
    fragility within 0.15. Reported objects that do not exist count against precision."""
    objs = parse_scene_completion(completion)
    if objs is None:
        return 0.0
    visible = {n: o for n, o in world.objs.items() if o.region not in blind_regions}
    if not visible:
        return 1.0 if not objs else 0.0
    got = {o["name"]: o for o in objs}
    pts = 0.0
    for n, o in visible.items():
        g = got.get(n)
        if g is None:
            continue
        pts += 0.25 * (g["cls"] == o.cls) + 0.25 * (g["region"] == o.region)
        pts += 0.25 * (abs(g["mass_guess"] - o.mass) / max(0.05, o.mass) <= 0.15)
        pts += 0.25 * (abs(g["fragility_guess"] - o.fragility) <= 0.15)
    recall = pts / len(visible)
    precision = sum(1 for n in got if n in world.objs) / max(1, len(got))
    return 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)


def r_eye_credit(completion, seed: int, view: str, harness, n_eyes: int = 2, noisy_other: bool = True,
                 cfg=None) -> float:
    """Counterfactual team credit in [-1, 1]-ish: R(episode with this report) - R(episode with empty report).
    The other Eye (if any) keeps its parametric noise; all draws are common random numbers."""
    from .datagen import sample_world, sample_spec, default_eyes
    from .loop import LoopConfig, run_episode
    from .brain import ScriptedBrain
    objs = parse_scene_completion(completion)
    if objs is None:
        return -1.0
    cfg = cfg or LoopConfig()

    def episode(report_objs):
        w = sample_world(seed)
        spec = sample_spec(w, seed)
        eyes = [e for e in default_eyes(seed, noisy_other)[:n_eyes] if e.name != view]
        eyes.append(FixedEye(view, report_objs))
        return run_episode(w, eyes, ScriptedBrain(), spec, harness, cfg).score

    return episode(objs) - episode([])


DEFAULT_WEIGHTS = {"format": 0.1, "accuracy": 0.0, "credit": 0.9}


def trl_eye_reward_funcs(harness, which=("format", "accuracy", "credit")):
    """TRL adapters. Dataset columns: seed (int), view ('A'|'B'). Pick `which` to ablate signals."""
    from .datagen import sample_world

    def fmt(prompts, completions, **kw):
        return [r_eye_format(c) for c in completions]

    def acc(prompts, completions, **kw):
        out = []
        for c, s, v in zip(completions, kw["seed"], kw["view"]):
            out.append(r_eye_accuracy(c, sample_world(int(s)), default_blind(v)))
        return out

    def credit(prompts, completions, **kw):
        return [r_eye_credit(c, int(s), v, harness) for c, s, v in zip(completions, kw["seed"], kw["view"])]

    fmt.__name__, acc.__name__, credit.__name__ = "eye_format", "eye_accuracy", "eye_credit"
    return [f for f, n in ((fmt, "format"), (acc, "accuracy"), (credit, "credit")) if n in which]
