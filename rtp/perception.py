"""Eye agents (the only agents with 'pixels') in the symbolic world.

In Isaac Lab the Eye is a VLM that maps rendered images to SCENE JSON. Here an Eye is a
parameterised noisy observer of ground truth, used to test the pipeline, inject faults, and
compute counterfactual credit. VLMEye is the adapter for the real model (needs images).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

from .messages import Message
from .world import World

CONFUSABLE = {"book": "notebook", "notebook": "book", "laptop": "book", "bottle": "lunchbox",
              "lunchbox": "bottle", "pencilcase": "notebook", "mug": "bottle", "plate": "lunchbox"}


@dataclass
class EyeConfig:
    name: str = "A"
    blind_regions: tuple = ()        # viewpoint cannot see these regions
    misclass_p: float = 0.0
    mass_sigma: float = 0.0          # multiplicative noise on guessed mass
    frag_sigma: float = 0.0
    stale_p: float = 0.0             # returns the previous report
    miss_occluded_p: float = 1.0     # prob. an object in a blind region is omitted (else flagged occluded)
    region_error_p: float = 0.0      # reports a wrong region


class Eye:
    def __init__(self, cfg: EyeConfig, seed: int = 0):
        self.cfg, self.seed = cfg, seed
        self._last: Message | None = None

    @property
    def name(self) -> str:
        return self.cfg.name

    def _rng(self, t: int, tag: str = "") -> random.Random:
        return random.Random(f"{self.seed}|{self.cfg.name}|{t}|{tag}")

    def report(self, world: World, t: int = 0) -> Message:
        c, rng = self.cfg, self._rng(t)
        if self._last is not None and rng.random() < c.stale_p:
            m = Message(**{**self._last.__dict__, "t": t, "note": "stale"})
            return m
        objs = []
        for n, o in sorted(world.objs.items()):
            region = o.region
            occluded = region in c.blind_regions
            if occluded and rng.random() < c.miss_occluded_p:
                continue
            cls = o.cls
            if rng.random() < c.misclass_p:
                cls = CONFUSABLE.get(cls, cls)
            if rng.random() < c.region_error_p and region.startswith("table"):
                region = rng.choice([r for r in ("table_left", "table_right", "table_center") if r != region])
            mass = o.mass * (1 + c.mass_sigma * rng.gauss(0, 1)) if c.mass_sigma else o.mass
            frag = min(1.0, max(0.0, o.fragility + c.frag_sigma * rng.gauss(0, 1))) if c.frag_sigma else o.fragility
            objs.append({"name": n, "cls": cls, "region": region, "size": o.size,
                         "mass_guess": round(max(0.05, mass), 3), "fragility_guess": round(frag, 3),
                         "occluded": occluded, "confidence": 0.5 if occluded else 0.9})
        msg = Message("SCENE", f"eye_{c.name}", "all", {"objects": objs, "view": c.name}, t=t)
        self._last = msg
        return msg


def oracle_report(world: World, name: str, t: int = 0) -> Message:
    return Eye(EyeConfig(name=name), 0).report(world, t)


def replace_report(msg: Message, world: World, mode: str, rng: random.Random, fields=None) -> Message:
    """Counterfactual replacement of an Eye report.
    mode: 'empty' | 'degraded' | 'oracle' | 'oracle_fields' (replace only `fields` by ground truth)."""
    objs = [dict(o) for o in msg.content["objects"]]
    if mode == "empty":
        objs = []
    elif mode == "degraded":
        objs = [o for o in objs if rng.random() < 0.5]
        for o in objs:
            o["mass_guess"] = round(o["mass_guess"] * (1 + 0.5 * rng.gauss(0, 1)), 3)
    elif mode == "oracle":
        return oracle_report(world, msg.content["view"], msg.t)
    elif mode == "oracle_fields":
        for o in objs:
            tru = world.objs.get(o["name"])
            if tru is None:
                continue
            truth = {"cls": tru.cls, "region": tru.region, "mass_guess": tru.mass,
                     "fragility_guess": tru.fragility, "size": tru.size}
            for f in (fields or []):
                if f in truth:
                    o[f] = truth[f]
    else:
        raise ValueError(mode)
    return Message("SCENE", msg.sender, msg.receiver, {**msg.content, "objects": objs}, msg.note, msg.t)


def merge_scenes(msgs) -> dict:
    """Union of Eye reports; on conflict prefer the non-occluded, higher-confidence report."""
    merged: dict = {}
    for m in sorted(msgs, key=lambda m: str(m.content.get("view", ""))):   # order-independent tie-break
        for o in m.content["objects"]:
            cur = merged.get(o["name"])
            if cur is None or o["confidence"] > cur["confidence"]:
                merged[o["name"]] = o
    return merged


class FixedEye:
    """An Eye that emits a given list of object dicts (e.g. parsed from a model completion)."""

    def __init__(self, name: str, objects: list):
        self.name, self._objects = name, objects

    def report(self, world, t: int = 0) -> Message:
        return Message("SCENE", f"eye_{self.name}", "all", {"objects": [dict(o) for o in self._objects], "view": self.name}, t=t)


def default_blind(view: str) -> tuple:
    """Blind regions of the default noisy Eyes (see datagen.default_eyes)."""
    return {"A": ("table_right",), "B": ("table_left",)}.get(view, ())
