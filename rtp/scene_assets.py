"""Turn a downloaded SimReady catalog into attributed objects for our environment.

This module is the bridge between the asset files fetched from Hugging Face and the rest of the
pipeline. It deliberately keeps two levels separate:

* :func:`resolve_assets` validates that the selected packages really exist locally and records
  their source/licence — no network, no guessing.
* :func:`to_world_objects` maps them to ``rtp.world.Obj`` for the *pose-level* environment, keeping
  the asset path on each object so every episode can be traced back to a real package.

The Newton/Isaac scene consumes the same records; this symbolic mapping is NOT a claim that a USD
file was simulated. A package with no measured mass/dims uses the class default and is marked as
such in the manifest. "Nearest match" classes are marked explicitly — a storage box standing in
for a lunchbox is not called a lunchbox.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from rtp.world import Obj

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "assets" / "catalog.json"
DEFAULT_SELECTION = ROOT / "assets" / "selection.json"

# Task classes -> SimReady naming hints. Only "exact" matches are claimed as such.
CLASS_HINTS = {
    "book": ("book", "decor_book"),
    "notebook": ("notebook",),
    "bottle": ("bottle",),
    "lunchbox": ("storage_box", "lunchbox"),
    "laptop": ("laptop", "cart_computer", "computer_monitor"),
    "pencilcase": ("pencilcase", "pencil_case", "stationery_pencil"),
    "table": ("table", "counter"),
    "chair": ("chair",),
    "cabinet": ("cabinet",),
    "shelf": ("shelf",),
    "bag": ("bag", "paperbag", "backpack"),
}


def _pick(d: dict, *names: str, default=None):
    for name in names:
        if name in d and d[name] not in (None, "", []):
            return d[name]
    return default


@dataclass
class AssetRecord:
    package: str
    repo: str
    cls: str
    licence: str
    paths: list[str] = field(default_factory=list)
    nearest_match: bool = False
    mass: float | None = None
    size: float | None = None
    source_of_measurement: str = "class-default"

    def as_dict(self) -> dict:
        return {"package": self.package, "repo": self.repo, "class": self.cls,
                "license": self.licence, "files": len(self.paths),
                "nearest_match": self.nearest_match, "mass_kg": self.mass,
                "size_m": self.size, "measurement": self.source_of_measurement,
                "example_path": self.paths[0] if self.paths else None}


def _load_json(path: Path):
    data = json.loads(Path(path).read_text())
    return data.get("assets", data.get("packages", data)) if isinstance(data, dict) else data


def resolve_assets(catalog: Path | str = DEFAULT_CATALOG,
                   selection: Path | str = DEFAULT_SELECTION) -> list[AssetRecord]:
    """Validate every selected package against the catalog and the files on disk."""
    catalog_entries = _load_json(catalog)
    selected = _load_json(selection)
    wanted = set()
    for item in selected:
        wanted.add(_pick(item, "package", "id", "name") if isinstance(item, dict) else item)
    if not wanted:
        raise ValueError(f"no packages selected in {selection}")

    by_pkg = {}
    for item in catalog_entries:
        if not isinstance(item, dict):
            continue
        pkg = _pick(item, "package", "id", "name")
        if pkg in wanted:
            by_pkg[pkg] = item
    records = []
    for pkg in sorted(wanted):
        entry = by_pkg.get(pkg)
        if entry is None:
            raise ValueError(f"selected package {pkg!r} is missing from {catalog}")
        repo = _pick(entry, "repo", "source", "repo_id", default="unknown")
        licence = _pick(entry, "license", "licence", default="see source")
        cls = _pick(entry, "class", "class_guess", "task_class", default="unknown")
        paths = [p for p in (_pick(entry, "paths", "files", "local_files", default=[]) or [])]
        local = _pick(entry, "local_path", "path", default=None)
        if local and not paths:
            root = Path(local)
            paths = [str(p) for p in root.rglob("*") if p.is_file()] if root.is_dir() else [str(root)]
        absent = [p for p in paths if not Path(p).exists()]
        if absent:
            raise FileNotFoundError(f"{pkg}: {len(absent)} catalog files missing (first: {absent[0]})")
        nearest = bool(_pick(entry, "nearest_match", default=False))
        mass = _pick(entry, "mass_kg", "mass", default=None)
        size = _pick(entry, "size_m", "size", default=None)
        records.append(AssetRecord(package=pkg, repo=str(repo), cls=str(cls),
                                   licence=str(licence), paths=paths, nearest_match=nearest,
                                   mass=mass, size=size,
                                   source_of_measurement="package-metadata" if mass or size
                                   else "class-default"))
    return records


def to_world_objects(records: Iterable[AssetRecord], *, seed: int = 0) -> list[Obj]:
    """Map resolved assets onto the pose-level world's objects with stable names."""
    from rtp.datagen import CLASSES

    out = []
    for i, rec in enumerate(records):
        cls = rec.cls if rec.cls in CLASSES else "book"
        mrange, frange, size = CLASSES[cls]
        # deterministic within a class, from the package name (no hidden randomness)
        import hashlib
        h = hashlib.sha256(f"{seed}|{rec.package}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2 ** 64
        mass = float(rec.mass) if rec.mass is not None else mrange[0] + u * (mrange[1] - mrange[0])
        fragility = frange[0] + (1 - u) * (frange[1] - frange[0])
        region = ("table_left", "table_center", "table_right")[i % 3]
        out.append(Obj(name=f"{rec.package}_{i:02d}", cls=cls, mass=round(mass, 3),
                       fragility=round(fragility, 3),
                       size=float(rec.size) if rec.size else size,
                       region=region, friction=1.0))
    return out


def write_manifest(records: Iterable[AssetRecord], objs: Iterable[Obj], path: Path | str) -> Path:
    """Write the asset-to-object provenance table used by the environment and the data generator."""
    payload = {
        "evidence_class": "asset-backed pose-level environment (NOT a Newton physics result)",
        "assets": [r.as_dict() for r in records],
        "objects": [{"name": o.name, "class": o.cls, "mass_kg": o.mass,
                     "fragility": o.fragility, "size_m": o.size, "region": o.region}
                    for o in objs],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p
