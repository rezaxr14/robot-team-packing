#!/usr/bin/env python3
"""Fetch a focused, texture-free SimReady subset from Hugging Face (geometry + physics only).

The full nvidia/simready-assets packages are ~300 MB each (mostly 4K textures). For the
environment we need the root USD, geometry payload and the Newton/MuJoCo physics payloads plus the
licence metadata; those are ~2 MB per package. This script downloads exactly those files, writes a
manifest with source/licence/bytes, and never touches textures or thumbnails.

    python ops/fetch_simready_min.py --budget-mb 80
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "assets" / "cache" / "nvidia__simready-assets"
MANIFEST = ROOT / "assets" / "simready_downloaded.json"
REPO = "nvidia/simready-assets"
API_TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

# Packages chosen to cover our task classes; nearest matches are labelled in the selection file.
PACKAGES = (
    "book_open_a01", "misc_notebook_a01", "bottle_laboratory_a01", "storage_box_a01",
    "stationery_pencil_watercolor_black_a01", "cart_computer_a01", "dishware_mug_a01",
    "dishware_plate_a01", "chair_office_amf_black_a01", "fixture_counter_a01",
    "storage_shelf_basic_a01", "cabinet_file_a01", "paperbag_a01", "food_produce_dragonFruit_a01",
)

KEEP = (".metadata/", "com.nvidia.simready.packaging.json", "simready_usd/sm_",
        "simready_usd/payloads/")


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "rtp-simready-min/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def wanted(path: str) -> bool:
    if "/Textures/" in path or "/.thumbs/" in path or "/source_assets/" in path:
        return False
    return any(token in path for token in KEEP) and path.endswith((".usd", ".usda", ".json", ".txt"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-mb", type=float, default=80.0)
    ap.add_argument("--packages", default=",".join(PACKAGES))
    a = ap.parse_args()
    budget = int(a.budget_mb * 1e6)
    packages = [p.strip() for p in a.packages.split(",") if p.strip()]
    CACHE.mkdir(parents=True, exist_ok=True)
    entries = []
    total = 0
    for pkg in packages:
        try:
            tree = _get_json(f"{API_TREE}/{pkg}?recursive=true")
        except Exception as exc:                            # noqa: BLE001
            print(f"{pkg}: tree failed ({type(exc).__name__}); skipping")
            continue
        files = [f for f in tree if f.get("type") == "file" and wanted(f["path"])]
        pkg_bytes = sum(f.get("size", 0) for f in files)
        if total + pkg_bytes > budget:
            print(f"{pkg}: {pkg_bytes / 1e6:.1f} MB would exceed the budget; stopping")
            break
        paths = []
        for f in files:
            dest = CACHE / f["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or dest.stat().st_size != f.get("size", 0):
                req = urllib.request.Request(f"{RESOLVE}/{f['path']}",
                                             headers={"User-Agent": "rtp-simready-min/1.0"})
                with urllib.request.urlopen(req, timeout=180) as response, open(dest, "wb") as out:
                    out.write(response.read())
            paths.append(str(dest.relative_to(ROOT)))
            total += f.get("size", 0)
        licence = "CC-BY-4.0 (package .metadata/LICENSE.txt)"
        entries.append({"package": pkg, "repo": REPO, "class_guess": "mixed",
                        "license": licence, "local_path": str((CACHE / pkg).relative_to(ROOT)),
                        "file_count": len(paths), "bytes": sum(f.get("size", 0) for f in files),
                        "paths": paths, "fetched_utc": datetime.now(timezone.utc).isoformat()})
        print(f"{pkg}: {len(paths)} files, {entries[-1]['bytes'] / 1e6:.2f} MB", flush=True)

    manifest = {"schema": 1, "source": REPO, "note": ("texture-free subset: USD geometry + "
                "Newton/MuJoCo physics payloads + licence metadata; no 4K textures"),
                "total_bytes": total, "entries": entries}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"total {total / 1e6:.2f} MB -> {MANIFEST}")
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
