#!/usr/bin/env python3
"""Download the manufacturer/Menagerie robot models the two-arm scene needs (G-5, D-34).

Sources (all Apache-2.0, no gated token):
  * MuJoCo Menagerie `franka_fr3_v2` and `franka_fr3_duo` — ready-made MJCF, meshes and license.
  * The same project's FR3 assets are the simulation stand-in for the Franka FR3 + Franka Hand
    specified in D-29; the real-twin override is recorded in docs/ASSETS.md.

Only the named directories are fetched (never a whole repository). Output goes to
``assets/cache/menagerie/<model>/`` with an attribution CSV that names the source commit.

    python ops/fetch_robots.py --fetch
    python ops/fetch_robots.py --verify
"""
from __future__ import annotations

import argparse
import csv
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "assets" / "cache" / "menagerie"
API = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents"
MODELS = ("franka_fr3_v2", "franka_fr3_duo")
REPORT = ROOT / "reports" / "robot_assets.json"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rtp-asset-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def _tree(path: str) -> list[dict]:
    """One GitHub contents listing; directories are recursed with their own API url."""
    items = json.loads(_get(f"{API}/{path}"))
    out = []
    for item in items:
        if item["type"] == "dir":
            out.extend(_tree(item["path"]))
        elif item["type"] == "file":
            out.append({"path": item["path"], "download_url": item["download_url"],
                        "size": item["size"], "sha": item["sha"]})
    return out


def fetch() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    summary = {"source": "google-deepmind/mujoco_menagerie", "license": "Apache-2.0", "models": {}}
    rows = []
    for model in MODELS:
        files = _tree(model)
        dest = CACHE / model
        for entry in files:
            rel = Path(entry["path"]).relative_to(model)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != entry["size"]:
                target.write_bytes(_get(entry["download_url"]))
            rows.append([model, rel.as_posix(), entry["size"], entry["sha"],
                         f"{API}/{entry['path']}"])
        xmls = [p for p in dest.rglob("*.xml")]
        for p in xmls:
            ET.parse(p)                                   # fail loudly on a truncated download
        summary["models"][model] = {"files": len(files), "xml_files": len(xmls),
                                    "bytes": sum(e["size"] for e in files)}
    (CACHE / "ATTRIBUTION.csv").write_text(
        "model,path,bytes,github_blob_sha,source_url\n" +
        "\n".join(",".join(str(c) for c in row) for row in rows) + "\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def verify() -> dict:
    report = json.loads(REPORT.read_text())
    for model, info in report["models"].items():
        dest = CACHE / model
        actual = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
        if actual != info["bytes"]:
            raise SystemExit(f"{model}: {actual} bytes on disk, {info['bytes']} recorded")
        for p in dest.rglob("*.xml"):
            ET.parse(p)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        report = verify()
    elif a.fetch:
        report = fetch()
    else:
        ap.print_help()
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
