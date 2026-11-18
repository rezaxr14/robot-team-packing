#!/usr/bin/env python3
"""G-13 (partial, honest): SimReady USD packages import into Newton 1.6.1.

Run with the Isaac venv. For each downloaded texture-free package, load its root USD with
``ModelBuilder.add_usd`` and record body/shape counts and mass. This proves ingestion of real
SimReady geometry, NOT the full G-13 gate (which also needs mass-within-5%-of-package-metadata,
a 10 s drop test, ≥80% grasp on ≥10 objects and visible textures in the Newton renderer). Those
remain pending and are listed in the report.
"""
from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets" / "simready_downloaded.json"
OUT = ROOT / "reports" / "g13_usd_ingest.json"


def main() -> int:
    import numpy as np
    import newton

    data = json.loads(MANIFEST.read_text())
    report = {"gate": "G-13 (USD ingestion only)",
              "scope": ("Real SimReady USD geometry imported into Newton; NOT the full G-13 gate "
                        "(no drop test, grasp test, mass-vs-datasheet check or rendered textures)"),
              "versions": {"python": platform.python_version(), "newton": metadata.version("newton")},
              "source": data["source"], "packages": {}, "pending": []}
    ok = 0
    for entry in data["entries"]:
        pkg = entry["package"]
        root = ROOT / entry["local_path"] / "simready_usd"
        usds = sorted(root.glob("sm_*")) if root.is_dir() else []
        if not usds:
            report["packages"][pkg] = {"error": "no root USD found"}
            continue
        try:
            builder = newton.ModelBuilder()
            try:
                builder.add_usd(str(usds[0]))
            except ValueError as exc:
                # Articulated SimReady packages declare joints that Newton's plain add_usd cannot
                # attach without an explicit articulation; dropping the joint layer still imports
                # the real geometry, which is what this ingestion-only gate checks.
                if "not belonging to any articulation" not in str(exc):
                    raise
                builder = newton.ModelBuilder()
                builder.add_usd(str(usds[0]), ignore_paths=["/RootNode/Joints"],
                                only_load_enabled_joints=False)
            model = builder.finalize()
            mass = float(np.sum(model.body_mass.numpy()))
            finite = bool(np.isfinite(model.body_mass.numpy()).all())
            report["packages"][pkg] = {"file": str(usds[0].relative_to(ROOT)),
                                       "bodies": model.body_count, "shapes": model.shape_count,
                                       "mass_kg": round(mass, 5), "masses_finite": finite}
            ok += int(finite and model.shape_count > 0)
        except Exception as exc:                              # noqa: BLE001
            report["packages"][pkg] = {"error": f"{type(exc).__name__}: {str(exc)[:180]}"}
    report["imported"] = ok
    report["attempted"] = len(data["entries"])
    report["pending"].extend([
        "Compare imported mass with the package metadata within 5%.",
        "10 s drop test stable in Newton.",
        "Scripted Franka Hand grasp >= 80% on >= 10 objects.",
        "Textures visible in the Newton Warp renderer (textures were deliberately not downloaded).",
        "Build the full two-FR3 kitchen scene from these packages.",
    ])
    report["result"] = "PASS" if ok == report["attempted"] else "PARTIAL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
