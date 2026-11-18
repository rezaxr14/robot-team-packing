#!/usr/bin/env python3
"""Assemble a two-arm Newton scene from the Menagerie FR3 Duo and downloaded SimReady USD objects.

This is the first real 3D scene in the repo (the earlier G-3 gate was a single falling cube). It
builds one Newton model containing both FR3 arms, the ground, and a placed subset of the real
SimReady packages, then writes a manifest with measured body/shape/mass counts.

Honest scope: the model is assembled and validated, but this script does NOT run manipulation
episodes, render the scene, or claim safety. Those require the full P2/P3 integration.
"""
from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / "assets" / "cache" / "menagerie" / "franka_fr3_duo" / "scene.xml"
MANIFEST = ROOT / "assets" / "simready_downloaded.json"
OUT = ROOT / "reports" / "scene_manifest.json"


def main() -> int:
    import numpy as np
    import newton
    import warp as wp

    data = json.loads(MANIFEST.read_text())
    report = {"scene": "two FR3 arms + placed SimReady objects + ground",
              "scope": "Newton model assembled and measured; NOT stepped, rendered or safety-tested",
              "versions": {"python": platform.python_version(), "newton": metadata.version("newton")},
              "robot": str(ROBOT.relative_to(ROOT)), "objects": [], "pending": []}
    builder = newton.ModelBuilder()
    import os
    cwd = Path.cwd()
    os.chdir(ROBOT.parent)
    try:
        builder.add_mjcf("scene.xml")
    finally:
        os.chdir(cwd)
    placed = 0
    for entry in data["entries"]:
        root = ROOT / entry["local_path"] / "simready_usd"
        usds = sorted(root.glob("sm_*")) if root.is_dir() else []
        if not usds:
            continue
        x = -0.4 + 0.2 * (placed % 5)
        y = -0.2 + 0.2 * (placed // 5)
        try:
            builder.add_usd(str(usds[0]), xform=wp.transform(wp.vec3(x, y, 0.8), wp.quat_identity()),
                            ignore_paths=["/RootNode/Joints"], only_load_enabled_joints=False)
            report["objects"].append({"package": entry["package"], "usd": str(usds[0].relative_to(ROOT)),
                                      "placement_xyz": [x, y, 0.8]})
            placed += 1
        except Exception as exc:                            # noqa: BLE001
            report["objects"].append({"package": entry["package"],
                                      "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
    builder.add_ground_plane()
    model = builder.finalize()
    report.update({"bodies": model.body_count, "joints": model.joint_count,
                   "shapes": model.shape_count,
                   "total_mass_kg": round(float(np.sum(model.body_mass.numpy())), 3),
                   "masses_finite": bool(np.isfinite(model.body_mass.numpy()).all()),
                   "placed_objects": placed})
    report["pending"].extend([
        "Render the scene with the Newton Warp renderer and inspect contacts.",
        "Step manipulation episodes with the two FR3 arms and PTZ eyes.",
        "Validate grasps, drop tests and mass-vs-datasheet (G-13/G-5).",
    ])
    report["result"] = "PASS" if report["masses_finite"] and placed >= 5 else "PARTIAL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
