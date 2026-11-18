#!/usr/bin/env python3
"""G-5 (partial): load the Menagerie FR3 v2 / FR3 Duo MJCF into Newton and measure the model.

Run with the isolated Isaac venv (``$ISAAC_PY`` from ops/env.sh), not the training venv. This verifies
the *robot model* half of G-5: the MJCF parses in Newton 1.6.1 with MuJoCo-Warp, has the expected
joint DoF and a finite mass. It does NOT verify the datasheet within 2% or a zero-pose reach test —
those need the real datasheet and are left as explicit pending items in the report.

    source ops/env.sh && $ISAAC_PY ops/g5_fr3_load.py
"""
from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "assets" / "cache" / "menagerie"
OUT = ROOT / "reports" / "g5_fr3.json"
MODELS = {"franka_fr3_v2": "fr3v2.xml", "franka_fr3_duo": "scene.xml"}


def main() -> int:
    import numpy as np
    import newton

    report = {"gate": "G-5 (robot-model ingestion only)",
              "scope": "Newton MJCF parse + measured mass/DoF; NOT a datasheet or reach test",
              "versions": {"python": platform.python_version(), "newton": metadata.version("newton")},
              "models": {}, "pending": []}
    ok = True
    for name, entry in MODELS.items():
        path = CACHE / name / entry
        if not path.exists():
            report["models"][name] = {"error": f"missing {path}"}
            ok = False
            continue
        builder = newton.ModelBuilder()
        # mesh assets are referenced relative to the MJCF, so run from its directory
        import os
        cwd = Path.cwd()
        os.chdir(path.parent)
        try:
            builder.add_mjcf(entry)
            model = builder.finalize()
            mass = float(np.sum(model.body_mass.numpy()))
            finite = bool(np.isfinite(model.body_mass.numpy()).all())
            info = {"file": str(path.relative_to(ROOT)), "bodies": model.body_count,
                    "joints": model.joint_count, "joint_dof": model.joint_dof_count,
                    "shapes": model.shape_count, "total_mass_kg": round(mass, 4),
                    "masses_finite": finite,
                    "checks": {"parsed": True, "finite_mass": finite,
                               "has_arm_dof": model.joint_dof_count >= 7}}
            report["models"][name] = info
            ok &= finite and model.joint_dof_count >= 7
        except Exception as exc:                              # noqa: BLE001
            report["models"][name] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            ok = False
        finally:
            os.chdir(cwd)

    report["pending"].append("Measure real FR3 masses/limits against the datasheet within 2%.")
    report["pending"].append("Zero-pose reach test and agreement between Franka official and Menagerie kinematics within 1 mm.")
    report["result"] = "PASS" if ok else "FAIL"
    report["error"] = None if ok else "see models"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
