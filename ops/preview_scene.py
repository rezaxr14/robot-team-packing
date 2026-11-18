#!/usr/bin/env python3
"""Stage-1 review preview of the assembled kitchen scene (pure Newton render).

Builds one Newton model with:
  * two separate Franka FR3 v2 arms, bases bolted to the table ends with margin,
  * a ground plane and a kitchen countertop,
  * real-scale SimReady USD objects from HuggingFace placed on the counter / behind it,
  * two eye cameras (left/right) plus review cameras.

Objects keep their real-world scale. The chair sits behind the counter on the floor.
Real albedo maps (downloaded from nvidia/simready-assets) are loaded so the assets render in their
true colors; the chair has no albedo map so it gets an explicit color.

Frames -> reports/scene_preview/*.png ; manifest -> reports/scene_preview.json.

    source ops/env.sh && $ISAAC_PY ops/preview_scene.py --width 640 --height 480
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets" / "simready_downloaded.json"
G13 = ROOT / "reports" / "g13_usd_ingest.json"
ARM_DIR = ROOT / "assets" / "cache" / "menagerie" / "franka_fr3_v2"
OUT = ROOT / "reports" / "scene_preview"
REPORT = ROOT / "reports" / "scene_preview.json"

COUNTER_TOP = 0.90
COUNTER_T = 0.06
COUNTER_HX, COUNTER_HY = 0.75, 0.45
ARM_X = 0.60      # arm base x; 0.15 m margin from the +/-0.75 m table ends
ARM_Y = -0.12     # near the front edge so bases visibly sit on the table
ARM_Z = COUNTER_TOP          # measured: fr3v2 base-plate bottom = body origin (offset 0.0 m)

# (package, (x, y, "on"|"floor"), rgb used only if the asset has no albedo map)
ITEMS = [
    ("dishware_mug_a01",                       (-0.42, -0.10, "on"),   (0.20, 0.20, 0.20)),
    ("dishware_plate_a01",                     (-0.12, -0.10, "on"),   (0.90, 0.90, 0.90)),
    ("book_open_a01",                          ( 0.18, -0.10, "on"),   (0.20, 0.20, 0.20)),
    ("misc_notebook_a01",                      ( 0.42, -0.10, "on"),   (0.20, 0.20, 0.20)),
    ("stationery_pencil_watercolor_black_a01", ( 0.42,  0.02, "on"),   (0.20, 0.20, 0.20)),
    ("food_produce_dragonFruit_a01",           ( 0.18,  0.05, "on"),   (0.90, 0.20, 0.40)),
    ("chair_office_amf_black_a01",             ( 0.00,  0.75, "floor"),(0.15, 0.45, 0.55)),
]


def root_usd(pkg: str):
    d = ROOT / "assets" / "cache" / "nvidia__simready-assets" / pkg / "simready_usd"
    files = sorted(d.glob("sm_*")) if d.is_dir() else []
    return files[0] if files else None


def bbox_size(usd: Path):
    try:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(usd))
        prim = stage.GetDefaultPrim()
        if prim is None:
            return None
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        return tuple(float(box.max[i] - box.min[i]) for i in range(3))
    except Exception:
        return None


def look_quat(pos, target):
    import numpy as np
    import warp as wp
    f = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, f)
    m = np.stack([right, cam_up, -f], axis=1)
    return wp.quat_from_matrix(wp.mat33f(m.astype("float32")))


def build():
    import newton
    import warp as wp
    import numpy as np

    builder = newton.ModelBuilder()
    cwd = Path.cwd()
    os.chdir(ARM_DIR)
    try:
        # Two separate single-arm rigs, bases bolted near the two table ends.
        for label, x in (("arm_left", -ARM_X), ("arm_right", ARM_X)):
            builder.add_mjcf("fr3v2.xml",
                             xform=wp.transform(wp.vec3f(x, ARM_Y, ARM_Z), wp.quat_identity()))
    finally:
        os.chdir(cwd)

    # Countertop as a static box with a stone-grey color.
    counter_body = builder.add_body(is_kinematic=True, label="countertop")
    counter_shape = builder.add_shape_box(
        body=counter_body,
        xform=wp.transform(wp.vec3f(0.0, 0.0, COUNTER_TOP - COUNTER_T / 2), wp.quat_identity()),
        hx=COUNTER_HX, hy=COUNTER_HY, hz=COUNTER_T / 2,
        color=(0.55, 0.57, 0.60), label="countertop")

    # Real floor at z=0 (fr3v2.xml has no floor, so nothing floats at counter height).
    ground_body = builder.add_body(is_kinematic=True, label="ground")
    builder.add_shape_box(body=ground_body,
                          xform=wp.transform(wp.vec3f(0.0, 0.0, -0.025), wp.quat_identity()),
                          hx=5.0, hy=5.0, hz=0.025,
                          color=(0.70, 0.70, 0.72), label="ground")

    placed = []
    fallback_colors = []
    for pkg, (x, y, where), rgb in ITEMS:
        usd = root_usd(pkg)
        if usd is None:
            placed.append({"package": pkg, "error": "no root USD"})
            continue
        size = bbox_size(usd)
        z = COUNTER_TOP + (size[2] / 2 if size else 0.05) if where == "on" else (size[2] / 2 if size else 0.40)
        start = builder.shape_count
        try:
            builder.add_usd(str(usd),
                            xform=wp.transform(wp.vec3f(x, y, z), wp.quat_identity()),
                            ignore_paths=["/RootNode/Joints"],
                            only_load_enabled_joints=False)
            # Detect albedo from the USD's OWN texture references, not from directory
            # contents (earlier debug placeholders left stray files in these folders).
            import re
            has_albedo = False
            texdir = usd.parent
            for usd_file in list(texdir.rglob("*.usda")) + list(texdir.rglob("*.usd")):
                refs = re.findall(r"@([^@]+\.(?:png|jpg|exr))@", usd_file.read_text(errors="ignore"), re.I)
                if any("_alb" in r.lower() or "albedo" in r.lower() for r in refs):
                    has_albedo = True
                    break
            placed.append({"package": pkg, "usd": str(usd.relative_to(ROOT)),
                           "pos": [round(x, 3), round(y, 3), round(z, 3)],
                           "bbox_m": [round(s, 4) for s in size] if size else None,
                           "scale": 1.0, "placement": where,
                           "has_albedo": has_albedo, "color": list(rgb) if not has_albedo else None})
            if not has_albedo:
                fallback_colors.append((start, builder.shape_count, rgb))
        except Exception as exc:                          # noqa: BLE001
            placed.append({"package": pkg, "error": f"{type(exc).__name__}: {str(exc)[:120]}"})

    model = builder.finalize()
    # Only override color for shapes that have no real texture (chair, countertop).
    colors = np.asarray(model.shape_color.numpy()).copy()
    for start, end, rgb in fallback_colors:
        colors[start:end, :3] = np.asarray(rgb, dtype=colors.dtype)
    model.shape_color = wp.array(colors, dtype=model.shape_color.dtype, device=model.device)
    return model, placed


def render(model, width, height):
    import warp as wp
    import numpy as np
    from PIL import Image
    from newton.sensors import SensorTiledCamera

    state = model.state()
    # Each FR3 v2 contributes 8 scalar joint coords (joints 1..7). Fold the arms into a
    # "ready" pose instead of leaving them stretched straight up at the MJCF default.
    import newton
    q = np.asarray(state.joint_q.numpy()).copy()
    # Mirrored poses: each arm leans INWARD over the counter toward the workspace.
    poses = [
        [0.0, -1.15, 0.35, -0.75, 1.10, 0.80, 0.20],  # left arm
        [0.0,  1.15, 0.35,  0.75, 1.10, 0.80, 0.20],  # right arm (mirrored)
    ]
    for arm_i, pose in enumerate(poses):
        offset = arm_i * 8
        for i, val in enumerate(pose):
            if offset + i < len(q):
                q[offset + i] = val
    state.joint_q = wp.array(q, dtype=state.joint_q.dtype, device=model.device)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    sensor = SensorTiledCamera(model, load_textures=True)   # real SimReady albedo maps
    sensor.default_render_config.enable_textures = True
    sensor.default_render_config.enable_shadows = True
    sensor.default_render_config.enable_ambient_lighting = True
    rays = sensor.utils.compute_camera_rays_pinhole(width, height,
                                                     camera_fovs=math.radians(55.0))
    image = sensor.utils.create_color_image_output(width, height, 1)

    tgt = (0.0, 0.0, COUNTER_TOP + 0.05)
    cams = {
        "oblique":   ((1.6, -1.9, 1.9), tgt),
        "front":     ((0.0, -2.2, 1.45), tgt),
        "top":       ((0.0, 0.0, 3.0), (0.0, 0.0, 0.0)),
        "eye_left":  ((-0.55, -0.45, 1.55), (0.0, 0.0, COUNTER_TOP)),
        "eye_right": ((0.55, -0.45, 1.55), (0.0, 0.0, COUNTER_TOP)),
    }
    out = {}
    for name, (pos, target) in cams.items():
        xf = wp.array([[wp.transformf(wp.vec3f(*pos), look_quat(pos, target))]], dtype=wp.transformf)
        model.bvh_refit_shapes(state)
        sensor.update(state, xf, rays, color_image=image,
                      clear_data=SensorTiledCamera.GRAY_CLEAR_DATA)
        rgba = sensor.utils.to_rgba_from_color(image).numpy().reshape(1, 1, height, width, 4)[0, 0]
        frame = np.asarray(rgba)[..., :3].astype("uint8")
        path = OUT / f"{name}.png"
        Image.fromarray(frame).save(path)
        out[name] = str(path.relative_to(ROOT))
        print(f"rendered {name}: {frame.shape} -> {path}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    g13 = json.loads(G13.read_text())["packages"]
    model, placed = build()
    frames = render(model, a.width, a.height)
    report = {
        "kind": "stage-1 preview: two FR3 arms bolted to table ends + real-textured SimReady objects + 2 eye cams",
        "scope": "rendered and measured; NOT stepped, NOT a manipulation/safety result",
        "versions": {"python": platform.python_version(), "newton": metadata.version("newton")},
        "robot": str(ARM_DIR.relative_to(ROOT) / "scene.xml"),
        "arm_bases": {"left": [-ARM_X, ARM_Y, ARM_Z], "right": [ARM_X, ARM_Y, ARM_Z],
                      "table_end_margin_m": round(COUNTER_HX - ARM_X, 3)},
        "bodies": model.body_count, "shapes": model.shape_count,
        "counter_top_z": COUNTER_TOP,
        "objects": placed, "frames": frames,
        "g13_status": {pkg: ("ok" if g13.get(pkg, {}).get("bodies") == 1 else "excluded")
                       for pkg, _, _ in ITEMS},
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"-> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
