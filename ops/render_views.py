#!/usr/bin/env python3
"""R1: scene bundle + eye view library (code written; NOT run on a GPU here).

For one scene seed, render rgb+depth at 640x480 for the 12 eye stations from
``reports/eye_placement.json`` x zoom {1, 2, 4} x a small pan/tilt grid x two arm
poses (start / reach), using the verified Isaac Lab 3.0 + Newton Warp camera
recipe from ``ops/g3_sim.py``:

    SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
    CameraCfg(..., data_types=["rgb", "depth"], renderer_cfg=NewtonWarpRendererCfg())

Successful output (only when a USD scene really exists)::

    reports/scene_<seed>/views/*.jpg          rgb + depth frames
    reports/scene_<seed>/visibility.json      per-view visibility record
    reports/scene_<seed>/objects.json         scene objects + eye pair
    reports/scene_<seed>/manifest.json        sha256 of every bundle file, versions,
                                              measured fps per eye

HONEST FALLBACK. A USD scene cannot be built from the cached MJCF in this
Newton-only venv: the MJCF->USD importer lives in ``isaacsim``, which is absent.
In that case the CLI writes **no images**, returns exit code 1 with
``result: FAIL`` and ``reason: "MJCF->USD conversion unavailable in Newton-only
venv"``, and leaves ``fps_per_eye`` null. It never fabricates a render and never
claims the >= 10 fps acceptance (G-8). The bundle sha256 covers file bytes only,
so it stays stable across runs of the same seed even though wall-clock fps moves.

Usage::

    source ops/env.sh && $ISAAC_PY ops/render_views.py --seed 100001
    python ops/render_views.py --seed 0 --out /path/to/render_views.json   # CPU fallback check
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EYE_PLACEMENT = ROOT / "reports" / "eye_placement.json"
SCENE_MANIFEST = ROOT / "reports" / "scene_manifest.json"
MJCF_DUO = ROOT / "assets" / "cache" / "menagerie" / "franka_fr3_duo" / "scene.xml"
MJCF_V2 = ROOT / "assets" / "cache" / "menagerie" / "franka_fr3_v2" / "fr3v2.xml"

# The exact reason string required by the honest fallback rule (D-81).
MJCF_USD_UNAVAILABLE = "MJCF->USD conversion unavailable in Newton-only venv"

DISTRIBUTIONS = ("isaaclab", "isaaclab-newton", "newton", "warp-lang", "mujoco-warp",
                 "torch", "numpy")

DEFAULT_ZOOMS = (1, 2, 4)
DEFAULT_PAN_DEG = (-10.0, 0.0, 10.0)
DEFAULT_TILT_DEG = (-5.0, 0.0, 5.0)
ARM_POSES = ("start", "reach")
FPS_THRESHOLD = 10.0  # G-8 acceptance from AGENT_RUNBOOK.md R1; read, not changed


class FpsNotMeasured(ValueError):
    """Raised when code tries to declare a frame rate without real measurements."""


def versions() -> dict:
    """Versions that decide whether a render means anything (same pattern as ops/g3_sim.py)."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in DISTRIBUTIONS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "MISSING"
    return out


# --------------------------------------------------------------------------------------
# Pure-CPU manifest helpers (no isaaclab / warp / torch import at module import time).
# --------------------------------------------------------------------------------------

def sha256_file(path: Path | str) -> str:
    """sha256 of a file's raw bytes, streamed so large jpgs do not load into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def measured_fps(fps_per_eye: dict) -> dict:
    """Validate *real* measured fps values; raise :class:`FpsNotMeasured` otherwise.

    A missing, non-numeric, non-finite or non-positive value is never rounded into a
    claim. This is the choke point that keeps ``fps_per_eye`` honest.
    """
    if not isinstance(fps_per_eye, dict) or not fps_per_eye:
        raise FpsNotMeasured("no measured fps supplied; refusing to declare a frame rate")
    out: dict[str, float] = {}
    for eye, value in sorted(fps_per_eye.items()):
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise FpsNotMeasured(f"fps for {eye!r} is not a number: {value!r}") from None
        if not math.isfinite(f) or f <= 0.0:
            raise FpsNotMeasured(f"fps for {eye!r} is not a positive finite measurement: {value!r}")
        out[str(eye)] = round(f, 3)
    return out


def fps_verdict(manifest: dict, *, threshold: float = FPS_THRESHOLD) -> str:
    """Return PASS/FAIL from numbers in a manifest; raise if none were measured."""
    if not manifest.get("fps_measured") or not manifest.get("fps_per_eye"):
        raise FpsNotMeasured("cannot declare fps: manifest contains no measured fps numbers")
    fps = measured_fps(manifest["fps_per_eye"])
    return "PASS" if all(v >= threshold for v in fps.values()) else "FAIL"


def build_manifest(root: Path | str, files, versions: dict, *, fps_per_eye: dict | None = None) -> dict:
    """Build a deterministic manifest: sorted relative paths -> sha256 of each file.

    ``manifest["sha256"]`` is the sha256 of the canonical file-digest map *only*.
    Wall-clock fps is stored separately (and only when real numbers are supplied), so
    the bundle hash is stable across two runs of the same seed.
    """
    root = Path(root)
    rels = sorted(str(Path(f)) for f in files)
    digests = {rel: sha256_file(root / rel) for rel in rels}
    canonical = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
    fps = measured_fps(fps_per_eye) if fps_per_eye is not None else None
    return {
        "schema": 1,
        "versions": dict(sorted(dict(versions).items())),
        "files": digests,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "fps_per_eye": fps,
        "fps_measured": fps is not None,
        "fps_threshold": FPS_THRESHOLD if fps is not None else None,
    }


def verify_manifest(root: Path | str, manifest: dict) -> list[str]:
    """Recompute every digest; return the sorted list of tampered/missing paths.

    An empty list means the bundle is intact. Includes ``<manifest.sha256>`` if the
    file-digest map itself was edited.
    """
    root = Path(root)
    bad: list[str] = []
    files = manifest.get("files", {})
    for rel in sorted(files):
        path = root / rel
        if not path.is_file() or sha256_file(path) != files[rel]:
            bad.append(rel)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != manifest.get("sha256"):
        bad.append("<manifest.sha256>")
    return sorted(bad)


# --------------------------------------------------------------------------------------
# Honest USD-scene preflight (pure checks: no GPU, no package import).
# --------------------------------------------------------------------------------------

def preflight_usd_scene() -> tuple[bool, str]:
    """Return ``(can_build, reason)`` for a USD scene from the cached MJCF.

    Runtime MJCF->USD conversion needs the ``isaacsim`` package. It is absent from the
    Newton-only venv (D-72), so this returns False with the exact reason string. No
    import of CUDA/Warp packages happens here: ``find_spec`` only locates a module.
    """
    try:
        spec = importlib.util.find_spec("isaacsim")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return False, MJCF_USD_UNAVAILABLE
    mjcf = MJCF_DUO if MJCF_DUO.is_file() else (MJCF_V2 if MJCF_V2.is_file() else None)
    if mjcf is None:
        return False, "cached MJCF not found (looked for franka_fr3_duo/scene.xml and franka_fr3_v2/fr3v2.xml)"
    return True, "ok"


def load_stations(path: Path | str = EYE_PLACEMENT) -> dict:
    """Read the chosen 12-station ring and the eye pair from the eye placement report."""
    report = json.loads(Path(path).read_text())
    rec = report["detail"]["recommended"]
    return {
        "layout": report["chosen_layout"],
        "R": rec["R"],
        "H": rec["H"],
        "stations": rec["station_table"],
        "eye_pair": rec["eye_A_B_stations"]["stations"],
        "look_at": report["detail"]["assumptions"]["look_at"],
    }


# --------------------------------------------------------------------------------------
# Render path (only reachable in a venv that actually has isaacsim). Written against
# the verified ops/g3_sim.py + Isaac Lab camera-contract recipe; not executed here.
# --------------------------------------------------------------------------------------

def _mjpeg_to_usd(mjcf: Path, usd_path: Path) -> Path:
    """Convert cached MJCF to USD with the Isaac Sim MJCF importer (needs ``isaacsim``)."""
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from isaacsim.asset.importer.mjcf import MjcfImporter  # type: ignore
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError(MJCF_USD_UNAVAILABLE) from exc
    MjcfImporter(str(mjcf)).convert(str(usd_path))
    return usd_path


def _to_numpy(buf):
    """Camera outputs are ProxyArrays: (N,H,W,C) via .warp.numpy() or .torch."""
    if hasattr(buf, "warp"):
        return buf.warp.numpy()
    if hasattr(buf, "torch"):
        return buf.torch.detach().cpu().numpy()
    return buf


def _rotate(v, axis, deg):
    """Rodrigues rotation of a 3-vector about a unit axis (numpy, CPU maths)."""
    import numpy as np
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    t = math.radians(deg)
    v = np.asarray(v, dtype=float)
    return (v * math.cos(t) + np.cross(a, v) * math.sin(t)
            + a * np.dot(a, v) * (1.0 - math.cos(t)))


def _look_target(pos, look_at, pan_deg, tilt_deg):
    """Apply a small pan (about world z) then tilt (about camera right) to the look point."""
    import numpy as np
    pos = np.asarray(pos, dtype=float)
    f = np.asarray(look_at, dtype=float) - pos
    f = _rotate(f, (0.0, 0.0, 1.0), pan_deg)
    right = np.cross(f, (0.0, 0.0, 1.0))
    if np.linalg.norm(right) > 1e-6:
        f = _rotate(f, right, tilt_deg)
    return tuple(pos + f)


def _look_quat(pos, target):
    """OpenGL-convention look quaternion (x, y, z, w): camera looks along -Z, up +Y."""
    import numpy as np
    import warp as wp
    f = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    f /= np.linalg.norm(f)
    right = np.cross(f, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, f)
    m = np.column_stack([right, cam_up, -f]).astype("float32")
    q = wp.quat_from_matrix(wp.mat33f(m))
    return (float(q.x), float(q.y), float(q.z), float(q.w))


def _set_arm_poses(scene, pose_name: str) -> None:
    """Write start/reach joint targets to any articulations from the converted USD."""
    import numpy as np
    reach = {
        "start": [0.0, -0.6, 0.0, -1.4, 0.0, 1.0, 0.0],
        "reach": [0.0, -1.15, 0.35, -0.75, 1.10, 0.80, 0.20],
    }[pose_name]
    for articulation in getattr(scene, "articulations", []):
        n = min(len(reach), articulation.num_joints)
        target = np.asarray(reach[:n], dtype=float)
        setter = getattr(articulation, "set_joint_position_target", None) or getattr(
            articulation, "write_joint_position_target", None)
        if setter is not None:
            setter(target)


def _view_grid(stations, zooms, pans, tilts, poses):
    for st in stations:
        for z in zooms:
            for pi, pan in enumerate(pans):
                for ti, tilt in enumerate(tilts):
                    for pose in poses:
                        yield st, z, pi, pan, ti, tilt, pose


def render_bundle(seed: int, *, views_dir: Path, out_dir: Path, width: int, height: int,
                  stations_meta: dict, zooms, pans, tilts, poses, device: str) -> dict:
    """Render the full grid and write the bundle. Raises if any render/measurement fails."""
    import numpy as np
    import torch
    from PIL import Image
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg, SimulationCfg, build_simulation_context
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    views_dir = Path(views_dir)
    out_dir = Path(out_dir)
    views_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    mjcf = MJCF_DUO if MJCF_DUO.is_file() else MJCF_V2
    usd = _mjpeg_to_usd(mjcf, ROOT / ".tmp" / f"scene_{seed}.usd")

    sim_cfg = SimulationCfg(device=device, dt=1.0 / 120.0,
                            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.ground = AssetBaseCfg(prim_path="/World/Ground",
                                    spawn=sim_utils.GroundPlaneCfg(size=(12.0, 12.0)))
    scene_cfg.scene_usd = AssetBaseCfg(prim_path="/World/SceneUsd",
                                       spawn=sim_utils.UsdFileCfg(usd_path=str(usd)))
    first = stations_meta["stations"][0]
    scene_cfg.eye = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Eye",
        width=width, height=height, update_period=0,
        data_types=["rgb", "depth"],
        offset=CameraCfg.OffsetCfg(pos=tuple(first["cam_xyz"]), convention="opengl"),
        spawn=PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                               clipping_range=(0.05, 50.0)),
        renderer_cfg=NewtonWarpRendererCfg(),
    )

    visibility: dict[str, dict] = {}
    render_seconds: dict[int, float] = {}
    frame_count: dict[int, int] = {}
    jpgs: list[str] = []
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(scene_cfg)
        camera = scene["eye"]
        sim.reset()
        base_K = getattr(camera.data, "intrinsic_matrices", None)
        if base_K is not None:
            base_K = base_K.torch.clone()
        else:
            fx = width * 24.0 / 20.955
            base_K = torch.tensor([[[fx, 0.0, width / 2.0],
                                    [0.0, fx, height / 2.0],
                                    [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)

        for st, z, pi, pan, ti, tilt, pose in _view_grid(
                stations_meta["stations"], zooms, pans, tilts, poses):
            pos = st["cam_xyz"]
            target = _look_target(pos, stations_meta["look_at"], pan, tilt)
            quat = _look_quat(pos, target)
            camera.set_world_poses(
                torch.tensor([list(map(float, pos))], device=device, dtype=torch.float32),
                torch.tensor([list(quat)], device=device, dtype=torch.float32),
                convention="opengl",
            )
            K = base_K.clone()
            K[:, 0, 0] *= float(z)
            K[:, 1, 1] *= float(z)
            camera.set_intrinsic_matrices(K)
            _set_arm_poses(scene, pose)

            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            t0 = time.perf_counter()
            camera.update(sim.get_physics_dt(), force_recompute=True)
            k = int(st["k"])
            render_seconds[k] = render_seconds.get(k, 0.0) + (time.perf_counter() - t0)
            frame_count[k] = frame_count.get(k, 0) + 1

            out = camera.data.output
            rgb = _to_numpy(out["rgb"])[0]
            depth = _to_numpy(out["depth"])[0, :, :, 0]
            name = f"s{k:02d}_z{z}_p{pi}_t{ti}_{pose}"
            Image.fromarray(rgb.astype("uint8")).save(views_dir / f"{name}_rgb.jpg", quality=90)
            Image.fromarray((np.clip(depth / (float(np.nanmax(depth)) or 1.0), 0, 1) * 255
                             ).astype("uint8")).save(views_dir / f"{name}_depth.jpg", quality=90)
            jpgs += [f"{name}_rgb.jpg", f"{name}_depth.jpg"]
            hits = int((depth > 0).sum())
            visibility[name] = {
                "station": k, "az_deg": st["az_deg"], "zoom": z,
                "pan_deg": pan, "tilt_deg": tilt, "arm_pose": pose,
                "depth_hits": hits,
                "rgb_nonblank": bool(int(rgb.max()) > int(rgb.min())),
                "visible": bool(hits > 0 and int(rgb.max()) > int(rgb.min())),
            }

    # fps per eye comes only from measured render seconds -- never from a constant.
    fps_per_eye: dict[str, float | None] = {}
    for eye, k in (("eye_A", stations_meta["eye_pair"][0]),
                   ("eye_B", stations_meta["eye_pair"][1])):
        secs = render_seconds.get(k, 0.0)
        fps_per_eye[eye] = (frame_count.get(k, 0) / secs) if secs > 0 else None
    checked_fps = measured_fps(fps_per_eye)  # raises (FAIL) rather than inventing a number

    (out_dir / "visibility.json").write_text(json.dumps(visibility, indent=2, sort_keys=True) + "\n")
    objects = {"seed": seed, "layout": stations_meta["layout"],
               "eye_pair": stations_meta["eye_pair"],
               "scene_objects": (json.loads(SCENE_MANIFEST.read_text()).get("objects", [])
                                 if SCENE_MANIFEST.is_file() else [])}
    (out_dir / "objects.json").write_text(json.dumps(objects, indent=2, sort_keys=True) + "\n")

    bundle_files = [f"views/{name}" for name in sorted(jpgs)]
    bundle_files += ["visibility.json", "objects.json"]
    manifest = build_manifest(out_dir, bundle_files, versions(), fps_per_eye=checked_fps)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verdict = fps_verdict(manifest)
    return {
        "result": verdict,
        "seed": seed,
        "usd_scene_available": True,
        "usd": str(usd.relative_to(ROOT)),
        "views": len(visibility),
        "images_written": len(jpgs),
        "fps_per_eye": checked_fps,
        "fps_measured": True,
        "fps_verdict": verdict,
        "bundle_sha256": manifest["sha256"],
        "manifest": str((out_dir / "manifest.json").relative_to(ROOT)),
        "files": bundle_files,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True, help="scene seed (train >= 100000, test < 100000)")
    ap.add_argument("--out", default=None, help="run report path (default reports/scene_<seed>/render_views.json)")
    ap.add_argument("--views", default=None, help="views dir (default reports/scene_<seed>/views)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--zooms", type=int, nargs="+", default=list(DEFAULT_ZOOMS))
    ap.add_argument("--pan-deg", type=float, nargs="+", default=list(DEFAULT_PAN_DEG))
    ap.add_argument("--tilt-deg", type=float, nargs="+", default=list(DEFAULT_TILT_DEG))
    ap.add_argument("--poses", nargs="+", default=list(ARM_POSES), choices=list(ARM_POSES))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eye-placement", default=str(EYE_PLACEMENT))
    a = ap.parse_args(argv)

    scene_dir = ROOT / "reports" / f"scene_{a.seed}"
    out = Path(a.out) if a.out else scene_dir / "render_views.json"
    views_dir = Path(a.views) if a.views else scene_dir / "views"
    report: dict = {
        "seed": a.seed, "versions": versions(), "width": a.width, "height": a.height,
        "zooms": a.zooms, "pan_deg": a.pan_deg, "tilt_deg": a.tilt_deg, "poses": a.poses,
        "views_dir": str(views_dir), "fps_threshold": FPS_THRESHOLD,
    }

    ok, reason = preflight_usd_scene()
    if not ok:
        # Honest fallback: report FAIL, write NO images, claim NO fps.
        report.update({
            "result": "FAIL", "reason": reason, "usd_scene_available": False,
            "images_written": 0, "fps_per_eye": None, "fps_measured": False,
            "manifest": None,
            "detail": ("No USD scene can be built from the cached MJCF because the isaacsim "
                       "MJCF->USD importer is absent from the Newton-only venv. No frames were "
                       "rendered, no bundle was written, and no >= 10 fps result is claimed."),
        })
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: report[k] for k in ("seed", "result", "reason", "images_written",
                                                 "fps_per_eye", "fps_measured")}, indent=2))
        print(f"  -> {out}")
        return 1

    try:
        stations = load_stations(a.eye_placement)
        report.update(render_bundle(
            a.seed, views_dir=views_dir, out_dir=out.parent, width=a.width, height=a.height,
            stations_meta=stations, zooms=a.zooms, pans=a.pan_deg, tilts=a.tilt_deg,
            poses=a.poses, device=a.device,
        ))
    except Exception as exc:                                     # noqa: BLE001
        import traceback
        report.update({"result": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                       "fps_per_eye": None, "fps_measured": False,
                       "traceback": traceback.format_exc()[-4000:]})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report.get(k) for k in ("seed", "result", "usd_scene_available",
                                                 "views", "images_written", "fps_per_eye",
                                                 "fps_verdict", "bundle_sha256", "error")}, indent=2))
    print(f"  -> {out}")
    return 0 if report.get("result") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
