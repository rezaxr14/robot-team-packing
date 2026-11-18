#!/usr/bin/env python3
"""Gate G-3: a Newton-only Isaac Lab scene steps and renders rgb+depth on this GPU.

The gate in `IMPLEMENTATION_PLAN.md` §12 is "Newton-only Isaac Lab install runs a camera task with
`physics=newton_mjwarp` + Newton Warp renderer; task steps with rgb+depth on each profile GPU;
versions logged". This script does exactly that with the API the checkout itself uses in
`source/isaaclab/test/renderers/test_camera_output_contract.py`:

    SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))   # no PhysX, no Isaac Sim
    CameraCfg(..., data_types=["rgb", "depth"], renderer_cfg=NewtonWarpRendererCfg())

It asserts more than "nothing crashed": the cube actually falls under Newton's MuJoCo-Warp solver
(so physics stepped), the camera returns `(envs, H, W, 3)` uint8 rgb and `(envs, H, W, 1)` float32
depth (so the Warp ray tracer ran), some pixels hit the cube, and the two frames differ after the
cube moves. Versions and timings go into the JSON report; the images are written next to it.

    python ops/g3_sim.py --out reports/g3_sim.json --images reports/g3_sim
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

DISTRIBUTIONS = ("isaaclab", "isaaclab-newton", "newton", "warp-lang", "mujoco-warp",
                 "torch", "numpy")


def versions() -> dict:
    """Every version that decides whether this run means anything, without importing more than needed."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in DISTRIBUTIONS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "MISSING"
    return out


def build_scene(width: int, height: int, envs: int, device: str):
    """The scene cfg: a cube above the origin, a ground plane, a camera that sees both."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    sim_cfg = SimulationCfg(device=device, dt=1.0 / 120.0,
                            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
    scene_cfg = InteractiveSceneCfg(num_envs=envs, env_spacing=2.0)
    scene_cfg.cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.5),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    scene_cfg.ground = AssetBaseCfg(prim_path="/World/Ground",
                                    spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)))
    scene_cfg.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        width=width, height=height,
        data_types=["rgb", "depth"],
        offset=CameraCfg.OffsetCfg(pos=(0.0, -2.0, 1.2), rot=(-0.3536, 0.3536, -0.1464, 0.8536),
                                   convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.05, 50.0)),
        renderer_cfg=NewtonWarpRendererCfg(),
    )
    return sim_cfg, scene_cfg


def _z(asset) -> float:
    """Root height in metres; asset state is a ProxyArray, so read it through .torch."""
    return float(asset.data.root_pos_w.torch[0, 2].item())


def to_numpy(buf):
    """Camera outputs are ProxyArrays: (N,H,W,C) via .warp.numpy() (or .torch)."""
    if hasattr(buf, "warp"):
        return buf.warp.numpy()
    if hasattr(buf, "torch"):
        return buf.torch.detach().cpu().numpy()
    return buf


def save_pngs(images: Path, rgb, depth, step: int) -> list[str]:
    """Write the frames next to the report when an image library is present (never a hard dep)."""
    images.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        import numpy as np
        from PIL import Image
    except Exception:                                            # noqa: BLE001
        return written
    r = rgb[0]
    Image.fromarray(r.astype("uint8")).save(images / f"rgb_{step:03d}.png")
    written.append(f"rgb_{step:03d}.png")
    d = depth[0, :, :, 0].astype("float32")
    span = float(np.nanmax(d)) or 1.0
    Image.fromarray((np.clip(d / span, 0.0, 1.0) * 255).astype("uint8")).save(
        images / f"depth_{step:03d}.png")
    written.append(f"depth_{step:03d}.png")
    return written


def run(steps: int, width: int, height: int, envs: int, device: str, images: Path) -> dict:
    import numpy as np

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import build_simulation_context

    sim_cfg, scene_cfg = build_scene(width, height, envs, device)
    report = {"renderer": "newton_warp", "physics": "newton_mjwarp",
              "device": device, "steps": steps, "width": width, "height": height, "envs": envs}

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(scene_cfg)
        camera = scene["camera"]
        cube = scene["cube"]
        sim.reset()
        z0 = _z(cube)

        frames, t0 = [], time.time()
        for i in range(steps):
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            camera.update(sim.get_physics_dt(), force_recompute=True)
            if i in (0, steps - 1):
                out = camera.data.output
                rgb, depth = to_numpy(out["rgb"]), to_numpy(out["depth"])
                frames.append({"step": i, "rgb": rgb, "depth": depth})
        dt = time.time() - t0
        z1 = _z(cube)

    first, last = frames[0], frames[-1]
    rgb, depth = last["rgb"], last["depth"]
    hits = int((depth[..., 0] > 0).sum())
    report.update({
        "seconds": round(dt, 3),
        "steps_per_s": round(steps / dt, 1),
        "cube_z_start": round(z0, 4),
        "cube_z_end": round(z1, 4),
        "rgb": {"shape": list(rgb.shape), "dtype": str(rgb.dtype),
                "min": int(rgb.min()), "max": int(rgb.max())},
        "depth": {"shape": list(depth.shape), "dtype": str(depth.dtype),
                  "hits": hits, "max": round(float(depth.max()), 4)},
        "frames_differ": bool(not np.array_equal(first["rgb"], last["rgb"])),
        "images": save_pngs(images, rgb, depth, steps - 1),
    })

    checks = {
        "physics_stepped": z1 < z0 - 1e-3,
        "rgb_contract": list(rgb.shape) == [envs, height, width, 3] and rgb.dtype == np.uint8,
        "depth_contract": list(depth.shape) == [envs, height, width, 1]
                          and depth.dtype == np.float32,
        "rgb_not_blank": int(rgb.max()) > int(rgb.min()),
        "geometry_rendered": hits > 0,
        "frames_differ": report["frames_differ"],
    }
    report["checks"] = checks
    report["result"] = "PASS" if all(checks.values()) else "FAIL"
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/g3_sim.json")
    ap.add_argument("--images", default="", help="directory for the rgb/depth PNGs")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--envs", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)

    report = {"versions": versions()}
    try:
        images = Path(a.images) if a.images else Path(a.out).with_suffix("")
        report.update(run(a.steps, a.width, a.height, a.envs, a.device, images))
    except Exception as exc:                                     # noqa: BLE001
        # A gate that cannot run must say why, with the traceback, not exit silently.
        import traceback
        report.update({"result": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()[-4000:]})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("result", "physics", "renderer", "rgb", "depth",
                                                 "checks", "error")}, indent=1)[:2000])
    if report.get("result") == "PASS":
        print(f"  {report['steps']} steps in {report['seconds']} s "
              f"({report['steps_per_s']} steps/s), cube z {report['cube_z_start']} -> "
              f"{report['cube_z_end']}, {report['depth']['hits']} depth hits")
    print(f"  -> {out}")
    return 0 if report.get("result") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
