#!/usr/bin/env python3
"""Run SFT on episodes generated from the asset-backed pose-level environment.

This is the pipeline the user asked for: real SimReady packages -> asset-backed worlds ->
teacher-validated plans -> QLoRA SFT on the 4B Brain, with resumable checkpoints in our private
HF repo. The environment evidence class is **SIM-POSE** (real assets, pose-level dynamics), NOT
Newton physics; the report says so explicitly.

    python ops/train_asset_env.py --profile local_4050 --n 64 --steps 4 --stop-after 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local_4050")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed0", type=int, default=200000)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--stop-after", type=int, default=2)
    ap.add_argument("--stage", default="asset_sft")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-store", action="store_true")
    a = ap.parse_args(argv)

    from rtp import assetcache as ac
    from rtp.asset_datagen import gen_asset_sft, gen_asset_hindsight
    from rtp.brain_train import run_sft
    from rtp.datagen import write_jsonl
    from rtp.harness import Harness
    from rtp.profiles import get_profile
    from rtp.remotes import from_env
    from rtp.scene_assets import resolve_assets, to_world_objects, write_manifest

    t0 = time.time()
    try:
        records = resolve_assets()
    except (FileNotFoundError, ValueError) as exc:
        print(f"assets not ready: {exc}")
        return 2
    profile = get_profile(a.profile, root=ROOT)
    profile.ensure_dirs()
    manifest = write_manifest(records, to_world_objects(records),
                              ROOT / "reports" / "simready_scene.json")
    print(f"assets: {len(records)} packages -> {manifest}", flush=True)

    harness = Harness(ROOT / "harness_files")
    sft_rows, funnel = gen_asset_sft(records, a.n, a.seed0, harness, manifest=str(manifest))
    hind_rows, hind_stats = gen_asset_hindsight(records, max(2, a.n // 8), a.seed0 + 50000,
                                                 harness, manifest=str(manifest))
    train_rows = sft_rows + hind_rows
    if not train_rows:
        print("no teacher-validated rows from the asset environment")
        return 1
    data = profile.paths.data / "asset_sft.jsonl"
    write_jsonl(train_rows, data)
    print(f"rows: {len(sft_rows)} SFT + {len(hind_rows)} hindsight -> {data}", flush=True)

    spec = ac.model_spec("Qwen/Qwen3.5-4B", revision="851bf6e8", name="qwen35-4b")
    model_dir = ac.asset_dir(spec, profile)
    if ac.verify_manifest(model_dir, spec):
        print(f"model not verified at {model_dir}")
        return 2
    store = None
    if not a.no_store:
        try:
            store = from_env(with_kaggle=False)
        except Exception as exc:                          # noqa: BLE001
            print(f"store unavailable ({type(exc).__name__}); local-only run")
    result = run_sft(profile, model_dir=model_dir, data=data, store=store, stage=a.stage,
                     epochs=10.0, lr=a.lr, every=1, keep=2, seed=a.seed0,
                     stop_after=a.stop_after)
    report = {"evidence_class": "SIM-POSE (real SimReady assets, pose-level dynamics; NOT Newton physics)",
              "profile": profile.name, "stage": a.stage, "assets": len(records),
              "manifest": str(manifest), "sft_funnel": funnel, "hindsight_funnel": hind_stats,
              "train_rows": len(train_rows), "model": str(model_dir),
              "step": result.state.step, "epoch": result.state.epoch,
              "resumed_from": result.resumed_from,
              "losses": [h["loss"] for h in result.state.history],
              "store": bool(store), "wall_s": round(time.time() - t0, 1)}
    out = ROOT / "reports" / "asset_train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
