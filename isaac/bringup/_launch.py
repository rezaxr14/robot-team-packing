"""Shared launcher for the bring-up scripts. Starts the Kit app BEFORE any other isaaclab import."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def launch(description: str, extra=None):
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=description)
    AppLauncher.add_app_launcher_args(p)               # adds --headless, --enable_cameras, --device, ...
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/isaac")
    for args, kw in (extra or []):
        p.add_argument(*args, **kw)
    a = p.parse_args()
    app = AppLauncher(a).app
    return a, app


def check(name: str, ok: bool, detail: str = ""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok
