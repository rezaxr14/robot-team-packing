#!/usr/bin/env python3
"""Push/monitor a private Kaggle kernel: one command per state, proxy + retries built in.

    ops/kaggle_kernel.py push ops/kaggle/g3_t4
    ops/kaggle_kernel.py status rezanadimikj/rtp-g3-t4
    ops/kaggle_kernel.py output rezanadimikj/rtp-g3-t4

Used for GPU work that must not run on the laptop (G-3 on a T4, model publishing from Kaggle).
No credential value is printed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtp.remotes import configure_kaggle_proxy, load_env            # noqa: E402


def kaggle(*args: str) -> subprocess.CompletedProcess:
    load_env(ROOT / ".env")
    configure_kaggle_proxy()
    exe = Path(sys.executable).with_name("kaggle")
    cmd = [str(exe) if exe.exists() else "kaggle", *args]
    last = ""
    for attempt in range(4):
        r = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if r.returncode == 0:
            return r
        last = (r.stderr or r.stdout).strip()[-800:]
        if "SSLError" not in last and "Max retries" not in last:
            break
        time.sleep(2 + 2 * attempt)
    raise SystemExit(f"kaggle {' '.join(args)} failed:\n{last}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "status", "output"])
    ap.add_argument("target", help="kernel directory for push, owner/slug for status/output")
    ap.add_argument("--out", default=str(ROOT / ".tmp" / "kaggle_output"), help="download dir for output")
    a = ap.parse_args(argv)
    if a.cmd == "push":
        print(kaggle("kernels", "push", "-p", str(Path(a.target).resolve())).stdout.strip())
        return 0
    if a.cmd == "status":
        # status can report a non-zero rc while still giving a useful state, so retry the SSL
        # flakiness here instead of going through kaggle()'s check=True path.
        load_env(ROOT / ".env")
        configure_kaggle_proxy()
        exe = Path(sys.executable).with_name("kaggle")
        last = ""
        for attempt in range(4):
            r = subprocess.run([str(exe), "kernels", "status", a.target], text=True, capture_output=True)
            if r.returncode == 0 or "SSLError" not in (r.stderr or ""):
                print((r.stdout or r.stderr).strip())
                return 0 if r.returncode == 0 else 1
            last = (r.stderr or r.stdout).strip()
            time.sleep(2 + 2 * attempt)
        print(last)
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = kaggle("kernels", "output", a.target, "-p", str(out))
    print(r.stdout.strip())
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"--- {p.relative_to(out)}")
            print(p.read_text(errors="replace")[-8000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
