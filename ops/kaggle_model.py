#!/usr/bin/env python3
"""Drive the Kaggle-side model publish (the laptop never uploads the 9.4 GB snapshot).

    ops/kaggle_model.py push       # upload the kernel script (~4 KB) and start the Kaggle run
    ops/kaggle_model.py status     # last run state + log tail
    ops/kaggle_model.py output     # download the kernel log
    ops/kaggle_model.py verify     # is the dataset private, and how big is it?

Every call configures the local proxy (``RTP_KAGGLE_PROXY``, default ``http://localhost:10808``);
Kaggle refuses the connection without it on this network. Credential values are never printed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtp.remotes import configure_kaggle_proxy, load_env            # noqa: E402

KERNEL_DIR = ROOT / "ops" / "kaggle" / "publish_model"
KERNEL_REF = "rezanadimikj/rtp-publish-qwen35-4b"
DATASET_SLUG = "rtp-model-qwen35-4b-851bf6e8"


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    load_env(ROOT / ".env")
    configure_kaggle_proxy()
    exe = Path(sys.executable).with_name("kaggle")
    cmd = [str(exe) if exe.exists() else "kaggle", *args]
    last = None
    for attempt in range(4):
        r = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if r.returncode == 0:
            return r
        last = f"{(r.stderr or r.stdout).strip()[-800:]}"
        if "SSLError" not in last and "Max retries" not in last:
            break                                   # a real error, not the proxy's TLS flakiness
        time.sleep(2 + 2 * attempt)
    if check:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{last}")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "status", "output", "verify"])
    a = ap.parse_args(argv)

    if a.cmd == "push":
        r = kaggle("kernels", "push", "-p", str(KERNEL_DIR))
        print(r.stdout.strip() or r.stderr.strip())
        return 0

    if a.cmd == "status":
        r = kaggle("kernels", "status", KERNEL_REF, check=False)
        print((r.stdout or r.stderr).strip())
        return 0 if r.returncode == 0 else 1

    if a.cmd == "output":
        with tempfile.TemporaryDirectory() as tmp:
            r = kaggle("kernels", "output", KERNEL_REF, "-p", tmp, check=False)
            if r.returncode:
                print((r.stdout or r.stderr).strip())
                return 1
            for p in sorted(Path(tmp).rglob("*")):
                if p.is_file():
                    print(f"--- {p.relative_to(tmp)}")
                    print(p.read_text(errors="replace")[-6000:])
        return 0

    # verify
    meta_dir = ROOT / ".tmp" / "kaggle_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    r = kaggle("datasets", "metadata", DATASET_SLUG, "-p", str(meta_dir), check=False)
    meta_path = meta_dir / "dataset-metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        # `kaggle datasets metadata` wraps the fields under "info"; a local dataset-metadata.json
        # written by publish_model.py has them at the top level.
        info = meta.get("info", meta)
        private = bool(info.get("isPrivate"))
        print(json.dumps({"id": info.get("datasetSlug") or info.get("id"),
                          "owner": info.get("ownerUser"), "isPrivate": private,
                          "title": info.get("title")}, indent=2))
        return 0 if private else 1
    print((r.stdout or r.stderr).strip())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
