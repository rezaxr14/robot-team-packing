#!/usr/bin/env python3
"""Push a private Kaggle kernel that carries the API token in its own source.

Kaggle's account secrets are not reachable from this account's kernels here
(``UserSecretsClient.get_secret`` returns HTTP 400 for every label, in both script and notebook
kernels). The owner authorised pushing the token into the project's **private** kernels, so this
helper reads `KAGGLE_API_TOKEN` from the local `.env`, writes a private kernel under a temporary
directory (never inside the git repo), pushes it, and deletes the directory afterwards.

The token is never printed, never staged and never committed. The kernel itself prints only the
dataset it created and the privacy flag.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rtp.remotes import read_env                                      # noqa: E402

KERNEL_ID = "rezanadimikj/rtp-publish-private-token"
PUBLISHER = ROOT / "ops" / "kaggle" / "publish_model" / "publish_model.py"


def main() -> int:
    token = read_env(ROOT / ".env").get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("KAGGLE_API_TOKEN missing from .env")
    src = PUBLISHER.read_text()
    # Replace the credential lookup with an environment assignment, but keep all download,
    # metadata and publish logic byte-for-byte identical to the reviewed publisher.
    start = src.index("def creds()")
    stop = src.index("\n\ndef download()", start)
    stub = f'''def creds() -> None:
    # Injected by ops/kaggle/push_token_kernel.py from the local .env. This kernel is private.
    os.environ["KAGGLE_API_TOKEN"] = {token!r}
    say("kaggle credentials: injected into this private kernel")
'''
    code = src[:start] + stub + src[stop:]

    with tempfile.TemporaryDirectory(prefix="rtp_kaggle_token_") as tmp:
        d = Path(tmp)
        (d / "publish.py").write_text(code, encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": KERNEL_ID, "title": "rtp publish private token", "code_file": "publish.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_internet": True, "enable_gpu": False, "enable_tpu": False,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}, indent=2))
        r = subprocess.run([sys.executable, str(ROOT / "ops" / "kaggle_kernel.py"),
                            "push", str(d)], text=True, capture_output=True)
        print((r.stdout or r.stderr).strip()[-500:])
        return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
