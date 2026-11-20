#!/usr/bin/env python3
"""Run the Kaggle CLI from the project-local .tools install, with the Kaggle-only proxy.

Everything this script writes is inside the repo. No credentials are printed.
The proxy is used ONLY for Kaggle commands (never HuggingFace or pip).

    python ops/kaggle_local.py kernels status rezanadimikj/rtp-g3-t4
    python ops/kaggle_local.py --benchmarks quota --format json
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtp.remotes import load_env  # noqa: E402

TOOLS = ROOT / ".tools"
PROXY = os.environ.get("RTP_KAGGLE_PROXY", "http://127.0.0.1:10808")
# Account recorded in the project resource notes; the token itself stays in .env.
DEFAULT_KAGGLE_USERNAME = "rezanadimikj"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    use_benchmarks = argv[0] == "--benchmarks"
    if use_benchmarks:
        argv = argv[1:]
        module = "kaggle_benchmarks"
    else:
        module = "kaggle"
    env = load_env(ROOT / ".env") or {}
    # load_env may return None and populate os.environ; tolerate both.
    def credential(name: str, default: str | None = None) -> str | None:
        value = env.get(name) or os.environ.get(name) or default
        if value:
            os.environ[name] = value
        return value

    user = credential("KAGGLE_USERNAME", DEFAULT_KAGGLE_USERNAME)
    key = credential("KAGGLE_KEY") or credential("KAGGLE_API_TOKEN")
    if not user or not key:
        print("KAGGLE_USERNAME and KAGGLE_KEY/KAGGLE_API_TOKEN must be set in .env", file=sys.stderr)
        return 2

    child_env = os.environ.copy()
    child_env.update({
        "PYTHONPATH": str(TOOLS) + os.pathsep + child_env.get("PYTHONPATH", ""),
        "HTTP_PROXY": PROXY, "HTTPS_PROXY": PROXY,
        "http_proxy": PROXY, "https_proxy": PROXY,
        "NO_PROXY": "localhost,127.0.0.1",
        "TMPDIR": str(ROOT / ".tmp"),
    })
    (ROOT / ".tmp").mkdir(exist_ok=True)
    return subprocess.run([sys.executable, "-m", module, *argv], env=child_env, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
