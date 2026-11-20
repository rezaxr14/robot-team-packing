"""Installer checks: per-profile plan, offline wheelhouse mode, and a real offline install.

The wheelhouse test builds a wheel by hand (no network, no build backend) and installs it into a
throw-away venv with ``--no-index``; if that works, the installer's offline branch has a working
target to talk to.
"""
import base64
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "ops" / "install.sh"


def run(*args, env=None, timeout=300):
    return subprocess.run(["bash", str(INSTALL), *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, env=env)


def make_wheel(dest: Path, name: str = "rtp_smoke", version: str = "0.0.1", value: int = 42) -> Path:
    """A valid pure-python wheel: package, METADATA, WHEEL and a RECORD with real hashes."""
    dest.mkdir(parents=True, exist_ok=True)
    dist = f"{name}-{version}"
    files = {
        f"{name}/__init__.py": f"VALUE = {value}\n".encode(),
        f"{dist}.dist-info/METADATA": (f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n").encode(),
        f"{dist}.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: rtp-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = []
    for path, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={digest},{len(data)}")
    record.append(f"{dist}.dist-info/RECORD,,")
    files[f"{dist}.dist-info/RECORD"] = ("\n".join(record) + "\n").encode()
    wheel = dest / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        for path, data in files.items():
            z.writestr(path, data)
    return wheel


def test_dry_run_prints_the_offline_flags(tmp_path):
    r = run("--profile", "local_4090", "--wheelhouse", str(tmp_path), "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "--no-index" in r.stdout and f"--find-links {tmp_path}" in r.stdout
    torch_line = next(line for line in r.stdout.splitlines() if " torch torchvision" in line)
    assert "--no-index" in torch_line and "--index-url" not in torch_line      # never the index offline
    assert str(Path.home() / "venvs" / "local_4090") in r.stdout  # per-profile venv in the single venv folder


def test_a_wheelhouse_that_is_not_a_directory_is_refused(tmp_path):
    r = run("--profile", "local_4090", "--wheelhouse", str(tmp_path / "absent"), "--dry-run")
    assert r.returncode == 2 and "not a directory" in r.stderr


def test_the_isaac_path_uses_the_tag_lock_not_a_hand_picked_newton(tmp_path, monkeypatch):
    """v3.0.0 pins newton[sim]==1.6.1 / warp-lang==1.17.0 and deprecates isaaclab.sh."""
    monkeypatch.setenv("RTP_HOME", str(tmp_path))
    monkeypatch.setenv("RTP_VENVS", str(tmp_path / "venvs"))
    r = run("--isaac", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "uv sync --project" in r.stdout and "--no-dev" in r.stdout
    assert "checkout v3.0.0" in r.stdout
    assert "pip install --python" not in r.stdout, "a hand-picked newton diverges from the tag's lock"
    assert "newton==1.8" not in r.stdout and "warp-lang==1.19" not in r.stdout
    assert str(tmp_path / "venvs" / "rtp") in r.stdout


def test_profiles_get_separate_plans_and_venvs(tmp_path):
    a = run("--profile", "local_4050", "--dry-run")
    b = run("--profile", "kaggle_t4x2", "--dry-run")
    assert a.returncode == b.returncode == 0
    assert "local_4050" in a.stdout and "kaggle_t4x2" in b.stdout
    assert "bitsandbytes" in a.stdout and "bitsandbytes" not in b.stdout        # QLoRA only on the 4050
    assert "flash-attn: skipped" in b.stdout                                   # no FA2 on Turing
    assert a.stdout.count("recipe hash") == b.stdout.count("recipe hash") == 1   # one recipe per plan


def test_offline_install_from_a_synthetic_wheelhouse(tmp_path):
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv not on PATH")
    wheelhouse = tmp_path / "wheelhouse"
    make_wheel(wheelhouse, "rtp_smoke", "0.0.1", 42)
    venv = tmp_path / "venv"
    subprocess.run([uv, "venv", str(venv), "--python", sys.executable], check=True,
                   capture_output=True, text=True)
    py = venv / "bin" / "python"
    r = subprocess.run([uv, "pip", "install", "--no-index", "--find-links", str(wheelhouse),
                        "--python", str(py), "rtp_smoke==0.0.1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    out = subprocess.run([str(py), "-c", "import rtp_smoke; print(rtp_smoke.VALUE)"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "42"
