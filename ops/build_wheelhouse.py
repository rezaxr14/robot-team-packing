#!/usr/bin/env python3
"""Build the offline wheelhouse that `ops/install.sh --wheelhouse DIR` installs from.

P1 promises a Kaggle run with caches attached performs **zero downloads**. The other half of that is
a directory of wheels that were resolved once, on a box with network, for the target's tags:

    ops/build_wheelhouse.py --profile kaggle_t4x2 --out ~/wheelhouses         # tags from this box
    ops/build_wheelhouse.py --profile kaggle_t4x2 --out ~/wheelhouses \\
        --python 3.12 --platform manylinux2014_x86_64 --cuda 12.8 --sm 75     # cross-build for a T4
    ops/build_wheelhouse.py --check ~/wheelhouses/rtp-wheels-py3.12-torch2.15.0-cu12.8-sm75

The package list is not repeated here: it is read from `ops/install.sh --print-packages`, so the
installer and the wheelhouse cannot drift. `--only-binary=:all:` is deliberate — an sdist in an
offline wheelhouse is a build failure waiting for a missing compiler, so it is rejected, not shipped.

The directory is named by the tags (`rtp-wheels-py<python>-torch<v>-cu<cuda>-sm<cc>`, plan §11) and
carries a manifest, so a wheelhouse built for py3.11/cu121 cannot be silently installed into a
py3.12/cu128 env: the slug does not match and `verify_dir` says which tags are missing.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # `rtp` imports when run as a script

from rtp import assetcache as ac                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "ops" / "install.sh"

#: wheels that PyPI does not publish and that need a real compile for the target's arch.
SOURCE_BUILT = {"flash-attn", "causal-conv1d", "flash-linear-attention"}

NORMALIZE = re.compile(r"[-_.]+")

#: Run inside the profile's interpreter: the tags that make a wheelhouse valid are that venv's,
#: not the tags of whatever interpreter happens to run this helper.
TAG_PROBE = r"""
import json, sys
out = {"python": "%d.%d" % sys.version_info[:2]}
try:
    import torch
    out["torch"] = torch.__version__.split("+")[0]
    if torch.cuda.is_available():
        out["cuda"] = str(torch.version.cuda or "?")
        cc = torch.cuda.get_device_capability(0)
        out["sm"] = "%d%d" % cc
except Exception:
    pass
print(json.dumps(out))
"""


def probe_tags(exe: Path | str) -> dict:
    """Version tags of ``exe``; an interpreter without torch reports its python version only."""
    out = subprocess.run([str(exe), "-c", TAG_PROBE], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"cannot read tags from {exe}: {out.stderr.strip()[-300:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def normalize(name: str) -> str:
    """PEP 503 name: ``Flash_Attn`` and ``flash-attn`` are the same distribution."""
    return NORMALIZE.sub("-", name.strip().lower())


def req_name(spec: str) -> str:
    """The distribution name of a requirement: ``huggingface_hub[cli]>=0.30`` -> ``huggingface-hub``."""
    name = spec.split(";")[0].split("[")[0]
    name = re.split(r"[<>=!~]", name, maxsplit=1)[0]
    return normalize(name)


def wheel_dist(filename: str) -> str | None:
    """``torch-2.15.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl`` -> ``torch``; None if not a wheel."""
    if not filename.endswith(".whl"):
        return None
    return normalize(filename.split("-")[0])


def scan(path: Path) -> tuple[set[str], list[str]]:
    """Distributions covered by the wheels in ``path``, and any non-wheel files that are not manifests."""
    have, strays = set(), []
    for p in sorted(Path(path).iterdir()):
        if p.name == ac.MANIFEST:
            continue
        dist = wheel_dist(p.name)
        if dist is None:
            strays.append(p.name)
        else:
            have.add(dist)
    return have, strays


def verify_dir(path: Path, required: list[str]) -> list[str]:
    """Problems that would make an offline install from ``path`` fail or silently fall back."""
    path = Path(path)
    if not path.is_dir():
        return [f"{path} is not a directory"]
    have, strays = scan(path)
    problems = [f"sdist/non-wheel {name}: an offline wheelhouse must contain wheels only"
                for name in strays]
    for spec in required:
        name = req_name(spec)
        if name in SOURCE_BUILT:
            problems.append(f"{name} has no PyPI wheel for the target arch: build it on that box "
                            f"(uv pip wheel {name} --no-build-isolation) and copy the .whl in")
        elif name not in have:
            problems.append(f"no wheel for {name}")
    return problems


def parse_plan(text: str) -> dict:
    """Parse `ops/install.sh --print-packages` output into a dict of shell-ish assignments."""
    plan = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        plan[key.strip()] = value.strip()
    for key in ("profile", "python", "core", "pkgs"):
        if key not in plan:
            raise ValueError(f"install.sh --print-packages did not print {key}")
    plan["core"] = plan["core"].split()
    plan["pkgs"] = plan["pkgs"].split()
    plan["extra"] = plan.get("extra", "").split()
    plan["flash"] = plan.get("flash", "0") == "1"
    return plan


def install_plan(profile: str) -> dict:
    """Ask the installer what this profile needs (one source of truth, no copy in this script)."""
    out = subprocess.run(["bash", str(INSTALLER), "--profile", profile, "--print-packages"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"install.sh --print-packages failed: {out.stderr.strip()[-400:]}")
    plan = parse_plan(out.stdout)
    if plan["profile"] != profile:
        raise RuntimeError(f"install.sh printed a plan for {plan['profile']!r}, asked for {profile!r}")
    return plan


def _pip_python(out: Path, python: str) -> Path:
    """A throwaway seed venv: the training venv has no pip, and pip is what can download wheels."""
    venv = out / ".pipenv"
    py = venv / "bin" / "python"
    if py.exists():
        return py
    venv.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("uv"):
        cmd = ["uv", "venv", "--seed", "--python", python, str(venv)]
    else:
        cmd = [sys.executable, "-m", "venv", str(venv)]
    subprocess.run(cmd, check=True)
    return py


def build_command(py: Path, dest: Path, requirements: list[str], *, index_url: str | None,
                  target: dict | None) -> list[str]:
    """The `pip download` argv for one group of requirements."""
    cmd = [str(py), "-m", "pip", "download", "--only-binary=:all:", "--dest", str(dest),
           "--disable-pip-version-check"]
    if index_url:
        cmd += ["--index-url", index_url]
    if target:
        cmd += ["--python-version", target["python"], "--implementation", "cp",
                "--abi", f"cp{target['python'].replace('.', '')}",
                "--platform", target["platform"]]
    return cmd + requirements


def build(plan: dict, dest: Path, *, python: str, torch: str, cuda: str, sm: str,
          platform: str = "", torch_index: str | None = None, dry_run: bool = False,
          skip_core: bool = False, only: list[str] | None = None) -> dict:
    """Download every wheel into ``dest`` and write its manifest; returns a report."""
    spec = ac.wheelhouse_spec(python=python, torch=torch, cuda=cuda, sm=sm)
    target = {"python": python, "platform": platform} if platform else None
    if only:
        groups = [("only", only, None)]
    else:
        groups = [] if skip_core else [("core", plan["core"], torch_index or plan.get("torch_index"))]
        groups.append(("pypi", plan["pkgs"] + plan["extra"], None))

    dest.mkdir(parents=True, exist_ok=True)
    py = _pip_python(dest.parent, python)
    commands = [build_command(py, dest, reqs, index_url=index, target=target)
                for _, reqs, index in groups]
    if dry_run:
        return {"slug": spec.slug, "dest": str(dest), "commands": commands, "downloaded": False}

    for cmd in commands:
        subprocess.run(cmd, check=True)

    required = [r for _, reqs, _ in groups for r in reqs]
    problems = verify_dir(dest, required)
    if problems:
        raise SystemExit("wheelhouse is incomplete:\n  " + "\n  ".join(problems))
    have, _ = scan(dest)
    if not only:
        # a partial build is a plumbing test; it must not look attachable to anyone else
        ac.write_manifest(dest, spec, deep=True)
    return {"slug": spec.slug, "dest": str(dest), "downloaded": True, "partial": bool(only),
            "files": len(have), "bytes": sum(p.stat().st_size for p in dest.glob("*.whl"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", help="which profile's package set to resolve")
    ap.add_argument("--out", help="parent directory for the wheelhouse (a tag-named subdir is created)")
    ap.add_argument("--check", metavar="DIR", help="verify an existing wheelhouse and ignore --out")
    ap.add_argument("--python", default="", help="target python (default: the profile's)")
    ap.add_argument("--python-exe", default="", help="interpreter whose tags to read (default: the profile venv)")
    ap.add_argument("--platform", default="", help="target platform tag, e.g. manylinux2014_x86_64")
    ap.add_argument("--torch", default="", help="torch version for the slug (default: the profile venv)")
    ap.add_argument("--cuda", default="", help="cuda version for the slug (default: the profile venv)")
    ap.add_argument("--sm", default="", help="compute capability for the slug, e.g. 75 or 89")
    ap.add_argument("--torch-index", default="", help="override the torch wheel index")
    ap.add_argument("--skip-core", action="store_true", help="skip torch/torchvision (debug runs)")
    ap.add_argument("--only", default="", help="comma list: build just these to test the plumbing "
                                               "(no manifest is written for a partial wheelhouse)")
    ap.add_argument("--dry-run", action="store_true", help="print the pip commands, download nothing")
    a = ap.parse_args(argv)

    if not a.profile:
        ap.error("--profile is required")
    plan = install_plan(a.profile)

    # The tags must describe the interpreter that will *run* the wheels. Reading them from this
    # helper's own interpreter (a bare `./ops/build_wheelhouse.py`) is how a py3.14/cpu slug happens.
    exe = Path(a.python_exe).expanduser() if a.python_exe else Path(plan.get("venv", "")) / "bin" / "python"
    if not exe.exists():
        exe = Path(sys.executable)
    tags = probe_tags(exe)
    if "torch" not in tags:
        print(f"WARNING: {exe} has no torch; building a wheelhouse without torch/sm tags is a guess",
              file=sys.stderr)
    python = a.python or plan.get("python") or tags["python"]
    torch = a.torch or tags.get("torch", "unknown")
    cuda = a.cuda or tags.get("cuda", "unknown")
    sm = a.sm or tags.get("sm", "unknown")
    spec = ac.wheelhouse_spec(python=python, torch=torch, cuda=cuda, sm=sm)

    if a.check:
        path = Path(a.check).expanduser()
        required = plan["core"] + plan["pkgs"] + plan["extra"]
        problems = verify_dir(path, required)
        problems += ac.verify_manifest(path, spec) if path.name == spec.slug else [
            f"directory name {path.name!r} does not match this profile's tags ({spec.slug})"]
        print(json.dumps({"dir": str(path), "expected_slug": spec.slug, "problems": problems}, indent=2))
        return 1 if problems else 0

    if not a.out:
        ap.error("--out is required unless --check is used")
    dest = Path(a.out).expanduser() / spec.slug
    print(f"wheelhouse: {dest}\n  tags: python={python} torch={torch} cuda={cuda} sm={sm}"
          f" platform={a.platform or 'native'} flash-attn={'wanted' if plan['flash'] else 'not wanted'}")
    report = build(plan, dest, python=python, torch=torch, cuda=cuda, sm=sm,
                   platform=a.platform, torch_index=a.torch_index or None,
                   dry_run=a.dry_run, skip_core=a.skip_core,
                   only=[s.strip() for s in a.only.split(",") if s.strip()] or None)
    if report["downloaded"]:
        print(f"  {report['files']} wheels, {report['bytes'] / 1024 ** 3:.2f} GB")
        if report.get("partial"):
            print("  PARTIAL (--only): no manifest written; not attachable as a cache dataset")
        else:
            print(f"  install offline: ops/install.sh --profile {a.profile} --wheelhouse {dest}")
            print(f"  upload as a private Kaggle dataset, then attach {dest.name}")
    else:
        for cmd in report["commands"]:
            print("  " + " ".join(cmd))
    if plan["flash"]:
        print("  NOTE: flash-attn has no portable wheel; build it on the target box and copy the .whl "
              "into the directory (install.sh falls back to SDPA without it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
