"""The stage plan (D-35): the runnable steps of IMPLEMENTATION_PLAN.md, as data.

  python ops/plan.py list                    # all groups and stages
  python ops/plan.py run <group> [--only NAME] [--force]
  python ops/plan.py lint                    # static checks (paths, undefined $VARS, producer order)

Stages are skipped if already done (rc 0) unless --force; a failing stage stops the group.
Only stages that exist today are listed: phases are added here as they land (P1 issue), never
as placeholders for files that do not exist.
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import OrderedDict

PY = os.environ.get("RTP_PY") or sys.executable   # the interpreter of the ACTIVE profile venv
MODEL = "$MODELS/Qwen3.5-4B"                      # $MODELS comes from ops/env.sh

GROUPS: "OrderedDict[str, list]" = OrderedDict([
    # P0 — foundations: environment report + the three P0 gates (plan §12)
    ("p0", [
        ("preflight", f"{PY} ops/preflight.py"),
        ("model_integrity", f"{PY} ops/check_model.py {MODEL}"),
        ("g1_module_dump", f"{PY} ops/g1_module_dump.py --model {MODEL} --out reports/qwen35_modules.json"),
        ("g2_fastpath", f"{PY} ops/g2_fastpath.py --model {MODEL} --out reports/g2_fastpath.json"),
    ]),
    # P2 — simulation stack: the Newton-only Isaac Lab gate (plan §12, G-3). Runs in the separate
    # --isaac venv, so it uses $ISAAC_PY, not the training interpreter.
    ("p2", [
        ("g3_sim", f"$ISAAC_PY ops/g3_sim.py --out reports/g3_sim.json --images reports/g3_sim"),
    ]),
    # P8/P1 — storage + resumable training: identity, a real-weight SFT/resume gate, and the setup notebook.
    ("p3", [
        ("hub_status", f"{PY} ops/hub.py status"),
        ("sft_smoke", f"{PY} ops/sft_smoke.py --out reports/sft_smoke.json"),
        ("notebook_setup", f"{PY} ops/nbrun.py notebooks/00_setup_cache.ipynb --timeout 1800"),
    ]),
    # package the small reports into one archive (contents: see ops/bundle.py)
    ("bundle", [
        ("bundle", f"{PY} ops/bundle.py"),
    ]),
])

# variables a stage cannot run without (checked BEFORE the stage starts). $MODELS/$DATA/$RUNS
# are exported by ops/env.sh, so only machine-specific inputs belong here.
REQUIRES: dict = {
    "g3_sim": ["ISAAC_PY"],   # the Newton-only venv; ops/env.sh exports it, --isaac creates it
}

# paths a stage reads that earlier stages must have written, by stage name
PRODUCES = {
    "g1_module_dump": ["reports/qwen35_modules.json"],
    "g2_fastpath": ["reports/g2_fastpath.json"],
    "g3_sim": ["reports/g3_sim.json"],
    "sft_smoke": ["reports/sft_smoke.json"],
}
SHELL_VARS = {"HOME", "PATH"}


def missing_vars(stage: str, environ) -> list:
    return [v for v in REQUIRES.get(stage, []) if not environ.get(v)]


def lint() -> list:
    """Static checks: undefined $VARS, and $RUNS/$DATA inputs that no earlier stage produces."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    defined = set(SHELL_VARS)
    defined |= set(re.findall(r"export (\w+)=", (root / "ops" / "env.sh").read_text()))
    problems, produced = [], set()
    for g, name, cmd in all_stages():
        for v in set(re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)", cmd)):
            if v not in defined:
                problems.append(f"{name}: undefined variable ${v}")
        for v in REQUIRES.get(name, []):
            if v not in defined:
                problems.append(f"{name}: REQUIRES unknown variable {v}")
        outs = set(re.findall(r"--out (\$[A-Z]+/[\w.]+)", cmd)) | set(PRODUCES.get(name, []))
        for tok in set(re.findall(r"(\$(?:RUNS|DATA)/[\w.]+)", cmd)):
            if tok in outs or any(tok == p or tok.startswith(p + "/") for p in produced):
                continue
            problems.append(f"{name}: reads {tok} but no earlier stage produces it")
        produced |= outs
    return problems


def all_stages():
    return [(g, n, c) for g, items in GROUPS.items() for n, c in items]


def _config_env() -> dict:
    """Environment as the stages will see it: the process env plus whatever ops/env.sh exports."""
    import subprocess
    try:
        out = subprocess.run(["bash", "-c", "source ops/env.sh >/dev/null 2>&1; env -0"], capture_output=True,
                             timeout=30).stdout
        env = dict(kv.split("=", 1) for kv in out.decode(errors="replace").split("\0") if "=" in kv)
    except Exception:                                                   # noqa: BLE001
        env = {}
    return {**env, **os.environ}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("lint")
    r = sub.add_parser("run")
    r.add_argument("group", choices=list(GROUPS))
    r.add_argument("--only", default=None)
    r.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "lint":
        probs = lint()
        print("\n".join(probs) if probs else "plan lint: ok")
        return 1 if probs else 0
    if a.cmd == "list":
        for g, items in GROUPS.items():
            print(f"[{g}]")
            for n, _ in items:
                print(f"   {n}")
        return 0
    from ops.stage import run_stage
    for n, cmd in GROUPS[a.group]:
        if a.only and n != a.only:
            continue
        env = _config_env()
        miss = missing_vars(n, env)
        if miss:
            print(f"stage {n} needs {', '.join(miss)}: set it in ops/env.sh or the environment, then rerun")
            return 2
        rc = run_stage(n, ["bash", "-lc", cmd], skip_if_done=not a.force)
        if rc != 0:
            print(f"stage {n} failed (rc={rc}); fix it and rerun: python ops/plan.py run {a.group} --only {n}")
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
