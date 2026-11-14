"""Run a named stage with logging, timing and resume.

  python ops/stage.py run NAME [--skip-if-done] -- <command...>
  python ops/stage.py status
Every run appends one JSON line to reports/stage_log.jsonl (name, command, seconds, returncode, log path, tail).
Per-run output is tee'd to reports/logs/<name>.log. The measured seconds are what recalibrates the day plan.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("reports/stage_log.jsonl")


def _read():
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()] if LOG.exists() else []


def latest() -> dict:
    out = {}
    for r in _read():
        out[r["name"]] = r
    return out


def run_stage(name: str, cmd: list, skip_if_done: bool = False, log_dir: str = "reports/logs") -> int:
    if skip_if_done and latest().get(name, {}).get("returncode") == 0:
        print(f"[stage {name}] already done, skipping")
        return 0
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{name}.log"
    t0 = time.time()
    tail = []
    with open(log_path, "w") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
            tail.append(line.rstrip())
            tail = tail[-25:]
        rc = p.wait()
    rec = {"name": name, "cmd": " ".join(cmd), "start": t0, "seconds": round(time.time() - t0, 1), "returncode": rc,
           "log": str(log_path), "tail": tail}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[stage {name}] rc={rc} in {rec['seconds']}s")
    return rc


def status_table() -> str:
    rows = ["| stage | rc | seconds | log |", "|---|---|---|---|"]
    for n, r in latest().items():
        rows.append(f"| {n} | {r['returncode']} | {r['seconds']} | {r['log']} |")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--skip-if-done", action="store_true")
    r.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(status_table())
        return 0
    command = a.command[1:] if a.command and a.command[0] == "--" else a.command
    return run_stage(a.name, command, a.skip_if_done)


if __name__ == "__main__":
    sys.exit(main())
