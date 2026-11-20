#!/usr/bin/env python3
"""Drive the Kaggle-hosted frontier CoT task: push / status / collect (no Qwen, no local model).

    $V/bin/python ops/kaggle/cot_frontier/run.py push
    $V/bin/python ops/kaggle/cot_frontier/run.py status
    $V/bin/python ops/kaggle/cot_frontier/run.py collect

`push` uses the KAGGLE_API_TOKEN from .env and the localhost:10808 proxy. A push also starts one
run on the account's default frontier model (Kaggle Benchmarks behaviour): do not schedule a
second run. `collect` downloads that run's /kaggle/working output, copies real traces to
data/<profile>/cot_frontier_traces.jsonl and writes reports/cot_frontier.json with the actual
model and exact API call. If nothing real came back, the report records the error verbatim.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rtp.profiles import get_profile  # noqa: E402
from rtp.remotes import configure_kaggle_proxy, load_env  # noqa: E402

TASK_NAME = "rtp-cot-frontier-fullplan"
TASK_FILE = ROOT / "ops" / "kaggle" / "cot_frontier" / "task.py"
DOWNLOAD_DIR = ROOT / ".tmp" / "cot_frontier_out"
TRACE_NAME = "cot_frontier_traces.jsonl"
MANIFEST_NAME = "cot_frontier_manifest.json"
EXPECTED_SOURCE = "kaggle-benchmarks-proxy"
TRACE_FIELDS = ("id", "seed", "prompt", "completion", "retrieved_ids", "source",
                "model", "latency_s")


def kaggle(*args: str) -> tuple[int, str]:
    load_env(ROOT / ".env")
    configure_kaggle_proxy()
    exe = Path(sys.executable).with_name("kaggle")
    cmd = [str(exe) if exe.exists() else "kaggle", *args]
    last = ""
    for attempt in range(5):
        r = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if r.returncode == 0:
            return 0, (r.stdout or "").strip()
        last = (r.stderr or r.stdout).strip()
        if "SSLError" not in last and "Max retries" not in last:
            break
        if attempt + 1 < 5:
            time.sleep(2 + 2 * attempt)
    return r.returncode, last[-2000:]


def _newest(out_dir: Path, name: str) -> Path | None:
    found = [p for p in out_dir.rglob(name) if p.is_file()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def _read_traces(path: Path | None) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    if path is None or not path.exists():
        return rows, problems
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{TRACE_NAME}:{i}: invalid JSON: {exc}")
            continue
        missing = [k for k in TRACE_FIELDS if k not in row]
        if missing:
            problems.append(f"{TRACE_NAME}:{i}: missing {missing}")
            continue
        if row.get("source") != EXPECTED_SOURCE:
            problems.append(f"{TRACE_NAME}:{i}: unexpected source {row.get('source')!r}")
        rows.append(row)
    return rows, problems


def _report(rows, manifest, problems, status, out_rc, out_text, dest, profile, quota) -> dict:
    errors = []
    if manifest:
        errors.extend(f"{e.get('id', '?')}: {e.get('error')}" for e in (manifest.get("errors") or []))
    errors.extend(problems)
    if out_rc != 0:
        errors.append("tasks download: " + out_text.strip()[-1200:])
    if not rows and not errors:
        errors.append("hosted task returned no traces and no error manifest")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": TASK_NAME,
        "profile": profile,
        "model": (manifest or {}).get("model"),
        "source_labels": sorted({r["source"] for r in rows}) or [EXPECTED_SOURCE],
        "api_call": {
            "method": "POST",
            "url": (manifest or {}).get("endpoint"),
            "route": (manifest or {}).get("api"),
            "auth": "Bearer $MODEL_PROXY_API_KEY (platform-injected in the hosted worker; value never stored)",
            "body": {"model": "llm.model from kaggle_benchmarks", "messages": [{"role": "user", "content": "<harness prompt>"}],
                     "max_tokens": (manifest or {}).get("request_params", {}).get("max_tokens"),
                     "temperature": (manifest or {}).get("request_params", {}).get("temperature"),
                     "reasoning_effort": (manifest or {}).get("request_params", {}).get("reasoning_effort")},
            "sdk": "kaggle_benchmarks @kbench.task with requests to Kaggle Model Proxy",
        },
        "counts": {"total": len(rows),
                   "completed": sum(1 for r in rows if r["completion"] and not r.get("error")),
                   "by_source": dict(Counter(r["source"] for r in rows))},
        "usage": (manifest or {}).get("total_usage"),
        "finish_reasons": (manifest or {}).get("finish_reasons"),
        "error_summary": {"count": len(errors), "errors": errors[:10]},
        "kernel_status": status.strip()[-1500:],
        "quota": quota,
        "trace_path": str(dest) if rows else None,
        "prompt_path": str(ROOT / "data" / profile / "cot_prompts.jsonl"),
        "caveats": [
            "The interactive Model Proxy from this machine returned HTTP 403: "
            "User location \"RU\" is not supported for this model/API; the hosted-task route was used instead.",
            "Earlier hosted task versions are superseded and are not cited as CoT evidence: versions "
            "with `-> None` and postponed `-> dict` annotations failed at import, and the 400/600-token "
            "runs ended with finish_reason=length (truncated Plan JSON). The final task uses "
            "reasoning_effort=none and max_tokens=1200.",
            "No Qwen model, Qwen kernel or local GPU generation was used for these traces.",
            "These are verbatim transport records; plans were not replayed or scored for correctness here.",
        ],
        "note": ("Completions came from the named Kaggle-hosted frontier model via the Model Proxy; "
                 "transport records are verbatim." if rows else
                 "No real frontier traces were available. Nothing was fabricated; see error_summary."),
        "success": bool(rows) and not problems and not ((manifest or {}).get("errors") or []),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["push", "status", "collect"])
    ap.add_argument("--profile", default=None)
    args = ap.parse_args(argv)
    prof = get_profile(args.profile)
    prof.ensure_dirs()

    if args.cmd == "push":
        if not TASK_FILE.exists():
            raise SystemExit(f"missing {TASK_FILE}; run ops/kaggle/cot_frontier/build_task.py first")
        rc, text = kaggle("benchmarks", "tasks", "push", TASK_NAME, "-f", str(TASK_FILE))
        print(text)
        print("push rc:", rc)
        return 0 if rc == 0 else 1
    if args.cmd == "status":
        rc, text = kaggle("benchmarks", "tasks", "status", TASK_NAME)
        print(text)
        print("status rc:", rc)
        return 0 if rc == 0 else 1

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    rc_status, status = kaggle("benchmarks", "tasks", "status", TASK_NAME)
    rc_out, out_text = kaggle("benchmarks", "tasks", "download", TASK_NAME,
                              "-o", str(DOWNLOAD_DIR), "-f")
    trace_file = _newest(DOWNLOAD_DIR, TRACE_NAME)
    manifest_file = _newest(DOWNLOAD_DIR, MANIFEST_NAME)
    manifest = None
    if manifest_file:
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            pass
    rows, problems = _read_traces(trace_file)
    dest = prof.paths.data / TRACE_NAME
    if rows:
        shutil.copyfile(trace_file, dest)
    else:
        dest = prof.paths.data / TRACE_NAME        # report the intended path, but do not create it
    _, quota_text = kaggle("benchmarks", "quota", "--format", "json")
    try:
        quota = json.loads(quota_text)
    except json.JSONDecodeError:
        quota = quota_text[-1000:]
    report = _report(rows, manifest, problems, status, rc_out, out_text, dest, prof.name, quota)
    report_path = ROOT / "reports" / "cot_frontier.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"traces": len(rows), "model": report["model"], "errors": report["error_summary"]["count"],
                      "report": str(report_path), "trace_path": str(dest) if rows else None}, indent=2))
    if rows:
        ex = rows[0]
        print("example (truncated):", json.dumps({"id": ex["id"], "seed": ex["seed"],
              "completion": ex["completion"][:400], "latency_s": ex["latency_s"]}, ensure_ascii=False))
    return 0 if rows and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
