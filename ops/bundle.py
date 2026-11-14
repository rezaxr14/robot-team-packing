"""Bundle small result files into one archive to paste back, and print a compact text summary.

  python ops/bundle.py [--reports reports] [--out paste_back.tar.gz]
Includes .json/.md/.txt/.csv/.png (figures) under reports/ (< 2 MB each), the stage log and tails of failed stages; excludes
per-item model outputs (*_items.jsonl), weights and images.
"""
from __future__ import annotations
import argparse
import json
import tarfile
import time
from pathlib import Path

ALLOWED = {".json", ".md", ".txt", ".csv", ".png"}
MAX_BYTES = 2_000_000


def collect(reports: Path) -> list:
    files = []
    for p in sorted(reports.rglob("*")):
        if not p.is_file() or p.name.endswith("_items.jsonl") or "logs" in p.parts:
            continue
        if (p.suffix in ALLOWED or p.name == "stage_log.jsonl") and p.stat().st_size <= MAX_BYTES:
            files.append(p)
    return files


def _metric(d: dict, *keys):
    for k in keys:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d.get("rate") if isinstance(d, dict) else None


def summarize(reports: Path) -> str:
    lines = ["# paste-back summary", ""]
    log = reports / "stage_log.jsonl"
    if log.exists():
        last = {}
        for l in log.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                last[r["name"]] = r
        lines += ["## stages", "| stage | rc | seconds |", "|---|---|---|"]
        lines += [f"| {n} | {r['returncode']} | {r['seconds']} |" for n, r in last.items()]
        for n, r in last.items():
            if r["returncode"] != 0:
                lines += ["", f"### FAILED: {n} (last lines)"] + ["    " + t for t in r.get("tail", [])[-15:]]
    for p in sorted(reports.rglob("*_summary.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        full = d.get("full") or {}
        lines += ["", f"## {p.name}"]
        if full:
            lines.append(f"n_scored={full.get('n_scored')} errors={full.get('n_errors')} "
                         f"format={_metric(full, 'format')} strict_draw={_metric(full, 'strict_draw')} "
                         f"strict_all={_metric(full, 'strict_all')} correct={_metric(full, 'correct')}")
        for cond, v in (d.get("conditions") or {}).items():
            lines.append(f"{cond}: n_scored={v.get('n_scored')} strict_draw={_metric(v, 'strict_draw')} strict_all={_metric(v, 'strict_all')}")
    for p in sorted(reports.rglob("tables.md")) + sorted(reports.rglob("credit.md")):
        lines += ["", f"## {p.relative_to(reports)}"] + p.read_text().splitlines()[:40]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = Path(a.reports)
    out = Path(a.out or f"paste_back_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz")
    files = collect(rep)
    with tarfile.open(out, "w:gz") as tf:
        for f in files:
            tf.add(f, arcname=str(f))
    summary = summarize(rep)
    (rep / "paste_back_summary.md").write_text(summary)
    print(summary)
    print(f"\nbundled {len(files)} files -> {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
