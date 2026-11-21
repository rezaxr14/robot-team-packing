#!/usr/bin/env python3
"""Audit a corpus for seed-namespace violations (RUNBOOK rule 8).

Training artefacts must only contain seeds >= 100000; held-out evaluation seeds are < 100000 and
may never appear in training data, memory or prompts. This script reads files and writes a report;
it never edits or deletes corpus data.

    python ops/audit_seeds.py                       # data/ + runs/datagen/
    python ops/audit_seeds.py path/to/corpus.jsonl  # explicit files

Report: ``reports/seed_audit.json``. Evidence class: FIXTURE (bookkeeping, no model claim).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtp import seeds  # noqa: E402

DEFAULT_DIRS = [ROOT / "data", ROOT / "runs" / "datagen"]
SUFFIXES = {".jsonl", ".ndjson", ".json"}


def collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in SUFFIXES))
        elif p.exists():
            files.append(p)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "seed_audit.json")
    a = ap.parse_args()

    files = collect(a.paths or DEFAULT_DIRS)
    findings = seeds.audit_paths(files)
    by_file: dict[str, int] = {}
    for f in findings:
        by_file[f["file"]] = by_file.get(f["file"], 0) + 1
    report = {
        "step": "M3 seed namespace audit",
        "evidence_class": "FIXTURE",
        "python": platform.python_version(),
        "train_min": seeds.TRAIN_MIN,
        "rule": "training seeds >= 100000; test seeds < 100000, never in training data, memory or prompts",
        "files_scanned": len(files),
        "violations": len(findings),
        "by_file": by_file,
        "findings": findings[:200],
        "checks": {"no_test_seed_in_scanned_corpus": not findings},
        "result": "PASS" if not findings else "FAIL",
        "note": ("A violation means the recorded seed is in the held-out namespace. Such records may not be "
                 "used for SFT/GRPO/memory; regenerate them with seeds >= 100000. Nothing is deleted here."),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"files": len(files), "violations": len(findings),
                      "by_file": by_file, "result": report["result"]}, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
