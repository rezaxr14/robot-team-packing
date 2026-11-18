#!/usr/bin/env python3
"""Build real CoT prompt records locally from the harness + datagen (no model call).

Each record is a prompt the model will actually complete on Kaggle:

    {"id": "cot-0", "seed": 0, "prompt": "...", "retrieved_ids": [...],
     "task_class": "pack_backpack", ...}

``task_class`` is the harness task-document id that the prompt instantiates (retrieved with
``Harness.retrieve(..., types=("task",))``), not an object class. Prompt text and retrieved ids
come from ``rtp.frontier_cot.build_cot_prompt`` / ``cot_request``; nothing here calls a model,
a frontier endpoint, or Kaggle. The private GPU kernel embeds the resulting JSONL verbatim.

    $V/bin/python ops/cot_prompts.py                 # 8 prompts -> data/local_4050/cot_prompts.jsonl
    $V/bin/python ops/cot_prompts.py --n 4 --style full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtp import frontier_cot as fc  # noqa: E402
from rtp.datagen import default_eyes, sample_spec, sample_world  # noqa: E402
from rtp.harness import Harness  # noqa: E402
from rtp.perception import merge_scenes  # noqa: E402
from rtp.profiles import get_profile  # noqa: E402

REQUIRED_FIELDS = ("id", "seed", "prompt", "retrieved_ids", "task_class")
DEFAULT_N = 8
DEFAULT_SEED0 = 0


def _task_class(harness: Harness, spec) -> str:
    """The harness task-doc id for this spec, grounded in real retrieval (never hard-coded)."""
    tasks = harness.retrieve(spec.text(), k=1, types=("task",))
    if tasks:
        return tasks[0].id
    docs = sorted((d for d in harness.docs.values() if d.type == "task"), key=lambda d: d.id)
    return docs[0].id if docs else "unknown"


def build_records(n: int = DEFAULT_N, seed0: int = DEFAULT_SEED0, k: int = 6,
                  style: str = "brief", *, harness_root: Path | str | None = None) -> list[dict]:
    """Return N prompt records from real scenes/specs; no model is consulted."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    harness = Harness(str(harness_root or ROOT / "harness_files"))
    out = []
    for seed in range(seed0, seed0 + n):
        world = sample_world(seed)
        spec = sample_spec(world, seed)
        scene = merge_scenes([e.report(world, 0) for e in default_eyes(seed)])
        prompt = fc.build_cot_prompt(harness, spec, scene, k=k, style=style)
        req = fc.cot_request(harness, spec, scene, k=k, style=style)
        out.append({
            "id": f"cot-{seed}",
            "seed": seed,
            "prompt": prompt,
            "retrieved_ids": list(req.doc_ids),
            "task_class": _task_class(harness, spec),
            "style": style,
            "k": k,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        })
    return out


def write_records(records: list[dict], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def read_records(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            missing = [k for k in REQUIRED_FIELDS if k not in row]
            if missing:
                raise ValueError(f"{path}: record missing {missing}")
            out.append(row)
    return out


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="number of prompts (default: 8)")
    ap.add_argument("--seed0", type=int, default=DEFAULT_SEED0, help="first datagen seed (default: 0)")
    ap.add_argument("--k", type=int, default=6, help="retrieval budget per pass (default: 6)")
    ap.add_argument("--style", choices=sorted(fc.STYLES), default="brief")
    ap.add_argument("--profile", default=None, help="rtp profile (default: $RTP_PROFILE or local_4050)")
    ap.add_argument("--out", default=None, help="override prompt JSONL path")
    args = ap.parse_args(argv)
    if args.n < 1 or args.k < 1:
        ap.error("--n and --k must be >= 1")
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    prof = get_profile(args.profile)
    prof.ensure_dirs()
    out = Path(args.out) if args.out else prof.paths.data / "cot_prompts.jsonl"
    records = build_records(n=args.n, seed0=args.seed0, k=args.k, style=args.style)
    write_records(records, out)
    classes = sorted({r["task_class"] for r in records})
    print(json.dumps({"count": len(records), "path": str(out), "seeds": [args.seed0, args.seed0 + args.n - 1],
                      "style": args.style, "task_classes": classes,
                      "sources": ["harness+datagen (no model call)"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
