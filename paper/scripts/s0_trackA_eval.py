"""Stage 0 / item 7: evaluate a served VIKI-R (or any OpenAI-compatible VLM) on VIKI-Bench L1/L2.

Reuses the upstream prompts (extracted from the repo's eval scripts via AST) and upstream scorers, and adds
position-robust L2 metrics. Reports BOTH the upstream-style single-draw number and the position-robust one.

Example:
  vllm serve <checkpoint> --served-model-name viki-r-7b --port 8000
  python scripts/s0_trackA_eval.py --repo "$VIKI_R_REPO" --level 2 --data <path>/test.parquet \
      --model viki-r-7b --label L2_id --out-dir reports/trackA --upstream-truncation
Paste back: reports/trackA/<label>_summary.json (small). Per-item results stay in <label>_items.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtp.llm_client import ChatClient
from rtp.trackA import load_upstream, evaluate_rows, summarize_items


def load_rows(path: str):
    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="clone of MARS-EAI/VIKI-R")
    ap.add_argument("--level", type=int, choices=(1, 2), required=True)
    ap.add_argument("--data", required=True, help=".parquet or .json with prompt/reward_model/images columns")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--label", required=True, help="e.g. L2_id, L2_ood (YOU decide which split is which)")
    ap.add_argument("--out-dir", default="reports/trackA")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    ap.add_argument("--upstream-truncation", action="store_true",
                    help="also report the subset the upstream script uses (first 900 for L2, first 1000 for L1)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=2000)
    a = ap.parse_args()

    up = load_upstream(a.repo)
    rows = load_rows(a.data)
    client = ChatClient(a.endpoint, a.model)
    t0 = time.time()
    items = evaluate_rows(rows, a.level, up,
                          lambda m: client.chat(m, temperature=0, max_tokens=a.max_tokens),
                          workers=a.workers, limit=a.limit)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.label}_items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, default=str) + "\n")
    summary = {"label": a.label, "level": a.level, "model": a.model, "data": a.data, "n_rows_in_file": len(rows),
               "seconds": round(time.time() - t0, 1), "full": summarize_items(items, a.level)}
    if a.upstream_truncation:
        n = 900 if a.level == 2 else 1000
        summary["upstream_subset"] = {"first_n": n, **summarize_items(items[:n], a.level)}
    json.dump(summary, open(out / f"{a.label}_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
