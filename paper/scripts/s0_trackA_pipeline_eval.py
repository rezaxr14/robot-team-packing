"""Track A decomposition: Eye->Brain on VIKI L2 with real-Eye, oracle-Eye and no-Eye conditions.

  # text-only Brain served on :8001, optional VLM Eye on :8000
  python scripts/s0_trackA_pipeline_eval.py --repo "$VIKI_R_REPO" --data <L2 test .parquet> --label L2_A \
      --brain-endpoint http://localhost:8001/v1 --brain-model brain \
      --eye-endpoint http://localhost:8000/v1 --eye-model eye --conditions oracle,none,eye
Paste back: reports/trackA_pipeline/<label>_pipeline_summary.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtp.llm_client import ChatClient
from rtp.trackA import load_upstream, summarize_items
from rtp.trackA_pipeline import run_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--brain-endpoint", required=True)
ap.add_argument("--brain-model", required=True)
ap.add_argument("--eye-endpoint", default=None)
ap.add_argument("--eye-model", default=None)
ap.add_argument("--conditions", default="oracle,none")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--out-dir", default="reports/trackA_pipeline")
a = ap.parse_args()

import pandas as pd
rows = pd.read_parquet(a.data).to_dict("records") if a.data.endswith(".parquet") else json.load(open(a.data))
up = load_upstream(a.repo)
brain = ChatClient(a.brain_endpoint, a.brain_model)
eye = ChatClient(a.eye_endpoint, a.eye_model) if a.eye_endpoint else None
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
summary = {"label": a.label, "data": a.data, "brain": a.brain_model, "eye": a.eye_model, "conditions": {}}
for cond in a.conditions.split(","):
    t0 = time.time()
    items = run_pipeline(rows, up, cond, lambda m: brain.chat(m, max_tokens=2000),
                         (lambda m: eye.chat(m, max_tokens=1000)) if eye else None, a.workers, a.limit)
    with open(out / f"{a.label}_{cond}_items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, default=str) + "\n")
    summary["conditions"][cond] = {"seconds": round(time.time() - t0, 1), **summarize_items(items, 2)}
    print(cond, json.dumps({k: summary["conditions"][cond][k] for k in ("n_scored", "n_errors", "strict_draw", "strict_all")}))
json.dump(summary, open(out / f"{a.label}_pipeline_summary.json", "w"), indent=2)
