"""Eye rejection-sampling fine-tuning data from a served Eye.

  ops/with_server.sh eye $RUNS/eye_sft_merged 8002 0.30 -- bash -c 'python scripts/eye_rft.py \
      --rows $DATA/eye/eye_grpo.jsonl --endpoint http://localhost:8002/v1 --model eye --mode credit --k 8 --out $DATA/eye/rft_credit.jsonl'
Run once per mode (accuracy | credit | mixed) and train each with LLaMA-Factory (ops/make_lf_config.py after pointing the
dataset at the new jsonl). Also writes <out>.stats.json including the accuracy-vs-credit selection disagreement rate.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtp.harness import Harness
from rtp.llm_client import ChatClient
from rtp.vlm_eye import eye_messages
from rtp.eye_rft import scorer_for, build_rft_rows, selection_disagreement, write_sft_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--endpoint", default="http://localhost:8002/v1")
ap.add_argument("--model", default="eye")
ap.add_argument("--mode", choices=("accuracy", "credit", "mixed"), default="credit")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--min-reward", type=float, default=0.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
rows = [json.loads(l) for l in open(a.rows)][: a.limit]
client = ChatClient(a.endpoint, a.model)
cache = {}


def sample(row, k):
    key = (row["seed"], row["view"])
    if key not in cache:
        msgs = eye_messages(Path(row["images"][0]).read_bytes())
        cache[key] = [client.chat(msgs, temperature=a.temperature, max_tokens=1024) for _ in range(k)]
    return cache[key]


sft, stats = build_rft_rows(rows, sample, a.k, scorer_for(a.mode, H), a.min_reward)
write_sft_jsonl(sft, a.out)
comps = [cache[(r["seed"], r["view"])] for r in rows if (r["seed"], r["view"]) in cache]
used = [r for r in rows if (r["seed"], r["view"]) in cache]
stats["accuracy_vs_credit"] = selection_disagreement(used, comps, scorer_for("accuracy", H), scorer_for("credit", H))
Path(str(a.out) + ".stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
