"""Counterfactual Eye credit by fault type and field (paper Fig. 3), on the symbolic world.

  python scripts/run_credit_analysis.py --seeds 60 --out reports/credit
Reads: for each Eye fault type, how much score would be recovered if that Eye were perfect (oracle loss), what the
Eye's reports are worth versus saying nothing, and which single field (class, region, mass, fragility) carries
the loss. Symbolic-fixture mechanics until the real Eye and Isaac Lab are in the loop.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtp.harness import Harness
from rtp.credit import credit_by_fault, credit_table_markdown

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=40)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--out", default="reports/credit")
a = ap.parse_args()
H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
rows = credit_by_fault(H, range(a.seed0, a.seed0 + a.seeds))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(out / "credit.json", "w"), indent=1)
md = credit_table_markdown(rows)
(out / "credit.md").write_text(md + "\n")
print(md)
