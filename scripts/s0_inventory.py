"""Stage 0 / item 2: inventory a local copy of the VIKI-R dataset (henggg/VIKI-R) and quantify how
exposed the L2 scorer is to its init-position randomness.

Usage: python scripts/s0_inventory.py --data /path/to/dataset_dir [--repo /path/to/VIKI-R]
STATUS: schema-tolerant but UNTESTED on the real parquet files (the authoring sandbox could not reach
Hugging Face). Fix column names here after the first real run.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def maybe_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return x if isinstance(x, (dict, list)) else None


def find_gt(row: dict):
    """Return a ground-truth dict containing init_pos, wherever it lives in the row."""
    for v in row.values():
        d = maybe_json(v)
        if isinstance(d, dict):
            if "init_pos" in d:
                return d
            for vv in d.values():
                dd = maybe_json(vv)
                if isinstance(dd, dict) and "init_pos" in dd:
                    return dd
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--repo", default=None)
    a = ap.parse_args()
    import pandas as pd
    files = sorted(Path(a.data).rglob("*.parquet"))
    print(f"{len(files)} parquet files")
    multi_pos_rows = total_gt = 0
    for f in files:
        df = pd.read_parquet(f)
        print(f"\n{f.relative_to(a.data)}: {len(df)} rows; columns={list(df.columns)}")
        print({c: str(df[c].dtype) for c in df.columns})
        for c in df.columns:
            try:                                           # nested columns (lists/dicts/arrays) are unhashable: skip them
                if df[c].nunique() <= 12 and len(df) > 12:
                    print(f"  {c}: {df[c].value_counts().to_dict()}")
            except TypeError:
                print(f"  {c}: (nested values, not summarised)")
        for _, row in df.head(2000).iterrows():
            gt = find_gt(row.to_dict())
            if gt is None:
                continue
            from rtp.trackA import to_native
            gt = to_native(gt)                              # parquet arrays -> plain lists/dicts
            total_gt += 1
            movable = {k: v for k, v in gt["init_pos"].items() if not (k.startswith("R") and k[1:].isdigit())}
            multi_pos_rows += any(len(v) > 1 for v in movable.values())
    if total_gt:
        print(f"\nrows with a parsable init_pos: {total_gt}; with >=1 asset having >1 candidate start: "
              f"{multi_pos_rows} ({multi_pos_rows / total_gt:.1%}) -> exposure to scorer nondeterminism")
    else:
        print("\nno init_pos found; adapt find_gt() to the real schema")


if __name__ == "__main__":
    main()
