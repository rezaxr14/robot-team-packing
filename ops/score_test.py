#!/usr/bin/env python3
"""Score a Brain on the fixed held-out test set and build an honest omni-model comparison.

Everything scored here is the **symbolic CPU fixture**: the only Brain loaded by this CLI
is ``ScriptedBrain`` (a deterministic reference planner / data-generation teacher). Its
numbers are fixture numbers, NOT a trained-model result. No adapter is loaded, so the
"Our Brain - Qwen3.5-4B (trained adapter)" row stays ``not run`` unless a real
prediction file for it is supplied via ``--predictions``.

Examples
--------
    python ops/score_test.py --profile local_4050 --n 30 --out reports/test_scores.json
    python ops/score_test.py --n 30 --predictions data/competitor_predictions
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtp import benchmark as bm                      # noqa: E402
from rtp.brain import ScriptedBrain                  # noqa: E402
from rtp.experiments import default_grid             # noqa: E402
from rtp.harness import Harness                      # noqa: E402
from rtp.profiles import get_profile                 # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:                                # noqa: BLE001
        return "unknown"


def _display(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def _ci(s) -> str:
    return "—" if s is None else f"[{_pct(s['lo'])}, {_pct(s['hi'])}]"


def markdown_table(models) -> str:
    out = [
        "| Model | n | Success % [95% Wilson CI] | Plan-pass % | Mean msg tokens | "
        "Paired vs ours (ours − model) | Status | How run / note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        if m["status"] != "scored":
            out.append(f"| {m['model']} | not run | — | — | — | — | not run | "
                       f"{m['how_run']}{'; ' + m['note'] if m['note'] else ''} |")
            continue
        sc, pp, paired = m["success"], m["plan_pass"], m["paired_vs_ours"]
        pr = ("—" if paired is None else
              f"{100 * paired['mean_diff']:+.1f} pp [{100 * paired['lo']:+.1f}, "
              f"{100 * paired['hi']:+.1f}] (n={paired['n']})")
        tok = "—" if m["mean_msg_tokens"] is None else f"{m['mean_msg_tokens']:.0f}"
        note = "; ".join(x for x in (m["how_run"], m["note"]) if x)
        out.append(f"| {m['model']} | {m['n']} | {_pct(sc['rate'])} {_ci(sc)} | "
                   f"{_pct(pp['rate']) if pp else '—'} | {tok} | {pr} | scored | {note} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="local_4050",
                    help="hardware profile for provenance only (this benchmark is CPU-only)")
    ap.add_argument("--n", type=int, default=30, help="number of fixed held-out test seeds (default: 30)")
    ap.add_argument("--out", default="reports/test_scores.json", help="where to write our scored records")
    ap.add_argument("--config", default="full", choices=[c.name for c in default_grid(noisy=True)],
                    help="experiment config from rtp.experiments.default_grid (default: full)")
    ap.add_argument("--predictions", default=None,
                    help="directory of competitor JSONL prediction files (model, seed, success, plan, reasoning?)")
    args = ap.parse_args(argv)
    if args.n < 1:
        ap.error("--n must be >= 1")

    try:
        prof = get_profile(args.profile, root=ROOT)
    except KeyError as e:
        ap.error(str(e))

    cfg = next(c for c in default_grid(noisy=True) if c.name == args.config)
    seeds = bm.test_seeds(args.n)
    print(f"profile: {prof.name} | config hash: {prof.config_hash()} | git: {_git_sha()[:8]}")
    print(f"config: {cfg.name} | noisy_eyes={cfg.noisy_eyes} | n={args.n} | held-out test seeds: "
          f"0..{args.n - 1} (train namespace >= {bm.TRAIN_SEED_BASE})")
    if not set(prof.eval_seeds) <= set(seeds):
        print("note: profile eval_seeds are not all inside this test set")

    harness = Harness(ROOT / "harness_files")
    brain = ScriptedBrain()
    brain.name = "Ours - ScriptedBrain (fixture)"      # explicit label; never a trained model
    scored = bm.score_brain(brain, seeds, harness, cfg=cfg)
    print(f"scored model: {scored['model']} (kind={scored['kind']})")
    s, pp = scored["success"], scored["plan_pass"]
    print(f"success: {s['k']}/{s['n']} = {100*s['rate']:.1f}% "
          f"[{100*s['lo']:.1f}, {100*s['hi']:.1f}] | "
          f"plan-pass: {pp['k']}/{pp['n']} = {100*pp['rate']:.1f}% "
          f"[{100*pp['lo']:.1f}, {100*pp['hi']:.1f}] | "
          f"mean msg tokens: {scored['mean_msg_tokens']:.1f}")
    print("IMPORTANT: no trained Brain adapter was loaded. These are ScriptedBrain FIXTURE numbers, "
          "not a trained-model score; no Isaac Lab/Newton or hardware claim is made.")

    # predictions must be on the held-out seeds; rows on other seeds cannot count
    prediction_files, ignored = [], 0
    predictions = []
    if args.predictions:
        pdir = Path(args.predictions)
        if not pdir.is_absolute():
            pdir = ROOT / pdir
        prediction_files = sorted(_display(p) for p in pdir.glob("*.jsonl"))
        predictions = bm.load_predictions(pdir)
        held_out = set(seeds)
        ignored = sum(r["seed"] not in held_out for r in predictions)
        if ignored:
            print(f"WARNING: ignoring {ignored} prediction row(s) whose seed is outside the held-out test set")
        predictions = [r for r in predictions if r["seed"] in held_out]

    comp = bm.compare(scored["records"], predictions)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "profile": prof.name,
        "config_hash": prof.config_hash(),
        "config": scored["config"],
        "split": "held-out test",
        "seed_namespace": f"0..n-1 (training seeds >= {bm.TRAIN_SEED_BASE})",
        "n": args.n,
        "seeds": scored["seeds"],
        "model": scored["model"],
        "model_kind": scored["kind"],
        "trained_brain_scored": False,
        "honesty_note": ("ScriptedBrain is a deterministic fixture. The trained Qwen3.5-4B adapter "
                         "was NOT scored: no adapter was loaded by this CLI."),
        "environment": "symbolic CPU fixture (not Isaac Lab/Newton, not hardware)",
        "success": scored["success"],
        "plan_pass": scored["plan_pass"],
        "wilson": scored["wilson"],
        "mean_msg_tokens": scored["mean_msg_tokens"],
        "records": scored["records"],
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    comparison = {
        "created_utc": payload["created_utc"],
        "git_sha": payload["git_sha"],
        "profile": prof.name,
        "config_hash": payload["config_hash"],
        "config": cfg.name,
        "n": args.n,
        "split": comp["split"],
        "direction": comp["direction"],
        "seed_namespace": payload["seed_namespace"],
        "prediction_files": prediction_files,
        "ignored_prediction_rows_outside_test": ignored,
        "trained_brain_scored": False,
        "ours_are_fixture": scored["kind"] == "fixture",
        "models": comp["models"],
    }
    cjson = ROOT / "reports" / "omni_comparison.json"
    cmd = ROOT / "reports" / "omni_comparison.md"
    cjson.parent.mkdir(parents=True, exist_ok=True)
    cjson.write_text(json.dumps(comparison, indent=2) + "\n")
    cmdtxt = ("# Omni-model comparison (held-out symbolic fixture)\n\n"
              "This table uses the fixed test seeds and the same symbolic CPU fixture for every scored row. "
              "No trained Brain adapter was scored by this CLI; rows with no prediction file are `not run` "
              "with no numbers.\n\n" + markdown_table(comp["models"]) + "\n")
    cmd.write_text(cmdtxt)

    print("\n" + cmdtxt.strip())
    print(f"\nwrote {_display(out)} | {_display(cjson)} | {_display(cmd)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
