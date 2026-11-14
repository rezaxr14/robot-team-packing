"""Assemble every result in a reports/ directory into one document with an EVIDENCE LEDGER.

  python scripts/build_report.py --reports reports --out reports/results.md
Each section is labelled with what its numbers can and cannot support:
  TRACK A         real benchmark data, real model                -> supports Track A claims
  ISAAC           physics episodes (records under isaac/)        -> supports Track B claims
  LLM-SYMBOLIC    a trained/served LLM Brain on the symbolic task -> supports claims scoped to the symbolic task
  FIXTURE         scripted Brain, flaky-planner fixture, parametric Eyes -> mechanics only, NEVER a paper claim
Missing inputs are listed, not silently skipped.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtp import tables
from rtp.experiments import read_records, summarize, paired_compare
from rtp.stats import summarize_binary

LEVELS = {
    "TRACK A": "real benchmark data and a real model; supports Track A claims",
    "ISAAC": "physics simulation; supports Track B claims",
    "LLM-SYMBOLIC": "a served LLM Brain in the symbolic task; supports claims scoped to the symbolic task only",
    "FIXTURE": "scripted Brain / flaky-planner fixture / parametric Eyes; checks mechanics, NOT evidence for any claim",
}


def load_group(reports: Path, sub: str, pattern: str):
    d = reports / sub
    return sorted(d.rglob(pattern)) if d.exists() else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", default=None, help="directory with hindsight_funnel.json and eye rft *.stats.json (e.g. $DATA)")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args(argv)
    rep = Path(a.reports)
    out_path = Path(a.out or rep / "results.md")
    sections, ledger, missing = [], [], []

    def add(level, title, body, source, n=None):
        sections.append(f"## [{level}] {title}\n\n*Scope: {LEVELS[level]}.*\n\n{body}\n")
        ledger.append(f"| {level} | {title} | `{source}` | {n if n is not None else ''} |")

    # ---- Track A ----
    ta = [json.loads(p.read_text()) for p in load_group(rep, "trackA", "*_summary.json")]
    if ta:
        add("TRACK A", "VIKI-Bench checkpoints through our runner", tables.table_trackA(ta), "trackA/*_summary.json",
            sum(s["full"]["n_scored"] for s in ta))
    else:
        missing.append("trackA/*_summary.json")
    pipe = [json.loads(p.read_text()) for p in load_group(rep, "trackA_pipeline", "*_pipeline_summary.json")]
    if pipe:
        rows = ["| Label | Condition | n scored | upstream-style % | position-robust % |", "|---|---|---|---|---|"]
        for s in pipe:
            for cond, v in s["conditions"].items():
                rows.append(f"| {s['label']} | {cond} | {v['n_scored']} | {100 * v['strict_draw']['rate']:.1f} | {100 * v['strict_all']['rate']:.1f} |")
        add("TRACK A", "Eye->Brain decomposition (oracle / none / eye)", "\n".join(rows) + "\n\nperception value = oracle - none; perception loss = oracle - eye.",
            "trackA_pipeline/*", sum(v["n_scored"] for s in pipe for v in s["conditions"].values()))
    else:
        missing.append("trackA_pipeline/*_pipeline_summary.json")

    # ---- symbolic ----
    for sub, level, title in (("sym", "FIXTURE", "Ablation grid, scripted Brain"), ("verify", "FIXTURE", "Plan verification vs flaky-planner fixture"),
                              ("llm", "LLM-SYMBOLIC", "LLM Brain, one-shot plans"), ("llm_tools", "LLM-SYMBOLIC", "LLM Brain with tools")):
        files = load_group(rep, sub, "records.jsonl")
        if not files:
            missing.append(f"{sub}/records.jsonl")
            continue
        recs = read_records(files[0])
        ref = "full" if any(r["config"] == "full" for r in recs) else recs[0]["config"]
        add(level, title, tables.table_main(recs) + f"\n\nAblations vs `{ref}`:\n\n" + tables.table_ablation(recs, ref) +
            "\n\nFatal-failure attribution (automatic labels; validate against hand labels before reporting):\n\n" + tables.table_attribution(recs),
            f"{sub}/records.jsonl", len(recs))
        if a.figures:
            from rtp import figures
            figures.fig_success(recs, rep / "figures" / f"{sub}_success.png")
            figures.fig_gap(recs, rep / "figures" / f"{sub}_gap.png")
            figures.fig_tokens_vs_success(recs, rep / "figures" / f"{sub}_tokens.png")

    cr = load_group(rep, "credit", "credit.json")
    if cr:
        from rtp.credit import credit_table_markdown
        rows = json.loads(cr[0].read_text())
        add("FIXTURE", "Counterfactual Eye credit by fault and field", credit_table_markdown(rows), "credit/credit.json", len(rows))
        if a.figures:
            from rtp import figures
            figures.fig_credit(rows, rep / "figures" / "credit.png")
    else:
        missing.append("credit/credit.json")

    # ---- Isaac: records per tag, paired comparisons on shared seeds ----
    iso = load_group(rep, "isaac", "episodes.jsonl")
    if iso:
        recs = [r for p in iso for r in read_records(p)]
        body = tables.table_main(recs)
        tags = sorted({r["config"] for r in recs})
        base = "isaac_scripted_arm" if "isaac_scripted_arm" in tags else tags[0]
        pairs = ["| Config vs " + base + " | paired n | success diff (pp) | 95% CI (pp) |", "|---|---|---|---|"]
        for t in tags:
            if t == base:
                continue
            c = paired_compare(recs, t, base)
            if c["n"]:
                pairs.append(f"| {t} | {c['n']} | {100 * c['mean_diff']:+.1f} | [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}] |")
        add("ISAAC", "Closed-loop episodes in Isaac", body + "\n\nPaired comparisons (same seeds only):\n\n" + "\n".join(pairs), "isaac/episodes.jsonl", len(recs))
        if a.figures:
            from rtp import figures
            figures.fig_success(recs, rep / "figures" / "isaac_success.png", "Isaac: success by configuration")
    else:
        missing.append("isaac/episodes.jsonl")

    for p in load_group(rep, "isaac", "regression_*.json"):
        d = json.loads(p.read_text())
        add("ISAAC", f"Skill regression ({p.stem})", f"{d['passed']}/{d['total']} cases passed.\n\nFailing: " +
            (", ".join(n for n, v in d["cases"].items() if not v["passed"]) or "none"), str(p.relative_to(rep)), d["total"])

    # ---- extras ----
    if a.extra:
        ex = Path(a.extra)
        f = ex / "hindsight_funnel.json"
        if f.exists():
            d = json.loads(f.read_text())
            add("FIXTURE", "Hindsight relabeling funnel", f"episodes {d['episodes']} -> failed {d['failed']} -> relabeled {d['relabeled']} -> replay-validated {d['validated']}", str(f), d["episodes"])
        for f in sorted(ex.rglob("*.stats.json")):
            d = json.loads(f.read_text())
            dis = d.get("accuracy_vs_credit", {})
            add("ISAAC" if "isaac" in f.name else "FIXTURE", f"Eye rejection sampling ({f.name})",
                f"prompts {d['prompts']}, kept {d['kept']}, mean best score {d['mean_best']:.3f} vs mean sample {d['mean_sample']:.3f}; "
                f"accuracy-vs-credit selection disagreement {dis.get('disagree_rate', float('nan')):.2f} over {dis.get('n', 0)} prompts", str(f), d["prompts"])

    head = ["# Results (auto-assembled)", "", "## Evidence ledger", "", "| level | what | source | n |", "|---|---|---|---|"] + ledger
    if missing:
        head += ["", "**Not found (not run yet, or not copied into reports/):** " + ", ".join(f"`{m}`" for m in missing)]
    head += ["", "**Levels:** " + "; ".join(f"{k} = {v}" for k, v in LEVELS.items()), ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(head) + "\n" + "\n".join(sections))
    print(f"wrote {out_path} ({len(sections)} sections, {len(missing)} missing inputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
