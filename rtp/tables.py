"""Markdown tables from episode records (feeds the paper's Section 6)."""
from __future__ import annotations
from .experiments import summarize, paired_compare


def _pct(x):
    return f"{100 * x:.1f}"


def table_main(records) -> str:
    s = summarize(records)
    out = ["| Config | n | Success % [95% CI] | Valid items % | Plan-pass % | Gap (pp) | Msg tokens | Retrieval recall |",
           "|---|---|---|---|---|---|---|---|"]
    for name, d in s.items():
        sc = d["success"]
        out.append(f"| {name} | {d['n']} | {_pct(sc['rate'])} [{_pct(sc['lo'])}, {_pct(sc['hi'])}] | "
                   f"{_pct(d['valid_item_frac'])} | {_pct(d['plan_pass']['rate'])} | "
                   f"{100 * d['gap_plan_minus_success']:+.1f} | {d['mean_msg_tokens']:.0f} | {d['mean_retrieval_recall']:.2f} |")
    return "\n".join(out)


def table_ablation(records, reference: str = "full", key: str = "success") -> str:
    names = sorted({r["config"] for r in records} - {reference})
    out = [f"| Config vs {reference} | paired n | mean diff in {key} (pp) | 95% paired-bootstrap CI (pp) |", "|---|---|---|---|"]
    for n in names:
        c = paired_compare(records, n, reference, key)
        if c["n"]:
            out.append(f"| {n} | {c['n']} | {100 * c['mean_diff']:+.1f} | [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}] |")
    return "\n".join(out)


def table_attribution(records) -> str:
    s = summarize(records)
    out = ["| Config | failed episodes | with Eye | with Brain | with Arm |", "|---|---|---|---|---|"]
    for name, d in s.items():
        failed = sum(1 for r in records if r["config"] == name and not r["success"])
        f = d["episodes_with_fatal_agent"]
        out.append(f"| {name} | {failed} | {f.get('Eye', 0)} | {f.get('Brain', 0)} | {f.get('Arm', 0)} |")
    return "\n".join(out)


def table_trackA(summaries: list) -> str:
    """summaries: list of dicts loaded from s0_trackA_eval.py *_summary.json files."""
    out = ["| Label | Model | n scored (errors) | Format % | Upstream-style % | Position-robust % | Position-dependent items |",
           "|---|---|---|---|---|---|---|"]
    for s in summaries:
        f = s["full"]
        if s["level"] == 2:
            out.append(f"| {s['label']} | {s['model']} | {f['n_scored']} ({f['n_errors']}) | {_pct(f['format']['rate'])} | "
                       f"{_pct(f['strict_draw']['rate'])} [{_pct(f['strict_draw']['lo'])}, {_pct(f['strict_draw']['hi'])}] | "
                       f"{_pct(f['strict_all']['rate'])} [{_pct(f['strict_all']['lo'])}, {_pct(f['strict_all']['hi'])}] | "
                       f"{f['items_position_dependent']} |")
        else:
            c = f["correct"]
            out.append(f"| {s['label']} | {s['model']} | {f['n_scored']} ({f['n_errors']}) | {_pct(f['format']['rate'])} | "
                       f"{_pct(c['rate'])} [{_pct(c['lo'])}, {_pct(c['hi'])}] | n/a | n/a |")
    return "\n".join(out)
