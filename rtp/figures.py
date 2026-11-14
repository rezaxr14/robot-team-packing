"""Figures for the paper (matplotlib, Agg backend). Each function writes one PNG and returns its path.
All take the same inputs as the tables: episode records, credit rows, Track A summaries."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .experiments import summarize


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def fig_success(records, path, title="Success rate by configuration (Wilson 95% CI)"):
    s = summarize(records)
    names = list(s)
    rate = [100 * s[n]["success"]["rate"] for n in names]
    lo = [max(0.0, 100 * (s[n]["success"]["rate"] - s[n]["success"]["lo"])) for n in names]
    hi = [max(0.0, 100 * (s[n]["success"]["hi"] - s[n]["success"]["rate"])) for n in names]
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(names) + 2), 4))
    ax.bar(range(len(names)), rate, yerr=[lo, hi], capsize=3, color="#4c72b0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    return _save(fig, path)


def fig_gap(records, path, title="Plan-pass vs success (gap = plan-pass minus success)"):
    s = summarize(records)
    names = list(s)
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(names) + 2), 4))
    ax.bar([i - w / 2 for i in range(len(names))], [100 * s[n]["plan_pass"]["rate"] for n in names], w, label="first plan passes dry run", color="#55a868")
    ax.bar([i + w / 2 for i in range(len(names))], [100 * s[n]["success"]["rate"] for n in names], w, label="episode success", color="#c44e52")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("%")
    ax.legend()
    ax.set_title(title)
    return _save(fig, path)


def fig_credit(rows, path, title="Score recoverable by fixing one Eye field (oracle gain)"):
    from .credit import FIELDS
    faults = [r["fault"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(faults) + 2), 4))
    w = 0.8 / len(FIELDS)
    for k, f in enumerate(FIELDS):
        ax.bar([i + k * w for i in range(len(faults))], [r["fields"][f]["mean"] for r in rows], w, label=f)
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(faults))])
    ax.set_xticklabels(faults, rotation=30, ha="right")
    ax.set_ylabel("score gain")
    ax.legend(fontsize=8)
    ax.set_title(title)
    return _save(fig, path)


def fig_tokens_vs_success(records, path, title="Message tokens vs success"):
    s = summarize(records)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for n, d in s.items():
        ax.scatter(d["mean_msg_tokens"], 100 * d["success"]["rate"], s=40)
        ax.annotate(n, (d["mean_msg_tokens"], 100 * d["success"]["rate"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean message tokens per episode")
    ax.set_ylabel("success (%)")
    ax.set_title(title)
    return _save(fig, path)


def fig_trackA(summaries, path, title="Track A level 2: upstream-style vs position-robust"):
    l2 = [s for s in summaries if s.get("level") == 2]
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(l2) + 2), 4))
    w = 0.38
    for i, s in enumerate(l2):
        f = s["full"]
        ax.bar(i - w / 2, 100 * f["strict_draw"]["rate"], w, color="#4c72b0", label="upstream-style (one draw)" if i == 0 else None)
        ax.bar(i + w / 2, 100 * f["strict_all"]["rate"], w, color="#dd8452", label="position-robust" if i == 0 else None)
    ax.set_xticks(range(len(l2)))
    ax.set_xticklabels([s["label"] for s in l2])
    ax.set_ylabel("correct (%)")
    ax.legend()
    ax.set_title(title)
    return _save(fig, path)
