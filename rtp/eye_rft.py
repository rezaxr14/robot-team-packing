"""Eye rejection-sampling fine-tuning (best-of-K): a GRPO-free way to train the Eye from a reward signal.

For each rendered image: draw K reports from the served Eye, score each (field accuracy, counterfactual team credit, or a
mix), keep the best one if it clears a threshold, and emit a LLaMA-Factory SFT row. Because accuracy and credit pick
different winners, `selection_disagreement` measures how often the credit signal changes what the Eye is taught.
"""
from __future__ import annotations
import json

from .eye_rewards import r_eye_format, r_eye_accuracy, r_eye_credit
from .perception import default_blind


def scorer_for(mode: str, harness, w_acc: float = 0.5):
    """Return score(row, completion) for mode in {'accuracy','credit','mixed'}. row needs seed and view."""
    from .datagen import sample_world

    def acc(row, c):
        return r_eye_accuracy(c, sample_world(int(row["seed"])), default_blind(row["view"]))

    def cred(row, c):
        return r_eye_credit(c, int(row["seed"]), row["view"], harness)

    def score(row, c):
        if r_eye_format(c) == 0.0:
            return -1.0
        if mode == "accuracy":
            return acc(row, c)
        if mode == "credit":
            return cred(row, c)
        if mode == "mixed":
            return w_acc * acc(row, c) + (1 - w_acc) * cred(row, c)
        raise ValueError(mode)
    return score


def best_of_k(row, completions: list, score) -> tuple:
    scores = [score(row, c) for c in completions]
    i = max(range(len(completions)), key=lambda j: (scores[j], -j))
    return completions[i], scores[i], scores


def build_rft_rows(rows: list, sample_fn, k: int, score, min_reward: float = 0.0) -> tuple:
    """Returns (sft_rows, stats). sample_fn(row, k) -> list[str]."""
    out, best_all, mean_all, kept = [], [], [], 0
    for row in rows:
        comps = sample_fn(row, k)
        if not comps:
            continue
        best, s, scores = best_of_k(row, comps, score)
        best_all.append(s)
        mean_all.append(sum(scores) / len(scores))
        if s > min_reward:
            kept += 1
            out.append({"messages": [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": best}],
                        "images": row["images"], "meta": {"seed": row["seed"], "view": row["view"], "score": s}})
    stats = {"prompts": len(rows), "kept": kept, "mean_best": sum(best_all) / max(1, len(best_all)),
             "mean_sample": sum(mean_all) / max(1, len(mean_all))}
    return out, stats


def selection_disagreement(rows: list, completions_by_row: list, score_a, score_b) -> dict:
    """How often do two scorers pick a different best completion, and what does each lose under the other's metric?"""
    diff = n = 0
    a_under_b, b_under_b = [], []
    for row, comps in zip(rows, completions_by_row):
        if len(comps) < 2:
            continue
        n += 1
        ca, _, _ = best_of_k(row, comps, score_a)
        cb, _, _ = best_of_k(row, comps, score_b)
        diff += ca != cb
        a_under_b.append(score_b(row, ca))
        b_under_b.append(score_b(row, cb))
    return {"n": n, "disagree_rate": diff / max(1, n),
            "b_score_of_a_choice": sum(a_under_b) / max(1, len(a_under_b)),
            "b_score_of_b_choice": sum(b_under_b) / max(1, len(b_under_b))}


def write_sft_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
