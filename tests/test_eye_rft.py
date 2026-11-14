import json
from pathlib import Path

from rtp.harness import Harness
from rtp.datagen import sample_world
from rtp.perception import oracle_report
from rtp.eye_rft import scorer_for, best_of_k, build_rft_rows, selection_disagreement, write_sft_jsonl

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
SEED = 3


def completions():
    base = oracle_report(sample_world(SEED), "A").content["objects"]
    lap_wrong = [dict(o, cls="book") if o["cls"] == "laptop" else o for o in base]                       # 1 critical error
    frag_wrong = [dict(o, fragility_guess=0.95) if o["cls"] != "laptop" else o for o in base]            # many harmless errors
    return base, lap_wrong, frag_wrong


def row():
    return {"prompt": "<image>list", "images": ["x.png"], "seed": SEED, "view": "A"}


def test_perfect_report_wins_under_every_scorer_and_garbage_loses():
    base, lap, frag = completions()
    comps = [json.dumps(c) for c in (lap, base, frag)] + ["garbage"]
    for mode in ("accuracy", "credit", "mixed"):
        best, s, scores = best_of_k(row(), comps, scorer_for(mode, H))
        assert best == comps[1] and scores[3] == -1.0 and s == max(scores)


def test_credit_and_accuracy_choose_differently_when_errors_matter_unequally():
    base, lap, frag = completions()
    comps = [json.dumps(lap), json.dumps(frag)]
    acc, cred = scorer_for("accuracy", H), scorer_for("credit", H)
    assert best_of_k(row(), comps, acc)[0] == comps[0]          # accuracy: one wrong class costs less than many small errors
    assert best_of_k(row(), comps, cred)[0] == comps[1]         # credit: the laptop class is what the team needs
    d = selection_disagreement([row()], [comps], acc, cred)
    assert d["disagree_rate"] == 1.0 and d["b_score_of_b_choice"] > d["b_score_of_a_choice"]


def test_build_rows_threshold_stats_and_jsonl(tmp_path):
    base, lap, frag = completions()
    rows = [row(), {**row(), "seed": 4}]
    sample = lambda r, k: [json.dumps(base), "garbage"] if r["seed"] == SEED else ["garbage"] * k
    # seed 4's world differs, so the 'base' (seed 3) report scores differently there; garbage never qualifies
    out, st = build_rft_rows(rows, sample, 2, scorer_for("accuracy", H), min_reward=0.5)
    assert st["prompts"] == 2 and st["kept"] == 1 and len(out) == 1
    assert out[0]["messages"][1]["content"] == json.dumps(base) and out[0]["images"] == ["x.png"] and out[0]["meta"]["score"] == 1.0
    p = tmp_path / "r.jsonl"
    write_sft_jsonl(out, p)
    assert json.loads(p.read_text().splitlines()[0])["meta"]["seed"] == SEED
