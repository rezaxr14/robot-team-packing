import json
import os
from pathlib import Path

import pytest

VIKI = os.environ.get("VIKI_R_REPO", "")
pytestmark = pytest.mark.skipif(not VIKI or not Path(VIKI).exists(), reason="VIKI-R clone not available")

from rtp.trackA import load_upstream, build_messages, evaluate_rows, summarize_items, score_l2, score_l1  # noqa: E402

GT = {"robots": {"R1": "fetch"}, "task_id": "t1",
      "init_pos": {"R1": ["R1"], "cup_1": ["table", "shelf"], "tray_1": ["tray_pos"]},
      "goal_constraints": [],
      "temporal_constraints": [[[{"type": "asset", "name": "cup", "is_satisfied": True, "status": {"pos.name": "tray"}}]]],
      "time_steps": [{"step": i, "actions": {"R1": ["x"]}} for i in range(1, 6)]}


def answer(first_target):
    steps = [["Move", first_target], ["Reach", "cup"], ["Grasp", "cup"], ["Move", "tray"], ["Place", "tray"]]
    return "<think>t</think><answer>" + json.dumps([{"step": i + 1, "actions": {"R1": s}} for i, s in enumerate(steps)]) + "</answer>"


def row(gt, level=2):
    return {"prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "<image>Put the cup on the tray."}],
            "reward_model": {"ground_truth": gt if level == 2 else str(gt)}, "images": [{"bytes": b"\x89PNGfake"}]}


@pytest.fixture(scope="module")
def up():
    return load_upstream(VIKI)


def test_prompts_extracted_and_messages_built(up):
    msgs, gt, task = build_messages(2, row(GT), up)
    assert "plan creator" in msgs[0]["content"] and "fetch" in msgs[0]["content"]
    assert task == "Put the cup on the tray." and msgs[1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    m1, _, _ = build_messages(1, row(["fetch", "panda"], 1), up)
    assert "Possible robots" in m1[0]["content"] and "Stompy" in m1[0]["content"]


def test_l2_scoring_separates_draw_from_position_robust(up):
    robust = score_l2(answer("cup"), GT, up, idx=0)
    assert robust["strict_all"] and robust["strict_draw"] and robust["format"] == 1.0 and robust["upstream_score"] == 1.0
    fragile = [score_l2(answer("table"), GT, up, idx=i) for i in range(20)]
    assert not any(f["strict_all"] for f in fragile)                       # never robust
    assert all(f["position_dependent"] and f["feasible_any"] for f in fragile)
    draws = {f["strict_draw"] for f in fragile}
    assert draws == {True, False}                                          # upstream-style single draw flips
    assert fragile[3] == score_l2(answer("table"), GT, up, idx=3)          # but is repeatable per idx


def test_l2_length_and_parse_failures(up):
    too_long = answer("cup").replace('"step": 5', '"step": 5').replace("]}}]", "]}}, {\"step\": 6, \"actions\": {\"R1\": [\"Move\", \"cup\"]}}]")
    r = score_l2(too_long, GT, up, 0)
    assert r["feasible_all"] and not r["length_ok"] and not r["strict_all"]
    for bad in ("", "no tags", "<answer>not a list</answer>", "<think>x</think><answer>[]</answer>"):
        s = score_l2(bad, GT, up, 0)
        assert not s["strict_all"] and not s["parsed"]


def test_l1_scoring(up):
    assert score_l1("<think>a</think><answer>['panda','fetch']</answer>", str(["fetch", "panda"]), up)["correct"]
    assert not score_l1("<think>a</think><answer>['panda']</answer>", str(["fetch", "panda"]), up)["correct"]


def test_evaluate_rows_with_fake_model_keeps_errors(up):
    rows = [row(GT) for _ in range(6)]
    calls = {"n": 0}

    def chat(msgs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return answer("cup")

    items = evaluate_rows(rows, 2, up, chat, workers=1)
    s = summarize_items(items, 2)
    assert s["n_items"] == 6 and s["n_errors"] == 1 and s["n_scored"] == 5
    assert s["strict_all"]["k"] == 5 and s["strict_all"]["n"] == 5 and s["strict_all"]["lo"] > 0.5
    assert [x for x in items if "error" in x][0]["error"].startswith("RuntimeError")
    s1 = summarize_items(evaluate_rows([row(["fetch"], 1)], 1, up, lambda m: "<think>a</think><answer>['fetch']</answer>"), 1)
    assert s1["correct"]["k"] == 1
