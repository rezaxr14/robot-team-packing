import json
import os
from pathlib import Path

import pytest

VIKI = os.environ.get("VIKI_R_REPO", "/home/claude/VIKI-R")
pytestmark = pytest.mark.skipif(not Path(VIKI).exists(), reason="VIKI-R clone not available")

from rtp.trackA import load_upstream, summarize_items  # noqa: E402
from rtp.trackA_pipeline import oracle_scene, brain_messages, eye_messages, run_pipeline  # noqa: E402
from tests.test_trackA import GT, row, answer  # noqa: E402


@pytest.fixture(scope="module")
def up():
    return load_upstream(VIKI)


def test_oracle_scene_lists_assets_not_robots():
    sc = oracle_scene(GT)
    assert {s["name"] for s in sc} == {"cup_1", "tray_1"}
    assert [s for s in sc if s["name"] == "cup_1"][0]["position"] == ["table", "shelf"]
    assert [s for s in sc if s["name"] == "tray_1"][0]["position"] == "tray_pos"


def test_brain_messages_are_text_only_and_carry_the_scene(up):
    msgs, gt, task = brain_messages(row(GT), up, "SCENE-X")
    assert isinstance(msgs[1]["content"], str) and "SCENE-X" in msgs[1]["content"] and "plan creator" in msgs[0]["content"]
    assert "image_url" not in json.dumps(msgs)
    none_msgs, _, _ = brain_messages(row(GT), up, None)
    assert "Scene description" not in none_msgs[1]["content"]
    assert eye_messages(row(GT))[0]["content"][0]["type"] == "image_url"


def test_three_conditions_use_their_own_scene_source(up):
    rows = [row(GT) for _ in range(4)]
    seen = []

    def brain(msgs):
        seen.append(msgs[1]["content"])
        return answer("cup")

    eye_calls = []
    items = {}
    items["oracle"] = run_pipeline(rows, up, "oracle", brain, workers=1)
    items["none"] = run_pipeline(rows, up, "none", brain, workers=1)
    items["eye"] = run_pipeline(rows, up, "eye", brain, lambda m: (eye_calls.append(1), '[{"name":"cup_1","position":"table"}]')[1], workers=1)
    assert sum("cup_1" in s and "shelf" in s for s in seen[:4]) == 4                 # oracle shows candidates
    assert not any("Scene description" in s for s in seen[4:8])                      # none: no description
    assert len(eye_calls) == 4 and "table" in items["eye"][0]["scene"]
    for c in items.values():
        assert summarize_items(c, 2)["strict_all"]["k"] == 4
    with pytest.raises(ValueError):
        run_pipeline(rows, up, "eye", brain)


def test_errors_kept_and_decomposition_signal(up):
    rows = [row(GT) for _ in range(6)]

    def smart(msgs):                       # plans well only if it is told where the cup is
        return answer("cup") if "Scene description" in msgs[1]["content"] else "I cannot see anything"

    oracle = summarize_items(run_pipeline(rows, up, "oracle", smart, workers=1), 2)
    none = summarize_items(run_pipeline(rows, up, "none", smart, workers=1), 2)
    assert oracle["strict_all"]["rate"] - none["strict_all"]["rate"] == 1.0          # perception is worth 100pp here
    bad = run_pipeline(rows[:2], up, "eye", smart, lambda m: (_ for _ in ()).throw(RuntimeError("eye down")), workers=1)
    assert all("RuntimeError" in b["error"] for b in bad)
