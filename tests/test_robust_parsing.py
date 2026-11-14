import json

import pytest

from rtp.plan import extract_json, parse_plan, strip_thinking, json_candidates, PlanError
from rtp.eye_rewards import parse_scene_completion
from rtp.llm_client import ChatClient

PLAN = {"tasks": [{"id": "g", "arm": "L", "skill": "grasp", "obj": "a", "params": {"force": 5}},
                  {"id": "p", "arm": "L", "skill": "place", "obj": "a", "params": {"compartment": "main"}, "deps": ["g"]}], "notes": "ok"}
SCENE = [{"name": "book1", "cls": "book", "region": "table_left", "size": 2.0, "mass_guess": 1.0, "fragility_guess": 0.1}]


def test_think_blocks_are_stripped_even_when_they_contain_json():
    draft = json.dumps({"tasks": [{"id": "x", "arm": "L", "skill": "fly"}]})
    reply = f"<think>Let me try {draft} ... no that is wrong, use {{braces}} carefully</think>\n```json\n{json.dumps(PLAN)}\n```"
    assert len(parse_plan(reply).tasks) == 2
    assert strip_thinking("a<think>x</think>b") == "ab" and strip_thinking("answer <think>cut off while reasoning") == "answer "
    with pytest.raises(PlanError):
        parse_plan("<think>" + json.dumps(PLAN))                              # truncated mid-reasoning: no answer exists


def test_last_object_wins_over_earlier_drafts():
    draft, final = {"tasks": []}, PLAN
    reply = f"First attempt: {json.dumps(draft)}\nActually, better: {json.dumps(final)}"
    assert len(parse_plan(reply).tasks) == 2
    assert extract_json('{"tool":"dry_run","args":{}} then {"tasks":[]}', prefer_keys=("tasks",)) == {"tasks": []}
    assert extract_json('{"a":1} {"b":2}') == {"b": 2}                       # no preference: the last object
    assert extract_json('{"tasks":[]} and later {"note":"x"}', prefer_keys=("tasks",)) == {"tasks": []}   # prefers the keyed one


def test_prose_braces_strings_and_trailing_commas():
    reply = 'I will use {curly} words. Plan:\n{"tasks": [{"id": "g", "arm": "L", "skill": "grasp", "obj": "a",},], "notes": "has } and { inside",}'
    p = parse_plan(reply)
    assert p.tasks[0].id == "g" and "} and {" in p.notes
    assert extract_json('{"k": "va\\"lue}"}') == {"k": 'va"lue}'}               # escaped quote and brace inside a string


def test_multiple_fences_and_language_tags():
    reply = "```python\nprint(1)\n```\n```json\n" + json.dumps(PLAN) + "\n```"
    assert len(parse_plan(reply).tasks) == 2
    assert len(parse_plan("```JSON\n" + json.dumps(PLAN) + "\n```").tasks) == 2


@pytest.mark.parametrize("bad", ["", "no braces here", "{unclosed", "{'single': 'quotes'}", "[1, 2]", None, 5])
def test_unparseable_replies_raise_plan_error(bad):
    with pytest.raises(PlanError):
        parse_plan(bad)


def test_scene_parsing_with_think_prose_and_drafts():
    final = json.dumps(SCENE)
    assert parse_scene_completion(f"<think>maybe [1,2,3] or {final}</think>Here you go:\n{final}") == parse_scene_completion(final)
    assert parse_scene_completion("```json\n" + final[:-1] + ",]\n```") is not None                      # trailing comma in a list
    assert parse_scene_completion('{"objects": ' + final + "}")[0]["name"] == "book1"
    bad_first = '[{"name":"x"}]'
    assert parse_scene_completion(bad_first + "\n" + final)[0]["name"] == "book1"                       # invalid draft skipped
    assert parse_scene_completion("<think>" + final) is None and parse_scene_completion("nothing") is None


def test_chat_template_kwargs_are_sent_only_when_asked(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            out = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    try:
        monkeypatch.delenv("RTP_NO_THINK", raising=False)
        ChatClient(url, "m").chat([{"role": "user", "content": "x"}])
        monkeypatch.setenv("RTP_NO_THINK", "1")
        ChatClient(url, "m").chat([{"role": "user", "content": "x"}])
        ChatClient(url, "m", chat_template_kwargs={"k": 1}).chat([{"role": "user", "content": "x"}])
    finally:
        srv.shutdown()
    assert "chat_template_kwargs" not in seen[0] and seen[1]["chat_template_kwargs"] == {"enable_thinking": False} and seen[2]["chat_template_kwargs"] == {"k": 1}


def test_training_rows_fit_the_context_budget():
    """Conservative 2 chars/token. SFT max_length is 4096 tokens; GRPO max_completion_length is 1536 tokens."""
    from pathlib import Path
    from rtp.harness import Harness
    from rtp.datagen import gen_brain_sft, gen_grpo_prompts
    H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
    rows = gen_brain_sft(12, 700, H, noisy_eyes=True)
    for r in rows:
        assert len(r["prompt"]) + len(r["completion"]) <= 2 * 4096 * 0.9, "SFT row may be truncated at max_length"
        assert len(r["completion"]) <= 2 * 1536 * 0.8, "GRPO completion may be cut off mid-JSON"
        assert '"tasks":[' in r["completion"] and '"tasks": ' not in r["completion"]          # compact separators
    assert all(len(g["prompt"]) <= 2 * 4096 * 0.5 for g in gen_grpo_prompts(8, 800, H))


def test_training_and_evaluation_seed_ranges_are_disjoint():
    """Every default training/data seed offset is >= 100000; every default evaluation seed is below it."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    mk = (root / "scripts" / "make_datasets.py").read_text()
    base = int(re.search(r'"--seed0", type=int, default=(\d+)', mk).group(1))
    offsets = [int(x) for x in re.findall(r"a\.seed0 \+ (\d+)", mk)] + [0]
    assert base >= 100000 and all(base + o >= 100000 for o in offsets)
    for f, pat in (("isaac/bringup/06_render_eye_data.py", r'"--seed0"\], dict\(type=int, default=(\d+)'),
                   ("isaac/bringup/07_collect_demos.py", r'"--seed0"\], dict\(type=int, default=(\d+)')):
        assert int(re.search(pat, (root / f).read_text()).group(1)) >= 100000, f
    run = (root / "scripts" / "run_experiments.py").read_text()
    assert int(re.search(r'"--seed0", type=int, default=(\d+)', run).group(1)) < 100000
    assert 'default=0' in (root / "isaac" / "bringup" / "_launch.py").read_text()            # --seed default for evaluation episodes
    plan = (root / "ops" / "plan.py").read_text()
    for m in re.findall(r"--seed (\d+)", plan):
        assert int(m) < 100000
