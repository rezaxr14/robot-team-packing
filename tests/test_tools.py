import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from rtp.harness import Harness
from rtp.experiments import ExpConfig, run_config, summarize, paired_compare, default_grid, write_records, read_records
from rtp.llm_client import ChatClient, ChatError
from rtp import tables
from rtp.attribution_eval import sample_for_labeling, write_label_csv, agreement

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")


class _Handler(BaseHTTPRequestHandler):
    fail_first = 0
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append((self.path, body, self.headers.get("Authorization")))
        if _Handler.fail_first > 0:
            _Handler.fail_first -= 1
            self.send_response(503); self.end_headers(); return
        out = json.dumps({"choices": [{"message": {"content": "echo:" + body["messages"][-1]["content"]}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    _Handler.seen, _Handler.fail_first = [], 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_chat_client_roundtrip_and_retry(server):
    c = ChatClient(server, "m", retries=3)
    _Handler.fail_first = 2
    assert c.chat([{"role": "user", "content": "hi"}], max_tokens=7) == "echo:hi"
    path, body, auth = _Handler.seen[-1]
    assert path.endswith("/chat/completions") and body["model"] == "m" and body["max_tokens"] == 7 and auth == "Bearer EMPTY"
    gen = c.generate(system="sys")
    assert gen("p") == "echo:p" and _Handler.seen[-1][1]["messages"][0] == {"role": "system", "content": "sys"}


def test_chat_client_gives_up():
    with pytest.raises(ChatError):
        ChatClient("http://127.0.0.1:9/v1", "m", retries=0, timeout=1).chat([{"role": "user", "content": "x"}])


def test_llm_brain_failure_is_recorded_not_crashing():
    recs = run_config(ExpConfig("llm_bad", brain="llm"), range(3), H, generate=lambda p: "garbage")
    assert all(not r["success"] for r in recs) and all("Brain" in r["fatal_agents"] or r["missing"] for r in recs)


def test_run_config_summary_and_paired(tmp_path):
    cfgs = [ExpConfig("full", noisy_eyes=True), ExpConfig("one_eye", n_eyes=1, noisy_eyes=True)]
    recs = sum((run_config(c, range(10), H) for c in cfgs), [])
    write_records(recs, tmp_path / "r.jsonl")
    assert read_records(tmp_path / "r.jsonl") == recs
    s = summarize(recs)
    assert s["full"]["n"] == 10 and 0 <= s["full"]["success"]["lo"] <= s["full"]["success"]["hi"] <= 1
    assert s["full"]["gap_plan_minus_success"] == s["full"]["plan_pass"]["rate"] - s["full"]["success"]["rate"]
    cmp_ = paired_compare(recs, "one_eye", "full")
    assert cmp_["n"] == 10 and cmp_["lo"] <= cmp_["mean_diff"] <= cmp_["hi"]
    assert "| full |" in tables.table_main(recs) and "one_eye" in tables.table_ablation(recs)
    assert "failed episodes" in tables.table_attribution(recs)


def test_default_grid_names_unique():
    names = [c.name for c in default_grid()]
    assert len(names) == len(set(names)) and "full" in names


def test_attribution_labeling_roundtrip(tmp_path):
    recs = run_config(ExpConfig("noisy", noisy_eyes=True, n_eyes=1), range(20), H, keep_story=True)
    rows = sample_for_labeling(recs, 5)
    assert rows and "EYE ERRORS" in rows[0]["story"] and "INSTRUCTION" in rows[0]["story"]
    p = tmp_path / "l.csv"
    write_label_csv(rows, p)
    assert agreement(p) == {"n": 0}                                   # nothing labelled yet
    lines = list(csv.DictReader(open(p)))
    for r in lines:
        r["human_agent"] = r["auto_agents"].split("|")[0] or "Unclear"  # pretend the human agrees
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0])); w.writeheader(); w.writerows(lines)
    a = agreement(p)
    assert a["n"] == len(rows) and 0 <= a["accuracy"] <= 1
    lines[0]["human_agent"] = "Nonsense"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0])); w.writeheader(); w.writerows(lines)
    with pytest.raises(ValueError):
        agreement(p)
