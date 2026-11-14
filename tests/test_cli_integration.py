"""Run the command-line scripts as real processes against a fake OpenAI server and synthetic VIKI-style parquet files."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from ops.fake_server import FakeServer
from tests.test_trackA import GT, answer

ROOT = Path(__file__).resolve().parents[1]
VIKI = os.environ.get("VIKI_R_REPO", "/home/claude/VIKI-R")
needs_viki = pytest.mark.skipif(not Path(VIKI).exists(), reason="VIKI-R clone not available")


def run(*args, timeout=300, env=None):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **(env or {})})


def viki_parquet(path, n=6, gt=GT, with_ids=False):
    rows = []
    for i in range(n):
        g = dict(gt, task_id=f"t{i}")
        rows.append({"prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "<image>Put the cup on the tray."}],
                     "reward_model": {"ground_truth": g}, "images": [{"bytes": b"\x89PNGfake" + bytes([i])}]})
    pd.DataFrame(rows).to_parquet(path)
    return path


@needs_viki
def test_trackA_eval_cli_end_to_end(tmp_path):
    data = viki_parquet(tmp_path / "l2.parquet")
    with FakeServer(lambda m, b: answer("cup")) as url:
        r = run("scripts/s0_trackA_eval.py", "--repo", VIKI, "--level", "2", "--data", str(data), "--endpoint", url,
                "--model", "m", "--label", "L2_t", "--out-dir", str(tmp_path / "out"), "--upstream-truncation", "--workers", "2")
    assert r.returncode == 0, r.stderr[-1500:]
    s = json.loads((tmp_path / "out" / "L2_t_summary.json").read_text())
    assert s["full"]["n_scored"] == 6 and s["full"]["n_errors"] == 0 and s["full"]["strict_all"]["k"] == 6
    assert s["upstream_subset"]["first_n"] == 900 and s["level"] == 2 and s["model"] == "m"
    items = [json.loads(l) for l in (tmp_path / "out" / "L2_t_items.jsonl").read_text().splitlines()]
    assert len(items) == 6 and all(i["strict_all"] and "response" in i for i in items)


@needs_viki
def test_trackA_eval_cli_records_server_errors_instead_of_crashing(tmp_path):
    data = viki_parquet(tmp_path / "l2.parquet", n=3)

    def boom(m, b):
        raise RuntimeError("model down")

    with FakeServer(boom) as url:
        r = run("scripts/s0_trackA_eval.py", "--repo", VIKI, "--level", "2", "--data", str(data), "--endpoint", url,
                "--model", "m", "--label", "e", "--out-dir", str(tmp_path / "o"), "--workers", "1")
    assert r.returncode == 0, r.stderr[-800:]
    s = json.loads((tmp_path / "o" / "e_summary.json").read_text())
    assert s["full"]["n_errors"] == 3 and s["full"]["n_scored"] == 0


@needs_viki
def test_trackA_pipeline_cli_three_conditions(tmp_path):
    data = viki_parquet(tmp_path / "l2.parquet", n=4)
    brain = lambda m, b: answer("cup") if "Scene description" in m[1]["content"] else "I see nothing"
    eye = lambda m, b: '[{"name":"cup_1","position":"table"}]'
    with FakeServer(brain) as burl, FakeServer(eye) as eurl:
        r = run("scripts/s0_trackA_pipeline_eval.py", "--repo", VIKI, "--data", str(data), "--label", "P", "--brain-endpoint", burl,
                "--brain-model", "b", "--eye-endpoint", eurl, "--eye-model", "e", "--conditions", "oracle,none,eye",
                "--out-dir", str(tmp_path / "po"), "--workers", "2")
    assert r.returncode == 0, r.stderr[-1500:]
    s = json.loads((tmp_path / "po" / "P_pipeline_summary.json").read_text())
    rate = lambda c: s["conditions"][c]["strict_all"]["rate"]
    assert rate("oracle") == 1.0 and rate("eye") == 1.0 and rate("none") == 0.0           # perception is worth everything here


def test_inventory_cli_reports_schema_and_position_exposure(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    viki_parquet(d / "train.parquet", n=5)
    viki_parquet(d / "test.parquet", n=3, gt={**GT, "init_pos": {"R1": ["R1"], "cup_1": ["table"], "tray_1": ["tray_pos"]}})
    r = run("scripts/s0_inventory.py", "--data", str(d))
    assert r.returncode == 0, r.stderr[-800:]
    assert "2 parquet files" in r.stdout and "reward_model" in r.stdout
    assert "rows with a parsable init_pos: 8" in r.stdout and "5 (62.5%)" in r.stdout          # only the 5 train rows have >1 start


def test_run_experiments_cli_against_an_llm_endpoint_records_brain_failures(tmp_path):
    with FakeServer(lambda m, b: "I refuse to plan") as url:
        r = run("scripts/run_experiments.py", "--seeds", "2", "--out", str(tmp_path / "r"), "--llm-endpoint", url, "--llm-model", "x", "--label-sample", "2")
    assert r.returncode == 0, r.stderr[-1500:]
    recs = [json.loads(l) for l in (tmp_path / "r" / "records.jsonl").read_text().splitlines()]
    assert recs and all(not x["success"] for x in recs) and any("Brain" in x["fatal_agents"] or x["missing"] for x in recs)
    assert (tmp_path / "r" / "tables.md").exists() and (tmp_path / "r" / "attribution_to_label.csv").exists()


def test_run_experiments_cli_tool_brain_flag_uses_the_chat_endpoint(tmp_path):
    with FakeServer(lambda m, b: '{"tool": "query_scene", "args": {}}') as url:
        r = run("scripts/run_experiments.py", "--seeds", "1", "--out", str(tmp_path / "t"), "--llm-endpoint", url, "--llm-model", "x",
                "--tool-brain", "--grid", "verify")
    assert r.returncode == 0, r.stderr[-1200:]
    recs = [json.loads(l) for l in (tmp_path / "t" / "records.jsonl").read_text().splitlines()]
    assert recs and recs[0]["brain_totals"]["turns"] >= 1


def test_eye_rft_cli_end_to_end(tmp_path):
    from rtp.datagen import sample_world
    from rtp.perception import oracle_report
    (tmp_path / "x.png").write_bytes(b"\x89PNGfake")
    rows = [{"prompt": "<image>list", "images": [str(tmp_path / "x.png")], "seed": 3, "view": "A"}]
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    good = json.dumps(oracle_report(sample_world(3), "A").content["objects"])
    replies = iter([good, "garbage", good.replace('"laptop"', '"book"')] * 4)
    with FakeServer(lambda m, b: next(replies)) as url:
        r = run("scripts/eye_rft.py", "--rows", str(tmp_path / "rows.jsonl"), "--endpoint", url, "--model", "eye", "--mode", "accuracy",
                "--k", "3", "--out", str(tmp_path / "rft.jsonl"))
    assert r.returncode == 0, r.stderr[-1200:]
    out = [json.loads(l) for l in (tmp_path / "rft.jsonl").read_text().splitlines()]
    st = json.loads((tmp_path / "rft.jsonl.stats.json").read_text())
    assert len(out) == 1 and out[0]["messages"][1]["content"] == good and out[0]["meta"]["score"] == 1.0
    assert st["prompts"] == 1 and st["kept"] == 1 and st["accuracy_vs_credit"]["n"] == 1


def test_preflight_and_env_report_clis_run_without_a_gpu(tmp_path):
    r = run("ops/preflight.py")                                           # no GPU here: must report it, not crash
    assert r.returncode in (0, 1) and "== GPUs ==" in r.stdout and "RESULT:" in r.stdout
    r2 = run("scripts/s0_env_report.py")
    assert r2.returncode == 0 and "packages" in r2.stdout and json.loads(r2.stdout)["python"].startswith("3.")


def test_credit_analysis_and_report_clis_chain(tmp_path):
    r = run("scripts/run_credit_analysis.py", "--seeds", "4", "--out", str(tmp_path / "reports" / "credit"))
    assert r.returncode == 0, r.stderr[-800:]
    assert (tmp_path / "reports" / "credit" / "credit.json").exists() and "| omission |" in r.stdout
    r2 = run("scripts/build_report.py", "--reports", str(tmp_path / "reports"), "--figures")
    assert r2.returncode == 0, r2.stderr[-800:]
    txt = (tmp_path / "reports" / "results.md").read_text()
    assert "Counterfactual Eye credit" in txt and (tmp_path / "reports" / "figures" / "credit.png").exists()


def test_plan_and_lf_config_clis():
    r = run("ops/plan.py", "list")
    assert r.returncode == 0 and "[trackA]" in r.stdout and "isaac_01_scene" in r.stdout
    assert run("ops/plan.py", "lint").stdout.strip() == "plan lint: ok"


def test_lf_config_cli(tmp_path):
    (tmp_path / "eye").mkdir()
    r = run("ops/make_lf_config.py", "--data", str(tmp_path / "eye"), "--model", "/m", "--out", str(tmp_path / "o"),
            "--file", "rft_credit.jsonl", "--name", "eye_rft_credit")
    assert r.returncode == 0, r.stderr[-500:] and (tmp_path / "o" / "eye_sft.yaml").exists()
    assert "dataset: eye_rft_credit" in (tmp_path / "o" / "eye_sft.yaml").read_text()


def test_pipeline_check_and_dataset_clis(tmp_path):
    r = run("scripts/make_datasets.py", "--out", str(tmp_path / "d"), "--brain-sft", "3", "--grpo", "3", "--hindsight", "2", "--eye-grpo", "2", "--eye", "0")
    assert r.returncode == 0, r.stderr[-600:]
    names = {p.name for p in (tmp_path / "d").iterdir()}
    assert {"brain_sft.jsonl", "grpo.jsonl", "hindsight_sft.jsonl", "hindsight_funnel.json", "eye_grpo.jsonl"} <= names
