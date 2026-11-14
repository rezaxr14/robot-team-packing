import json
import os
import tarfile
from pathlib import Path

import pytest

from ops import preflight, stage, bundle

ADA48 = "NVIDIA RTX 6000 Ada Generation, 49140 MiB, 8.9, 550.54.15"
BW96 = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, 12.0, 580.65.06"
BW48 = "NVIDIA RTX PRO 5000 Blackwell, 49140 MiB, 12.0, 575.51"
H100 = "NVIDIA H100 80GB HBM3, 81559 MiB, 9.0, 550.54"


def test_parse_and_arch():
    g = preflight.parse_nvidia_smi(ADA48 + "\n" + BW96)
    assert [x.vram_gb for x in g] == [48.0, 95.6] and g[0].cc == (8, 9) and g[1].cc == (12, 0)
    assert [preflight.arch_of(x) for x in g] == ["ada", "blackwell"]
    assert preflight.parse_nvidia_smi("garbage") == []
    nocc = preflight.parse_nvidia_smi("NVIDIA RTX PRO 6000 Blackwell, 97887 MiB, [N/A], 580.1")[0]
    assert nocc.cc is None and preflight.arch_of(nocc) == "blackwell"            # falls back to the name


def test_recommendations_branch_on_architecture_and_memory():
    ada = preflight.recommend(preflight.parse_nvidia_smi(ADA48)[0])
    b96 = preflight.recommend(preflight.parse_nvidia_smi(BW96)[0])
    b48 = preflight.recommend(preflight.parse_nvidia_smi(BW48)[0])
    assert ada["arch"] == "ada" and b96["arch"] == b48["arch"] == "blackwell"
    assert ada["isaac_line"] == "2.3" and ada["torch_index"] == "cu128"
    assert any("predate Blackwell" in n for n in b48["notes"])                   # the upstream pins warning
    assert not ada["can_run_isaac_vllm_and_train_together"] and b96["can_run_isaac_vllm_and_train_together"]
    assert b96["vllm_gpu_memory_utilization_4b"] < ada["vllm_gpu_memory_utilization_4b"] <= 0.5
    old = preflight.recommend(preflight.parse_nvidia_smi(BW96.replace("580.65.06", "535.1"))[0])
    assert any("driver" in w for w in old["warnings"])


def test_datacenter_gpu_without_rt_cores_is_a_hard_fail():
    r = preflight.recommend(preflight.parse_nvidia_smi(H100)[0])
    assert any(w.startswith("FAIL") for w in r["warnings"])


def test_stage_runner_logs_resumes_and_reports_failures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stage, "LOG", Path("reports/stage_log.jsonl"))
    assert stage.run_stage("hello", ["python3", "-c", "print('hi')"]) == 0
    assert stage.run_stage("hello", ["python3", "-c", "raise SystemExit(5)"], skip_if_done=True) == 0     # skipped
    assert stage.run_stage("bad", ["python3", "-c", "import sys; print('boom'); sys.exit(3)"]) == 3
    lat = stage.latest()
    assert lat["hello"]["returncode"] == 0 and lat["bad"]["returncode"] == 3 and "boom" in lat["bad"]["tail"][-1]
    assert Path(lat["bad"]["log"]).read_text().strip() == "boom"
    assert "| bad | 3 |" in stage.status_table()
    assert stage.main(["run", "viacli", "--", "python3", "-c", "print(1)"]) == 0


def test_bundle_collects_small_files_excludes_items_and_summarizes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rep = Path("reports"); (rep / "trackA").mkdir(parents=True); (rep / "logs").mkdir()
    (rep / "trackA" / "L2_summary.json").write_text(json.dumps({"label": "L2", "full": {
        "n_scored": 10, "n_errors": 0, "format": {"rate": 1.0}, "strict_draw": {"rate": 0.5}, "strict_all": {"rate": 0.4}}}))
    (rep / "trackA" / "L2_items.jsonl").write_text("x" * 100)
    (rep / "logs" / "a.log").write_text("log")
    (rep / "big.json").write_text("x" * 3_000_000)
    (rep / "tables.md").write_text("| a |\n|---|\n")
    (rep / "stage_log.jsonl").write_text(json.dumps({"name": "s1", "returncode": 2, "seconds": 3, "tail": ["Traceback", "boom"]}) + "\n")
    files = [str(f) for f in bundle.collect(rep)]
    assert any("L2_summary.json" in f for f in files) and not any("_items" in f or "big.json" in f or "a.log" in f for f in files)
    s = bundle.summarize(rep)
    assert "strict_all=0.4" in s and "FAILED: s1" in s and "boom" in s
    assert bundle.main(["--reports", "reports", "--out", "out.tar.gz"]) == 0
    with tarfile.open("out.tar.gz") as tf:
        names = tf.getnames()
    assert any(n.endswith("L2_summary.json") for n in names) and not any("items" in n for n in names)


def test_plan_commands_reference_real_files_and_names_are_unique():
    import re
    from ops import plan
    root = Path(__file__).resolve().parents[1]
    names = [n for _, n, _ in plan.all_stages()]
    assert len(names) == len(set(names))
    for g, n, cmd in plan.all_stages():
        for m in re.findall(r"(?:ops|scripts|train|isaac/bringup)/[\w./-]+\.(?:sh|py)", cmd):
            assert (root / m).exists(), f"{n}: {m} does not exist"
    assert {"setup", "smoke", "trackA", "isaac", "data", "train", "eval", "bundle"} == set(plan.GROUPS)


def test_plan_run_executes_stages_in_order_and_stops_on_failure(tmp_path, monkeypatch):
    from ops import plan
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stage, "LOG", Path("reports/stage_log.jsonl"))
    monkeypatch.setitem(plan.GROUPS, "bundle", [("ok1", "echo one"), ("bad", "exit 4"), ("never", "echo never")])
    assert plan.main(["run", "bundle"]) == 4
    lat = stage.latest()
    assert lat["ok1"]["returncode"] == 0 and lat["bad"]["returncode"] == 4 and "never" not in lat
    monkeypatch.setitem(plan.GROUPS, "bundle", [("ok1", "echo changed")])
    assert plan.main(["run", "bundle"]) == 0 and "changed" not in " ".join(stage.latest()["ok1"]["tail"])   # skipped (done)
    assert plan.main(["run", "bundle", "--force"]) == 0


def test_lf_config_writer(tmp_path):
    import json
    from ops.make_lf_config import build
    (tmp_path / "eye").mkdir()
    cfg = build(str(tmp_path / "eye"), "/m/Qwen2.5-VL-3B", "qwen2_vl", str(tmp_path / "out"))
    y = cfg.read_text()
    assert "template: qwen2_vl" in y and "dataset: eye_sft" in y and f"dataset_dir: {tmp_path / 'eye'}" in y
    info = json.loads((tmp_path / "eye" / "dataset_info.json").read_text())
    assert info["eye_sft"]["columns"]["images"] == "images" and info["eye_sft"]["formatting"] == "sharegpt"


def test_lf_config_merges_datasets_and_names_them(tmp_path):
    import json
    from ops.make_lf_config import build
    (tmp_path / "eye").mkdir()
    build(str(tmp_path / "eye"), "/m", "qwen2_vl", str(tmp_path / "o1"))
    cfg = build(str(tmp_path / "eye"), "/m2", "qwen2_vl", str(tmp_path / "o2"), file_name="rft_credit.jsonl", name="eye_rft_credit")
    info = json.loads((tmp_path / "eye" / "dataset_info.json").read_text())
    assert set(info) == {"eye_sft", "eye_rft_credit"} and info["eye_rft_credit"]["file_name"] == "rft_credit.jsonl"
    assert "dataset: eye_rft_credit" in cfg.read_text()


def test_plan_lint_is_clean_and_catches_problems(monkeypatch):
    from ops import plan
    assert plan.lint() == []
    bad = list(plan.GROUPS["train"]) + [("oops_stage", "source ops/env.sh; python x.py --model $RUNS/never_made --data $NOPE_VAR")]
    monkeypatch.setitem(plan.GROUPS, "train", bad)
    probs = plan.lint()
    assert any("undefined variable $NOPE_VAR" in p for p in probs) and any("$RUNS/never_made" in p for p in probs)


def test_missing_vars_blocks_stages_that_need_config(tmp_path, monkeypatch):
    from ops import plan
    assert plan.missing_vars("trackA_L2_id", {}) == ["VIKI_CKPT_L2", "L2_ID"]
    assert plan.missing_vars("trackA_L2_id", {"VIKI_CKPT_L2": "/m", "L2_ID": "/d.parquet"}) == []
    assert plan.missing_vars("env_train", {}) == []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stage, "LOG", Path("reports/stage_log.jsonl"))
    monkeypatch.setattr(plan, "_config_env", lambda: {})
    monkeypatch.setitem(plan.GROUPS, "bundle", [("trackA_L2_id", "echo should-not-run")])
    assert plan.main(["run", "bundle"]) == 2 and "trackA_L2_id" not in stage.latest()       # refused before running
