import json
from pathlib import Path

import pytest

from rtp.harness import Harness
from rtp.experiments import default_grid, verify_grid, run_config, write_records, ExpConfig
from rtp.credit import credit_by_fault, FAULTS

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
ROOT = Path(__file__).resolve().parents[1]


def build_reports(tmp):
    rep = tmp / "reports"
    for sub in ("sym", "isaac", "credit", "trackA", "trackA_pipeline"):
        (rep / sub).mkdir(parents=True)
    cfgs = [c for c in default_grid() if c.name in ("full", "probe", "one_eye")]
    write_records(sum((run_config(c, range(6), H) for c in cfgs), []), rep / "sym" / "records.jsonl")
    iso = []
    for tag in ("isaac_scripted_arm", "isaac_learned_flow"):
        iso += run_config(ExpConfig(tag, noisy_eyes=(tag != "isaac_scripted_arm")), range(6), H)
    write_records(iso, rep / "isaac" / "episodes.jsonl")
    (rep / "isaac" / "regression_scripted.json").write_text(json.dumps({"passed": 17, "total": 18, "cases": {"a": {"passed": True}, "b": {"passed": False}}}))
    (rep / "credit" / "credit.json").write_text(json.dumps(credit_by_fault(H, range(4), {k: FAULTS[k] for k in ("mass_noise",)}), default=str))
    f = lambda r: {"k": int(10 * r), "n": 10, "rate": r, "lo": max(0, r - .2), "hi": min(1, r + .2)}
    (rep / "trackA" / "L2_id_summary.json").write_text(json.dumps({"label": "L2_id", "level": 2, "model": "viki", "full": {
        "n_scored": 10, "n_errors": 0, "format": f(1.0), "strict_draw": f(0.6), "strict_all": f(0.5), "items_position_dependent": 2}}))
    (rep / "trackA_pipeline" / "L2_pipeline_summary.json").write_text(json.dumps({"label": "L2", "conditions": {
        "oracle": {"n_scored": 10, "strict_draw": f(0.8), "strict_all": f(0.7)}, "none": {"n_scored": 10, "strict_draw": f(0.2), "strict_all": f(0.1)}}}))
    ex = tmp / "data"; ex.mkdir()
    (ex / "hindsight_funnel.json").write_text(json.dumps({"episodes": 40, "failed": 6, "relabeled": 6, "validated": 6}))
    return rep, ex


def test_report_has_ledger_levels_sections_and_lists_missing(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_report", ROOT / "scripts" / "build_report.py")
    br = importlib.util.module_from_spec(spec); spec.loader.exec_module(br)
    rep, ex = build_reports(tmp_path)
    assert br.main(["--reports", str(rep), "--extra", str(ex), "--figures"]) == 0
    txt = (rep / "results.md").read_text()
    assert "## Evidence ledger" in txt
    for level in ("[TRACK A]", "[ISAAC]", "[FIXTURE]"):
        assert level in txt
    assert "NOT evidence for any claim" in txt                              # the fixture scope sentence is always attached
    assert "Paired comparisons" in txt and "isaac_learned_flow" in txt and "16/18" not in txt and "17/18" in txt
    assert "perception value = oracle - none" in txt and "Hindsight relabeling funnel" in txt
    assert "Not found" in txt and "llm_tools/records.jsonl" in txt and "verify/records.jsonl" in txt    # missing inputs are named
    assert "Failing: b" in txt
    for fig in ("sym_success.png", "sym_gap.png", "sym_tokens.png", "credit.png", "isaac_success.png"):
        p = rep / "figures" / fig
        assert p.exists() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", fig


def test_figures_trackA_and_empty_inputs(tmp_path):
    from rtp import figures
    s = {"label": "L2_id", "level": 2, "full": {"strict_draw": {"rate": .5}, "strict_all": {"rate": .4}}}
    p = figures.fig_trackA([s, {"level": 1}], tmp_path / "t.png")
    assert Path(p).read_bytes()[:4] == b"\x89PNG"


def test_report_on_an_empty_reports_dir_lists_everything_missing(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_report", ROOT / "scripts" / "build_report.py")
    br = importlib.util.module_from_spec(spec); spec.loader.exec_module(br)
    (tmp_path / "r").mkdir()
    assert br.main(["--reports", str(tmp_path / "r")]) == 0
    txt = (tmp_path / "r" / "results.md").read_text()
    assert "Not found" in txt and "trackA/*_summary.json" in txt and "isaac/episodes.jsonl" in txt
