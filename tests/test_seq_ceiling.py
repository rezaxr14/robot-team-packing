"""`max_len` in a profile must be backed by a measurement, not by a table someone typed.

`rtp/profiles.py` sets a sequence length per hardware profile and `ops/seq_ceiling.py` measures the
real ceiling (one forward + backward per length, until the card runs out). These tests keep the two
honest: any committed ceiling report must agree with the profile it names, and the probe's own
helpers must not quietly change what they measure.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rtp.profiles import PROFILE_NAMES, get_profile

from ops.seq_ceiling import SAMPLE, main, token_ids

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CEILINGS = sorted(REPORTS.glob("seq_ceiling_*.json"))


def test_there_is_a_ceiling_report_for_the_profile_this_box_runs():
    if not Path("/proc/driver/nvidia/version").exists():
        pytest.skip("needs the box the profile is for")
    assert CEILINGS, "no reports/seq_ceiling_*.json: run ops/seq_ceiling.py on this profile first"


@pytest.mark.parametrize("path", CEILINGS, ids=lambda p: p.name)
def test_a_measured_ceiling_never_sits_below_the_profile_max_len(path):
    r = json.loads(path.read_text())
    if not r.get("profile") or not r.get("longest_fitting"):
        pytest.skip("binding-only or empty report")
    prof = get_profile(r["profile"])
    assert prof.max_len <= r["longest_fitting"], (
        f"{path.name}: profile {prof.name} claims max_len={prof.max_len} but only "
        f"{r['longest_fitting']} tokens fit at batch {r['batch']} on {r['gpu']} "
        f"(lora_r={r.get('lora_r', '?')}); correct the profile or re-measure")


@pytest.mark.parametrize("path", CEILINGS, ids=lambda p: p.name)
def test_a_ceiling_report_records_enough_to_reproduce_it(path):
    r = json.loads(path.read_text())
    for key in ("gpu", "vram_total_gb", "batch", "loss_chunk_tokens", "lengths"):
        assert key in r, f"{path.name}: missing {key}"
    for seq, entry in r["lengths"].items():
        assert seq.isdigit()
        assert entry.get("fits") is True or "error" in entry
        if entry.get("fits"):
            assert 0 < entry["peak_vram_gb"] <= r["vram_total_gb"] + 0.01


def test_token_ids_fill_exactly_the_requested_length():
    class Tok:
        bos_token_id = 151643
        calls = 0

        def __call__(self, text, **kw):
            assert text == SAMPLE and kw["add_special_tokens"] is False
            Tok.calls += 1
            return SimpleNamespace(input_ids=torch.arange(1, 10).unsqueeze(0))

    ids = token_ids(Tok(), 40, batch=2)
    assert ids.shape == (2, 40) and torch.equal(ids[0], ids[1])
    assert ids[0, 0].item() == 151643 and Tok.calls <= 2, "one tokenisation, then tile"


def test_the_ceiling_probe_refuses_to_guess_without_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(["--model", "/nonexistent", "--lengths", "8"]) == 2


def test_every_profile_name_is_still_a_real_profile():
    assert "local_4050" in PROFILE_NAMES
