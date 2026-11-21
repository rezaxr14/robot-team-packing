"""CPU-only tests for the R1 bundle manifest helper and the honest USD fallback.

No isaaclab / warp / torch import happens here: ``ops.render_views`` keeps all GPU
imports inside functions, and the fallback is patched in the one CLI test. These
tests check determinism, tamper detection and that fps is never claimed without a
real measurement.
"""
import hashlib
import json

import pytest

from ops import render_views as rv


def _bundle(tmp_path):
    """A tiny stand-in for a rendered bundle (two jpgs + two json sidecars)."""
    (tmp_path / "views").mkdir()
    blobs = {
        "views/s03_z1_p0_t0_start_rgb.jpg": b"fake-rgb-bytes",
        "views/s03_z1_p0_t0_start_depth.jpg": b"fake-depth-bytes",
    }
    for rel, data in blobs.items():
        (tmp_path / rel).write_bytes(data)
    (tmp_path / "visibility.json").write_text(json.dumps({"v": {"visible": True}}))
    (tmp_path / "objects.json").write_text(json.dumps([{"package": "mug"}]))
    return sorted(list(blobs) + ["visibility.json", "objects.json"])


def test_build_manifest_is_deterministic(tmp_path):
    files = _bundle(tmp_path)
    versions = {"newton": "1.6.1", "warp-lang": "1.17.0", "isaaclab": "28.0.0"}
    m1 = rv.build_manifest(tmp_path, files, versions)
    m2 = rv.build_manifest(tmp_path, files, dict(reversed(list(versions.items()))))
    assert m1 == m2
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
    assert list(m1["files"]) == sorted(files)
    rel = "views/s03_z1_p0_t0_start_rgb.jpg"
    assert m1["files"][rel] == hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()
    assert len(m1["sha256"]) == 64
    # Bundle hash covers bytes only, so it is independent of any fps number.
    assert rv.build_manifest(tmp_path, files, versions,
                             fps_per_eye={"eye_A": 12.5})["sha256"] == m1["sha256"]


def test_manifest_detects_tampering(tmp_path):
    files = _bundle(tmp_path)
    m = rv.build_manifest(tmp_path, files, {"newton": "1.6.1"})
    assert rv.verify_manifest(tmp_path, m) == []

    # 1) a file is edited after the manifest was built
    edited = tmp_path / "views/s03_z1_p0_t0_start_rgb.jpg"
    edited.write_bytes(b"TAMPERED")
    assert "views/s03_z1_p0_t0_start_rgb.jpg" in rv.verify_manifest(tmp_path, m)

    # 2) a file is deleted
    edited.unlink()
    assert "views/s03_z1_p0_t0_start_rgb.jpg" in rv.verify_manifest(tmp_path, m)

    # 3) the digest map inside the manifest itself is edited
    m2 = json.loads(json.dumps(m))
    m2["files"]["objects.json"] = "0" * 64
    assert "objects.json" in rv.verify_manifest(tmp_path, m2)
    assert "<manifest.sha256>" in rv.verify_manifest(tmp_path, m2)


def test_no_fps_is_declared_without_measured_numbers(tmp_path):
    files = _bundle(tmp_path)
    m = rv.build_manifest(tmp_path, files, {})
    assert m["fps_per_eye"] is None
    assert m["fps_measured"] is False
    assert m["fps_threshold"] is None
    with pytest.raises(rv.FpsNotMeasured):
        rv.fps_verdict(m)

    # Bogus "measurements" are refused rather than rounded into a claim.
    for bogus in ({}, {"eye_A": 0}, {"eye_A": -3}, {"eye_A": None},
                  {"eye_A": "fast"}, {"eye_A": float("nan")}, {"eye_A": float("inf")}):
        with pytest.raises(rv.FpsNotMeasured):
            rv.build_manifest(tmp_path, files, {}, fps_per_eye=bogus)


def test_fps_verdict_uses_only_real_numbers(tmp_path):
    files = _bundle(tmp_path)
    m = rv.build_manifest(tmp_path, files, {}, fps_per_eye={"eye_A": 11.2, "eye_B": 8.4})
    assert m["fps_measured"] is True
    assert m["fps_per_eye"] == {"eye_A": 11.2, "eye_B": 8.4}
    assert rv.fps_verdict(m) == "FAIL"          # eye_B below the G-8 threshold, honestly
    ok = rv.build_manifest(tmp_path, files, {}, fps_per_eye={"eye_A": 12.0, "eye_B": 10.5})
    assert rv.fps_verdict(ok) == "PASS"

    # Claiming a measurement with no numbers cannot fool the verdict helper.
    forged = dict(ok)
    forged["fps_per_eye"] = None
    with pytest.raises(rv.FpsNotMeasured):
        rv.fps_verdict(forged)


def test_preflight_requires_isaacsim(monkeypatch):
    monkeypatch.setattr(rv.importlib.util, "find_spec", lambda name: None)
    ok, reason = rv.preflight_usd_scene()
    assert ok is False
    assert reason == rv.MJCF_USD_UNAVAILABLE
    assert reason == "MJCF->USD conversion unavailable in Newton-only venv"


def test_fallback_cli_writes_fail_and_no_images(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "preflight_usd_scene", lambda: (False, rv.MJCF_USD_UNAVAILABLE))
    out = tmp_path / "reports" / "scene_0" / "render_views.json"
    rc = rv.main(["--seed", "0", "--out", str(out)])
    assert rc == 1
    report = json.loads(out.read_text())
    assert report["result"] == "FAIL"
    assert report["reason"] == rv.MJCF_USD_UNAVAILABLE
    assert report["images_written"] == 0
    assert report["fps_per_eye"] is None
    assert report["fps_measured"] is False
    assert not list(out.parent.rglob("*.jpg"))
    assert not (out.parent / "views").exists()
