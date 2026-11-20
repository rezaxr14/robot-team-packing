"""Phase 1 cache: resolution order, manifest refusal, profile isolation, kill/resume, export stub."""
import json
import os
from pathlib import Path

import pytest

from rtp import assetcache as ac
from rtp.profiles import get_profile

SPEC = ac.model_spec("Qwen/Qwen3.5-4B", revision="26a93f06deadbeef", name="qwen35-4b")


def make_asset(root: Path, files: dict, spec: ac.AssetSpec = SPEC) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    ac.write_manifest(root, spec)
    return root


def prof(tmp_path: Path, name: str = "local_4050"):
    return get_profile(name, root=tmp_path)


def test_slugs_match_the_plan():
    assert SPEC.slug == "rtp-model-qwen35-4b-26a93f06"
    assert ac.model_spec("x", revision="", name="Qwen 3.5").slug == "rtp-model-qwen-3.5-local"
    wheels = ac.wheelhouse_spec(python="3.12", torch="2.15.0", cuda="13.2", sm="89")
    assert wheels.slug == "rtp-wheels-py3.12-torch2.15.0-cu13.2-sm89"
    p = prof(Path("unused"))
    assert ac.adapters_spec(p, "sft", "deadbeefcafe1234").slug == "rtp-adapters-local_4050-sft-deadbeef"
    assert ac.data_spec(p, "sft", "0123456789abcdef").slug == "rtp-data-local_4050-sft-01234567"


def test_guard_path_accepts_children_and_refuses_escapes(tmp_path):
    root = tmp_path / "profile"
    (root / "a" / "b").mkdir(parents=True)
    assert ac.guard_path(root / "a" / "b", root) == (root / "a" / "b").resolve()
    with pytest.raises(ac.PathEscape):
        ac.guard_path(root / ".." / "elsewhere", root)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, root / "link")
    with pytest.raises(ac.PathEscape):                      # resolve() follows the link
        ac.guard_path(root / "link" / "x", root)


def test_asset_dir_stays_inside_the_profile_tree(tmp_path):
    p = prof(tmp_path)
    d = ac.asset_dir(SPEC, p)
    assert d.is_relative_to(p.paths.models.resolve())


def test_resolve_prefers_the_attach_then_the_cache_and_never_reads_it_again(tmp_path):
    p = prof(tmp_path)
    make_asset(tmp_path / "kaggle" / SPEC.slug, {"config.json": b"{}", "model.bin": b"x" * 32})
    calls = []

    def fetch(dest):
        calls.append(dest)
        raise AssertionError("fetch must not run when an attach matches")

    dest = ac.resolve(SPEC, p, attached=tmp_path / "kaggle", fetch=fetch, verify="sha256", log=None)
    assert calls == []
    assert dest == ac.asset_dir(SPEC, p)
    assert (dest / "model.bin").read_bytes() == b"x" * 32
    assert ac.verify_manifest(dest, SPEC, verify="sha256") == []

    # second run: local cache is enough, even with the attach gone
    dest2 = ac.resolve(SPEC, p, fetch=fetch, verify="sha256", log=None)
    assert dest2 == dest and calls == []


def test_a_corrupt_cache_is_refused_and_not_left_looking_valid(tmp_path):
    p = prof(tmp_path)
    dest = make_asset(ac.asset_dir(SPEC, p), {"config.json": b"{}", "w.bin": b"y" * 16})
    (dest / "w.bin").write_bytes(b"z" * 16)

    problems = ac.verify_manifest(dest, SPEC, verify="sha256")
    assert any("sha256 mismatch w.bin" in x for x in problems)

    def bad_fetch(tmp):
        (tmp / "config.json").write_bytes(b"{}")
        (tmp / "w.bin").write_bytes(b"y" * 16)
        ac.write_manifest(tmp, SPEC)
        (tmp / "w.bin").write_bytes(b"tampered")          # manifest now lies

    with pytest.raises(ac.ManifestError) as exc:
        ac.resolve(SPEC, p, fetch=bad_fetch, verify="sha256", log=None)
    assert exc.value.problems and not ac.asset_dir(SPEC, p).exists()


def test_fetch_runs_once_and_a_killed_run_resumes_cleanly(tmp_path):
    p = prof(tmp_path)
    dest = ac.asset_dir(SPEC, p)
    stale = dest.with_name(f"{dest.name}.partial-4194303")       # a pid that is not running
    stale.mkdir(parents=True)
    (stale / "junk").write_bytes(b"half")

    calls = []

    def flaky(tmp):
        calls.append(tmp)
        if len(calls) == 1:
            raise KeyboardInterrupt("user hit stop")
        (tmp / "config.json").write_bytes(b"{}")
        ac.write_manifest(tmp, SPEC, deep=True)

    with pytest.raises(KeyboardInterrupt):
        ac.resolve(SPEC, p, fetch=flaky, log=None)
    assert not dest.exists() and not stale.exists()          # neither the stale partial nor a fake cache
    out = ac.resolve(SPEC, p, fetch=flaky, log=None)
    assert out == dest and len(calls) == 2 and (dest / "config.json").exists()


def test_two_profiles_get_separate_trees(tmp_path):
    a, b = prof(tmp_path, "local_4050"), prof(tmp_path, "local_4090")
    for p in (a, b):
        src = make_asset(tmp_path / f"attach-{p.name}", {"config.json": b"{}"})
        ac.resolve(SPEC, p, attached=src, verify="sha256", log=None)
    da, db = ac.asset_dir(SPEC, a), ac.asset_dir(SPEC, b)
    assert da != db and not da.is_relative_to(db) and not db.is_relative_to(da)
    assert da.is_relative_to((tmp_path / "models" / "local_4050").resolve())


def test_dry_run_touches_nothing_and_reports_a_miss(tmp_path):
    p = prof(tmp_path)
    lines = []
    dest = ac.resolve(SPEC, p, fetch=lambda d: pytest.fail("no fetch in dry-run"),
                      dry_run=True, log=lines.append)
    assert "cache miss" in lines[0] and not dest.exists()
    assert not ac.cache_root(p).exists()


def test_wheelhouse_tag_mismatch_is_not_used():
    built = ac.wheelhouse_spec(python="3.12", torch="2.15.0", cuda="13.2", sm="89",
                               files=[ac.FileSpec("x.whl", 3, None)])
    wanted = ac.wheelhouse_spec(python="3.12", torch="2.15.0", cuda="13.2", sm="75")      # T4
    assert built.slug != wanted.slug
    assert "tag sm" not in built.slug


def test_wheelhouse_manifest_with_another_sm_is_refused(tmp_path):
    p = prof(tmp_path, "kaggle_t4x2")
    built = ac.wheelhouse_spec(python="3.12", torch="2.15.0", cuda="13.2", sm="89")
    wanted = ac.wheelhouse_spec(python="3.12", torch="2.15.0", cuda="13.2", sm="75")
    src = make_asset(tmp_path / "attach" / built.slug, {"w.whl": b"abc"}, spec=built)
    # same slug is impossible across sm, so even a forged attach dir under the wanted name is refused by tags
    forged = make_asset(tmp_path / "attach" / wanted.slug, {"w.whl": b"abc"}, spec=built)
    problems = ac.verify_manifest(forged, wanted)
    assert any(x.startswith("tag sm") for x in problems)
    with pytest.raises(ac.AssetUnavailable):
        ac.resolve(wanted, p, attached=tmp_path / "attach", log=None)
    assert src.exists()


def test_spec_from_dir_describes_the_bytes(tmp_path):
    root = tmp_path / "m"
    (root / "sub").mkdir(parents=True)
    (root / "a.json").write_bytes(b"{}")
    (root / "sub" / "b.bin").write_bytes(b"1234")
    spec = ac.spec_from_dir(root, kind="model", name="qwen35-4b", revision="abc12345", source="Qwen/Qwen3.5-4B")
    assert spec.slug == "rtp-model-qwen35-4b-abc12345"
    assert {f.path for f in spec.files} == {"a.json", "sub/b.bin"}
    ac.write_manifest(root, spec)                         # seed the cache from a verified download
    assert ac.verify_manifest(root, spec, verify="sha256") == []


def test_seed_from_a_local_download_links_the_bytes_and_verifies(tmp_path):
    p = prof(tmp_path)
    src = tmp_path / "download"
    src.mkdir()
    (src / "config.json").write_bytes(b"{}")
    (src / "w.bin").write_bytes(b"k" * 64)
    dest = ac.seed_from_dir(src, SPEC, p, verify="sha256")
    assert dest == ac.asset_dir(SPEC, p) and (dest / "w.bin").read_bytes() == b"k" * 64
    assert ac.verify_manifest(dest, SPEC, verify="sha256") == []
    first = (dest / "w.bin").stat().st_ino
    assert ac.seed_from_dir(src, SPEC, p, verify="sha256") == dest        # idempotent
    assert (dest / "w.bin").stat().st_ino == first


def test_export_metadata_is_private_and_needs_auth(tmp_path, monkeypatch):
    meta = ac.kaggle_metadata(SPEC, "someone")
    assert meta["isPrivate"] is True and meta["id"] == f"someone/{SPEC.slug}"
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    with pytest.raises(ac.ExportNotConfigured):
        ac.export_private(tmp_path, SPEC)
    written = ac.write_export_metadata(tmp_path, SPEC, "someone")
    assert json.loads(written.read_text())["isPrivate"] is True
    seen = []
    ac.export_private(tmp_path, SPEC, upload=seen.append, owner="someone")
    assert seen == [tmp_path]
    assert ac.attach_hint(SPEC, "someone") == f"attach Kaggle dataset someone/{SPEC.slug} next time"


def test_env_tags_records_what_is_installed():
    tags = ac.env_tags()
    assert tags["python"].startswith("3.")
    assert "torch" in tags
