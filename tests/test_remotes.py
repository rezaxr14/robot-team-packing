"""Offline tests for the local -> Kaggle -> HF ordering and resumable checkpoint naming.

No network here: the clients are doubles. The one real fact pinned is our HF repo name/id shape;
live auth and uploads are exercised by the notebook and recorded in DECISIONS when actually run.
"""
from pathlib import Path

import pytest

from rtp import assetcache as ac
from rtp.remotes import (HF_REPO_NAME, RemoteError, RemoteStore, configure_kaggle_proxy, hf_repo_id,
                         kaggle_slug, read_env)

import pytest as _pytest


@_pytest.fixture
def profile_for_test(tmp_path):
    from rtp.profiles import get_profile
    return get_profile("local_4050", root=tmp_path)


class FakeKaggle:
    def __init__(self, has=(), username="rezanadimikj"):
        self.has = set(has)
        self.username = username
        self.calls = []

    def dataset_exists(self, slug, owner):
        return slug in self.has

    def dataset_create(self, slug, owner, root, message):
        self.calls.append(("create", slug, owner, message))

    def dataset_version(self, slug, owner, root, message):
        self.calls.append(("version", slug, owner, message))

    def dataset_download(self, slug, dest, owner):
        if slug not in self.has:
            raise FileNotFoundError(slug)
        self.calls.append(("download", slug, owner))
        # The test fixture pre-creates the bytes in dest; a real client would download them.
        (Path(dest) / ac.MANIFEST).write_text("{}")


class FakeHF:
    def __init__(self, files=()):
        self.files = list(files)
        self.uploads = []

    def create_repo(self, **kw):
        self.uploads.append(("repo", kw["repo_id"], kw.get("private")))

    def upload_folder(self, **kw):
        self.uploads.append(("upload", kw["path_in_repo"]))

    def snapshot_download(self, **kw):
        self.uploads.append(("download", kw["allow_patterns"]))

    def list_repo_files(self, **kw):
        return self.files

    def whoami(self):
        return {"name": "Reza-Nadimi"}


def _spec():
    return ac.model_spec("Qwen/Qwen3.5-4B", revision="851bf6e8", name="qwen35-4b")


def test_env_is_stripped_and_never_exposes_values(capsys):
    p = Path(".env")
    env = read_env(p)
    for key in ("HF_TOKEN", "KAGGLE_API_TOKEN"):
        if key in env:
            assert not env[key].endswith("\r")
            assert "\n" not in env[key]
    assert "HF_TOKEN" not in capsys.readouterr().out


def test_proxy_is_configured_from_env(monkeypatch):
    assert configure_kaggle_proxy("http://localhost:10808") == "http://localhost:10808"
    assert __import__("os").environ["HTTPS_PROXY"] == "http://localhost:10808"
    monkeypatch.setenv("RTP_KAGGLE_PROXY", "http://proxy.test:9")
    assert configure_kaggle_proxy() == "http://proxy.test:9"


def test_slugs_and_repo_ids_are_ours_only():
    assert kaggle_slug(_spec()).startswith("rtp-model-qwen35-4b-")
    assert hf_repo_id(client=FakeHF()) == f"Reza-Nadimi/{HF_REPO_NAME}"
    assert "robot-team-packing" in hf_repo_id(client=FakeHF())


def test_resolution_prefers_the_local_cache(profile_for_test):
    spec = _spec()
    cache = ac.asset_dir(spec, profile_for_test)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "x.bin").write_bytes(b"bytes")
    ac.write_manifest(cache, spec, deep=True)
    store = RemoteStore(kaggle=FakeKaggle(has=[kaggle_slug(spec)]), hf=FakeHF(),
                        hf_repo="Reza-Nadimi/robot-team-packing")
    assert store.resolve(spec, profile_for_test) == cache
    assert store.kaggle.calls == []                       # no remote touched on a local hit


def test_resolution_uses_kaggle_first_then_hf(profile_for_test, monkeypatch):
    spec = _spec()
    k = FakeKaggle(has=[])
    hf = FakeHF()
    store = RemoteStore(kaggle=k, hf=hf, hf_repo="Reza-Nadimi/robot-team-packing")

    # Kaggle wins when it has the bytes: no HF download.
    monkeypatch.setattr(FakeKaggle, "dataset_download", lambda self, slug, dest, owner: (
        k.calls.append(("download", slug)),
        (Path(dest) / "x.bin").write_bytes(b"k"),
        None)[-1])
    k.has.add(kaggle_slug(spec))
    p1 = store.resolve(spec, profile_for_test)
    assert (p1 / "x.bin").read_bytes() == b"k"
    assert any(c[0] == "download" for c in k.calls) and not hf.uploads

    # HF is the fallback when Kaggle does not have it.
    k.has.clear()
    spec2 = ac.model_spec("Qwen/Qwen3.5-4B", revision="851bf6e8", name="hf-only")
    monkeypatch.setattr(FakeHF, "snapshot_download", lambda self, **kw: (
        hf.uploads.append(("download", kw["allow_patterns"])),
        (Path(kw["local_dir"]) / "x.bin").write_bytes(b"h"),
        None)[-1])
    p2 = store.resolve(spec2, profile_for_test, fetch_kaggle=False)
    assert (p2 / "x.bin").read_bytes() == b"h"


def test_publish_creates_then_versions_a_private_dataset(profile_for_test, tmp_path):
    spec = _spec()
    root = tmp_path / "asset"
    root.mkdir()
    (root / "w.bin").write_bytes(b"k" * 100)
    ac.write_manifest(root, spec, deep=True)
    k = FakeKaggle()
    hf = FakeHF()
    store = RemoteStore(kaggle=k, hf=hf, hf_repo="Reza-Nadimi/robot-team-packing")
    out = store.publish(root, spec, to_hf=False)
    assert out["kaggle"] == "rezanadimikj/" + kaggle_slug(spec)
    assert k.calls[0][0] == "create"
    assert (root / "dataset-metadata.json").exists()
    k.has.add(kaggle_slug(spec))
    store.publish(root, spec, to_hf=False)
    assert k.calls[-1][0] == "version"


def test_publish_refuses_an_unverified_asset(tmp_path):
    spec = _spec()
    root = tmp_path / "bad"
    root.mkdir()
    (root / "w.bin").write_bytes(b"x")
    with pytest.raises(ac.ManifestError):
        RemoteStore().publish(root, spec, to_kaggle=False, to_hf=False)


def test_checkpoints_are_sortable_and_resume_reads_the_latest_step(tmp_path):
    hf = FakeHF(files=[
        "checkpoints/local_4050/sft/step-000000100/model.safetensors",
        "checkpoints/local_4050/sft/step-000000350/optimizer.pt",
        "checkpoints/local_4090/sft/step-000000900/x",
        "checkpoints/local_4050/grpo/step-000000010/x",
    ])
    store = RemoteStore(hf=hf, hf_repo="Reza-Nadimi/robot-team-packing")
    assert store.latest_checkpoint("local_4050", "sft") == 350
    assert store.latest_checkpoint("local_4090", "sft") == 900
    assert store.latest_checkpoint("local_4050", "none") is None
    url = store.push_checkpoint(tmp_path, "local_4050", "sft", 351)
    assert "step-000000351" in url
    assert any("step-000000351" in str(u) for _, u, *_ in hf.uploads)


def test_publish_needs_a_real_client(tmp_path):
    spec = _spec()
    root = tmp_path / "a"
    root.mkdir()
    (root / "x").write_bytes(b"x")
    ac.write_manifest(root, spec, deep=True)
    with pytest.raises(RemoteError):
        RemoteStore().publish(root, spec, to_kaggle=True, to_hf=False)


def test_publish_uses_the_manifest_identity_not_a_guessed_source(tmp_path):
    """A cached model's manifest says source=Qwen/Qwen3.5-4B; publish must not rewrite it to 'local'."""
    spec = ac.model_spec("Qwen/Qwen3.5-4B", revision="851bf6e8", name="qwen35-4b")
    root = tmp_path / "model"
    root.mkdir()
    (root / "w.bin").write_bytes(b"k" * 64)
    ac.write_manifest(root, spec, deep=True)
    from rtp.remotes import RemoteStore as RS
    k, hf = FakeKaggle(), FakeHF()
    store = RS(kaggle=k, hf=hf, hf_repo="Reza-Nadimi/robot-team-packing")
    out = store.publish(root, spec, to_hf=False)
    assert out["kaggle"].endswith(spec.slug)
    assert ac.read_manifest(root)["source"] == "Qwen/Qwen3.5-4B"
