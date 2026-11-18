"""Offline tests for ops/fetch_simready.py: a fake HfApi/file list, no network calls."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

import fetch_simready as fs  # noqa: E402

NVIDIA = "nvidia/simready-assets"


class FakeHub:
    """Minimal injectable stand-in for HubClient / huggingface_hub."""

    def __init__(self, files: dict[str, list[str]], trees: dict[str, list[tuple[str, int | None]]],
                 texts: dict[str, str] | None = None):
        self.files = files
        self.trees = trees
        self.texts = texts or {}
        self.downloaded: list[tuple[str, str]] = []

    def list_files(self, repo_id):
        return self.files.get(repo_id, [])

    def list_tree(self, repo_id, path):
        return [types.SimpleNamespace(path=p, size=s) for p, s in self.trees.get(path, [])]

    def download(self, repo_id, filename, local_dir):
        self.downloaded.append((repo_id, filename))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.texts.get(f"{repo_id}/{filename}", "{}"), encoding="utf-8")
        return target


def small_tree(pkg):
    return [
        (f"{pkg}/.metadata/LICENSE.txt", 100),
        (f"{pkg}/com.nvidia.simready.packaging.json", 200),
        (f"{pkg}/simready_usd/sm_{pkg}_01.usd", 300),
        (f"{pkg}/simready_usd/payloads/geometries.usd", 1000),
        (f"{pkg}/simready_usd/payloads/base.usda", 400),
        (f"{pkg}/simready_usd/Textures/huge_albedo.png", 30_000_000),
        (f"{pkg}/simready_usd/Textures/small_orm.png", 500),
        (f"{pkg}/simready_usd/.thumbs/256x256/thumb.png", 600),
    ]


@pytest.fixture
def selection_data():
    return {
        "schema": 1,
        "byte_budget": 100_000_000,
        "packages": [
            {"repo_id": NVIDIA, "package_id": "book_open_a01", "task_class": "book",
             "match": "exact", "is_furniture": False, "why": "exact book"},
            {"repo_id": NVIDIA, "package_id": "chair_office_amf_black_a01", "task_class": "chair",
             "match": "exact", "is_furniture": True, "why": "furniture"},
            {"repo_id": NVIDIA, "package_id": "cart_computer_a01", "task_class": "laptop/computer",
             "match": "nearest", "is_furniture": False, "why": "no laptop",
             "nearest_note": "cart computer, not a laptop"},
        ],
    }


@pytest.fixture
def fake_hub(selection_data):
    files = {
        NVIDIA: [
            f"{p['package_id']}/{f}" for p in selection_data["packages"]
            for f in (".metadata/LICENSE.txt", "simready_usd/payloads/geometries.usd")
        ] + ["README.md", "other_pkg/a.usd"],
        "imagine-io/PhysicalAI-Simready-Kitchen-Assets": [
            "packages/Models/appliances/bbq_grill/bbq_grill.usd", "README.md"],
    }
    trees = {p["package_id"]: small_tree(p["package_id"]) for p in selection_data["packages"]}
    texts = {f"{NVIDIA}/{p['package_id']}/com.nvidia.simready.packaging.json":
             json.dumps({"license": "CC-BY-4.0"}) for p in selection_data["packages"]}
    return FakeHub(files, trees, texts)


# --- catalog construction -----------------------------------------------------------------
def test_discover_packages_respects_repo_layout():
    top = fs.discover_packages(["a/x.usd", "a/b/y.usd", "z.usd", ".gitattributes"], "top")
    assert set(top) == {"a"} and len(top["a"]) == 2
    kitchen = fs.discover_packages([
        "packages/Models/appliances/bbq_grill/bbq_grill.usd",
        "packages/Models/props/apple/apple.usd"], "kitchen")
    assert set(kitchen) == {"packages/Models/appliances/bbq_grill", "packages/Models/props/apple"}


@pytest.mark.parametrize("name,expected", [
    ("misc_notebook_a01", "notebook"),
    ("book_open_a01", "book"),
    ("bottle_laboratory_a01", "bottle"),
    ("chair_office_amf_black_a01", "chair"),
    ("fixture_counter_a01", "counter"),
    ("unknown_thing_a01", "other"),
])
def test_class_guess(name, expected):
    assert fs.class_guess(name) == expected


def test_build_catalog_uses_fake_file_list_and_verifies_selected_licenses(fake_hub, selection_data, tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "CACHE", tmp_path / "cache")
    rows = fs.build_catalog(fake_hub, selection=selection_data,
                            repos=[r for r in fs.REPOS if r["repo_id"] in fake_hub.files])
    assert len(rows) == 5  # 3 selected nvidia + 1 unselected nvidia + 1 kitchen package
    by_pkg = {r["package"]: r for r in rows}
    assert by_pkg["book_open_a01"]["license"] == "CC-BY-4.0"
    assert by_pkg["book_open_a01"]["license_verified"] is True
    assert by_pkg["book_open_a01"]["license_source"] == "com.nvidia.simready.packaging.json"
    assert by_pkg["book_open_a01"]["file_count"] == 2
    assert by_pkg["book_open_a01"]["class_guess"] == "book"
    assert by_pkg["chair_office_amf_black_a01"]["is_furniture"] is True
    assert by_pkg["packages/Models/appliances/bbq_grill"]["license_verified"] is False
    assert by_pkg["packages/Models/appliances/bbq_grill"]["local_path"].startswith("assets/cache/")


def test_write_catalog_emits_json_and_csv(tmp_path):
    rows = [{"repo": NVIDIA, "package": "p", "class_guess": "book", "is_furniture": False,
             "license": "CC-BY-4.0", "license_source": "packaging.json", "license_verified": True,
             "file_count": 3, "usd_count": 1, "local_path": "assets/cache/x/p",
             "source_url": "https://huggingface.co/datasets/nvidia/simready-assets"}]
    jp, cp = tmp_path / "catalog.json", tmp_path / "catalog.csv"
    fs.write_catalog(rows, jp, cp)
    assert json.loads(jp.read_text())["count"] == 1
    assert "book" in cp.read_text() and "license" in cp.read_text()


# --- selection filtering / licence extraction -------------------------------------------
def test_license_extraction_prefers_packaging_json_then_license_text():
    lic, src, ok = fs.extract_license_from_text('{"license": "CC-BY-4.0"}', "Creative Commons Attribution 4.0")
    assert (lic, src, ok) == ("CC-BY-4.0", "com.nvidia.simready.packaging.json", True)
    lic, src, ok = fs.extract_license_from_text("not json", "Creative Commons Attribution 4.0")
    assert ok and "LICENSE.txt" in src and "Creative Commons" in lic
    assert fs.extract_license_from_text(None, None) == ("unknown", "none", False)


def test_selection_loading_and_id_set(selection_data, tmp_path):
    p = tmp_path / "selection.json"
    p.write_text(json.dumps(selection_data))
    sel = fs.load_selection(p)
    assert fs.selection_ids(sel) == {
        (NVIDIA, "book_open_a01"),
        (NVIDIA, "chair_office_amf_black_a01"),
        (NVIDIA, "cart_computer_a01"),
    }
    bad = dict(selection_data)
    bad["packages"] = [{"repo_id": NVIDIA, "package_id": "x"}]
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        fs.load_selection(p)


def test_choose_files_drops_large_textures_and_keeps_one_thumbnail(fake_hub):
    chosen = fs.choose_files(fake_hub.list_tree(NVIDIA, "book_open_a01"))
    paths = [p for p, _ in chosen]
    assert any("geometries.usd" in p for p in paths)
    assert any(".metadata/LICENSE.txt" in p for p in paths)
    assert any("small_orm.png" in p for p in paths)
    assert not any("huge_albedo.png" in p for p in paths)
    assert sum(1 for p in paths if p.endswith("thumb.png")) == 1
    assert sum(s for _, s in chosen) < 3_000_000


def test_fetch_only_downloads_selected_packages(fake_hub, selection_data, tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "FETCH_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(fs, "CACHE", tmp_path / "cache")
    result = fs.fetch_packages(fake_hub, selection_data,
                               [(NVIDIA, "book_open_a01")], local_root=tmp_path / "cache")
    assert result["files"] and result["bytes_downloaded"] > 0
    assert {f["package"] for f in result["files"]} == {"book_open_a01"}
    assert all((NVIDIA, "book_open_a01") == call[:2] or "book_open_a01" in call[1]
               for call in fake_hub.downloaded)
    assert json.loads((tmp_path / "manifest.json").read_text())["files"] == result["files"]


def test_fetch_refuses_unselected_package(fake_hub, selection_data, tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "FETCH_MANIFEST", tmp_path / "manifest.json")
    with pytest.raises(fs.UnselectedPackage, match="not in assets/selection.json"):
        fs.fetch_packages(fake_hub, selection_data, [(NVIDIA, "secret_not_selected_a01")],
                          local_root=tmp_path / "cache")
    assert fake_hub.downloaded == []


def test_cli_fetch_rejects_unselected_package_early(selection_data, tmp_path, monkeypatch):
    p = tmp_path / "selection.json"
    p.write_text(json.dumps(selection_data))
    assert fs.main(["--fetch", "--selection", str(p), "--package", "not_selected"]) == 2


# --- generated attribution / report ------------------------------------------------------
def test_attribution_and_report_are_generated_from_catalog(fake_hub, selection_data, tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "FETCH_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(fs, "CACHE", tmp_path / "cache")
    manifest = fs.fetch_packages(fake_hub, selection_data, local_root=tmp_path / "cache")
    catalog = {(NVIDIA, "book_open_a01"): {
        "license": "CC-BY-4.0", "license_source": "packaging.json",
        "local_path": "assets/cache/book", "source_url": fs.REPOS[0]["source_url"]}}
    md = fs.write_attribution(manifest, selection_data, catalog, tmp_path / "ATTRIBUTION.md")
    assert "book_open_a01" in md and "CC-BY-4.0" in md
    assert "Nearest-match notices" in md and "cart_computer_a01" in md
    report = fs.write_ingest_report(manifest, selection_data, catalog, tmp_path / "ingest.json")
    assert report["totals"]["packages"] == 3
    assert report["coverage"]["object_packages"] == 2
    assert report["coverage"]["furniture_pieces"] == 1
    assert report["coverage"]["object_classes"] == 2


def test_cache_scan_marks_absent_packages_pending_and_reads_present_bytes(selection_data, tmp_path,
                                                                         monkeypatch):
    monkeypatch.setattr(fs, "FETCH_MANIFEST", tmp_path / "manifest.json")
    cache = tmp_path / "cache"
    present = cache / "nvidia__simready-assets" / "book_open_a01" / "simready_usd" / "payloads"
    present.mkdir(parents=True)
    (present / "geometries.usd").write_bytes(b"x" * 4096)
    result = fs.scan_cache(selection_data, local_root=cache)
    assert result["method"].startswith("cache-scan")
    assert result["bytes_downloaded"] == 4096
    assert {f["package"] for f in result["files"]} == {"book_open_a01"}
    assert {p["package"] for p in result["pending"]} == {
        "chair_office_amf_black_a01", "cart_computer_a01"}
    assert all(f["sha256"] for f in result["files"])


def test_enrich_catalog_marks_selection():
    rows = [{"repo": NVIDIA, "package": "book_open_a01", "class_guess": "book"},
            {"repo": NVIDIA, "package": "other_a01", "class_guess": "other"}]
    sel = {"packages": [{"repo_id": NVIDIA, "package_id": "book_open_a01",
                         "task_class": "book", "match": "exact"}]}
    out = fs.enrich_catalog_with_selection(rows, sel)
    assert out[0]["selected"] is True and out[0]["task_class"] == "book"
    assert out[1]["selected"] is False and out[1]["task_class"] == ""


def test_pending_packages_are_not_reported_as_downloaded(selection_data, tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "FETCH_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(fs, "CACHE", tmp_path / "cache")
    manifest = fs.scan_cache(selection_data, local_root=tmp_path / "cache")
    md = fs.write_attribution(manifest, selection_data, {}, tmp_path / "ATTRIBUTION.md")
    report = fs.write_ingest_report(manifest, selection_data, {}, tmp_path / "ingest.json")
    assert "NOT downloaded" in md and "cart_computer_a01" in md
    assert report["totals"]["bytes"] == 0 and report["totals"]["pending_packages"] == 3
    assert report["coverage"]["meets_20_objects_6_classes_3_furniture"] is False


def test_verify_download_checks_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "FETCH_MANIFEST", tmp_path / "manifest.json")
    manifest = {"files": [{"local_path": str(tmp_path / "f.usd"), "size": 3,
                           "sha256": __import__("hashlib").sha256(b"abc").hexdigest()}],
                "bytes_downloaded": 3}
    (tmp_path / "f.usd").write_bytes(b"abc")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert fs.verify_download(tmp_path / "manifest.json")["problems"] == []
    (tmp_path / "f.usd").write_bytes(b"abcd")
    assert any("size mismatch" in x for x in fs.verify_download(tmp_path / "manifest.json")["problems"])


def test_token_is_read_without_leaking_it(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text("HF_USERNAME=someone\nHF_TOKEN=secret_value_123\n")
    assert fs.load_token(env) == "secret_value_123"
    assert "secret_value_123" not in (fs.__doc__ or "")
