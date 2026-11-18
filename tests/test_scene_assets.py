"""Offline tests for the asset-to-environment bridge (no HF, no USD files needed)."""
import json
from pathlib import Path

import pytest

from rtp.scene_assets import AssetRecord, resolve_assets, to_world_objects, write_manifest


def make_catalog(tmp_path):
    files = []
    for i in range(3):
        p = tmp_path / f"obj{i}.usd"
        p.write_text("# fake usd")
        files.append(str(p))
    catalog = [{"package": f"pkg{i}", "repo": "nvidia/simready-assets",
                "class": cls, "license": "CC-BY-4.0", "paths": [files[i]],
                "nearest_match": nearest,
                "mass_kg": 1.5 if i == 0 else None}
               for i, (cls, nearest) in enumerate((("book", False), ("lunchbox", True),
                                                   ("chair", False)))]
    (tmp_path / "catalog.json").write_text(json.dumps({"assets": catalog}))
    (tmp_path / "selection.json").write_text(json.dumps(
        [{"package": "pkg0"}, {"package": "pkg1"}, {"package": "pkg2"}]))
    return tmp_path


def test_resolve_validates_packages_and_licences(tmp_path):
    d = make_catalog(tmp_path)
    recs = resolve_assets(d / "catalog.json", d / "selection.json")
    assert [r.package for r in recs] == ["pkg0", "pkg1", "pkg2"]
    assert recs[0].mass == 1.5 and recs[0].source_of_measurement == "package-metadata"
    assert recs[1].nearest_match and recs[2].source_of_measurement == "class-default"
    assert all(r.licence == "CC-BY-4.0" for r in recs)


def test_unselected_and_missing_packages_are_refused(tmp_path):
    d = make_catalog(tmp_path)
    (d / "selection.json").write_text(json.dumps([{"package": "not_in_catalog"}]))
    with pytest.raises(ValueError, match="missing from"):
        resolve_assets(d / "catalog.json", d / "selection.json")
    cat = json.loads((d / "catalog.json").read_text())
    cat["assets"][0]["paths"] = [str(d / "gone.usd")]
    (d / "catalog.json").write_text(json.dumps(cat))
    (d / "selection.json").write_text(json.dumps([{"package": "pkg0"}]))
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_assets(d / "catalog.json", d / "selection.json")


def test_world_objects_and_manifest_keep_provenance(tmp_path):
    d = make_catalog(tmp_path)
    recs = resolve_assets(d / "catalog.json", d / "selection.json")
    objs = to_world_objects(recs, seed=0)
    assert len(objs) == 2 and len({o.name for o in objs}) == 2   # chair is scene furniture
    assert any(o.cls == "lunchbox" for o in objs)       # nearest-match class preserved
    assert objs[0].mass == 1.5                           # measured asset mass wins
    p = write_manifest(recs, objs, d / "scene.json")
    data = json.loads(p.read_text())
    assert "NOT a Newton physics result" in data["evidence_class"]
    assert data["assets"][1]["nearest_match"] is True
    assert data["objects"][0]["mass_kg"] == 1.5
    assert all(o["class"] in ("book", "lunchbox") for o in data["objects"])
