"""The asset-backed data generator must produce real teacher-validated rows with provenance."""
import json
from pathlib import Path

import pytest

from rtp.asset_datagen import gen_asset_sft, spec_for_assets
from rtp.harness import Harness
from rtp.scene_assets import resolve_assets, to_world_objects, write_manifest

ROOT = Path(__file__).resolve().parents[1]


def fake_catalog(tmp_path):
    entries = []
    for i, cls in enumerate(("book", "notebook", "lunchbox", "table", "chair")):
        f = tmp_path / f"a{i}.usd"
        f.write_text("fake")
        entries.append({"package": f"pkg_{cls}_{i}", "repo": "nvidia/simready-assets",
                        "class": cls, "license": "CC-BY-4.0", "paths": [str(f)],
                        "nearest_match": cls == "lunchbox"})
    (tmp_path / "catalog.json").write_text(json.dumps({"assets": entries}))
    (tmp_path / "selection.json").write_text(json.dumps(
        [{"package": e["package"]} for e in entries]))
    return tmp_path


def test_spec_uses_real_asset_classes(tmp_path):
    d = fake_catalog(tmp_path)
    recs = resolve_assets(d / "catalog.json", d / "selection.json")
    objs = to_world_objects(recs)
    spec = spec_for_assets(objs, 0)
    assert any("lunchbox" in n for n in spec.items)
    assert spec.last_item and "lunchbox" in spec.last_item


def test_asset_sft_rows_carry_the_manifest(tmp_path):
    d = fake_catalog(tmp_path)
    recs = resolve_assets(d / "catalog.json", d / "selection.json")
    manifest = write_manifest(recs, to_world_objects(recs), d / "scene.json")
    rows, funnel = gen_asset_sft(recs, 2, 200000, Harness(ROOT / "harness_files"),
                                 noisy_eyes=False, manifest=str(manifest))
    assert len(rows) == 2
    assert funnel["kept"] == 2 and funnel["evidence"] == "SIM-POSE"
    assert all(r["meta"]["asset_manifest"] == str(manifest) for r in rows)
    assert all(r["meta"]["n_assets"] == 5 for r in rows)
