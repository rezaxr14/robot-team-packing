"""Notebook hygiene: valid nbformat, compiling cells, real profiles and real scripts.

Execution is not done here: a notebook needs a GPU (fast-path assertion) and a resolved model, so
it is run by hand and its output is recorded in the commit/DECISIONS. What this test guarantees is
that a notebook cannot rot silently (bad JSON, a syntax error, a path to a deleted script, a typo in
a profile name) before anyone starts a long Kaggle session.
"""
import ast
import json
import re
from pathlib import Path

import pytest

from rtp.profiles import PROFILE_NAMES

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))


def source(cell) -> str:
    return "".join(cell.get("source", []))


def test_at_least_the_setup_notebook_exists():
    assert (NB_DIR / "00_setup_cache.ipynb").exists()
    assert NOTEBOOKS, "no notebooks"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_and_cells_compile(path):
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5
    assert nb["cells"], "empty notebook"
    ids = [c.get("id") for c in nb["cells"]]
    assert len(ids) == len(set(ids)) and all(ids), "every cell needs a unique id"
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        if cell["cell_type"] == "code":
            compile(source(cell), f"{path.name}:{cell['id']}", "exec")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_names_only_real_profiles_and_scripts(path):
    text = path.read_text()
    for name in re.findall(r'ACTIVE\s*=\s*"([^"]+)"', text):
        assert name in PROFILE_NAMES, f"{path.name}: unknown profile {name}"
    # absolute paths are fine for mounts; repo-relative scripts must exist
    for rel in re.findall(r'["\']((?:ops|train|rtp)/[\w./-]+\.py)["\']', text):
        assert (ROOT / rel).exists(), f"{path.name}: {rel} does not exist"


def test_setup_notebook_uses_the_cache_api_not_raw_paths():
    tree = ast.parse(source(next(c for c in json.loads((NB_DIR / "00_setup_cache.ipynb").read_text())["cells"]
                               if c["cell_type"] == "code" and "assetcache" in source(c))))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "ac" in names
    text = (NB_DIR / "00_setup_cache.ipynb").read_text()
    assert "resolve(" in text and "verify_manifest(" in text
    assert "hf_home" not in text, "the notebook must not build its own model paths"


# --------------------------------------------------------------------------------------------------
# generated-notebook discipline: a builder script exists, it regenerates the same file, and every
# `from rtp.x import y` names something that really exists (no invented APIs in a notebook).
# --------------------------------------------------------------------------------------------------
import importlib                                                        # noqa: E402
import subprocess                                                       # noqa: E402
import sys                                                              # noqa: E402

SRC_DIR = NB_DIR / "src"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_notebook_has_a_builder_script(path):
    src = SRC_DIR / (path.stem + ".py")
    assert src.exists(), f"{path.name} has no notebooks/src/{path.stem}.py; a notebook must be generated"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_builder_regenerates_the_committed_notebook(path):
    src = SRC_DIR / (path.stem + ".py")
    if not src.exists():
        pytest.skip("no builder")
    before = path.read_text()
    subprocess.run([sys.executable, str(src)], cwd=ROOT, check=True,
                   capture_output=True, text=True)
    assert path.read_text() == before, f"{path.name} is stale: rerun notebooks/src/{src.name}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_imports_from_rtp_really_exist(path):
    nb = json.loads(path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse(source(cell))
        except SyntaxError as exc:                       # the compile test reports it properly
            raise AssertionError(f"{path.name}:{cell['id']}: {exc}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module != "rtp" and not node.module.startswith("rtp."):
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(module, alias.name), \
                    f"{path.name}:{cell['id']}: {node.module} has no {alias.name!r}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_placeholder_markers(path):
    text = path.read_text()
    for marker in ("TODO", "FIXME", "XXX", "your-code-here", "lorem ipsum"):
        assert marker.lower() not in text.lower(), f"{path.name} contains {marker!r}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_notebook_installs_what_it_needs(path):
    text = path.read_text()
    if path.stem.startswith("00"):
        assert "install_cell" in text or "REQUIRED" in text
    assert "SKIP" in text.upper() or "skip" in text, \
        f"{path.name}: needs an explicit SKIPPED path for hardware it cannot assume"
