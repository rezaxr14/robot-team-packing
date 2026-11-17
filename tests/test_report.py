"""Track-A figure helper that still lives in `rtp.figures`.

The results-report builder itself moved to `paper/scripts/` (frozen, user-owned), so its tests are
not part of the code test suite any more.
"""
from pathlib import Path

from rtp import figures


def test_figures_trackA_and_empty_inputs(tmp_path):
    s = {"label": "L2_id", "level": 2, "full": {"strict_draw": {"rate": .5}, "strict_all": {"rate": .4}}}
    p = figures.fig_trackA([s, {"level": 1}], tmp_path / "t.png")
    assert Path(p).read_bytes()[:4] == b"\x89PNG"
