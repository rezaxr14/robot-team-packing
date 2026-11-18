"""Build all 12 figures: SVG sources in figures/svg, PDF twins in figures/pdf.   python scripts/build_diagrams.py"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagram_common import PAPER_DIR, SVG_DIR  # noqa: E402
import figs_a, figs_b, figs_v2, fig2, fig3, fig4, fig5  # noqa: E402

BUILDERS = [figs_a.fig_mechanism, figs_a.fig_overview, figs_a.fig_messages, figs_a.fig_harness, figs_a.fig_loop, figs_a.fig_tools,
            figs_b.fig_signals, figs_b.fig_credit, figs_b.fig_bridge, figs_b.fig_design, figs_b.fig_scorer_audit, figs_b.fig_results]


def to_pdf(name: str) -> None:
    txt = (SVG_DIR / f"{name}.svg").read_text(encoding="utf-8")
    if "engine:cairo" in txt:                         # modern figures: gradients, clipping, Inter
        import cairosvg
        out = PAPER_DIR / "figures" / "pdf"
        out.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2pdf(bytestring=txt.encode("utf-8"), write_to=str(out / f"{name}.pdf"))
        return
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    out = PAPER_DIR / "figures" / "pdf"
    out.mkdir(parents=True, exist_ok=True)
    d = svg2rlg(str(SVG_DIR / f"{name}.svg"))
    renderPDF.drawToFile(d, str(out / f"{name}.pdf"))


def main() -> None:
    for fn in BUILDERS:
        mods = (fig5, fig4, fig3, fig2, figs_v2)
        for m_ in mods:                                    # the newest redesign wins
            if hasattr(m_, fn.__name__):
                fn = getattr(m_, fn.__name__)
                break
        fn()
    for f in sorted(SVG_DIR.glob("fig_*.svg")):
        to_pdf(f.stem)
        print("built", f.stem)


if __name__ == "__main__":
    main()
