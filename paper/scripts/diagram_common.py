"""Shared helpers for the generated paper diagrams (build_diagrams.py).

Everything here is plain-SVG string assembly with two hard constraints from
the toolchain (svglib -> reportlab -> PDF):

- Only Helvetica-family fonts survive conversion; no text wrapping happens
  automatically, so every label is hand-broken into lines.
- reportlab base-14 fonts use the WinAnsi charset: use "-" not U+2192,
  draw check/cross marks as paths, never as glyphs.

All figures here are vector only; every number or trace they show is read from generated/figdata.json.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent          # the paper/ directory (this file is paper/scripts/)
SVG_DIR = PAPER_DIR / "figures" / "svg"

FONT = "Helvetica"
MONO = "Courier"

# Muted Q1-style palette (matplotlib "deep"-adjacent, print-safe).
C_BLUE = "#4C72B0"
C_BLUE_D = "#34548A"
C_BLUE_FILL = "#F4F8FD"
C_GREEN = "#55A868"
C_GREEN_D = "#2F6B3C"
C_GREEN_FILL = "#F4FBF6"
C_ORANGE = "#ff7f0e"
C_ORANGE_L = "#FFF4E8"
C_ORANGE_D = "#8A4B06"
C_RED = "#B5383C"
C_SAND_FILL = "#FDFBF2"
C_SAND_D = "#8A6D1F"
C_TXT = "#444444"
C_SUB = "#666666"
C_MUT = "#999999"
# fills for pills that carry white bold text (>= 4.5:1 contrast)
PILL_GREEN = "#2F7D45"
PILL_ORANGE = "#B8561F"
PILL_GREY = "#5F6B7A"


_AMP = chr(38)
_LT = chr(60)
_GT = chr(62)


def esc(s: str) -> str:
    return (
        s.replace(_AMP, _AMP + "amp;")
        .replace(_LT, _AMP + "lt;")
        .replace(_GT, _AMP + "gt;")
    )


def T(x, y, s, size=9, fill=C_TXT, anchor="middle", weight=None, style=None, font=None):
    a = f'<text x="{x}" y="{y}" font-family="{font or FONT}" font-size="{size}"'
    if weight:
        a += f' font-weight="{weight}"'
    if style:
        a += f' font-style="{style}"'
    if anchor != "start":
        a += f' text-anchor="{anchor}"'
    return a + f' fill="{fill}">{esc(s)}</text>'


def R(x, y, w, h, rx=6, fill="#FFFFFF", stroke=C_MUT, sw=1.1, dash=None, opacity=None):
    a = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
    if dash:
        a += f' stroke-dasharray="{dash}"'
    if opacity is not None:
        a += f' opacity="{opacity}"'
    return a + "/>"


def L(x1, y1, x2, y2, color=C_MUT, sw=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>'


def IMG(x, y, w, h, b64):
    mime = "jpeg" if b64.startswith("/9j/") else "png"  # base64 of a JPEG always starts with /9j/
    return f'<image x="{x}" y="{y}" width="{w}" height="{h}" xlink:href="data:image/{mime};base64,{b64}"/>'


def arrow(points, color="#666666", sw=1.5, dash=None, free=""):
    """Orthogonal polyline ending in an explicit triangular arrowhead.

    ``free`` ("tail", "tip", "both", or "vector" for a displacement vector drawn inside
    a schematic) marks an end that is deliberately not on a
    box (written as an SVG comment that audit_diagram_arrows.py reads); prefer a
    small source/sink box and connect() instead.

    svglib silently drops SVG <marker> elements, so heads drawn with
    marker-end vanish in the PDF the paper actually includes. The head is
    therefore a plain polygon, which every converter renders identically.
    """
    import math

    (x0, y0), (x1, y1) = points[-2], points[-1]
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    head_len, head_hw = 5.6 * sw, 2.6 * sw
    bx, by = x1 - ux * head_len, y1 - uy * head_len  # base centre of the head
    line_pts = list(points[:-1]) + [(bx, by)]
    pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in line_pts)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    tri = (
        f"{x1:.1f},{y1:.1f} "
        f"{bx - uy * head_hw:.1f},{by + ux * head_hw:.1f} "
        f"{bx + uy * head_hw:.1f},{by - ux * head_hw:.1f}"
    )
    mark = f"<!--free:{free}-->" if free else ""
    return (
        mark
        + f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}"{d}/>'
        + f'<polygon points="{tri}" fill="{color}" stroke="none"/>'
    )


def anchor(box, side, off=None):
    """Point on a box edge. box = (x, y, w, h); side in l/r/t/b; ``off`` is the distance
    from the box's left (for t/b) or top (for l/r) edge, default the edge midpoint."""
    x, y, w, h = box
    if side == "l":
        return (x, y + (h / 2 if off is None else off))
    if side == "r":
        return (x + w, y + (h / 2 if off is None else off))
    if side == "t":
        return (x + (w / 2 if off is None else off), y)
    if side == "b":
        return (x + (w / 2 if off is None else off), y + h)
    raise ValueError(side)


def route(a, sa, b, sb, oa=None, ob=None, mid=None):
    """Orthogonal points from box a (side sa) to box b (side sb), both ends exactly on
    the edges: straight when aligned, a Z (shared gutter coordinate ``mid``) between
    parallel sides, an L between perpendicular sides. Never diagonal."""
    p0, p1 = anchor(a, sa, oa), anchor(b, sb, ob)
    horiz_a, horiz_b = sa in "lr", sb in "lr"
    if horiz_a and horiz_b:
        if abs(p0[1] - p1[1]) < 0.05:
            return [p0, p1]
        mx = (p0[0] + p1[0]) / 2 if mid is None else mid
        return [p0, (mx, p0[1]), (mx, p1[1]), p1]
    if not horiz_a and not horiz_b:
        if abs(p0[0] - p1[0]) < 0.05:
            return [p0, p1]
        my = (p0[1] + p1[1]) / 2 if mid is None else mid
        return [p0, (p0[0], my), (p1[0], my), p1]
    if horiz_a:  # leave sideways, enter vertically
        return [p0, (p1[0], p0[1]), p1]
    return [p0, (p0[0], p1[1]), p1]  # leave vertically, enter sideways


def connect(a, sa, b, sb, oa=None, ob=None, mid=None, color="#666666", sw=1.5, dash=None, free=""):
    """arrow() between two boxes' edges (see route()); the single way diagrams connect cards."""
    return arrow(route(a, sa, b, sb, oa, ob, mid), color=color, sw=sw, dash=dash, free=free)


def marker_defs() -> str:
    """Kept for API compatibility; arrowheads are now explicit polygons."""
    return ""


def svg(w, h, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>'
        + body
        + marker_defs()
        + "</svg>"
    )


def badge(n, x, y, r=8.0, fill=C_BLUE):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>' + T(
        x, y + 3.2, str(n), size=9.5, fill="#FFFFFF", weight="bold"
    )


# ---------------------------------------------------------------------------
# Icons (vector paths -- glyphs like U+2713 do not survive WinAnsi)
# ---------------------------------------------------------------------------


def icon_check(x, y, s=6.0, color=C_GREEN_D):
    return (
        f'<path d="M {x} {y + s * 0.55} L {x + s * 0.36} {y + s} L {x + s} {y - s * 0.25}" '
        f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def icon_cross(x, y, s=5.0, color=C_RED):
    return (
        f'<path d="M {x} {y} L {x + s} {y + s} M {x + s} {y} L {x} {y + s}" '
        f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>'
    )


def icon_db(x, y, w=14.0, h=16.0, color=C_BLUE):
    rx = w / 2
    return (
        f'<ellipse cx="{x + rx}" cy="{y + 3}" rx="{rx}" ry="2.8" fill="none" stroke="{color}" stroke-width="1.4"/>'
        f'<path d="M {x} {y + 3} L {x} {y + h - 3} A {rx} 2.8 0 0 0 {x + w} {y + h - 3} L {x + w} {y + 3}" '
        f'fill="none" stroke="{color}" stroke-width="1.4"/>'
        f'<path d="M {x} {y + 8.5} A {rx} 2.8 0 0 0 {x + w} {y + 8.5}" fill="none" stroke="{color}" stroke-width="1.1"/>'
    )


def icon_img(x, y, w=16.0, h=12.0, color="#888888"):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="1.5" fill="none" stroke="{color}" stroke-width="1.3"/>'
        f'<circle cx="{x + w * 0.3}" cy="{y + h * 0.35}" r="1.2" fill="{color}"/>'
        f'<path d="M {x + 1.5} {y + h - 2} L {x + w * 0.45} {y + h * 0.45} '
        f'L {x + w * 0.65} {y + h * 0.7} L {x + w - 2} {y + h * 0.4}" '
        f'fill="none" stroke="{color}" stroke-width="1.2"/>'
    )


def icon_magnifier(x, y, r=5.0, color="#555555"):
    return (
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="{color}" stroke-width="1.6"/>'
        f'<line x1="{x + r * 0.75}" y1="{y + r * 0.75}" x2="{x + r * 1.9}" y2="{y + r * 1.9}" '
        f'stroke="{color}" stroke-width="2.2" stroke-linecap="round"/>'
    )


def icon_shield(x, y, w=13.0, h=15.0, color=C_GREEN_D):
    return (
        f'<path d="M {x} {y} L {x + w} {y} L {x + w} {y + h * 0.55} '
        f'Q {x + w} {y + h * 0.85} {x + w / 2} {y + h} '
        f'Q {x} {y + h * 0.85} {x} {y + h * 0.55} Z" '
        f'fill="none" stroke="{color}" stroke-width="1.4"/>'
        + icon_check(x + w * 0.28, y + h * 0.3, s=4.6, color=color)
    )


def icon_doc(x, y, w=11.0, h=14.0, color="#888888"):
    return (
        f'<path d="M {x} {y} L {x + w * 0.65} {y} L {x + w} {y + h * 0.25} L {x + w} {y + h} L {x} {y + h} Z" '
        f'fill="none" stroke="{color}" stroke-width="1.3"/>'
        f'<path d="M {x + w * 0.65} {y} L {x + w * 0.65} {y + h * 0.25} L {x + w} {y + h * 0.25}" '
        f'fill="none" stroke="{color}" stroke-width="1.1"/>'
        f'<line x1="{x + 2}" y1="{y + h * 0.5}" x2="{x + w - 2}" y2="{y + h * 0.5}" stroke="{color}" stroke-width="1"/>'
        f'<line x1="{x + 2}" y1="{y + h * 0.68}" x2="{x + w - 2}" y2="{y + h * 0.68}" stroke="{color}" stroke-width="1"/>'
    )


def icon_gear(x, y, r=6.0, color="#666666"):
    import math

    parts = [f'<circle cx="{x}" cy="{y}" r="{r * 0.55}" fill="none" stroke="{color}" stroke-width="1.5"/>']
    for i in range(8):
        a = math.pi / 4 * i
        x1, y1 = x + r * 0.72 * math.cos(a), y + r * 0.72 * math.sin(a)
        x2, y2 = x + r * 1.08 * math.cos(a), y + r * 1.08 * math.sin(a)
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="1.7"/>'
        )
    return "".join(parts)




# ---------------------------------------------------------------------------
# Paper-specific helpers (palette, metric-based wrapping, auto-sized cards)
# ---------------------------------------------------------------------------
from reportlab.pdfbase.pdfmetrics import stringWidth  # noqa: E402

BLUE, NAVY, TEAL, PURPLE, ORANGE, GREEN, RED = "#2865A7", "#17324A", "#087F83", "#76519A", "#BD641B", "#337857", "#B5383C"
GREY, GREY_D = "#8A94A0", "#4A5560"


def tint(hex_color: str, a: float = 0.90) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = lambda c: int(round(c + (255 - c) * a))
    return f"#{f(r):02X}{f(g):02X}{f(b):02X}"


def tw(s: str, size: float, bold: bool = False, mono: bool = False) -> float:
    return stringWidth(s, "Courier" if mono else ("Helvetica-Bold" if bold else "Helvetica"), size)


def wrap(s: str, width: float, size: float = 13, bold: bool = False, mono: bool = False) -> list[str]:
    out: list[str] = []
    for para in s.split("\n"):
        cur = ""
        for word in para.split(" "):
            t = (cur + " " + word).strip()
            if tw(t, size, bold, mono) <= width or not cur:
                cur = t
            else:
                out.append(cur)
                cur = word
        out.append(cur)
    return out


def card(x, y, w, title=None, body="", h=None, color=BLUE, fill=None, tsize=16.5, bsize=14.5, mono=False, dash=None,
         pad=10, lh=1.28, title_color=None, body_color=C_TXT, rx=8, sw=1.4, center=False):
    """Rounded card with a bold title and a metric-wrapped body. Returns (svg, (x, y, w, h))."""
    inner = w - 2 * pad
    tl = wrap(title, inner, tsize, True) if title else []
    bl = wrap(body, inner, bsize, False, mono) if body else []
    need = pad + len(tl) * tsize * lh + (4 if tl and bl else 0) + len(bl) * bsize * lh + pad - 2
    h = max(h or 0, need)
    s = R(x, y, w, h, rx=rx, fill=fill or tint(color), stroke=color, sw=sw, dash=dash)
    yy = y + pad + tsize * 0.85
    ax, anc = (x + w / 2, "middle") if center else (x + pad, "start")
    for ln in tl:
        s += T(ax, round(yy, 1), ln, size=tsize, fill=title_color or color, anchor=anc, weight="bold")
        yy += tsize * lh
    if tl and bl:
        yy += 4
    for ln in bl:
        s += T(ax, round(yy, 1), ln, size=bsize, fill=body_color, anchor=anc, font=MONO if mono else None)
        yy += bsize * lh
    return s, (x, y, w, h)


def pending_tag(x, y, text="pending", w=None, size=14.5):
    w = w or tw(text, size, True) + 14
    return R(x, y, w, size + 8, rx=4, fill=tint(RED, 0.9), stroke=RED, sw=1.1) + T(x + w / 2, y + size + 1, text, size=size, fill=RED, weight="bold")


def tag(x, y, text, color, size=14.5, w=None):
    w = w or tw(text, size, True) + 14
    return R(x, y, w, size + 8, rx=4, fill=tint(color, 0.88), stroke=color, sw=1.1) + T(x + w / 2, y + size + 1, text, size=size, fill=color, weight="bold")


def panel(x, y, letter, title, color=NAVY, size=16):
    return T(x, y, letter, size=size + 1, fill=color, anchor="start", weight="bold") + T(x + 20, y, title, size=size, fill=color, anchor="start", weight="bold")


def save_svg(name: str, w: int, h: int, body: str) -> None:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    (SVG_DIR / f"{name}.svg").write_text(svg(w, h, body), encoding="utf-8")
