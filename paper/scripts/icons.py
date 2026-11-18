"""Small vector pictograms (schematic, NOT rendered images). Each returns an SVG fragment centred on (cx, cy)."""
from __future__ import annotations
from diagram_common import R, L, T, tint, NAVY, GREY, GREY_D, RED, GREEN

DARK = "#2B3A48"


def _c(cx, cy, r, fill, stroke="none", sw=0):
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def _poly(pts, fill, stroke, sw=1.5):
    return f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def _ln(x1, y1, x2, y2, color, sw):
    """Pictogram stroke drawn as a path (so the arrow audit does not read it as a leader line)."""
    return f'<path d="M {x1:.1f},{y1:.1f} L {x2:.1f},{y2:.1f}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>'


def eye(cx, cy, s, color):
    w, h = s * 0.55, s * 0.30
    d = f"M {cx - w:.1f},{cy:.1f} Q {cx:.1f},{cy - 2 * h:.1f} {cx + w:.1f},{cy:.1f} Q {cx:.1f},{cy + 2 * h:.1f} {cx - w:.1f},{cy:.1f} Z"
    return f'<path d="{d}" fill="#FFFFFF" stroke="{color}" stroke-width="2.4"/>' + _c(cx, cy, s * 0.2, color) + _c(cx, cy, s * 0.085, "#FFFFFF")


def plan(cx, cy, s, color):
    x, y, w, h = cx - s * 0.34, cy - s * 0.44, s * 0.68, s * 0.88
    o = R(x, y, w, h, rx=5, fill="#FFFFFF", stroke=color, sw=2.2)
    for i in range(3):
        yy = y + h * (0.22 + 0.27 * i)
        o += R(x + w * 0.12, yy - 4, 8, 8, rx=2, fill=tint(color, 0.5), stroke=color, sw=1.0)
        o += L(x + w * 0.12 + 14, yy, x + w * 0.88, yy, color=color, sw=2.2)
    return o


def arm(cx, cy, s, color):
    p0 = (cx - s * 0.34, cy + s * 0.36)
    p1 = (cx - s * 0.02, cy - s * 0.22)
    p2 = (cx + s * 0.34, cy + s * 0.02)
    o = R(p0[0] - s * 0.14, p0[1], s * 0.28, s * 0.1, rx=2, fill=color, stroke="none", sw=0)
    o += _ln(p0[0], p0[1], p1[0], p1[1], color, 7) + _ln(p1[0], p1[1], p2[0], p2[1], color, 7)
    o += _ln(p2[0], p2[1], p2[0] - s * 0.07, p2[1] + s * 0.2, color, 4.5) + _ln(p2[0], p2[1], p2[0] + s * 0.1, p2[1] + s * 0.2, color, 4.5)
    return o + _c(*p1, 5, "#FFFFFF", color, 2) + _c(*p2, 5, "#FFFFFF", color, 2)


def notebook(cx, cy, s, color):
    x, y, w, h = cx - s * 0.3, cy - s * 0.42, s * 0.6, s * 0.84
    o = R(x, y, w, h, rx=3, fill=tint(color, 0.8), stroke=color, sw=2.0)
    for i in range(4):
        o += _c(x, y + h * (0.16 + 0.22 * i), 3.2, "#FFFFFF", color, 1.6)
    for i in range(3):
        o += L(x + w * 0.28, y + h * (0.3 + 0.2 * i), x + w * 0.85, y + h * (0.3 + 0.2 * i), color=color, sw=1.6)
    return o


def pencilcase(cx, cy, s, color):
    x, y, w, h = cx - s * 0.46, cy - s * 0.2, s * 0.92, s * 0.4
    o = R(x, y, w, h, rx=9, fill=tint(color, 0.8), stroke=color, sw=2.0)
    o += L(x + 8, y + h * 0.32, x + w - 8, y + h * 0.32, color=color, sw=1.6, dash="4 3")
    o += _c(x + w * 0.78, y + h * 0.32, 3.4, color)
    o += _poly([(x + w * 0.3, y - 2), (x + w * 0.38, y - s * 0.22), (x + w * 0.46, y - 2)], tint(color, 0.5), color, 1.6)
    return o


def book(cx, cy, s, color):
    x, y, w, h = cx - s * 0.32, cy - s * 0.42, s * 0.64, s * 0.84
    o = R(x, y, w, h, rx=3, fill=tint(color, 0.78), stroke=color, sw=2.0)
    o += L(x + w * 0.2, y, x + w * 0.2, y + h, color=color, sw=1.8)
    o += R(x + w * 0.38, y + h * 0.2, w * 0.4, h * 0.18, rx=2, fill="#FFFFFF", stroke=color, sw=1.2)
    return o


def laptop(cx, cy, s, color):
    o = R(cx - s * 0.36, cy - s * 0.3, s * 0.72, s * 0.5, rx=3, fill=tint(color, 0.85), stroke=color, sw=2.0)
    o += R(cx - s * 0.46, cy + s * 0.22, s * 0.92, s * 0.1, rx=3, fill=color, stroke="none", sw=0)
    return o


def bottle(cx, cy, s, color):
    o = R(cx - s * 0.16, cy - s * 0.46, s * 0.32, s * 0.1, rx=2, fill=color, stroke="none", sw=0)
    o += R(cx - s * 0.1, cy - s * 0.38, s * 0.2, s * 0.14, rx=2, fill=tint(color, 0.7), stroke=color, sw=1.6)
    o += R(cx - s * 0.2, cy - s * 0.26, s * 0.4, s * 0.72, rx=8, fill=tint(color, 0.8), stroke=color, sw=2.0)
    return o


def bag(x, y, w, h, color):
    o = R(x, y, w, h, rx=14, fill="#FFFFFF", stroke=color, sw=2.6)
    o += R(x + w * 0.12, y - 9, w * 0.76, 14, rx=7, fill="#FFFFFF", stroke=color, sw=2.2)
    o += R(x + w * 0.16, y + h * 0.72, w * 0.68, h * 0.2, rx=5, fill=tint(color, 0.9), stroke=color, sw=1.6)
    return o


def mark(cx, cy, ok, r=11):
    col = GREEN if ok else RED
    o = _c(cx, cy, r, col)
    if ok:
        o += f'<polyline points="{cx - r * 0.5:.1f},{cy:.1f} {cx - r * 0.1:.1f},{cy + r * 0.42:.1f} {cx + r * 0.55:.1f},{cy - r * 0.4:.1f}" fill="none" stroke="#FFFFFF" stroke-width="2.6"/>'
    else:
        o += _ln(cx - r * 0.4, cy - r * 0.4, cx + r * 0.4, cy + r * 0.4, "#FFFFFF", 2.6) + _ln(cx - r * 0.4, cy + r * 0.4, cx + r * 0.4, cy - r * 0.4, "#FFFFFF", 2.6)
    return o


def qmark(cx, cy, color, r=15):
    return _c(cx, cy, r, color) + T(cx, cy + 7, "?", size=21, fill="#FFFFFF", weight="bold")


def docs(cx, cy, s, color):
    o = ""
    for k, (dx, dy) in enumerate(((-0.12, 0.10), (0.0, 0.0), (0.12, -0.10))):
        o += R(cx + s * (dx - 0.26), cy + s * (dy - 0.34), s * 0.5, s * 0.66, rx=3, fill="#FFFFFF" if k == 2 else tint(color, 0.88), stroke=color, sw=1.8)
    for i in range(3):
        o += L(cx + s * (0.12 - 0.20), cy + s * (-0.34 - 0.10 + 0.16 + 0.13 * i), cx + s * (0.12 + 0.12), cy + s * (-0.34 - 0.10 + 0.16 + 0.13 * i), color=color, sw=1.6)
    return o


def person(cx, cy, s, color):
    return _c(cx, cy - s * 0.18, s * 0.17, color) + f'<path d="M {cx - s * 0.32:.1f},{cy + s * 0.42:.1f} Q {cx:.1f},{cy - s * 0.12:.1f} {cx + s * 0.32:.1f},{cy + s * 0.42:.1f} Z" fill="{color}"/>'


def gate(cx, cy, s, color):
    d = s * 0.42
    o = _poly([(cx, cy - d), (cx + d, cy), (cx, cy + d), (cx - d, cy)], "#FFFFFF", color, 2.2)
    return o + f'<polyline points="{cx - d * 0.45:.1f},{cy:.1f} {cx - d * 0.1:.1f},{cy + d * 0.38:.1f} {cx + d * 0.5:.1f},{cy - d * 0.34:.1f}" fill="none" stroke="{color}" stroke-width="3"/>'


def ladder(cx, cy, s, color):
    o = L(cx - s * 0.2, cy - s * 0.4, cx - s * 0.2, cy + s * 0.4, color=color, sw=3) + L(cx + s * 0.2, cy - s * 0.4, cx + s * 0.2, cy + s * 0.4, color=color, sw=3)
    for i in range(4):
        y = cy - s * 0.3 + i * s * 0.2
        o += L(cx - s * 0.2, y, cx + s * 0.2, y, color=color, sw=2.4)
    return o


def world(cx, cy, w, h, color):
    o = R(cx - w / 2, cy - h / 2, w, h, rx=10, fill="#F6F7F9", stroke=color, sw=2.0)
    o += laptop(cx - w * 0.3, cy - 4, h * 0.46, GREY_D) + bottle(cx - w * 0.08, cy - 2, h * 0.46, GREY_D) + notebook(cx + w * 0.12, cy - 2, h * 0.46, GREY_D) + pencilcase(cx + w * 0.34, cy + 2, h * 0.46, GREY_D)
    return o
