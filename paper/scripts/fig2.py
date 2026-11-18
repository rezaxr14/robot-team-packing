"""Figure 2 (v6): the team as a closed loop around a drawn world, with real data in every module.

Rendered by cairosvg (gradients, clipping, rotation, Inter). All numbers come from generated/figdata.json:
  - detections: the Eye masses the planner used (mechanism, seed 24)
  - message stream: the recorded loop episode (seed 11), capsule width = approximate tokens
  - grip windows: friction, mass and fragility of the seed-24 items, with every recorded grasp force
Design system: Inter; ink #14202B; teal = perceive, purple = plan, orange = act, green = verify, gold = responsibility.
"""
from __future__ import annotations
import json, math
from html import escape
from PIL import ImageFont
from diagram_common import PAPER_DIR, SVG_DIR, arrow, tint

FD = json.loads((PAPER_DIR / "generated" / "figdata.json").read_text())
INK, MUTE, LINE = "#14202B", "#5B6875", "#CBD3DB"
TEAL, PURPLE, ORANGE, GREEN, GOLD, RED, NAVY, BLUE = "#087F83", "#76519A", "#BD641B", "#337857", "#A87900", "#B5383C", "#17324A", "#2865A7"
FAM = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
_W = {400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold"}
_FC: dict = {}


def tw(s, size, w=400):
    k = (size, w)
    if k not in _FC:
        _FC[k] = ImageFont.truetype(f"/usr/share/fonts/opentype/inter/Inter-{_W[w]}.otf", int(size * 8))
    return _FC[k].getlength(s) / 8.0


def wrap(s, width, size, w=400):
    out, cur = [], ""
    for word in s.split(" "):
        t = (cur + " " + word).strip()
        if tw(t, size, w) <= width or not cur:
            cur = t
        else:
            out.append(cur)
            cur = word
    return out + [cur]


def clip(s, width, size, w=400):
    if tw(s, size, w) <= width:
        return s
    while s and tw(s + "...", size, w) > width:
        s = s[:-1]
    return s + "..."


def T(x, y, s, size=13.5, w=400, fill=INK, anchor="start", rot=None, italic=False):
    tr = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot else ""
    st = ' font-style="italic"' if italic else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAM}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}"{st}{tr}>{escape(s)}</text>'


def R(x, y, w, h, r=0, fill="none", stroke="none", sw=1, dash=None, op=None, extra=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' opacity="{op}"' if op is not None else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o} {extra}/>'


def C(x, y, r, fill="none", stroke="none", sw=1, op=None, dash=None):
    o = f' opacity="{op}"' if op is not None else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{o}{d}/>'


def Ln(x1, y1, x2, y2, col=LINE, sw=1.2, dash=None, cap="round"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="M {x1:.1f},{y1:.1f} L {x2:.1f},{y2:.1f}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="{cap}"{d}/>'


def P(d, fill="none", stroke="none", sw=1, op=None, cap="round"):
    o = f' opacity="{op}"' if op is not None else ""
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}" stroke-linejoin="round"{o}/>'


def poly(pts, fill="none", stroke="none", sw=1, op=None):
    o = f' opacity="{op}"' if op is not None else ""
    return f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"{o}/>'


DEFS = f"""<defs>
<linearGradient id="wood" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#D9B88E"/><stop offset="1" stop-color="#BE9566"/></linearGradient>
<linearGradient id="floor" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F3F5F7"/><stop offset="1" stop-color="#E6EAEE"/></linearGradient>
<linearGradient id="bagc" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#34597F"/><stop offset="1" stop-color="#1D3750"/></linearGradient>
<linearGradient id="screen" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2C3A4A"/><stop offset="1" stop-color="#0F1822"/></linearGradient>
<linearGradient id="metal" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#E7924A"/><stop offset="1" stop-color="#BD641B"/></linearGradient>
<linearGradient id="nbk" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#4A7FB5"/><stop offset="1" stop-color="#2F5F92"/></linearGradient>
<linearGradient id="bookg" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#3E8666"/><stop offset="1" stop-color="#2A6048"/></linearGradient>
<linearGradient id="pcg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#EBC45A"/><stop offset="1" stop-color="#D2A33A"/></linearGradient>
<linearGradient id="lbg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#D86A5E"/><stop offset="1" stop-color="#B44A40"/></linearGradient>
<radialGradient id="botg" cx="0.35" cy="0.35" r="0.8"><stop offset="0" stop-color="#CFE7F5"/><stop offset="1" stop-color="#7FB3D3"/></radialGradient>
<linearGradient id="cone" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="{TEAL}" stop-opacity="0.28"/><stop offset="1" stop-color="{TEAL}" stop-opacity="0.04"/></linearGradient>
<linearGradient id="tube" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#EEF2F6"/><stop offset="1" stop-color="#DCE3EA"/></linearGradient>
</defs>"""


# ------------------------------------------------------------------ drawn objects (top-down), centred at (cx, cy)
def o_laptop(cx, cy, s):
    w, h = s, s * 0.72
    o = R(cx - w / 2, cy - h / 2, w, h, 4, "#C9D1D9", "#8E99A5", 1.2)
    o += R(cx - w * 0.46, cy - h * 0.46, w * 0.92, h * 0.5, 2, "url(#screen)")
    o += Ln(cx - w * 0.36, cy - h * 0.3, cx - w * 0.1, cy - h * 0.3, "#6C7F93", 1.4)
    for i in range(3):
        o += Ln(cx - w * 0.4, cy + h * (0.12 + 0.11 * i), cx + w * 0.4, cy + h * (0.12 + 0.11 * i), "#9AA5B1", 1.6)
    return o


def o_notebook(cx, cy, s):
    w, h = s * 0.62, s * 0.84
    o = R(cx - w / 2, cy - h / 2, w, h, 3, "url(#nbk)", "#274E78", 1.2)
    for i in range(5):
        o += C(cx - w / 2 + 1, cy - h * 0.38 + i * h * 0.19, 2.2, "#F2F5F8", "#274E78", 0.8)
    o += R(cx - w * 0.1, cy - h * 0.2, w * 0.45, h * 0.26, 2, "#F2F5F8", op=0.85)
    o += Ln(cx + w * 0.3, cy - h / 2, cx + w * 0.3, cy + h / 2, "#274E78", 1.4)
    return o


def o_pencilcase(cx, cy, s):
    w, h = s, s * 0.4
    o = poly([(cx - w * 0.22, cy - h / 2), (cx - w * 0.14, cy - h / 2 - s * 0.2), (cx - w * 0.06, cy - h / 2)], "#E8D6B0", "#9A7B3A", 1)
    o += R(cx - w / 2, cy - h / 2, w, h, 9, "url(#pcg)", "#9A7B3A", 1.2)
    o += Ln(cx - w * 0.4, cy - h * 0.12, cx + w * 0.4, cy - h * 0.12, "#8A6D2E", 1.2, "4 3")
    o += C(cx + w * 0.34, cy - h * 0.12, 2.6, "#8A6D2E")
    return o


def o_book(cx, cy, s):
    w, h = s * 0.66, s * 0.86
    o = R(cx - w / 2, cy - h / 2, w, h, 3, "url(#bookg)", "#214A38", 1.2)
    o += R(cx - w / 2, cy - h / 2, w * 0.16, h, 2, "#214A38", op=0.55)
    o += R(cx - w * 0.2, cy - h * 0.28, w * 0.5, h * 0.2, 2, "#E9F1EC", op=0.85)
    o += Ln(cx - w * 0.2, cy + h * 0.02, cx + w * 0.28, cy + h * 0.02, "#E9F1EC", 1.4)
    return o


def o_bottle(cx, cy, s):
    r = s * 0.24
    return C(cx, cy, r, "url(#botg)", "#4F86A8", 1.2) + C(cx, cy, r * 0.5, "#DCEFFA", "#4F86A8", 1) + C(cx - r * 0.3, cy - r * 0.3, r * 0.14, "#FFFFFF", op=0.9)


def o_lunchbox(cx, cy, s):
    w, h = s * 0.84, s * 0.56
    o = R(cx - w / 2, cy - h / 2, w, h, 6, "url(#lbg)", "#8E352D", 1.2)
    o += R(cx - w * 0.14, cy - h / 2 - 3, w * 0.28, 7, 3, "#8E352D")
    o += Ln(cx - w * 0.4, cy, cx + w * 0.4, cy, "#8E352D", 1, "3 3")
    return o


# ------------------------------------------------------------------ modules (v7: three tight columns, text >= 14.5 units)
BODY, SMALL, SUB, TITLE = 14.5, 14, 14.5, 18


def hero():
    """The packing scene, top-down; drawn in 350..650 x 88..540 and scaled into the centre column."""
    o = R(350, 88, 300, 452, 14, "url(#floor)", "#C3CCD5", 1.4)
    for gx in range(375, 650, 25):
        o += Ln(gx, 92, gx, 536, "#E2E7EC", 1)
    for gy in range(113, 540, 25):
        o += Ln(354, gy, 646, gy, "#E2E7EC", 1)
    o += R(452, 106, 96, 112, 12, "url(#bagc)", "#10243A", 1.6)
    o += R(458, 112, 84, 24, 6, "#5E83A8", op=0.55) + R(458, 140, 84, 52, 6, "#0F2133", op=0.9) + R(458, 196, 84, 18, 6, "#5E83A8", op=0.55)
    o += Ln(462, 205, 538, 205, "#C9D6E3", 1.2, "3 3")
    o += T(444, 130, "laptop slot", 16, 500, MUTE, "end") + T(444, 174, "main", 16, 500, MUTE, "end") + T(556, 210, "front pocket", 16, 500, MUTE)
    o += R(366, 262, 268, 120, 10, "url(#wood)", "#9C7A52", 1.6)
    o += Ln(455, 268, 455, 376, "#F1E2CB", 1, "4 4") + Ln(545, 268, 545, 376, "#F1E2CB", 1, "4 4")
    o += o_laptop(410, 304, 58) + o_book(500, 304, 50) + o_bottle(590, 304, 50)
    o += o_notebook(410, 352, 46) + o_pencilcase(500, 358, 54) + o_lunchbox(590, 352, 52)
    for cx_, sgn in ((372, 1), (628, -1)):
        o += poly([(cx_, 508), (500 - sgn * 120, 270), (500 + sgn * 20, 270)], "url(#cone)", "none")
        o += Ln(cx_, 508, 500 - sgn * 120, 270, TEAL, 1, "5 4") + Ln(cx_, 508, 500 + sgn * 20, 270, TEAL, 1, "5 4")
    for bx, ex, gx in ((410, 372, 404), (590, 628, 596)):
        o += Ln(bx, 476, ex, 424, "#8C4710", 13) + Ln(bx, 476, ex, 424, "url(#metal)", 9) + Ln(ex, 424, gx, 384, "#8C4710", 12) + Ln(ex, 424, gx, 384, "url(#metal)", 8)
        o += C(bx, 478, 15, "#3A2A1C", "#1F160D", 1.5) + C(bx, 478, 6, "#BD641B") + C(ex, 424, 6, "#F7E6D4", "#8C4710", 1.5)
        o += Ln(gx - 7, 384, gx - 7, 370, "#3A2A1C", 3.4) + Ln(gx + 7, 384, gx + 7, 370, "#3A2A1C", 3.4)
    for cx_ in (372, 628):
        o += R(cx_ - 13, 504, 26, 20, 5, "#26323E", "#0F1822", 1.2) + C(cx_, 514, 6, "#0A0F15", "#6FD0D6", 1.6)
    o += T(372, 537, "Eye A", 16, 600, TEAL, "middle") + T(628, 537, "Eye B", 16, 600, TEAL, "middle")
    return f'<g transform="translate(500 88) scale(0.933) translate(-500 -88)">{o}</g>'


def eyes_module():
    items = {i["name"]: i for i in FD["mechanism"]["items"]}
    o = T(20, 42, "Eyes read the scene", TITLE, 600, TEAL) + T(20, 63, "pixels in, one typed claim per object out", SUB, 400, MUTE)
    x0, y0, w, h = 20, 76, 290, 170
    o += R(x0, y0, w, h, 6, "url(#wood)", INK, 2)
    o += Ln(x0 + 97, y0 + 4, x0 + 97, y0 + h - 4, "#F1E2CB", 1, "4 4") + Ln(x0 + 193, y0 + 4, x0 + 193, y0 + h - 4, "#F1E2CB", 1, "4 4")
    place = [(o_laptop, 50, 56, 56), (o_book, 146, 56, 50), (o_notebook, 242, 58, 46), (o_pencilcase, 50, 124, 54), (o_bottle, 146, 124, 50), (o_lunchbox, 242, 124, 52)]
    for fn, px, py, s in place:
        o += fn(x0 + px, y0 + py, s)
    shown = [("book1", 146, 56, 52, 56, True), ("notebook1", 242, 58, 50, 56, True), ("pencilcase1", 50, 124, 62, 34, False), ("bottle1", 146, 124, 44, 44, True)]
    labels = {"book1": "book", "notebook1": "notebook", "pencilcase1": "pencil case", "bottle1": "bottle"}
    for name, px, py, bw, bh, above in shown:
        it = items[name]
        col = RED if abs(it["eye_mass"] - it["true_mass"]) / it["true_mass"] > 0.3 else TEAL
        bx, by = x0 + px - bw / 2, y0 + py - bh / 2
        for (ax, ay, dx, dy) in ((bx, by, 1, 1), (bx + bw, by, -1, 1), (bx, by + bh, 1, -1), (bx + bw, by + bh, -1, -1)):
            o += P(f"M {ax + dx * 9:.1f},{ay:.1f} L {ax:.1f},{ay:.1f} L {ax:.1f},{ay + dy * 9:.1f}", stroke=col, sw=2.4)
        txt = f"{it['eye_mass']:.2f} kg"
        cw = tw(txt, BODY, 600) + 10
        cxp = min(max(x0 + px - cw / 2, x0 + 3), x0 + w - cw - 3)
        cyp = by - 21 if above else by + bh + 3
        o += R(cxp, cyp, cw, 19, 4, col) + T(cxp + 5, cyp + 14.5, txt, BODY, 600, "#FFFFFF")
    o += T(20, 270, "Eye mass estimate; red = far off", BODY, 400, MUTE)
    o += R(x0 + w - 62, y0 + h + 8, 62, 21, 4, INK, op=0.85) + T(x0 + w - 31, y0 + h + 23.5, "Eye A", BODY, 600, "#FFFFFF", "middle")
    x, y = 20, 284
    for tag_ in ("class", "region", "mass", "fragility", "confidence"):
        wd = tw(tag_, SMALL, 500) + 16
        if x + wd > 312:
            x, y = 20, y + 26
        o += R(x, y, wd, 22, 11, "#E5F2F2", TEAL, 1) + T(x + wd / 2, y + 16, tag_, SMALL, 500, TEAL, "middle")
        x += wd + 6
    return o


def stream_module():
    ev = [e for e in FD["loop"]["events"] if e["ev"] == "msg"]
    grp = {"SCENE": TEAL, "LOOK": PURPLE, "ABILITY": ORANGE, "TOUCH": ORANGE, "STATUS": ORANGE, "CHALLENGE": "#9C3F12", "CLAIM": ORANGE, "AMEND": "#4A5560", "LOCK": ORANGE}
    y = 372
    o = T(20, y, "One bus carries every claim", TITLE, 600, NAVY) + T(20, y + 21, "a recorded episode; capsule width = tokens", SUB, 400, MUTE)
    tx0, ty0, tw_, th = 50, y + 40, 240, 40
    o += R(tx0, ty0, tw_, th, 20, "url(#tube)", NAVY, 1.6)
    scale, x, seen, toks = 0.052, tx0 + 12, [], {}
    for e in ev:
        t = max(10, math.ceil(len(json.dumps(e["content"])) / 4) + 8)
        wcap = max(t * scale, 3.4)
        if x + wcap > tx0 + tw_ - 12:
            break
        o += R(x, ty0 + 7, wcap - 1.2, th - 14, 2, grp.get(e["type"], "#4A5560"), op=0.95)
        toks.setdefault(e["type"], []).append(t)
        if e["type"] not in seen:
            seen.append(e["type"])
        x += wcap
    o += R(16, ty0 + 5, 26, 30, 6, "#FFFFFF", NAVY, 1.4) + P(f"M 22,{ty0 + 20} L 27,{ty0 + 26} L 37,{ty0 + 12}", stroke=GREEN, sw=2.6)
    o += T(29, ty0 + 56, "schema", SMALL, 500, MUTE, "middle")
    o += R(tx0 + tw_ + 8, ty0 + 5, 22, 30, 4, "#FFFFFF", NAVY, 1.4) + R(tx0 + tw_ + 10, ty0 + 22, 18, 11, 2, NAVY, op=0.3)
    o += T(tx0 + tw_ + 19, ty0 + 56, "budget", SMALL, 500, MUTE, "middle")
    xl, yl = 20, ty0 + 84
    for t_ in seen:
        wd = tw(t_, SMALL, 600) + 26
        if xl + wd > 312:
            xl, yl = 20, yl + 22
        o += C(xl + 6, yl - 5, 5, grp.get(t_, "#4A5560")) + T(xl + 16, yl, t_, SMALL, 600, INK)
        xl += wd
    sc_ = toks.get("SCENE", [0]); small = [t for k, v in toks.items() if k != "SCENE" for t in v]
    yy = yl + 30
    o += T(20, yy, f"SCENE reports: {min(sc_)}-{max(sc_)} tokens each;", BODY, 400, INK) + T(20, yy + 19, f"every other message: {min(small)}-{max(small)}.", BODY, 400, INK)
    o += T(20, yy + 46, "Capitalised arrows are bus messages.", BODY, 600, NAVY)
    return o


def brain_module():
    ins = FD["mechanism"]["instruction"].split(": ", 1)[1]
    ins = ins[ins.index("Heavy"):]
    rules = " ".join(FD["harness"]["retrieved_rules_new"][:4])
    sc = FD["messages"]["examples"]["SCENE"]["content"]["objects"][0]
    X = 690
    o = T(X, 42, "The Brain plans in text", TITLE, 600, PURPLE) + T(X, 63, "no pixels, only these three inputs", SUB, 400, MUTE)
    rows = [("instruction", "#6B7684", ins), ("harness", BLUE, rules), ("scene", TEAL, f"{sc['name']} {sc['region']} {sc['mass_guess']} kg, ...")]
    for k, (lab, col, txt) in enumerate(rows):
        y = 76 + k * 38
        o += R(X, y, 290, 32, 5, tint(col, 0.9), col, 1) + R(X, y, 92, 32, 5, col) + T(X + 46, y + 21.5, lab, BODY, 600, "#FFFFFF", "middle")
        o += T(X + 100, y + 21.5, clip(txt, 182, BODY), BODY, 400, INK)
    o += Ln(X + 145, 194, X + 145, 210, PURPLE, 2.4) + poly([(X + 139, 208), (X + 151, 208), (X + 145, 218)], PURPLE)
    cols = [("laptop1", X + 52), ("book2", X + 145), ("book1", X + 238)]
    for name, cx_ in cols:
        o += C(cx_, 244, 12, "#F8E9DA", ORANGE, 1.8) + T(cx_, 249.5, "g", BODY, 600, ORANGE, "middle")
        o += C(cx_, 290, 12, "#EFE7F6", PURPLE, 1.8) + T(cx_, 295.5, "p", BODY, 600, PURPLE, "middle")
        o += Ln(cx_, 256, cx_, 274, "#8A94A0", 1.6) + poly([(cx_ - 4, 272), (cx_ + 4, 272), (cx_, 279)], "#8A94A0")
        o += T(cx_, 322, name, BODY, 500, MUTE, "middle")
    for a_, c_ in ((X + 52, X + 145), (X + 145, X + 238)):
        o += Ln(a_ + 14, 290, c_ - 18, 290, "#8A94A0", 1.6) + poly([(c_ - 19, 286), (c_ - 19, 294), (c_ - 12, 290)], "#8A94A0")
    pw = tw("dry run: no violations", BODY, 600) + 24
    o += R(X, 334, pw, 26, 13, "#E6F2EC", GREEN, 1.2) + T(X + pw / 2, 352, "dry run: no violations", BODY, 600, GREEN, "middle")
    return o


SHORT = {"pencilcase1": "pencil", "notebook2": "note 2", "notebook1": "note 1", "lunchbox1": "lunch", "bottle1": "bottle", "book2": "book 2", "book1": "book 1", "laptop1": "laptop"}


def arms_module():
    m = FD["mechanism"]
    items = sorted([i for i in m["items"] if not i["distractor"]], key=lambda i: i["true_mass"])
    ex = [e for e in m["challenge"]["executed"] if e["task"]["skill"] == "grasp"]
    X = 690
    o = T(X, 466, "Arms hold inside a force window", TITLE, 600, ORANGE) + T(X, 487, "set by mass, friction and fragility", SUB, 400, MUTE)
    cx0, cy0, cw, ch = X + 34, 514, 256, 104
    y_of = lambda f: cy0 + ch - f / 32.0 * ch
    for f in (0, 10, 20, 30):
        o += Ln(cx0, y_of(f), cx0 + cw, y_of(f), "#E1E6EB", 1) + T(cx0 - 7, y_of(f) + 5, str(f), SMALL, 400, MUTE, "end")
    o += T(X - 2, cy0 - 8, "N", SMALL, 500, MUTE)
    slot = cw / len(items)
    for k, it in enumerate(items):
        cx_ = cx0 + slot * (k + 0.5)
        floor, ceil_ = (4 + 6 * it["true_mass"]) / it["friction"], 30 - 22 * it["true_frag"]
        o += R(cx_ - 9, y_of(ceil_), 18, y_of(floor) - y_of(ceil_), 5, "#DDEEE5", GREEN, 1.4)
        att = [e for e in ex if e["task"]["obj"] == it["name"]]
        for j, e in enumerate(att):
            o += C(cx_ + (j - (len(att) - 1) / 2) * 7, y_of(e["task"]["params"]["force"]), 4.6, GREEN if e["ok"] else RED, "#FFFFFF", 1.2)
        o += T(cx_ + 6, cy0 + ch + 17, SHORT[it["name"]], SMALL, 500, MUTE, "end", rot=-48)
    ly = cy0 + ch + 70
    o += C(X + 6, ly - 5, 6, GREEN) + T(X + 18, ly, "held", SMALL, 500, INK) + C(X + 66, ly - 5, 6, RED) + T(X + 78, ly, "slipped, retried", SMALL, 500, INK)
    o += R(X + 192, ly - 11, 12, 12, 3, "#DDEEE5", GREEN, 1.2) + T(X + 210, ly, "safe window", SMALL, 500, INK)
    return o


def responsibility_module():
    cr = FD["credit"]
    y0 = 722
    o = T(20, y0, "A failure becomes evidence about who was responsible", TITLE, 600, GOLD)
    o += T(20, y0 + 22, "The episode is replayed once per agent with an oracle swapped in; the agent whose oracle repairs it carries the responsibility.", SUB, 400, MUTE)
    yh = y0 + 62
    o += T(262, yh, "oracle swapped in", BODY, 600, MUTE, "middle") + T(436, yh, "repaired?", BODY, 600, MUTE, "middle") + T(560, yh, "gain in score", BODY, 600, MUTE, "middle")
    rows = [("Eye", True, cr["oracle_gain"]), ("Brain", False, 0.0), ("Arm", False, 0.0)]
    cols = (("Eye", TEAL), ("Brain", PURPLE), ("Arm", ORANGE))
    yc = y0 + 112
    o += C(44, yc, 17, RED) + P(f"M 37,{yc - 7} L 51,{yc + 7} M 51,{yc - 7} L 37,{yc + 7}", stroke="#FFFFFF", sw=3)
    o += T(70, yc - 4, "shared checker", BODY, 600, INK) + T(70, yc + 15, "rule broken", BODY, 500, RED)
    o += arrow([(166, yc), (204, yc)], color=GOLD, sw=2.4, free="both")
    for k, (nm, fixed, g) in enumerate(rows):
        y = y0 + 90 + k * 48
        for j, (nm2, col2) in enumerate(cols):
            x = 226 + j * 36
            if nm2 == nm:
                o += C(x, y + 14, 17, "none", GOLD, 2.4, dash="4 3")
            o += C(x, y + 14, 12, tint(col2, 0.8), col2, 1.8)
        yy = y + 14
        if fixed:
            o += R(412, yy + 1, 40, 14, 3, "#2F5F92") + R(419, yy - 13, 26, 12, 3, "#D2A33A")
        else:
            o += R(419, yy + 3, 26, 12, 3, "#D2A33A") + R(412, yy - 13, 40, 14, 3, "#2F5F92")
        o += (C(474, yy - 2, 8, GREEN) + P(f"M 470,{yy - 2} L 473,{yy + 1} L 478,{yy - 5}", stroke="#FFFFFF", sw=2) if fixed else C(474, yy - 2, 8, RED) + P(f"M 471,{yy - 5} L 477,{yy + 1} M 477,{yy - 5} L 471,{yy + 1}", stroke="#FFFFFF", sw=2))
        bw = g / 0.15 * 90
        o += R(504, yy - 8, 120, 16, 4, "#F3F5F7", LINE, 1) + (R(504, yy - 8, bw, 16, 4, GOLD) if g > 0 else Ln(505, yy - 8, 505, yy + 8, MUTE, 2))
        o += T(632, yy + 5, f"+{g:.2f}" if g > 0 else "0", BODY, 600, GOLD if g > 0 else MUTE)
    routes = [(TEAL, "Eye", "LoRA fine-tuning, rejection sampling (K = 8)"),
              (PURPLE, "Brain", "SFT, GRPO on the plan reward, relabeled failures"),
              (ORANGE, "Arms", "flow-matching distillation")]
    o += arrow([(676, y0 + 138), (706, y0 + 138)], color=GOLD, sw=2.4, free="both")
    for k, (col, nm, txt) in enumerate(routes):
        y = y0 + 104 + k * 48
        o += C(726, y + 0, 12, tint(col, 0.8), col, 1.8)
        words, line1, rest = txt.split(" "), "", []
        first_w = 984 - 746 - tw(nm + " ", BODY, 600)
        for i_, wd in enumerate(words):
            t_ = (line1 + " " + wd).strip()
            if tw(t_, BODY) <= first_w:
                line1 = t_
            else:
                rest = words[i_:]
                break
        o += T(746, y - 3, nm, BODY, 600, col) + T(746 + tw(nm + " ", BODY, 600), y - 3, line1, BODY, 400, INK)
        for j, ln in enumerate(wrap(" ".join(rest), 238, BODY)[:2]):
            o += T(746, y + 15 + j * 17, ln, BODY, 400, INK)
    return o


def flows():
    o = ""
    o += arrow([(358, 176), (312, 176)], color=MUTE, sw=2.4, free="both") + T(336, 164, "images", SMALL, 600, MUTE, "middle")
    o += arrow([(314, 46), (686, 46)], color=NAVY, sw=2.4, free="both") + T(500, 38, "SCENE", BODY, 600, NAVY, "middle")
    o += arrow([(686, 70), (314, 70)], color=PURPLE, sw=2.4, free="both") + T(500, 86, "LOOK", BODY, 600, PURPLE, "middle")
    o += arrow([(722, 366), (722, 446)], color=GREEN, sw=2.4, free="both") + T(734, 394, "verified plan,", BODY, 600, GREEN) + T(734, 413, "tasks", BODY, 600, GREEN)
    o += arrow([(950, 446), (950, 366)], color=ORANGE, sw=2.4, free="both") + T(940, 394, "TOUCH,", BODY, 600, ORANGE, "end") + T(940, 413, "CHALLENGE", BODY, 600, ORANGE, "end")
    o += arrow([(688, 572), (664, 572), (664, 470), (643, 470)], color=ORANGE, sw=2.4, free="both") + T(650, 458, "skills", SMALL, 600, ORANGE)
    o += arrow([(540, 516), (540, 664), (8, 664), (8, y_checker()), (26, y_checker())], color=GOLD, sw=2.4, free="both") + T(274, 656, "episode outcome", BODY, 600, GOLD, "middle")
    return o


def y_checker():
    return 722 + 112


def build():
    H = 962
    body = DEFS + R(0, 0, 1000, H, 0, "#FFFFFF")
    body += hero() + eyes_module() + brain_module() + stream_module() + arms_module() + responsibility_module() + flows()
    body += T(364, 556, "illustrative render", SMALL, 500, MUTE)
    svg = f'<?xml version="1.0" encoding="UTF-8"?>\n<!-- engine:cairo -->\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {H}" width="1000" height="{H}">{body}</svg>'
    (SVG_DIR / "fig_overview.svg").write_text(svg, encoding="utf-8")


def fig_overview():
    build()
