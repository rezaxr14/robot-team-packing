"""Figure 4: the harness. (a) plain files and one opened, (b) ranked retrieval, (c) the tokenizer fix, (d) write-back loop.

Real data: the 19 harness files, the opened skill file, BM25 scores of the six rules for the recorded instruction,
before and after the tokenizer fix (the pre-fix tokenizer is a reconstruction, see the paper).
"""
from __future__ import annotations
import json, textwrap
from fig2 import (FD, DEFS, T, R, C, Ln, P, poly, tw, wrap, tint, arrow, SVG_DIR,
                  INK, MUTE, LINE, TEAL, PURPLE, ORANGE, GREEN, GOLD, RED, NAVY, BLUE)
from fig3 import M, Mseg

BODY, SMALL, SUB, TITLE = 14.5, 14, 14.5, 18
HC = {"rule": BLUE, "skill": GREEN, "object": TEAL, "agent": PURPLE, "memory": ORANGE, "task": "#4A5560"}
FOLDER = {"rule": "rules", "skill": "skills", "object": "objects", "agent": "agents", "memory": "memory", "task": "tasks"}


def files_section():
    h = FD["harness"]
    docs = h["docs"]
    o = T(20, 38, "The Brain's knowledge is plain files", TITLE, 600, NAVY)
    o += T(20, 60, f"{h['n_files']} markdown files; an id, a type and keywords for retrieval, and a body the Brain reads", SUB, 400, MUTE)
    cols = [("agent", "memory", "object", "task"), ("rule", "skill")]
    xs = [20, 190]
    y_end = 0
    for ci, folders in enumerate(cols):
        y = 98
        for ty in folders:
            ids = [d["id"] for d in docs if d["type"] == ty]
            o += R(xs[ci], y - 12, 14, 11, 2, HC[ty]) + T(xs[ci] + 22, y, f"{FOLDER[ty]}/", BODY, 600, HC[ty]) + T(xs[ci] + 22 + tw(f"{FOLDER[ty]}/ ", BODY, 600), y, f"{len(ids)}", SMALL, 400, MUTE)
            for k, i in enumerate(ids):
                yy = y + 21 * (k + 1)
                o += Ln(xs[ci] + 7, yy - 14 if k == 0 else yy - 21 + 4, xs[ci] + 7, yy - 4, tint(HC[ty], 0.5), 1.4) + Ln(xs[ci] + 7, yy - 4, xs[ci] + 17, yy - 4, tint(HC[ty], 0.5), 1.4)
                if i == "grasp":
                    o += R(xs[ci] + 20, yy - 16, tw(i, SMALL, 600) + 12, 19, 5, tint(GREEN, 0.85), GREEN, 1.2) + T(xs[ci] + 26, yy - 1, i, SMALL, 600, GREEN)
                else:
                    o += T(xs[ci] + 24, yy - 1, i, SMALL, 400, INK)
            y += 21 * (len(ids) + 1) + 10
        y_end = max(y_end, y)
    # the opened file
    g = next(d for d in docs if d["id"] == "grasp")
    X = 372
    o += Ln(344, 118 + 21 * 3 + 6, 362, 118 + 21 * 3 + 6, GREEN, 2.2) if False else ""
    o += T(X, 98, "skills/grasp.md", BODY, 600, GREEN) + R(X - 10, 108, 4, 232, 2, GREEN)
    body = textwrap.wrap(g["text"], 40)
    lines = [[("---", MUTE)], [("id: ", MUTE), ("grasp", INK)], [("type: ", MUTE), ("skill", GREEN)], [("keywords: ", MUTE)], [(g["keywords"], BLUE)], [("---", MUTE)]]
    kw = textwrap.wrap(g["keywords"], 30)
    lines = [[("---", MUTE)], [("id: ", MUTE), ("grasp", INK)], [("type: ", MUTE), ("skill", GREEN)], [("keywords: ", MUTE), (kw[0], BLUE)]] + [[(k, BLUE)] for k in kw[1:]] + [[("---", MUTE)]]
    lines += [[(b_, INK)] for b_ in body]
    for k, segs in enumerate(lines):
        o += Mseg(X, 128 + k * 18, segs)
    # annotations with leaders (one line each, right of the longest code line)
    ax = 744
    notes = [(1, "id", "the name the Brain can cite", GREEN), (2, "type", "one of six kinds", GREEN), (3, "keywords", "help retrieval find the file", BLUE),
             (len(lines) - len(body), "body", "what the Brain reads", INK)]
    for li, lab, txt, col in notes:
        y = 128 + li * 18 - 4
        o += Ln(704, y, ax - 8, y, tint(col, 0.5), 1.2, "3 3") + C(ax - 8, y, 3, col)
        o += T(ax, y + 5, lab, SMALL, 600, col) + T(ax + tw(lab + "  ", SMALL, 600), y + 5, txt, SMALL, 400, MUTE)
    return o, max(y_end, 128 + len(lines) * 18) + 20


def retrieval(y0):
    h = FD["harness"]
    new = h["rule_rank_new"]
    req = set(h["required_rules"])
    o = T(20, y0, "Retrieval ranks the rules for this instruction", TITLE, 600, NAVY)
    for j, ln in enumerate(wrap("pass 1: BM25 over the six rules; the top five enter the Brain's context. A dot marks a rule the instruction actually needs.", 470, SUB)):
        o += T(20, y0 + 21 + j * 18, ln, SUB, 400, MUTE)
    xb, sc = 196, 11.0
    y = y0 + 78
    for k, (rid, sc_) in enumerate(new):
        yy = y + k * 32
        o += T(xb - 22, yy + 14, rid, SMALL, 500 if k < 5 else 400, INK if k < 5 else MUTE, "end")
        if rid in req:
            o += C(xb - 12, yy + 10, 4.5, TEAL)
        o += R(xb, yy, sc_ * sc, 20, 4, tint(BLUE, 0.35) if k < 5 else "#E8ECF0", BLUE if k < 5 else LINE, 1.2)
        o += T(xb + sc_ * sc + 8, yy + 15, f"{sc_:.1f}", SMALL, 600, INK if k < 5 else MUTE)
        if k == 4:
            o += Ln(20, yy + 32 - 6, 500, yy + 32 - 6, RED, 1.6, "5 4") + T(500, yy + 32 - 10, "k = 5", SMALL, 600, RED, "end")
    yp = y + 6 * 32 + 30
    o += T(20, yp, "pass 2: notes for the scene's classes, k = 3", BODY, 600, NAVY)
    x = 20
    for i in h["retrieved_other_new"]:
        ty = next(d["type"] for d in h["docs"] if d["id"] == i)
        wd = tw(i, SMALL, 500) + 26
        o += C(x + 7, yp + 24, 5, HC[ty]) + T(x + 18, yp + 29, i, SMALL, 500, INK)
        x += wd + 10
    return o, yp + 50


def tokenizer(y0, X=540):
    h = FD["harness"]
    old, new = dict(h["rule_rank_old"]), dict(h["rule_rank_new"])
    order = [r for r, _ in h["rule_rank_new"]]
    o = T(X, y0, "A tokenizer fix moves one rule up a rank", TITLE, 600, NAVY)
    for j, ln in enumerate(wrap("BM25 score of each rule, before and after the tokenizer splits letters from digits (the pre-fix tokenizer is reconstructed)", 440, SUB)):
        o += T(X, y0 + 21 + j * 18, ln, SUB, 400, MUTE)
    yt = y0 + 78
    # token chips
    o += M(X, yt + 4, '"lunchbox1 last"', 13.5, MUTE)
    def chips(x, y, toks, col):
        for t in toks:
            wd = 8.13 * len(t) + 12
            o_ = R(x, y - 14, wd, 20, 4, tint(col, 0.88), col, 1) + M(x + 6, y + 1, t, 13.5, INK)
            yield o_, wd
            x += wd + 5
    xx = X + 160
    for lab, toks, col, yy in (("before", h["tok_old"], RED, yt - 8), ("after", h["tok_new"], GREEN, yt + 18)):
        pass
    x = X + 150
    o += T(X + 150, yt - 14, "before", SMALL, 600, RED)
    xc = X + 200
    for o_, wd in chips(xc, yt - 14, h["tok_old"], RED):
        o += o_
        xc += wd + 5
    o += T(X + 150, yt + 14, "after", SMALL, 600, GREEN)
    xc = X + 200
    for o_, wd in chips(xc, yt + 14, h["tok_new"], GREEN):
        o += o_
        xc += wd + 5
    ya = yt + 46
    ax0, ax1, smax = X + 176, X + 430, 30.0
    xs = lambda v: ax0 + v / smax * (ax1 - ax0)
    for v in (0, 10, 20, 30):
        o += Ln(xs(v), ya - 6, xs(v), ya + 6 * 32 - 14, "#E4E8EC", 1) + T(xs(v), ya + 6 * 32 + 6, str(v), SMALL, 400, MUTE, "middle")
    for k, rid in enumerate(order):
        yy = ya + k * 32 + 8
        hot = rid == "lunchbox_last"
        o += T(ax0 - 12, yy + 5, rid, SMALL, 600 if hot else 400, NAVY if hot else INK, "end")
        a_, b_ = old[rid], new[rid]
        if abs(a_ - b_) > 1e-6:
            o += Ln(xs(a_), yy, xs(b_), yy, tint(GREEN, 0.4) if not hot else GREEN, 3.2 if hot else 2.2)
            o += C(xs(a_), yy, 5.5, "#FFFFFF", RED, 2.2)
        o += C(xs(b_), yy, 5.5, GREEN, "#FFFFFF", 1.4)
        if hot:
            o += T(xs(a_), yy - 12, f"{a_:.1f}", SMALL, 600, RED, "middle") + T(xs(b_), yy - 12, f"{b_:.1f}", SMALL, 600, GREEN, "middle")
    rk_old = [r for r, _ in h["rule_rank_old"]].index("lunchbox_last") + 1
    rk_new = order.index("lunchbox_last") + 1
    o += C(X + 8, ya + 6 * 32 + 40, 5, "#FFFFFF", RED, 2) + T(X + 20, ya + 6 * 32 + 45, "before", SMALL, 500, INK) + C(X + 88, ya + 6 * 32 + 40, 5, GREEN) + T(X + 100, ya + 6 * 32 + 45, "after", SMALL, 500, INK)
    o += T(X + 176, ya + 6 * 32 + 45, f"lunchbox_last: rank {rk_old} of 6 to rank {rk_new}", SMALL, 600, NAVY)
    return o, ya + 6 * 32 + 62


def writeback(y0):
    o = T(20, y0, "The harness learns from episodes", TITLE, 600, NAVY)
    o += T(20, y0 + 21, "after episodes a monitor writes measured outcomes back into the skill files, and the next retrieval carries them to the Brain", SUB, 400, MUTE)
    xs = [110, 360, 610, 860]
    yc = y0 + 78
    labels = [("episodes", "finished runs with their outcomes and failure causes", NAVY),
              ("monitor", "measures each skill's success rate and what went wrong", GREEN),
              ("skill files", "the measurements are written into the file body", BLUE),
              ("retrieval", "the next Brain reads them in its context", PURPLE)]
    for k, (name, txt, col) in enumerate(labels):
        x = xs[k]
        o += C(x, yc, 28, tint(col, 0.88), col, 2.2)
        if k == 0:
            for i, h_ in enumerate((14, 22, 10, 26, 18)):
                o += R(x - 20 + i * 8.5, yc + 13 - h_, 6, h_, 2, col)
        elif k == 1:
            o += P(f"M {x - 14},{yc + 12} A 15 15 0 0 1 {x + 14},{yc + 12}", stroke=col, sw=3) + Ln(x, yc + 12, x + 9, yc - 5, col, 2.6) + C(x, yc + 12, 3, col)
        elif k == 2:
            o += R(x - 12, yc - 16, 24, 32, 3, "#FFFFFF", col, 2) + Ln(x - 7, yc - 8, x + 7, yc - 8, col, 1.8) + Ln(x - 7, yc, x + 7, yc, col, 1.8) + Ln(x - 7, yc + 8, x + 2, yc + 8, col, 1.8) + Ln(x + 6, yc + 10, x + 6, yc + 18, GREEN, 2.4) + Ln(x + 2, yc + 14, x + 10, yc + 14, GREEN, 2.4)
        else:
            o += C(x - 4, yc - 4, 10, "none", col, 2.6) + Ln(x + 3, yc + 3, x + 12, yc + 12, col, 3.2)
        o += T(x, yc + 52, name, BODY, 600, col, "middle")
        for j, ln in enumerate(wrap(txt, 200, SMALL)):
            o += T(x, yc + 72 + j * 17, ln, SMALL, 400, MUTE, "middle")
        if k < 3:
            o += arrow([(x + 36, yc), (xs[k + 1] - 36, yc)], color=GOLD, sw=2.4, free="both")
    o += arrow([(xs[3], yc + 112), (xs[3], yc + 128), (xs[0], yc + 128), (xs[0], yc + 112)], color=GOLD, sw=2.4, free="both")
    o += T((xs[0] + xs[3]) / 2, yc + 122, "next episode", SMALL, 600, GOLD, "middle")
    return o, yc + 150


def build():
    a, ya = files_section()
    b_, yb = retrieval(ya + 30)
    c_, yc = tokenizer(ya + 30)
    d_, yd = writeback(max(yb, yc) + 30)
    H = int(yd + 8)
    body = DEFS + R(0, 0, 1000, H, 0, "#FFFFFF") + a + b_ + c_ + d_
    svg = f'<?xml version="1.0" encoding="UTF-8"?>\n<!-- engine:cairo -->\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {H}" width="1000" height="{H}">{body}</svg>'
    (SVG_DIR / "fig_harness.svg").write_text(svg, encoding="utf-8")


def fig_harness():
    build()
