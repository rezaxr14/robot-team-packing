"""Figure 5: the control loop. (a) control flow with its interrupts, (b) a recorded episode as a swimlane timeline,
(c) the recovery ladder with one real slip, and who is charged for a failure. Illustrative: values come from the prototype."""
from __future__ import annotations
import json
from fig2 import (FD, DEFS, T, R, C, Ln, P, poly, tw, wrap, tint, arrow, SVG_DIR,
                  INK, MUTE, LINE, TEAL, PURPLE, ORANGE, GREEN, GOLD, RED, NAVY, BLUE)

BODY, SMALL, SUB, TITLE = 14.5, 14, 14.5, 18
DKO = "#9C3F12"
GREY_D = "#4A5560"


def node(x, y, r, col, glyph, dash=False):
    o = f'<circle cx="{x}" cy="{y}" r="{r}" fill="{tint(col, 0.9)}" stroke="{col}" stroke-width="2.2"' + (' stroke-dasharray="5 4"' if dash else "") + "/>"
    if glyph == "scale":
        o += Ln(x, y - 12, x, y + 12, col, 2.4) + Ln(x - 13, y - 6, x + 13, y - 6, col, 2.4) + P(f"M {x - 13},{y - 6} L {x - 18},{y + 6} L {x - 8},{y + 6} Z", stroke=col, sw=1.8) + P(f"M {x + 13},{y - 6} L {x + 8},{y + 6} L {x + 18},{y + 6} Z", stroke=col, sw=1.8)
    elif glyph == "plan":
        o += R(x - 11, y - 14, 22, 28, 3, "#FFFFFF", col, 2) + "".join(Ln(x - 6, y - 7 + i * 7, x + 6, y - 7 + i * 7, col, 1.8) for i in range(3))
    elif glyph == "gate":
        d = 14
        o += poly([(x, y - d), (x + d, y), (x, y + d), (x - d, y)], "#FFFFFF", col, 2.2) + P(f"M {x - 6},{y} L {x - 1},{y + 5} L {x + 7},{y - 5}", stroke=col, sw=2.6)
    elif glyph == "rounds":
        o += P(f"M {x + 12},{y - 4} A 13 13 0 1 0 {x + 8},{y + 10}", stroke=col, sw=3) + poly([(x + 6, y - 12), (x + 16, y - 6), (x + 5, y - 3)], col)
    elif glyph == "flag":
        o += Ln(x - 8, y - 14, x - 8, y + 14, col, 2.6) + poly([(x - 8, y - 14), (x + 13, y - 8), (x - 8, y - 2)], col)
    return o


def control_flow():
    o = T(20, 38, "One episode, step by step", TITLE, 600, NAVY)
    o += T(20, 60, "a verified plan runs in rounds; four events can interrupt it", SUB, 400, MUTE)
    yc = 150
    xs = {"probe": 80, "plan": 240, "verify": 400, "exec": 600, "check": 860}
    spec = [("probe", PURPLE, "scale", "probe (optional)", "weigh every item first", True), ("plan", PURPLE, "plan", "plan", "the Brain drafts a task graph", False),
            ("verify", GREEN, "gate", "verify", "a rejection sends the reasons back", False), ("exec", ORANGE, "rounds", "rounds", "each Arm runs its head task", False),
            ("check", NAVY, "flag", "end check", "shared checker scores the bag", False)]
    for k, col, g, name, desc, dash in spec:
        x = xs[k]
        o += node(x, yc, 28, col, g, dash)
        o += T(x, yc + 50, name, BODY, 600, col, "middle")
        for j, ln in enumerate(wrap(desc, 138, SMALL)):
            o += T(x, yc + 68 + j * 17, ln, SMALL, 400, MUTE, "middle")
    o += arrow([(xs["probe"] + 34, yc), (xs["plan"] - 34, yc)], color=PURPLE, sw=2.4, dash="5 4", free="both")
    o += arrow([(xs["plan"] + 34, yc), (xs["verify"] - 34, yc)], color=NAVY, sw=2.4, free="both")
    o += arrow([(xs["verify"] + 34, yc), (xs["exec"] - 34, yc)], color=GREEN, sw=2.4, free="both") + T(500, yc - 10, "verified", SMALL, 600, GREEN, "middle")
    o += arrow([(xs["exec"] + 34, yc), (xs["check"] - 34, yc)], color=NAVY, sw=2.4, free="both") + T(730, yc - 10, "all tasks done, or abort", SMALL, 600, NAVY, "middle")
    # loops above and below
    o += arrow([(xs["verify"] - 8, yc - 34), (xs["verify"] - 8, yc - 56), (xs["plan"] + 8, yc - 56), (xs["plan"] + 8, yc - 34)], color=TEAL, sw=2.2, free="both")
    o += T((xs["plan"] + xs["verify"]) / 2, yc - 64, "an item is missing: LOOK, then replan", SMALL, 600, TEAL, "middle")
    o += arrow([(xs["verify"] - 8, yc + 118), (xs["verify"] - 8, yc + 136), (xs["plan"] + 8, yc + 136), (xs["plan"] + 8, yc + 118)], color=RED, sw=2.2, free="both") if False else ""
    o += arrow([(xs["exec"] - 10, yc - 34), (xs["exec"] - 10, yc - 56), (xs["exec"] + 10, yc - 56), (xs["exec"] + 10, yc - 34)], color=NAVY, sw=2.2, free="both") + T(xs["exec"] + 22, yc - 52, "next round", SMALL, 600, NAVY)
    o += arrow([(xs["verify"] + 14, yc + 104), (xs["verify"] + 14, yc + 124), (xs["plan"] - 14, yc + 124), (xs["plan"] - 14, yc + 104)], color=RED, sw=2.2, free="both") if False else ""
    # interrupts under the rounds
    ev = [(DKO, "touch contradicts an Eye by more than 20%:", "CHALLENGE, correct the scene, replan once"), (GREY_D, "the human changes the instruction:", "AMEND, replan"),
          (RED, "a skill fails:", "recovery ladder, then replan or abort")]
    yy = yc + 128
    o += T(xs["exec"] - 130, yc + 112, "can interrupt a round", SMALL, 600, MUTE)
    o += arrow([(xs["exec"], yy - 18), (xs["exec"], yc + 34)], color=GREY_D, sw=2.2, free="both") if False else ""
    for k, (col, a_, b_) in enumerate(ev):
        y = yy + k * 36
        o += C(xs["exec"] - 130, y - 4, 5, col) + T(xs["exec"] - 118, y, a_, SMALL, 500, INK) + T(xs["exec"] - 118, y + 17, b_, SMALL, 600, col)
    o += T(xs["verify"], yc + 100, "rejected: reasons go back,", SMALL, 500, RED, "middle") + T(xs["verify"], yc + 118, "at most two retries", SMALL, 500, RED, "middle") if False else ""
    return o, yy + 3 * 36 + 8


def timeline(y0):
    lp = FD["loop"]
    st = lp["story"]
    ev = lp["events"]
    o = T(20, y0, "A recorded episode", TITLE, 600, NAVY)
    o += T(20, y0 + 21, "tasks by arm and loop round, and the messages that changed its course", SUB, 400, MUTE)
    lanes = [("round", None), ("Brain", PURPLE), ("Eyes", TEAL), ("Arm L", ORANGE), ("Arm R", ORANGE), ("bus", NAVY)]
    lx0, lx1 = 92, 980
    ly, y = {}, y0 + 44
    for n, c in lanes:
        hh = 26 if n == "round" else 46
        ly[n] = y
        if c:
            o += R(lx0 - 4, y, lx1 - lx0 + 8, hh, 8, tint(c, 0.95), None if False else "none")
            o += T(lx0 - 14, y + hh / 2 + 5, n, BODY, 600, c, "end")
        else:
            o += T(lx0 - 14, y + hh / 2 + 5, n, SMALL, 600, MUTE, "end")
        y += hh + 6
    keep = [e for e in ev if e["ev"] == "exec" or e.get("type") in ("LOOK", "CHALLENGE", "AMEND")]
    n = len(keep)
    step = (lx1 - lx0 - 10) / n
    short = {"notebook": "nb", "pencilcase": "pc", "lunchbox": "lb", "bottle": "bt", "laptop": "lp", "book": "bk"}
    abbr = lambda ob: next((v + ob[len(k):] for k, v in short.items() if ob.startswith(k)), ob[:3])
    xs = {k: lx0 + 4 + k * step for k in range(n)}
    rounds = [e["round"] for e in st["executed"]]
    ri, by_round = 0, {}
    for k, e in enumerate(keep):
        x, w_ = xs[k], step - 4
        if e["ev"] == "exec":
            by_round.setdefault(rounds[ri], []).append(k)
            ri += 1
            lane = "Arm " + e["arm"]
            col = ORANGE if e["ok"] else RED
            o += R(x, ly[lane] + 4, w_, 38, 5, tint(col, 0.82), col, 1.2)
            o += T(x + w_ / 2, ly[lane] + 20, "grasp" if e["skill"] == "grasp" else "place", 13, 600, col, "middle") if w_ >= 40 else T(x + w_ / 2, ly[lane] + 20, "g" if e["skill"] == "grasp" else "p", 13, 600, col, "middle")
            o += T(x + w_ / 2, ly[lane] + 36, abbr(e["obj"]), 13, 500, INK, "middle")
        else:
            t = e["type"]
            lane = {"LOOK": "Eyes", "CHALLENGE": "bus", "AMEND": "bus"}[t]
            col = {"LOOK": TEAL, "CHALLENGE": DKO, "AMEND": GREY_D}[t]
            o += R(x, ly[lane] + 4, w_, 38, 5, tint(col, 0.8), col, 1.4)
            o += T(x + w_ / 2, ly[lane] + 29, {"LOOK": "L", "CHALLENGE": "C", "AMEND": "A"}[t], BODY, 600, col, "middle")
    for r_, ks in by_round.items():
        xa, xb = xs[ks[0]], xs[ks[-1]] + step - 4
        o += R(xa, ly["round"] + 2, xb - xa, 22, 5, "#EEF1F4", LINE, 1) + T((xa + xb) / 2, ly["round"] + 18, str(r_), 13, 600, MUTE, "middle")
    il = [k for k, e in enumerate(keep) if e.get("type") == "LOOK"]
    ia = [k for k, e in enumerate(keep) if e.get("type") == "AMEND"]
    for k, lab in [(0, "P")] + [(k + 1, "R") for k in il + ia if k + 1 < n]:
        o += R(xs[k], ly["Brain"] + 4, step - 4, 38, 5, tint(PURPLE, 0.8), PURPLE, 1.3) + T(xs[k] + (step - 4) / 2, ly["Brain"] + 29, lab, BODY, 600, PURPLE, "middle")
    yl = y + 12
    leg = [("P", PURPLE, "first plan"), ("R", PURPLE, "replan"), ("L", TEAL, "LOOK"), ("C", DKO, "CHALLENGE"), ("A", GREY_D, "AMEND")]
    x = 20
    for lab, col, txt in leg:
        o += R(x, yl - 13, 20, 20, 5, tint(col, 0.8), col, 1.3) + T(x + 10, yl + 2, lab, SMALL, 600, col, "middle") + T(x + 28, yl + 3, txt, SMALL, 500, INK)
        x += 28 + tw(txt, SMALL, 500) + 22
    o += T(x + 6, yl + 3, f"{sum(1 for e in ev if e['ev'] == 'exec')} tasks in {st['rounds']} rounds; grasp or place, then the item (bk book, nb notebook, pc pencil case, bt bottle, lp laptop, lb lunchbox)", SMALL, 400, MUTE) if False else ""
    o += T(20, yl + 30, f"{sum(1 for e in ev if e['ev'] == 'exec')} tasks in {st['rounds']} rounds. Items: bk book, nb notebook, pc pencil case, bt bottle, lp laptop, lb lunchbox.", SMALL, 400, MUTE)
    return o, yl + 48


def ladder(y0):
    m = FD["mechanism"]
    it = next(i for i in m["items"] if i["name"] == "book1")
    floor, ceil_ = (4 + 6 * it["true_mass"]) / it["friction"], 30 - 22 * it["true_frag"]
    att = [e for e in m["challenge"]["executed"] if e["task"]["skill"] == "grasp" and e["task"]["obj"] == "book1"]
    o = T(20, y0, "When a skill fails", TITLE, 600, NAVY)
    o += T(20, y0 + 21, "a recovery ladder, shown with one recorded slip", SUB, 400, MUTE)
    rungs = ["retry", "alternate parameters", "retreat", "reassign to the other Arm", "replan", "abort"]
    yt = y0 + 88
    o += Ln(34, yt - 6, 34, yt + 6 * 34 - 14, GREY_D, 3.4) + Ln(70, yt - 6, 70, yt + 6 * 34 - 14, GREY_D, 3.4)
    for k, rname in enumerate(rungs):
        y = yt + k * 34 + 8
        used = k < 2
        o += Ln(34, y, 70, y, ORANGE if used else LINE, 3.2)
        o += T(86, y + 5, f"{k + 1}  {rname}", BODY, 600 if used else 400, INK if used else MUTE)
    # the recorded slip
    cx0 = 330
    yb, ymax, chh = yt + 6 * 34 - 14, 30.0, 6 * 34 - 4
    yo = lambda f: yb - f / ymax * chh
    o += T(cx0 + 62, yt - 28, "book1 grip force (N)", SMALL, 600, MUTE, "middle")
    for f in (0, 10, 20, 30):
        o += Ln(cx0 + 20, yo(f), cx0 + 100, yo(f), "#E4E8EC", 1) + T(cx0 + 14, yo(f) + 5, str(f), SMALL, 400, MUTE, "end")
    o += R(cx0 + 46, yo(ceil_), 28, yo(floor) - yo(ceil_), 6, "#DDEEE5", GREEN, 1.4)
    n_ = len(att)
    for j, e in enumerate(att):
        f = e["task"]["params"]["force"]
        x = cx0 + 60 + (j - (n_ - 1) / 2) * 0 + 14 * (j % 2) - 7
        o += C(cx0 + 60, yo(f) + (0 if True else 0), 5.4, GREEN if e["ok"] else RED, "#FFFFFF", 1.2) if False else ""
    xs_ = [cx0 + 54, cx0 + 60, cx0 + 66]
    for j, e in enumerate(att):
        f = e["task"]["params"]["force"]
        o += C(cx0 + 60, yo(f), 5.4 if e["ok"] else 5.4, GREEN if e["ok"] else RED, "#FFFFFF", 1.2)
    slips = [e for e in att if not e["ok"]]
    good = [e for e in att if e["ok"]]
    if slips:
        o += Ln(150, yt + 8, cx0 + 30, yo(slips[0]["task"]["params"]["force"]), tint(RED, 0.5), 1.2, "3 3") if False else ""
    if slips:
        o += T(cx0 + 90, yo(slips[0]["task"]["params"]["force"]) + 24, f"{slips[0]['task']['params']['force']:.1f} N slips" + (f" (x{len(slips)})" if len(slips) > 1 else ""), SMALL, 600, RED)
    o += T(cx0 + 90, yo(floor) + 5, f"floor {floor:.1f} N", SMALL, 500, GREEN)
    if good:
        o += T(cx0 + 90, yo(good[0]["task"]["params"]["force"]) - 10, f"{good[0]['task']['params']['force']:.1f} N holds", SMALL, 600, GREEN)
    yr = lambda k: yt + k * 34 + 8
    o += C(292, yr(0), 5.5, RED) + C(292, yr(1), 5.5, GREEN)
    return o, yb + 24


def charging(y0, X=560):
    o = T(X, y0, "Who is charged for a failure", TITLE, 600, NAVY)
    o += T(X, y0 + 21, "attribution rules at the end check", SUB, 400, MUTE)
    rows = [(TEAL, "Eye", "a slip or crush while the Eye's mass or fragility report was off by more than 15%"),
            (PURPLE, "Brain", "a rule violation in a plan built on correct reports"),
            (ORANGE, "Arm", "a slip or crush although the reports were accurate")]
    for k, (col, nm, txt) in enumerate(rows):
        y = y0 + 62 + k * 62
        o += C(X + 16, y, 14, tint(col, 0.85), col, 2.2) + T(X + 42, y - 2, nm, BODY, 600, col)
        for j, ln in enumerate(wrap(txt, 360, SMALL)):
            o += T(X + 42, y + 15 + j * 17, ln, SMALL, 400, INK)
    o += T(X, y0 + 62 + 3 * 62 + 6, "heuristic labels, to be checked against hand labels", SMALL, 500, MUTE)
    return o, y0 + 62 + 3 * 62 + 26


def build():
    a, ya = control_flow()
    b_, yb = timeline(ya + 34)
    c_, yc = ladder(yb + 36)
    d_, yd = charging(yb + 36)
    H = int(max(yc, yd) + 8)
    body = DEFS + R(0, 0, 1000, H, 0, "#FFFFFF") + a + b_ + c_ + d_
    svg = f'<?xml version="1.0" encoding="UTF-8"?>\n<!-- engine:cairo -->\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {H}" width="1000" height="{H}">{body}</svg>'
    (SVG_DIR / "fig_loop.svg").write_text(svg, encoding="utf-8")


def fig_loop():
    build()
