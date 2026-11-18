"""Figure 3 (v2): the message layer, in three different visual forms (no card grids).

(a) who may say what: a dot matrix, plus one real message annotated with what the bus does to it
(b) claims against evidence: touch vs the Eye's report for the five objects the Arms challenged (seed 24, recorded)
(c) what the traffic costs: mean cumulative tokens against tasks executed, four communication modes, 80 fixture episodes
"""
from __future__ import annotations
import json
from html import escape
from fig2 import (FD, DEFS, T, R, C, Ln, P, poly, tw, wrap, tint, arrow, SVG_DIR, PAPER_DIR,
                  INK, MUTE, LINE, TEAL, PURPLE, ORANGE, GREEN, GOLD, RED, NAVY, BLUE, FAM)

FX = json.loads((PAPER_DIR / "generated" / "fixture.json").read_text())
BODY, SMALL, SUB, TITLE = 14.5, 14, 14.5, 18
MONO = "'DejaVu Sans Mono', Menlo, Consolas, monospace"
DKO = "#9C3F12"
GROUPS = [("perceive", TEAL, ["SCENE", "LOOK"]), ("coordinate", NAVY, ["ABILITY", "CLAIM", "OFFER", "DECLINE", "HANDOVER", "LOCK"]),
          ("feedback", ORANGE, ["STATUS", "BLOCKED", "TOUCH", "CHALLENGE"]), ("human", "#4A5560", ["AMEND"])]
SENDER = {"SCENE": "Eye", "LOOK": "Brain", "AMEND": "Human"}                 # everything else is sent by the Arms (design)
SEEN = {"SCENE", "LOOK", "ABILITY", "TOUCH", "CHALLENGE", "AMEND"}            # exchanged in the recorded episodes


def M(x, y, s, size=13.5, fill=INK, anchor="start"):
    return f'<text xml:space="preserve" x="{x:.1f}" y="{y:.1f}" font-family="{MONO}" font-size="{size}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>'


def Mseg(x, y, segs, size=13.5):
    o, cw = "", size * 0.602
    for txt, col in segs:
        o += M(x, y, txt, size, col)
        x += cw * len(txt)
    return o


# ------------------------------------------------------------------------------------------------ (a)
def who_says_what():
    sch = FD["messages"]["schema"]
    o = T(20, 38, "Thirteen typed messages: who may say what", TITLE, 600, NAVY)
    o += T(20, 60, "a dot marks the sender (design); a tick marks types exchanged in the recorded episodes", SUB, 400, MUTE)
    cx = {"Eye": 206, "Brain": 262, "Arms": 318, "Human": 378}
    colr = {"Eye": TEAL, "Brain": PURPLE, "Arms": ORANGE, "Human": "#4A5560"}
    yh = 94
    for r_, x in cx.items():
        o += T(x, yh, r_, SMALL, 600, colr[r_], "middle")
    o += T(430, yh, "seen", SMALL, 600, MUTE, "middle") + T(462, yh, "required fields", SMALL, 600, MUTE)
    y = yh + 12
    for g, col, types in GROUPS:
        o += T(20, y + 17, g, BODY, 600, col) + Ln(20 + tw(g, BODY, 600) + 10, y + 12, 636, y + 12, tint(col, 0.55), 1.4)
        y += 26
        for t in types:
            o += T(34, y + 17, t, BODY, 600, INK)
            sender = SENDER.get(t, "Arms")
            for r_, x in cx.items():
                o += C(x, y + 12, 6.5, colr[r_]) if r_ == sender else C(x, y + 12, 2, LINE)
            if t in SEEN:
                o += P("M 424,%d L 429,%d L 438,%d" % (y + 12, y + 17, y + 6), stroke=GREEN, sw=2.4)
            fl = wrap(", ".join(sch[t]), 175, SMALL)
            for j, ln in enumerate(fl):
                o += T(462, y + 17 + j * 17, ln, SMALL, 400, MUTE)
            y += max(27, 10 + 17 * len(fl))
        y += 6
    ybot = y
    ch = next(c for c in FD["mechanism"]["challenge_msgs"] if c["content"]["obj"] == "notebook1")
    n_tok = FD["messages"]["examples"]["CHALLENGE"]["n_tokens_proxy"]
    X = 672
    o += T(X, yh, "one message, and what the bus does to it", BODY, 600, NAVY)
    lines = [[("{", INK), ('"type"', MUTE), (": ", INK), ('"CHALLENGE"', DKO), (",", INK)],
             [(' "sender"', MUTE), (": ", INK), (f'"{ch["from"]}"', ORANGE), (",", INK)],
             [(' "receiver"', MUTE), (": ", INK), (f'"{ch["to"]}"', PURPLE), (",", INK)],
             [(' "content"', MUTE), (": {", INK), ('"obj"', MUTE), (": ", INK), ('"notebook1"', INK), (",", INK)],
             [('   "evidence"', MUTE), (": {...}}}", INK)]]
    for k, segs in enumerate(lines):
        o += Mseg(X, yh + 28 + k * 20, segs)
    ys = yh + 28 + 5 * 20 + 26
    steps = [("validate", "against the schema: type, sender and every required field", GREEN),
             ("count", f"tokens at about four characters each; this message costs {n_tok}", BLUE),
             ("deliver", "to the receiver, or refuse it when the episode budget is spent", NAVY)]
    o += Ln(X + 11, ys + 8, X + 11, ys + 8 + 2 * 66, LINE, 2)
    for k, (verb, txt, col) in enumerate(steps):
        yy = ys + k * 66
        o += C(X + 11, yy + 8, 11, "#FFFFFF", col, 2.2)
        if k == 0:
            o += P("M %d,%d L %d,%d L %d,%d" % (X + 6, yy + 8, X + 10, yy + 12, X + 17, yy + 4), stroke=col, sw=2.4)
        elif k == 1:
            o += Ln(X + 5, yy + 11, X + 17, yy + 11, col, 2.2) + Ln(X + 7, yy + 6, X + 7, yy + 11, col, 1.6) + Ln(X + 11, yy + 4, X + 11, yy + 11, col, 1.6) + Ln(X + 15, yy + 6, X + 15, yy + 11, col, 1.6)
        else:
            o += P("M %d,%d L %d,%d M %d,%d L %d,%d L %d,%d" % (X + 5, yy + 8, X + 16, yy + 8, X + 12, yy + 4, X + 16, yy + 8, X + 12, yy + 12), stroke=col, sw=2.4)
        o += T(X + 32, yy + 13, verb, BODY, 600, col)
        for j, ln in enumerate(wrap(txt, 270, SMALL)):
            o += T(X + 32, yy + 32 + j * 17, ln, SMALL, 400, MUTE)
    # ---- who fills the bus in the recorded episode
    ev = [e for e in FD["loop"]["events"] if e["ev"] == "msg"]
    tok = {}
    cnt = {}
    for e in ev:
        t_ = e.get("tok") or (len(json.dumps(e["content"])) // 4 + 8)
        tok[e["type"]] = tok.get(e["type"], 0) + t_
        cnt[e["type"]] = cnt.get(e["type"], 0) + 1
    tot = sum(tok.values())
    ym = ys + 3 * 66 + 16
    o += T(X, ym, f"the recorded episode: {len(ev)} messages", BODY, 600, NAVY) + T(X, ym + 19, "share of all tokens, by type", SMALL, 400, MUTE)
    col = {"SCENE": TEAL, "LOOK": PURPLE, "AMEND": "#4A5560"}
    for k, (t_, v) in enumerate(sorted(tok.items(), key=lambda kv: -kv[1])):
        yy = ym + 40 + k * 24
        wbar = 130 * v / tot
        o += T(X, yy + 12, t_, SMALL, 600, INK) + R(X + 96, yy, max(wbar, 2), 16, 3, col.get(t_, ORANGE)) + T(X + 96 + max(wbar, 2) + 8, yy + 13, f"{100 * v / tot:.0f}%  ({cnt[t_]})", SMALL, 500, MUTE)
    return o, max(ybot, ym + 40 + len(tok) * 24)


# ------------------------------------------------------------------------------------------------ (b)
def claims_vs_evidence(y0, X=20, W=500):
    msgs = FD["mechanism"]["challenge_msgs"]
    o = T(X, y0, "A claim meets its evidence", TITLE, 600, NAVY)
    for j, ln in enumerate(wrap("five objects in one recorded episode where an Arm's touch contradicted the Eye's report by more than 20%", W, SUB)):
        o += T(X, y0 + 21 + j * 18, ln, SUB, 400, MUTE)
    yl = y0 + 68
    o += poly([(X + 10, yl - 6), (X + 16, yl), (X + 10, yl + 6), (X + 4, yl)], TEAL) + T(X + 24, yl + 5, "Eye's report", SMALL, 500, INK)
    o += C(X + 144, yl, 6, ORANGE) + T(X + 156, yl + 5, "Arm's touch", SMALL, 500, INK)
    o += R(X + 262, yl - 7, 28, 14, 3, "#FBE9D8", "#E5B07C", 1) + T(X + 298, yl + 5, "within 20%: accepted", SMALL, 500, INK)
    o += arrow([(X + 4, yl + 26), (X + 38, yl + 26)], color=DKO, sw=2.4, free="both") + T(X + 46, yl + 31, "CHALLENGE: the scene is corrected to the touch", SMALL, 500, INK)
    ax0, ax1, kmax = X + 118, X + W - 24, 2.4
    sc = (ax1 - ax0) / kmax
    rows = sorted(msgs, key=lambda m: m["content"]["evidence"]["measured_weight"])
    yr0, rh = yl + 66, 56
    for k_ in (0, 0.5, 1.0, 1.5, 2.0):
        o += Ln(ax0 + k_ * sc, yr0 - 14, ax0 + k_ * sc, yr0 + len(rows) * rh - 12, "#E4E8EC", 1) + T(ax0 + k_ * sc, yr0 + len(rows) * rh + 8, f"{k_:g}", SMALL, 400, MUTE, "middle")
    o += T(ax1 + 6, yr0 + len(rows) * rh + 8, "kg", SMALL, 500, MUTE)
    for i, m in enumerate(rows):
        e = m["content"]["evidence"]
        meas, rep = e["measured_weight"], e["reported"]
        y = yr0 + i * rh + 12
        o += T(ax0 - 12, y + 5, m["content"]["obj"], BODY, 500, INK, "end")
        o += R(ax0 + meas * 0.8 * sc, y - 10, meas * 0.4 * sc, 20, 4, "#FBE9D8", "#E5B07C", 1)
        x_r, x_m = ax0 + rep * sc, ax0 + meas * sc
        d = 1 if x_m > x_r else -1
        o += arrow([(x_r + 9 * d, y), (x_m - 8 * d, y)], color=DKO, sw=2.2, free="both")
        o += poly([(x_r, y - 8), (x_r + 7, y), (x_r, y + 8), (x_r - 7, y)], TEAL, "#FFFFFF", 1.4) + C(x_m, y, 6.5, ORANGE, "#FFFFFF", 1.4)
        o += T(x_r, y - 14, f"{rep:.2f}", SMALL, 600, TEAL, "middle") + T(x_m, y + 26, f"{meas:.2f}", SMALL, 600, ORANGE, "middle")
    yb = yr0 + len(rows) * rh + 34
    for j, ln in enumerate(wrap("After the CHALLENGE for notebook1 the Brain replans from 0.46 kg: the notebook goes below the pencil case, and the heavy-at-the-bottom rule holds.", W, BODY)):
        o += T(X, yb + j * 19, ln, BODY, 400, INK)
    return o, yb + 44


# ------------------------------------------------------------------------------------------------ (c)
def traffic(y0, X=560, W=420):
    cc = FD["comm_curves"]["modes"]
    fx = FX["comm"]
    o = T(X, y0, "What the traffic costs", TITLE, 600, NAVY)
    o += R(X + 196, y0 - 15, 92, 22, 11, "#FBEFD9", "#C98A1E", 1) + T(X + 242, y0 + 1, "FIXTURE", SMALL, 600, "#9A6410", "middle")
    for j, ln in enumerate(wrap("mean cumulative message tokens as an episode progresses; 80 symbolic-fixture episodes, noisy Eyes", W, SUB)):
        o += T(X, y0 + 21 + j * 18, ln, SUB, 400, MUTE)
    x0, x1, yt, yb = X + 58, X + 258, y0 + 74, y0 + 314
    kmax, ymax = 16, 11000.0
    xs = lambda k: x0 + k / kmax * (x1 - x0)
    ys = lambda v: yb - v / ymax * (yb - yt)
    for v in (0, 2500, 5000, 7500, 10000):
        o += Ln(x0, ys(v), x1, ys(v), "#E4E8EC", 1) + T(x0 - 8, ys(v) + 5, f"{v:,}", SMALL, 400, MUTE, "end")
    for k in (0, 4, 8, 12, 16):
        o += T(xs(k), yb + 20, str(k), SMALL, 400, MUTE, "middle")
    o += T((x0 + x1) / 2, yb + 42, "tasks executed", SMALL, 500, MUTE, "middle") + T(x0 - 8, yt - 14, "tokens", SMALL, 500, MUTE, "end")
    style = {"push": (NAVY, "full scene each round"), "delta": (BLUE, "only what changed"), "pull": (TEAL, "on demand"), "init_only": (PURPLE, "once at the start")}
    ends = []
    for mode, (col, desc) in style.items():
        pts = [(xs(k), ys(v)) for k, v in zip(cc[mode]["x"], cc[mode]["mean"]) if v is not None]
        o += P("M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts), stroke=col, sw=3.2) + C(pts[-1][0], pts[-1][1], 5, col, "#FFFFFF", 1.6)
        ends.append((pts[-1][1], mode, col, desc))
    ends.sort()
    placed = []
    for ye, mode, col, desc in ends:
        yy = ye if not placed else max(ye, placed[-1] + 62)
        placed.append(yy)
        v = fx[mode]
        o += Ln(x1 + 8, ye, x1 + 20, yy, tint(col, 0.5), 1.2)
        o += T(x1 + 26, yy - 3, f"{mode.replace('_', '-')} {v['tokens_per_episode']:,.0f}", BODY, 600, col)
        o += T(x1 + 26, yy + 14, desc, SMALL, 400, MUTE) + T(x1 + 26, yy + 31, f"{v['k']}/{v['n']} succeed", SMALL, 400, MUTE)
    return o, max(yb + 62, placed[-1] + 50)


def build():
    a, ya = who_says_what()
    b_, yb = claims_vs_evidence(ya + 48)
    c_, yc = traffic(ya + 48)
    H = int(max(yb, yc) + 8)
    body = DEFS + R(0, 0, 1000, H, 0, "#FFFFFF") + a + b_ + c_
    svg = f'<?xml version="1.0" encoding="UTF-8"?>\n<!-- engine:cairo -->\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {H}" width="1000" height="{H}">{body}</svg>'
    (SVG_DIR / "fig_messages.svg").write_text(svg, encoding="utf-8")


def fig_messages():
    build()
