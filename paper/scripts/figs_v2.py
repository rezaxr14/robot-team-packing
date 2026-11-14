"""Second-generation figures. build_diagrams.py prefers a function defined here over figs_a / figs_b."""
from __future__ import annotations
import json
from diagram_common import *  # noqa: F401,F403
from diagram_common import PAPER_DIR

FD = json.loads((PAPER_DIR / "generated" / "figdata.json").read_text())
FX = json.loads((PAPER_DIR / "generated" / "fixture.json").read_text())


def num(x, y, n, color=RED, r=11):
    """Numbered badge used to link the same object across panels."""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>' + T(x, y + 5, str(n), size=14, fill="#FFFFFF", weight="bold")


def fig_mechanism():
    m = FD["mechanism"]
    W_, H_ = 1000, 610
    b = ""
    items = sorted([i for i in m["items"] if not i["distractor"]], key=lambda i: -i["true_mass"])
    flagged = {"pencilcase1": 1, "notebook1": 2}
    # ---- a: reports vs truth
    b += panel(20, 30, "a", "What the Eye reported")
    x0, sc, y0, rh = 128, 98.0, 56, 40
    for k, it in enumerate(items):
        y = y0 + k * rh
        err = (it["eye_mass"] - it["true_mass"]) / it["true_mass"] * 100
        bad = it["name"] in flagged
        b += T(x0 - 10, y + 22, it["name"], size=15, anchor="end", fill=RED if bad else C_TXT, weight="bold" if bad else None)
        if bad:
            b += num(18, y + 18, flagged[it["name"]])
        b += R(x0, y + 4, it["true_mass"] * sc, 13, rx=2, fill="#C7CDD4", stroke="none", sw=0)
        b += R(x0, y + 19, it["eye_mass"] * sc, 13, rx=2, fill=RED if bad else TEAL, stroke="none", sw=0)
        if bad:
            b += T(x0 + max(it["true_mass"], it["eye_mass"]) * sc + 8, y + 31, f"{err:+.0f}%", size=15, anchor="start", fill=RED, weight="bold")
    ly = y0 + len(items) * rh + 8
    b += R(20, ly, 14, 12, rx=2, fill="#C7CDD4", stroke="none", sw=0) + T(40, ly + 11, "true mass (kg)", size=14.5, anchor="start")
    b += R(170, ly, 14, 12, rx=2, fill=TEAL, stroke="none", sw=0) + T(190, ly + 11, "Eye report", size=14.5, anchor="start")
    b += L(x0 + sc, y0 - 6, x0 + sc, ly - 8, color="#E2E6EA", sw=1.0, dash="3 4") + T(x0 + sc, y0 - 10, "1 kg", size=13, fill=C_SUB)
    # ---- b: the plan
    b += panel(375, 30, "b", "The plan built on it")
    plans = m["no_fix"]["plans"][0]
    plans = json.loads(plans) if isinstance(plans, str) else plans
    order = [t["obj"] for t in plans["tasks"] if t["skill"] == "place" and t["obj"] != "laptop1"]
    tm = {i["name"]: i["true_mass"] for i in m["items"]}
    bx, bw, bh, by0 = 405, 190, 38, 66
    b += R(bx - 10, by0 - 10, bw + 20, len(order) * bh + 20, rx=10, fill="#F6F7F9", stroke=NAVY, sw=1.6)
    for k, o in enumerate(reversed(order)):
        y = by0 + k * bh
        fl = o in flagged
        b += R(bx, y + 2, bw, bh - 4, rx=5, fill=tint(RED, 0.85) if fl else "#FFFFFF", stroke=RED if fl else GREY, sw=1.8 if fl else 1.0)
        b += T(bx + 12, y + 25, o, size=15, anchor="start", weight="bold" if fl else None, fill=RED if fl else C_TXT)
        b += T(bx + bw - 12, y + 25, f"{tm[o]:.2f} kg", size=14.5, anchor="end", fill=RED if fl else C_SUB)
        if fl:
            b += num(bx - 24, y + 20, flagged[o])
    bot = by0 + len(order) * bh
    b += T(bx - 10, by0 - 14, "top", size=13, anchor="start", fill=C_SUB) if False else ""
    b += T(bx + bw / 2, bot + 34, "bag, main compartment: last placed on top", size=14, fill=C_SUB)
    # ---- c: why, and how it is fixed
    b += panel(640, 30, "c", "Why it broke, and two fixes")
    viol = m["no_fix"]["violations"][0]
    s, p1 = card(640, 50, 340, "Rule broken", f"{viol['rule']}: {viol['detail']}. The planner reasoned correctly from numbers that were wrong.", color=RED)
    b += s
    s, p2 = card(640, p1[1] + p1[3] + 12, 340, "Charged to the Eye", "The loop's attribution rules blame the Eye: the mass it reported was far from the truth. A heuristic label, not yet checked against hand labels.", color=TEAL)
    b += s
    ch = m["challenge_msgs"]
    lines = "; ".join(f"{c['content']['obj']} {c['content']['evidence']['measured_weight']:.2f} vs {c['content']['evidence']['reported']:.2f}" for c in ch[:2])
    s, p3 = card(640, p2[1] + p2[3] + 12, 340, "Fix 1: CHALLENGE", f"The Arm weighs what it grasps and disputes the Eye ({len(ch)} challenges here; measured vs reported kg: {lines}). The scene is corrected and the rest replanned.", color=ORANGE)
    b += s
    s, p4 = card(640, p3[1] + p3[3] + 12, 340, "Fix 2: probe", "Both Arms weigh every item before planning; measurements override the Eye's mass guesses.", color=BLUE)
    b += s
    # ---- outcome strip
    ys = max(p4[1] + p4[3], bot + 50) + 20
    outs = [("no fix", m["no_fix"], RED), ("CHALLENGE", m["challenge"], GREEN), ("probe", m["probe"], GREEN)]
    for k, (lab, r, c) in enumerate(outs):
        x = 20 + k * 322
        b += R(x, ys, 308, 62, rx=8, fill=tint(c, 0.9), stroke=c, sw=1.4)
        b += T(x + 12, ys + 24, lab, size=16, anchor="start", weight="bold", fill=c)
        b += T(x + 12, ys + 48, f"score {r['score']:.2f}, {'success' if r['success'] else 'failure'}, {r['rounds']} rounds", size=14.5, anchor="start")
        b += R(x + 196, ys + 14, 100, 14, rx=3, fill="#FFFFFF", stroke=c, sw=1.0) + R(x + 196, ys + 14, 100 * r["score"], 14, rx=3, fill=tint(c, 0.4), stroke="none", sw=0)
    b += T(20, ys + 90, f"Recorded symbolic episode (seed {m['seed']}, Eye mass noise sigma {m['mass_sigma']}, scripted Brain): FIXTURE, a mechanism illustration, not a result.", size=14, anchor="start", fill=C_SUB)
    b += T(20, ys + 110, "Panels a and b show the scene the planner actually used; the challenge values are the Arms' own touch measurements.", size=14, anchor="start", fill=C_SUB)
    save_svg("fig_mechanism", W_, int(ys + 124), b)


def fig_overview():
    W_, H_ = 1000, 760
    cx = [20, 275, 530, 785]
    cw, ch = 215, 128
    r0, r1, r2, r3 = 26, 176, 350, 524
    b = ""

    def box(x, y, title, body, color, h=ch, fill=None, dash=None, n=None):
        s, bb = card(x, y, cw, title, body, h=h, color=color, fill=fill, dash=dash, tsize=16.5, bsize=14.5)
        if n:
            s += num(x + cw - 17, y + 17, n, color=color, r=11)
        return s, bb

    HAR = box(cx[1], r0, "Harness", "19 markdown files: rules, skills, notes, memory. Two-pass BM25 retrieval.", BLUE, h=112)
    HUM = box(cx[0], r1, "Human", "Gives the instruction; may AMEND it mid-episode.", GREY_D, fill="#F1F3F5", n=1)
    BRN = box(cx[1], r1, "Brain", "Text-only LLM, never sees pixels. Plans a DAG of tasks with tools.", PURPLE, n=4)
    VER = box(cx[2], r1, "Verifier", "none, static or dry run before any arm moves; reasons fed back, 2 retries.", GREEN, n=5)
    SCH = box(cx[3], r1, "Scheduler + ladder", "Head task per arm, bag lock; retry, retreat, reassign, replan, abort.", GREEN, n=6)
    BUS = box(cx[1], r2, "Message bus", "13 typed messages, schema-checked, token-budgeted.", NAVY, n=3)
    ARM = box(cx[3], r2, "Arm L, Arm R", "Blind: force and proprioception only. Report TOUCH; CHALLENGE the Eye.", ORANGE, n=7)
    EYE = box(cx[1], r3, "Eye A, Eye B", "VLMs, the only agents with pixels. Emit SCENE JSON.", TEAL, n=2)
    PRB = box(cx[2], r3, "Probe (optional)", "Arms weigh every item first; measured mass overrides Eye guesses.", BLUE, h=104, dash="6 4")
    WRD = box(cx[3], r3, "World", "Symbolic, pose-level or Isaac Lab; one checker.", GREY_D, fill="#F1F3F5")
    for s_, _ in (HAR, HUM, BRN, VER, SCH, BUS, ARM, EYE, PRB, WRD):
        b += s_
    har, hum, brn, ver, sch, bus, arm, eye, prb, wrd = (c[1] for c in (HAR, HUM, BRN, VER, SCH, BUS, ARM, EYE, PRB, WRD))
    b += connect(har, "b", brn, "t", color=BLUE, sw=2)
    b += connect(hum, "r", brn, "l", color=GREY_D, sw=2)
    b += connect(brn, "r", ver, "l", color=PURPLE, sw=2)
    b += connect(ver, "r", sch, "l", color=GREEN, sw=2)
    b += connect(hum, "b", bus, "l", color=GREY_D, sw=2)
    b += connect(bus, "t", brn, "b", color=NAVY, sw=2)
    b += connect(sch, "b", arm, "t", color=GREEN, sw=2)
    b += connect(arm, "l", bus, "r", color=ORANGE, sw=2)
    b += connect(arm, "b", wrd, "t", color=ORANGE, sw=2)
    b += connect(wrd, "l", eye, "r", oa=110, ob=110, color=GREY_D, sw=2)
    b += connect(eye, "t", bus, "b", color=TEAL, sw=2)
    b += T(660, r2 + 56, "TOUCH, CHALLENGE", size=14.5, fill=ORANGE, weight="bold")
    b += T(64, r2 + 50, "AMEND", size=14.5, fill=GREY_D, weight="bold")
    b += T(cx[1] + cw / 2 + 52, r3 - 22, "SCENE", size=14.5, fill=TEAL, weight="bold")
    b += T(cx[1] + cw / 2 + 52, r1 + ch + 26, "reports", size=14.5, fill=NAVY, weight="bold")
    b += T(660, r3 + 138, "images", size=14.5, fill=GREY_D)
    # status and legend
    yb = 686
    b += pending_tag(cx[1] - 255, yb, "Eye LoRA: pending") + pending_tag(cx[1] - 80, yb, "Brain LoRA: pending") + pending_tag(cx[1] + 105, yb, "Arm policy: pending")
    b += T(20, yb + 52, "Numbers give the order of one episode. Teal Eye, purple Brain, orange Arm; dashed = optional; red = no trained model yet.", size=14, anchor="start", fill=C_SUB)
    save_svg("fig_overview", W_, H_, b)


def fig_messages():
    sch = FD["messages"]["schema"]
    ex = FD["messages"]["examples"]["CHALLENGE"]
    comm = FX["comm"]
    W_ = 1000
    b = panel(20, 28, "a", "Thirteen typed messages and their required fields")
    sender = {"ABILITY": ORANGE, "SCENE": TEAL, "TOUCH": ORANGE, "LOOK": PURPLE, "CHALLENGE": ORANGE, "AMEND": GREY_D}
    cw, ch, gx, gy = 182, 92, 12, 10
    for k, n in enumerate(sch):
        r, c = divmod(k, 5)
        x, y = 20 + c * (cw + gx), 46 + r * (ch + gy)
        col = sender.get(n, GREY)
        seen = n in sender
        b += R(x, y, cw, ch, rx=8, fill=tint(col, 0.9 if seen else 0.94), stroke=col, sw=1.5 if seen else 1.0)
        b += T(x + 10, y + 23, n, size=16, anchor="start", fill=col if seen else GREY_D, weight="bold")
        for j, ln in enumerate(wrap(", ".join(sch[n]), cw - 20, 14.5)[:3]):
            b += T(x + 10, y + 44 + j * 18, ln, size=14.5, anchor="start", fill=C_TXT if seen else C_SUB)
    yn = 46 + 3 * (ch + gy) + 4
    b += T(20, yn + 14, "Colour = sender in the recorded episodes: Eye teal, Arm orange, Brain purple, human dark grey. Light grey: not exchanged in them.", size=14, anchor="start", fill=C_SUB)
    # ---- b and c below
    y0 = yn + 50
    b += panel(20, y0, "b", "A real CHALLENGE (recorded, seed 24)")
    js = json.dumps({"type": ex["type"], "sender": ex["sender"], "receiver": ex["receiver"], "content": ex["content"]}, separators=(", ", ": "))
    lines = wrap(js, 440, 13.5, mono=True)
    ht = 18 + len(lines) * 18
    b += R(20, y0 + 14, 470, ht + 38, rx=8, fill="#F6F7F9", stroke=C_MUT)
    for j, ln in enumerate(lines):
        b += T(32, y0 + 38 + j * 18, ln, size=13.5, anchor="start", font=MONO)
    b += T(32, y0 + 38 + len(lines) * 18 + 6, f"token proxy: {ex['n_tokens_proxy']} (about 4 characters per token)", size=14, anchor="start", fill=C_SUB)
    s, bb = card(20, y0 + 14 + ht + 38 + 16, 470, "Bus", "Validates the schema, charges tokens against a per-episode budget, and refuses an over-budget message while telling the sender.", color=NAVY)
    b += s
    b += panel(520, y0, "c", "Four communication modes")
    mx = max(v["tokens_per_episode"] for v in comm.values())
    for k, (mode, desc) in enumerate((("push", "full scene every round"), ("delta", "changed objects only"), ("pull", "on request and after failures"), ("init_only", "once at the start"))):
        y = y0 + 26 + k * 58
        v = comm[mode]
        b += T(520, y + 18, mode.replace("_", "-"), size=16, anchor="start", weight="bold", fill=NAVY)
        b += T(520, y + 38, desc, size=14, anchor="start", fill=C_SUB)
        w = 250 * v["tokens_per_episode"] / mx
        b += R(715, y + 6, max(w, 3), 26, rx=3, fill=tint(BLUE, 0.35), stroke=BLUE, sw=1.0)
        inside = w > 130
        b += T(715 + (w - 8 if inside else w + 8), y + 25, f"{v['tokens_per_episode']:,.0f}", size=14.5, anchor="end" if inside else "start", fill="#FFFFFF" if inside else C_TXT, weight="bold")
    yt = y0 + 26 + 4 * 58 + 10
    b += tag(520, yt, "FIXTURE", ORANGE)
    b += T(612, yt + 17, "tokens per episode; symbolic world, noisy Eyes, 80 seeds", size=14, anchor="start", fill=C_SUB)
    b += T(520, yt + 48, f"Success: push {comm['push']['k']}/80, delta {comm['delta']['k']}/80, pull {comm['pull']['k']}/80, init-only {comm['init_only']['k']}/80.", size=14, anchor="start", fill=C_SUB)
    b += T(520, yt + 68, "Checks the accounting only; says nothing about trained agents.", size=14, anchor="start", fill=C_SUB)
    save_svg("fig_messages", W_, int(max(yt + 84, bb[1] + bb[3] + 16)), b)


def fig_loop():
    lp = FD["loop"]
    st = lp["story"]
    ev = lp["events"]
    W_ = 1000
    b = panel(20, 28, "a", f"One recorded episode (seed {lp['seed']}, flaky planner, leaky Eyes)")
    lanes = [("Round", GREY_D), ("Brain", PURPLE), ("Eyes", TEAL), ("Arm L", ORANGE), ("Arm R", ORANGE), ("Bus", NAVY)]
    lx0, lx1 = 100, 985
    ly, y = {}, 52
    for n, c in lanes:
        hh = 30 if n == "Round" else 56
        ly[n] = y
        b += R(lx0 - 4, y, lx1 - lx0 + 8, hh, rx=6, fill=tint(c, 0.95), stroke=tint(c, 0.6), sw=1.0)
        b += T(lx0 - 12, y + hh / 2 + 5, n, size=15, anchor="end", weight="bold", fill=c)
        y += hh + 6
    keep = [e for e in ev if e["ev"] == "exec" or e.get("type") in ("LOOK", "CHALLENGE", "AMEND")]
    n = len(keep)
    step = (lx1 - lx0 - 16) / n
    short = {"notebook": "nb", "pencilcase": "pc", "lunchbox": "lb", "bottle": "bt", "laptop": "lp", "book": "bk"}
    abbr = lambda o: next((v + o[len(k):] for k, v in short.items() if o.startswith(k)), o[:3])
    xs = {k: lx0 + 6 + k * step for k in range(n)}
    rounds = [e["round"] for e in st["executed"]]
    ri = 0
    by_round: dict[int, list[int]] = {}
    for k, e in enumerate(keep):
        x = xs[k]
        cxm = round(x + (step - 3) / 2, 1)
        if e["ev"] == "exec":
            by_round.setdefault(rounds[ri], []).append(k)
            ri += 1
            lane = "Arm " + e["arm"]
            col = ORANGE if e["ok"] else RED
            b += R(x, ly[lane] + 4, step - 3, 48, rx=3, fill=tint(col, 0.8), stroke=col, sw=1.0)
            b += T(cxm, ly[lane] + 22, "g" if e["skill"] == "grasp" else "p", size=14.5, weight="bold")
            b += T(cxm, ly[lane] + 43, abbr(e["obj"]), size=13)
        else:
            t = e["type"]
            lane = {"LOOK": "Eyes", "CHALLENGE": "Bus", "AMEND": "Bus"}[t]
            col = {"LOOK": TEAL, "CHALLENGE": ORANGE, "AMEND": GREY_D}[t]
            b += R(x, ly[lane] + 4, step - 3, 48, rx=3, fill=tint(col, 0.75), stroke=col, sw=1.5)
            b += T(cxm, ly[lane] + 34, {"LOOK": "L", "CHALLENGE": "C", "AMEND": "A"}[t], size=14.5, weight="bold")
    for r_, ks in by_round.items():
        xa, xb = xs[ks[0]], xs[ks[-1]] + step - 3
        b += R(xa, ly["Round"] + 3, xb - xa, 24, rx=3, fill=tint(GREY_D, 0.8), stroke=GREY_D, sw=1.0)
        b += T((xa + xb) / 2, ly["Round"] + 20, str(r_), size=13, weight="bold", fill=GREY_D)
    il = [k for k, e in enumerate(keep) if e.get("type") == "LOOK"]
    ia = [k for k, e in enumerate(keep) if e.get("type") == "AMEND"]
    for k, lab in [(0, "P")] + [(k + 1, "R") for k in il + ia if k + 1 < n]:
        b += R(xs[k], ly["Brain"] + 4, step - 3, 48, rx=3, fill=tint(PURPLE, 0.78), stroke=PURPLE, sw=1.3)
        b += T(round(xs[k] + (step - 3) / 2, 1), ly["Brain"] + 34, lab, size=14.5, weight="bold")
    yl = y + 6
    b += T(20, yl + 12, "Arms: g grasp, p place, then the item (bk book, nb notebook, pc pencilcase, bt bottle, lp laptop, lb lunchbox). Round: the loop round of those tasks.", size=14, anchor="start", fill=C_SUB)
    b += T(20, yl + 32, "Eyes: L = LOOK. Bus: C = CHALLENGE, A = AMEND. Brain: P = first plan, R = replan after LOOK or AMEND. Red box = failed skill (none here).", size=14, anchor="start", fill=C_SUB)
    chips = [(f"{sum(1 for e in ev if e['ev'] == 'exec')} tasks", GREY_D), (f"{st['rounds']} rounds", GREY_D), (f"{st['looks']} LOOK", TEAL), (f"{st['challenges']} CHALLENGE", ORANGE),
             (f"AMEND at round {lp['amend_round']}", GREY_D), (f"{st['replans']} replans", PURPLE), ("success" if st["success"] else "failure", GREEN if st["success"] else RED)]
    x = 20
    for t, c in chips:
        w = tw(t, 14.5, True) + 16
        b += tag(x, yl + 46, t, c, size=14.5, w=w)
        x += w + 10
    yb = yl + 96
    b += panel(20, yb, "b", "Stages of the loop, and what this trace shows")
    rej = sum(1 for f in st["failures"] if f["cause"] == "plan_rejected")
    inf = sum(1 for f in st["failures"] if f["cause"] == "static_infeasible")
    stages = [("Probe", "optional; not used in this episode", BLUE, True),
              ("Verify + retry", f"dry run: {rej} drafts rejected with reasons; {inf} phases ended unsatisfied", GREEN, False),
              ("LOOK", f"an item is missing from the scene: {st['looks']} new look, then replan", TEAL, False),
              ("Rounds", "each Arm runs its head task; the bag is exclusive", ORANGE, False),
              ("CHALLENGE", f"touch vs Eye mass over 20%: {st['challenges']} here; scene corrected", ORANGE, False),
              ("AMEND", f"instruction changes at round {lp['amend_round']}: replan", GREY_D, False),
              ("End check", "rules.check_final; fatal failures attributed", NAVY, False)]
    sw_, bot = 126, 0
    for k, (t, d, c, opt) in enumerate(stages):
        s, bb = card(20 + k * (sw_ + 12), yb + 14, sw_, t, d, h=170, color=c, dash="6 4" if opt else None, tsize=16, bsize=14, pad=8)
        b += s
        bot = bb[1] + bb[3]
    b += T(20, bot + 26, "FIXTURE: symbolic world, scripted Brain behind a flaky-planner wrapper, parametric Eyes.", size=14, anchor="start", fill=C_SUB)
    b += T(20, bot + 46, "Recovery ladder on a failed skill: retry, alternate parameters, retreat, reassign to the other Arm, replan, abort.", size=14, anchor="start", fill=C_SUB)
    save_svg("fig_loop", W_, int(bot + 60), b)


def fig_signals():
    sg = FD["signals"]
    w = sg["weights"]
    ep = sg["episode_reward"]
    W_ = 1000
    b = T(20, 28, "From data to trained components", size=17, anchor="start", weight="bold", fill=NAVY)
    heads = [("Training data or signal", 20, 270), ("Component", 330, 250), ("Method", 620, 270)]
    for t, x, wd in heads:
        b += T(x, 56, t, size=14.5, anchor="start", weight="bold", fill=C_SUB)
    rows = [
        (TEAL, "Rendered table images with SCENE JSON targets", "Eye (VLM)", "LoRA fine-tuning, then rejection sampling: K = 8 reports per image, keep the best by field accuracy or by counterfactual credit."),
        (PURPLE, "Teacher plans validated by closed-loop success", "Brain (text LLM)", "Supervised fine-tuning to learn the format, then GRPO on the plan-level reward below."),
        (ORANGE, "Scripted differential-IK demonstrations", "Arms (blind policy)", "Small transformer with a flow-matching action head; an MLP head is the ablation."),
        (NAVY, "Finished episodes, including failed ones", "Team signals", "Counterfactual Eye credit and hindsight relabeling feed the Eye and Brain data above."),
    ]
    y = 66
    for c, src, comp, how in rows:
        s1, p1 = card(20, y, 270, None, src, h=100, color=c, fill="#FFFFFF", bsize=15)
        s2, p2 = card(330, y, 250, comp, "", h=100, color=c, tsize=17, center=True)
        s3, p3 = card(620, y, 270, None, how, h=100, color=c, fill="#FFFFFF", bsize=14.5)
        b += s1 + s2 + s3 + connect(p1, "r", p2, "l", color=c, sw=2) + connect(p2, "r", p3, "l", color=c, sw=2)
        b += pending_tag(905, y + 36, "pending", w=82)
        y += 118
    s, p = card(20, y + 4, 960, "Brain plan-level reward (weights read from the code)",
                f"R = {w['format']} r_format + {w['feasible']} r_feasible + {w['rules']} r_rules + {w['length']} r_length, optionally + {sg['rollout_weight']} r_rollout (closed-loop rollout).\n"
                f"r_rules: valid-item fraction from a deterministic dry run, minus {sg['r_rules_penalty']} per violation and per crush, floored at 0.   r_length: 1 up to {sg['length_cap']} tasks, falling to 0 at {2 * sg['length_cap']}.",
                color=PURPLE, bsize=14.5)
    b += s
    s, p2 = card(20, p[1] + p[3] + 12, 960, "Episode score used for credit",
                 f"valid-item fraction - {ep['w_viol']} per violation - {ep['w_crush']} per crushed item - {ep['w_replan']} per replan.   Credit C(i, r) = R(real) - R(Eye i replaced by r); oracle gain G = R(oracle) - R(real) = -C(i, oracle).",
                 color=NAVY, bsize=14.5)
    b += s
    b += T(20, p2[1] + p2[3] + 28, "Formulas are definitions. No trained model has been run with these signals yet, so every outcome is pending.", size=14, anchor="start", fill=C_SUB)
    save_svg("fig_signals", W_, int(p2[1] + p2[3] + 44), b)


def fig_bridge():
    br = FD["bridge"]
    lay = br["layout"]
    W_, H_ = 1000, 600
    b = panel(20, 28, "a", "Isaac scene, top view (layout constants, metres)")
    sc = 330.0
    ox, oy = 292, 548                                  # pixel of world (0, y = -0.8)
    px = lambda x: ox + x * sc
    py = lambda y: oy - (y + 0.8) * sc
    ytab0, ytab1 = -0.10, 0.23
    # table
    b += R(px(-0.62), py(ytab1), 1.24 * sc, (ytab1 - ytab0) * sc, rx=8, fill="#F6F7F9", stroke=GREY, sw=1.5)
    for xb in (lay["left_max"], lay["right_min"]):
        b += L(px(xb), py(ytab1), px(xb), py(ytab0), color=C_MUT, sw=1.2, dash="5 4")
    for nm, x in br["zone_x"].items():
        b += T(px(x), py(ytab0) - 8, nm.replace("table_", ""), size=14, fill=C_SUB)
        for yr in br["row_y"]:
            b += f'<circle cx="{px(x):.1f}" cy="{py(yr):.1f}" r="5" fill="{GREY}"/>'
    # bag with compartments
    bc, bh = lay["bag_center"], lay["bag_half"]
    bx0, bx1, by0, by1 = bc[0] - bh[0], bc[0] + bh[0], bc[1] - bh[1], bc[1] + bh[1]
    ld, fd = lay["laptop_slot_depth"], lay["front_pocket_depth"]
    b += R(px(bx0), py(by1), (bx1 - bx0) * sc, (by1 - by0) * sc, rx=8, fill="#FFFFFF", stroke=NAVY, sw=2.2)
    b += R(px(bx0), py(by0 + ld), (bx1 - bx0) * sc, ld * sc, rx=0, fill=tint(PURPLE, 0.85), stroke=PURPLE, sw=1.0)
    b += R(px(bx0), py(by1), (bx1 - bx0) * sc, fd * sc, rx=0, fill=tint(ORANGE, 0.85), stroke=ORANGE, sw=1.0)
    b += T(px(0), py((by0 + ld + by1 - fd) / 2) + 6, "main", size=14.5, fill=NAVY, weight="bold")
    b += T(px(bx1) + 10, py(by0 + ld / 2) + 6, "laptop slot (back)", size=14, anchor="start", fill=PURPLE)
    b += T(px(bx1) + 10, py(by1 - fd / 2) + 6, "front pocket", size=14, anchor="start", fill=ORANGE)
    b += T(px(bx0) - 10, py(bc[1]) + 6, f"bag {2 * bh[0]:.2f} x {2 * bh[1]:.2f} m", size=14, anchor="end", fill=NAVY)
    # arms
    for nm, (x, y, _) in br["arm_base"].items():
        b += f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="16" fill="{tint(ORANGE, 0.7)}" stroke="{ORANGE}" stroke-width="2"/>'
        b += T(px(x), py(y) + 6, nm, size=15, fill=ORANGE, weight="bold")
    # eyes with view lines
    tgt = None
    for nm, (pos, look) in br["eye_poses"].items():
        tgt = look
        ex, ey = pos[0], pos[1]
        b += arrow([(px(ex) + (18 if ex < 0 else -18), py(ey) - 14), (px(look[0]) + (-20 if ex < 0 else 20), py(look[1]) + 8)], color=TEAL, sw=1.6, dash="6 5", free="vector")
        b += R(px(ex) - 18, py(ey) - 14, 36, 28, rx=5, fill=tint(TEAL, 0.7), stroke=TEAL, sw=2)
        b += T(px(ex), py(ey) + 6, nm, size=15, fill=TEAL, weight="bold")
    b += T(px(0), py(-0.76) + 6, "Eyes: z = 0.75 m, looking at the table centre", size=14.5, fill=TEAL, weight="bold")
    # ---- b
    b += panel(620, 28, "b", "One success definition, three worlds")
    worlds = [("Symbolic world", "Fast; the loop is developed and tested here.", GREEN),
              ("Pose-level fake simulator", "Poses in; a bridge maps them to regions and compartments. Loop outcomes match the symbolic world in tests.", BLUE),
              ("Isaac Lab", "Physics. Skill outcomes are derived from the state before and after each skill.", ORANGE)]
    y = 48
    bxs = []
    for nm, d, c in worlds:
        s, bb = card(620, y, 360, nm, d, h=104, color=c)
        b += s
        bxs.append(bb)
        y += 118
    b += pending_tag(850, bxs[2][1] + 8, "render pending", w=118)
    s, chk = card(620, y + 14, 360, "rules.check_final", "Every requested item packed, zero rule violations, nothing crushed. One function scores all three worlds.", h=104, color=NAVY)
    b += s
    for bb in bxs:
        b += connect(bb, "b", chk, "t", color=C_MUT, sw=1.0) if False else ""
    yq = chk[1] + chk[3] + 26
    b += T(620, yq, "Ten skill outcomes derived from state:", size=14, anchor="start", fill=C_SUB, weight="bold")
    for j, ln in enumerate(wrap(", ".join(br["skill_outcomes"]), 360, 14)):
        b += T(620, yq + 20 + j * 18, ln, size=14, anchor="start", fill=C_TXT)
    b += T(620, yq + 74, "Quaternion order: wxyz (Isaac Lab 2.x), xyzw (3.0).", size=14, anchor="start", fill=C_SUB)
    b += T(620, yq + 94, "The identical-outcome test validates the integration,", size=14, anchor="start", fill=C_SUB)
    b += T(620, yq + 112, "not the physics.", size=14, anchor="start", fill=C_SUB)
    save_svg("fig_bridge", W_, max(H_, int(yq + 128)), b)


def fig_design():
    W_ = 1000
    b = T(20, 28, "Five questions, the experiment that answers each, and where it is reported", size=17, anchor="start", weight="bold", fill=NAVY)
    for t, x in (("Question", 20), ("Experiment", 345), ("Evidence class", 665), ("Reported in", 815)):
        b += T(x, 56, t, size=14.5, anchor="start", weight="bold", fill=C_SUB)
    rows = [
        (BLUE, "Does the upstream level-2 scorer's random start position change scores?", "Inventory the released data; rescore released checkpoints with a seeded and an exhaustive scorer.", "TRACK A", "Fig. scorer"),
        (BLUE, "How much of a level-1 error is perception rather than reasoning?", "Same text-only Brain given oracle descriptions, no description, or a real Eye description.", "TRACK A", "Table Track A"),
        (PURPLE, "Do tools, verification and the harness help the team, and which part matters?", "Full system against single-factor ablations: harness, verification, challenge, probe, one Eye, communication mode.", "LLM-SYMBOLIC, ISAAC", "Tables main, ablations"),
        (ORANGE, "How far does plan validity overstate physical success, and who is blamed?", "Plan pass against physical success, split by the agent the loop charges; attribution checked by hand labels.", "ISAAC", "Fig. results, Table attribution"),
        (TEAL, "Do counterfactual credit and hindsight relabeling improve training?", "Credit-guided rejection sampling against plain SFT; Brain data with and without relabeled records at fixed size.", "LLM-SYMBOLIC, ISAAC", "Table credit"),
    ]
    y = 66
    for c, q, e, cls, where in rows:
        s1, p1 = card(20, y, 300, None, q, h=104, color=c, fill="#FFFFFF", bsize=14.5)
        s2, p2 = card(345, y, 295, None, e, h=104, color=c, fill="#FFFFFF", bsize=14.5)
        s3, p3 = card(665, y, 130, cls, "", h=104, color=GREY_D, fill="#F1F3F5", tsize=14.5, center=True)
        s4, p4 = card(815, y, 165, where, "", h=104, color=GREY_D, fill="#F1F3F5", tsize=14.5, center=True)
        b += s1 + s2 + s3 + s4 + connect(p1, "r", p2, "l", color=c, sw=2) + connect(p2, "r", p3, "l", color=c, sw=2)
        b += pending_tag(823, y + 70, "pending", w=82)
        y += 120
    b += T(20, y + 14, "Every outcome is pending. Evidence classes are never pooled; fixture runs are kept out of this design (appendix F). Evaluation seeds are below 100000.", size=14, anchor="start", fill=C_SUB)
    save_svg("fig_design", W_, int(y + 32), b)


def fig_results(root=None):
    import math, os
    from pathlib import Path
    from figs_b import load_results, wilson
    root = root or Path(os.environ.get("RTP_RESULTS", PAPER_DIR / "results"))
    recs, cred = load_results(root) if root.exists() else ({}, None)
    rows = [("Full system, LLM-SYMBOLIC", "llm_full"), ("Full system, ISAAC", "isaac_full"), ("No harness", "llm_no_harness"),
            ("Verification: none", "llm_verify_none"), ("Verification: static", "llm_verify_static"), ("Verification: dry run", "llm_verify_dry_run"),
            ("No CHALLENGE, no probe", "llm_no_challenge"), ("Probe on", "llm_probe"), ("One Eye", "llm_one_eye"),
            ("Communication: push", "llm_comm_push"), ("Communication: delta", "llm_comm_delta"), ("Communication: pull", "llm_comm_pull"),
            ("Communication: init-only", "llm_comm_init_only")]
    W_ = 1000
    have = bool(recs)
    b = panel(20, 28, "a", "Success, with Wilson 95% interval")
    b += panel(520, 28, "b", "Plan pass (purple) vs success (green)")
    xa, xb, tw_ = 250, 750, 230
    for xt in (xa, xb):
        for v in (0, 0.5, 1.0):
            b += T(xt + v * tw_, 62, f"{v:g}", size=13, fill=C_SUB)
    y0 = 70
    for k, (lab, key) in enumerate(rows):
        y = y0 + k * 30
        b += T(240, y + 19, lab, size=14.5, anchor="end")
        for xt in (xa, xb):
            b += R(xt, y + 4, tw_, 20, rx=3, fill="#FAFBFC", stroke=C_MUT, sw=0.9, dash=None if have else "4 3")
            for v in (0.5,):
                b += L(xt + v * tw_, y + 4, xt + v * tw_, y + 24, color="#E2E6EA", sw=1.0)
        r = recs.get(key)
        if r:
            n = len(r)
            kk = sum(1 for e in r if e["success"])
            p, lo, hi = wilson(kk, n)
            pp = sum(1 for e in r if e["plan_pass"]) / n
            b += R(xa, y + 4, tw_ * p, 20, rx=3, fill=tint(GREEN, 0.4), stroke=GREEN, sw=1.0)
            b += L(xa + tw_ * lo, y + 14, xa + tw_ * hi, y + 14, color=NAVY, sw=2.0)
            b += R(xb, y + 4, tw_ * pp, 9, rx=2, fill=tint(PURPLE, 0.4), stroke=PURPLE, sw=0.9) + R(xb, y + 15, tw_ * p, 9, rx=2, fill=tint(GREEN, 0.4), stroke=GREEN, sw=0.9)
        else:
            b += T(xa + tw_ / 2, y + 19, "pending", size=13, fill=RED, weight="bold") + T(xb + tw_ / 2, y + 19, "pending", size=13, fill=RED, weight="bold")
    yc = y0 + len(rows) * 30 + 28
    b += panel(20, yc, "c", "Oracle gain of an Eye, by injected fault and field (0 to 0.5)")
    faults = [r["fault"] for r in FX["credit"]]
    rowsc = cred if isinstance(cred, list) else []
    xs = {"mass_guess": 250, "cls": 400, "fragility_guess": 550, "region": 700}
    for f_, x_ in xs.items():
        b += T(x_ + 60, yc + 28, f_.replace("_guess", ""), size=13, fill=C_SUB, weight="bold")
    for k, fl in enumerate(faults):
        y = yc + 38 + k * 30
        b += T(240, y + 19, fl, size=14.5, anchor="end")
        match = next((r for r in rowsc if r["fault"] == fl), None)
        for f_, x_ in xs.items():
            b += R(x_, y + 4, 120, 20, rx=3, fill="#FAFBFC", stroke=C_MUT, sw=0.9, dash=None if match else "4 3")
            if match:
                g = match["fields"].get(f_, 0.0)
                b += R(x_, y + 4, 120 * max(0.0, min(1.0, g / 0.5)), 20, rx=3, fill=tint(TEAL, 0.4), stroke=TEAL, sw=0.9)
            else:
                b += T(x_ + 60, y + 19, "pending", size=13, fill=RED, weight="bold")
    yend = yc + 38 + len(faults) * 30 + 14
    b += T(20, yend + 10, "Drawn by scripts/build_diagrams.py from the results bundle (reports/**/records.jsonl, reports/credit/credit.json).", size=14, anchor="start", fill=C_SUB)
    b += T(20, yend + 30, "With no bundle every bar is empty and marked pending; nothing here is simulated.", size=14, anchor="start", fill=C_SUB)
    save_svg("fig_results", W_, int(yend + 46), b)
    return have, bool(cred)
