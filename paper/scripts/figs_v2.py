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
    nbm = next(c for c in ch if c["content"]["obj"] == "notebook1")["content"]["evidence"]
    s, p3 = card(640, p2[1] + p2[3] + 12, 340, "Fix 1: CHALLENGE",
                 f"The Arm weighs what it grasps and disputes the Eye's report at that moment ({len(ch)} challenges in this run; e.g. notebook1 measured {nbm['measured_weight']:.2f} kg, reported {nbm['reported']:.2f} kg). The scene is corrected and the rest replanned.", color=ORANGE)
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
    b += T(20, ys + 110, "Panels a and b show the scene the planner actually used. Eye noise is redrawn at every observation, so a later report (as in c) can differ from the one in a.", size=14, anchor="start", fill=C_SUB)
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


def fig_mechanism():
    """Figure 1: the teaser. Problem, design, what it gives, what is new. Numbers from the recorded seed-24 episode."""
    import icons as I
    m, cr = FD["mechanism"], FD["credit"]
    it = {i["name"]: i for i in m["items"]}
    pc, nb = it["pencilcase1"], it["notebook1"]
    pce = (pc["eye_mass"] - pc["true_mass"]) / pc["true_mass"] * 100
    nbe = (nb["eye_mass"] - nb["true_mass"]) / nb["true_mass"] * 100
    nbm = next(c for c in m["challenge_msgs"] if c["content"]["obj"] == "notebook1")["content"]["evidence"]
    W_ = 1000
    b = ""
    # ============ column 1: the problem
    b += T(20, 30, "1", size=18, anchor="start", weight="bold", fill=RED) + T(42, 30, "Problem: a team fails. Who failed?", size=16.5, anchor="start", weight="bold", fill=RED)
    b += R(20, 46, 320, 82, rx=10, fill="#F6F7F9", stroke=C_MUT, sw=1.3)
    names = [("laptop", I.laptop, "laptop"), ("book", I.book, "book"), ("bottle", I.bottle, "bottle"), ("notebook", I.notebook, "notebook"), ("pencil case", I.pencilcase, "pencil case")]
    for k, (lab, fn, _) in enumerate(names):
        cx = 48 + k * 64
        hot = lab in ("notebook", "pencil case")
        b += fn(cx, 80, 40, RED if hot else GREY_D)
        b += T(cx, 118, lab, size=13, fill=RED if hot else C_SUB)
    b += T(180, 144, "the table the Eyes photograph", size=13, fill=C_SUB)
    tiles = [("Eye", I.eye, TEAL), ("Brain", I.plan, PURPLE), ("Arm", I.arm, ORANGE)]
    b += arrow([(78, 132), (78, 157)], color=GREY_D, sw=2.0, free="vector")
    tb = []
    for k, (lab, fn, col) in enumerate(tiles):
        x = 36 + k * 112
        b += R(x, 160, 84, 88, rx=10, fill=tint(col), stroke=col, sw=1.8)
        b += fn(x + 42, 192, 52, col)
        b += T(x + 42, 238, lab, size=15, fill=col, weight="bold")
        tb.append((x, 160, 84, 88))
    for k in range(2):
        b += connect(tb[k], "r", tb[k + 1], "l", color=GREY_D, sw=1.8)
    # bag with wrong order
    b += I.bag(36, 292, 96, 112, NAVY)
    b += I.notebook(84, 328, 44, RED) + I.pencilcase(84, 376, 44, RED)
    b += I.mark(128, 292, False, 13)
    b += T(36, 428, "bag", size=13, anchor="start", fill=C_SUB)
    b += T(150, 306, f"Eye: pencil case {pc['eye_mass']:.2f} kg", size=14, anchor="start", fill=TEAL, weight="bold")
    b += T(150, 324, f"(true {pc['true_mass']:.2f}, {pce:+.0f}%)", size=14, anchor="start", fill=TEAL)
    b += T(150, 348, f"Eye: notebook {nb['eye_mass']:.2f} kg", size=14, anchor="start", fill=TEAL, weight="bold")
    b += T(150, 366, f"(true {nb['true_mass']:.2f}, {nbe:+.0f}%)", size=14, anchor="start", fill=TEAL)
    b += T(150, 392, "Brain stacks heavy on light:", size=14, anchor="start", fill=PURPLE)
    b += T(150, 410, "heavy_bottom broken", size=14, anchor="start", fill=RED, weight="bold")
    b += tag(36, 446, f"one team score: {m['no_fix']['score']:.2f}, failure", RED)
    for k, col in enumerate((TEAL, PURPLE, ORANGE)):
        b += I.qmark(44 + k * 36, 498, col, 14)
    b += T(160, 503, "Eye, Brain or Arm?", size=15, anchor="start", weight="bold", fill=C_TXT)
    # ============ column 2: the design
    b += T(365, 30, "2", size=18, anchor="start", weight="bold", fill=NAVY) + T(387, 30, "Our team: split by interface", size=16.5, anchor="start", weight="bold", fill=NAVY)
    steps = [
        (TEAL, I.eye, "Eyes: typed reports", f"Only agents with pixels. Claims are typed messages, e.g. SCENE: pencil case {pc['eye_mass']:.2f} kg."),
        (PURPLE, I.plan, "Brain: plan, then dry run", "Text-only, guided by rules. A dry run replays the plan before any arm moves."),
        (ORANGE, I.arm, "Arms: touch can challenge", f"Blind. Weighing the notebook: {nbm['measured_weight']:.2f} kg vs the Eye's {nbm['reported']:.2f}, so CHALLENGE and replan."),
    ]
    y = 46
    sb = []
    for col, fn, t, body in steps:
        s, bb = card(365, y, 300, t, body, h=112, color=col, pad=12, rpad=58)
        b += s + fn(365 + 300 - 38, y + 40, 46, col)
        sb.append(bb)
        y += bb[3] + 30
    for k in range(2):
        b += connect(sb[k], "b", sb[k + 1], "t", color=GREY_D, sw=1.8)
    s, bb = card(365, y, 300, "Failure charged to: Eye", "Each failure gets an owner by rule (heuristic, to be checked against hand labels).", h=96, color=TEAL, fill=tint(TEAL, 0.82), pad=12)
    b += s
    # ============ column 3: what it gives
    b += T(690, 30, "3", size=18, anchor="start", weight="bold", fill=GREEN) + T(712, 30, "What the split gives", size=16.5, anchor="start", weight="bold", fill=GREEN)
    s, c1 = card(690, 46, 290, "Repair", "Challenge or probe repairs the plan.", h=150, color=GREEN, pad=12)
    b += s
    for k, (lab, v, col) in enumerate((("no fix", m["no_fix"]["score"], RED), ("CHALLENGE", m["challenge"]["score"], GREEN), ("probe", m["probe"]["score"], GREEN))):
        yy = 46 + 76 + k * 26
        b += T(702, yy + 14, lab, size=13.5, anchor="start", fill=C_SUB)
        b += R(800, yy + 2, 120, 15, rx=3, fill="#FFFFFF", stroke=col, sw=1.0) + R(800, yy + 2, 120 * v, 15, rx=3, fill=tint(col, 0.4), stroke="none", sw=0)
        b += T(926, yy + 14, f"{v:.2f}", size=13.5, anchor="start", weight="bold", fill=col)
    s, c2 = card(690, 208, 290, "Credit", f"Replace Eye A's report by the truth and replay: a perfect Eye would add {cr['oracle_gain']:+.2f}, all of it in the mass field.", h=118, color=ORANGE, pad=12)
    b += s
    s, c3 = card(690, 338, 290, "Training data", f"The failed episode is relabeled to what it achieved and admitted only after a replay check ({FD['hindsight']['n_tasks']} tasks).", h=118, color=PURPLE, pad=12)
    b += s
    s, c4 = card(690, 468, 290, "One checker", "The same success function scores the symbolic world, a pose-level simulator and Isaac Lab.", h=96, color=NAVY, pad=12)
    b += s
    for x1, x2 in ((340, 365), (665, 690)):
        b += arrow([(x1 + 3, 300), (x2 - 3, 300)], color=GREY_D, sw=2.4, free="vector")
    # ============ bottom: what is new
    yb = 590
    b += L(20, yb - 14, 980, yb - 14, color=C_MUT, sw=1.2)
    b += T(20, yb + 8, "What is new", size=16.5, anchor="start", weight="bold", fill=NAVY)
    news = [("Roles split by interface", "Only Eyes see pixels, only the Brain plans, Arms are blind: every failure has an owner.", TEAL),
            ("Evidence as typed messages", "13 budgeted types; dry run, CHALLENGE and probe repair errors before they cost the team.", PURPLE),
            ("Signals from the team itself", "Counterfactual Eye credit under common random numbers; hindsight relabeling with a replay check.", ORANGE),
            ("One checker, three worlds", "Symbolic, pose-level and Isaac Lab share a success function; plus an audit of a benchmark scorer.", NAVY)]
    xs = 20
    bots = []
    for k, (t, d, col) in enumerate(news):
        s, bb = card(20 + k * 245, yb + 22, 235, t, d, h=150, color=col, pad=11, tsize=15.5, bsize=14, rpad=22)
        b += s + I_num(20 + k * 245 + 235 - 16, yb + 38, k + 1, col)
        bots.append(bb)
    yend = yb + 22 + 150 + 22
    b += T(20, yend, "Icons are schematic pictograms, not rendered images. Numbers come from one recorded symbolic fixture episode (seed 24, Eye mass noise sigma 0.45).", size=13.5, anchor="start", fill=C_SUB)
    b += T(20, yend + 18, "No trained model and no physics result is shown or claimed in this paper yet.", size=13.5, anchor="start", fill=C_SUB)
    save_svg("fig_mechanism", W_, yend + 34, b)


def I_num(x, y, n, col):
    return num(x, y, n, color=col, r=11)


def fig_mechanism():
    """Figure 1 (teaser), v3: boxed problem on the left; message-sequence centre, open mini-graphics right, open footer."""
    import icons as I
    m, cr, hd = FD["mechanism"], FD["credit"], FD["hindsight"]
    it = {i["name"]: i for i in m["items"]}
    pc, nb = it["pencilcase1"], it["notebook1"]
    pce = (pc["eye_mass"] - pc["true_mass"]) / pc["true_mass"] * 100
    nbe = (nb["eye_mass"] - nb["true_mass"]) / nb["true_mass"] * 100
    nbm = next(c for c in m["challenge_msgs"] if c["content"]["obj"] == "notebook1")["content"]["evidence"]
    W_ = 1000
    b = ""
    # ================================================================ left: the problem (kept)
    b += T(20, 30, "1", size=18, anchor="start", weight="bold", fill=RED) + T(42, 30, "The problem: one score, no owner", size=16.5, anchor="start", weight="bold", fill=RED)
    b += R(20, 46, 320, 82, rx=10, fill="#F6F7F9", stroke=C_MUT, sw=1.3)
    for k, (lab, fn) in enumerate((("laptop", I.laptop), ("book", I.book), ("bottle", I.bottle), ("notebook", I.notebook), ("pencil case", I.pencilcase))):
        cx, hot = 48 + k * 64, lab in ("notebook", "pencil case")
        b += fn(cx, 80, 40, RED if hot else GREY_D) + T(cx, 118, lab, size=13, fill=RED if hot else C_SUB)
    b += T(180, 144, "the table the Eyes photograph", size=13, fill=C_SUB)
    b += arrow([(78, 132), (78, 157)], color=GREY_D, sw=2.0, free="vector")
    tb = []
    for k, (lab, fn, col) in enumerate((("Eye", I.eye, TEAL), ("Brain", I.plan, PURPLE), ("Arm", I.arm, ORANGE))):
        x = 36 + k * 112
        b += R(x, 160, 84, 88, rx=10, fill=tint(col), stroke=col, sw=1.8) + fn(x + 42, 192, 52, col) + T(x + 42, 238, lab, size=15, fill=col, weight="bold")
        tb.append((x, 160, 84, 88))
    for k in range(2):
        b += connect(tb[k], "r", tb[k + 1], "l", color=GREY_D, sw=1.8)
    b += I.bag(36, 292, 96, 112, NAVY) + I.notebook(84, 328, 44, RED) + I.pencilcase(84, 376, 44, RED) + I.mark(128, 292, False, 13)
    b += T(36, 428, "bag", size=13, anchor="start", fill=C_SUB)
    for y_, t_, w_ in ((306, f"Eye: pencil case {pc['eye_mass']:.2f} kg", "bold"), (324, f"(true {pc['true_mass']:.2f}, {pce:+.0f}%)", None),
                       (348, f"Eye: notebook {nb['eye_mass']:.2f} kg", "bold"), (366, f"(true {nb['true_mass']:.2f}, {nbe:+.0f}%)", None)):
        b += T(150, y_, t_, size=14, anchor="start", fill=TEAL, weight=w_)
    b += T(150, 392, "Brain stacks heavy on light:", size=14, anchor="start", fill=PURPLE) + T(150, 410, "heavy_bottom broken", size=14, anchor="start", fill=RED, weight="bold")
    b += tag(36, 446, f"one team score: {m['no_fix']['score']:.2f}, failure", RED)
    for k, col in enumerate((TEAL, PURPLE, ORANGE)):
        b += I.qmark(44 + k * 36, 498, col, 14)
    b += T(160, 503, "Eye, Brain or Arm?", size=15, anchor="start", weight="bold", fill=C_TXT)
    # ================================================================ centre: a message trace, no cards
    b += T(370, 30, "2", size=18, anchor="start", weight="bold", fill=NAVY) + T(392, 30, "Our team: split by interface", size=16.5, anchor="start", weight="bold", fill=NAVY)
    lx = {"Eye": 410, "Brain": 540, "Arm": 670}
    cols = {"Eye": TEAL, "Brain": PURPLE, "Arm": ORANGE}
    fns = {"Eye": I.eye, "Brain": I.plan, "Arm": I.arm}
    for n, x in lx.items():
        b += L(x, 150, x, 540, color=tint(cols[n], 0.55), sw=1.8, dash="5 5")
    for n, x in lx.items():
        b += f'<circle cx="{x}" cy="92" r="34" fill="{tint(cols[n], 0.85)}" stroke="{cols[n]}" stroke-width="2"/>' + fns[n](x, 92, 44, cols[n])
        b += T(x, 150, n, size=15.5, fill=cols[n], weight="bold") if False else T(x, 146, n, size=15.5, fill=cols[n], weight="bold")

    def msg(y, a, c, head, detail, col, below=True):
        x1, x2 = lx[a], lx[c]
        d = 1 if x2 > x1 else -1
        xm = (x1 + x2) / 2
        w1, w2 = tw(head, 14.5, True), tw(detail, 13.5)
        wmax = max(w1, w2) + 8
        s = R(xm - wmax / 2, y - 38, wmax, 34, rx=2, fill="#FFFFFF", stroke="none", sw=0)
        s += arrow([(x1 + 4 * d, y), (x2 - 4 * d, y)], color=col, sw=2.4, free="both")
        s += T(xm, y - 22, head, size=14.5, fill=col, weight="bold") + T(xm, y - 6, detail, size=13.5, fill=C_TXT)
        return s

    b += msg(206, "Eye", "Brain", "SCENE", f"pencil case {pc['eye_mass']:.2f} kg", TEAL)
    b += R(lx["Brain"] - 6, 226, 12, 62, rx=3, fill=PURPLE, stroke="none", sw=0)
    b += T(lx["Brain"] + 14, 246, "plans, then a dry run", size=13.5, anchor="start", fill=PURPLE, weight="bold") if False else ""
    b += T(lx["Brain"] + 16, 244, "dry run passes:", size=13.5, anchor="start", fill=PURPLE, weight="bold")
    b += T(lx["Brain"] + 16, 262, "plan follows the", size=13.5, anchor="start", fill=C_SUB) + T(lx["Brain"] + 16, 278, "Eye's reports", size=13.5, anchor="start", fill=C_SUB)
    b += msg(334, "Brain", "Arm", "TASKS", "pencil case, notebook", PURPLE)
    b += msg(396, "Arm", "Brain", "TOUCH", f"notebook {nbm['measured_weight']:.2f} kg", ORANGE)
    b += msg(458, "Arm", "Eye", "CHALLENGE", f"{nbm['measured_weight']:.2f} kg vs Eye's {nbm['reported']:.2f}", ORANGE)
    b += R(lx["Brain"] - 6, 476, 12, 34, rx=3, fill=PURPLE, stroke="none", sw=0) + T(lx["Brain"] + 16, 498, "replans", size=13.5, anchor="start", fill=PURPLE, weight="bold")
    b += f'<circle cx="{lx["Eye"]}" cy="536" r="12" fill="{TEAL}"/>' + T(lx["Eye"], 542, "!", size=17, fill="#FFFFFF", weight="bold")
    b += T(lx["Eye"] + 20, 541, "failure charged to the Eye", size=14.5, anchor="start", fill=TEAL, weight="bold")
    b += T(370, 566, "A dry run checks the plan against the reports, so only touch exposes the Eye's error.", size=13.5, anchor="start", fill=C_SUB)
    # ================================================================ right: open mini-graphics
    b += T(722, 30, "3", size=18, anchor="start", weight="bold", fill=GREEN) + T(744, 30, "What the split gives", size=16.5, anchor="start", weight="bold", fill=GREEN)
    # repair
    b += T(722, 62, "Repair", size=15.5, anchor="start", weight="bold", fill=GREEN) + T(790, 62, "challenge or probe fixes the plan", size=13.5, anchor="start", fill=C_SUB)
    for k, (lab, v, col) in enumerate((("no fix", m["no_fix"]["score"], RED), ("CHALLENGE", m["challenge"]["score"], GREEN), ("probe", m["probe"]["score"], GREEN))):
        yy = 76 + k * 30
        b += T(722, yy + 15, lab, size=13.5, anchor="start", fill=C_SUB)
        b += R(812, yy + 2, 120 * v, 18, rx=4, fill=tint(col, 0.35), stroke=col, sw=1.2) + T(812 + 120 * v + 6, yy + 16, f"{v:.2f}", size=14.5, anchor="start", weight="bold", fill=col)
    b += L(722, 176, 980, 176, color=C_MUT, sw=1.0)
    # credit
    b += T(722, 204, "Credit", size=15.5, anchor="start", weight="bold", fill=ORANGE) + T(782, 204, "what a perfect Eye adds", size=13.5, anchor="start", fill=C_SUB)
    gains = cr["field_gain"]
    top = max(max(gains.values()), 1e-9)
    for k, (f_, lab) in enumerate((("mass_guess", "mass"), ("cls", "class"), ("fragility_guess", "fragility"), ("region", "region"))):
        x = 730 + k * 62
        h = 40 * gains[f_] / top
        b += R(x, 282 - max(h, 2), 40, max(h, 2), rx=3, fill=ORANGE if h > 1 else GREY, stroke="none", sw=0)
        b += T(x + 20, 300, lab, size=13, fill=C_SUB) + T(x + 20, 282 - max(h, 2) - 6, f"{gains[f_]:+.2f}" if gains[f_] else "0", size=13.5, weight="bold", fill=ORANGE if h > 1 else C_SUB)
    b += L(722, 316, 980, 316, color=C_MUT, sw=1.0)
    # training data
    b += T(722, 340, "Training data", size=15.5, anchor="start", weight="bold", fill=PURPLE) + T(836, 340, "failures made useful", size=13.5, anchor="start", fill=C_SUB)
    b += I.mark(744, 384, False, 14) + T(744, 418, "failed", size=13, fill=C_SUB)
    b += arrow([(764, 384), (808, 384)], color=GREY_D, sw=2.0, free="vector") + T(786, 372, "relabel", size=13, fill=C_SUB)
    b += I.plan(836, 384, 40, PURPLE) + T(836, 418, f"{hd['n_tasks']} tasks", size=13, fill=C_SUB)
    b += arrow([(862, 384), (906, 384)], color=GREY_D, sw=2.0, free="vector") + T(884, 372, "replay", size=13, fill=C_SUB)
    b += I.mark(932, 384, True, 14) + T(932, 418, "kept", size=13, fill=C_SUB)
    b += L(722, 436, 980, 436, color=C_MUT, sw=1.0)
    # one checker
    b += T(722, 464, "One checker", size=15.5, anchor="start", weight="bold", fill=NAVY) + T(826, 464, "three worlds, one score", size=13.5, anchor="start", fill=C_SUB)
    gx = [750, 826, 902]
    for k in range(3):
        for i in range(3):
            for j in range(3):
                b += f'<circle cx="{gx[0] - 10 + i * 10}" cy="{498 - 10 + j * 10}" r="2.8" fill="{GREEN}"/>' if k == 0 else ""
    b += L(gx[1] - 14, 508, gx[1] + 14, 508, color=BLUE, sw=2.4) + L(gx[1] - 14, 508, gx[1] - 14, 482, color=BLUE, sw=2.4) + f'<circle cx="{gx[1] + 2}" cy="494" r="4" fill="{BLUE}"/>'
    b += R(gx[2] - 11, 486, 22, 22, rx=2, fill="none", stroke=ORANGE, sw=2.0) + R(gx[2] - 4, 479, 22, 22, rx=2, fill="none", stroke=ORANGE, sw=2.0)
    for x_, lab in zip(gx, ("symbolic", "pose-level", "Isaac Lab")):
        b += T(x_, 530, lab, size=13, fill=C_SUB)
    b += I.mark(954, 498, True, 12) if False else ""
    # ================================================================ footer: what is new (open columns)
    yb = 592
    b += L(20, yb, 980, yb, color=C_MUT, sw=1.4)
    b += T(20, yb + 30, "What is new", size=17, anchor="start", weight="bold", fill=NAVY)
    news = [("Roles split by interface", "Only Eyes see pixels, only the Brain plans, Arms are blind, so every failure has an owner.", TEAL),
            ("Evidence as typed messages", "13 budgeted message types: the dry run checks the plan, challenge and probe check the Eyes.", PURPLE),
            ("Signals from the team itself", "Counterfactual Eye credit under common random numbers; hindsight relabeling with a replay check.", ORANGE),
            ("One checker, three worlds", "Symbolic, pose-level and Isaac Lab share one success function; plus an audit of a benchmark scorer.", NAVY)]
    colw = 240
    for k, (t, d, col) in enumerate(news):
        x = 20 + k * colw
        if k:
            b += L(x - 10, yb + 46, x - 10, yb + 196, color=C_MUT, sw=1.0)
        b += T(x, yb + 92, str(k + 1), size=44, anchor="start", weight="bold", fill=tint(col, 0.35))
        y_ = yb + 62
        for ln in wrap(t, colw - 70, 15.5, True):
            b += T(x + 36, y_, ln, size=15.5, anchor="start", weight="bold", fill=col)
            y_ += 19
        y_ = yb + 124
        for ln in wrap(d, colw - 24, 14):
            b += T(x, y_, ln, size=14, anchor="start", fill=C_TXT)
            y_ += 18
    yend = yb + 224
    b += T(20, yend, "Icons are schematic pictograms, not rendered images. Numbers come from one recorded symbolic fixture episode (seed 24, Eye mass noise sigma 0.45).", size=13.5, anchor="start", fill=C_SUB)
    b += T(20, yend + 18, "No trained model and no physics result is shown or claimed in this paper yet.", size=13.5, anchor="start", fill=C_SUB)
    save_svg("fig_mechanism", W_, yend + 34, b)


GOLD = "#A87900"


def _stack(I, cx, cy, s, ok):
    """Two items in the bag column: ok = heavy notebook below, light pencil case on top; else reversed (rule broken)."""
    nb, pcs = s * 0.62, s * 0.78
    if ok:
        return I.notebook(cx, cy + s * 0.30, nb, GREY_D) + I.pencilcase(cx, cy - s * 0.30, pcs, GREY_D)
    return I.pencilcase(cx, cy + s * 0.34, pcs, RED) + I.notebook(cx, cy - s * 0.24, nb, RED)


def _node(I, x, y, r, kind, oracle=False, name=True):
    col = {"Eye": TEAL, "Brain": PURPLE, "Arm": ORANGE}[kind]
    fn = {"Eye": I.eye, "Brain": I.plan, "Arm": I.arm}[kind]
    o = ""
    if oracle:
        o += f'<circle cx="{x}" cy="{y}" r="{r + 7}" fill="none" stroke="{GOLD}" stroke-width="2.6" stroke-dasharray="5 4"/>'
    o += f'<circle cx="{x}" cy="{y}" r="{r}" fill="{tint(col, 0.85)}" stroke="{col}" stroke-width="2"/>' + fn(x, y, r * 1.25, col)
    return o


def fig_mechanism():
    """Figure 1 (teaser), v4: problem -> measurable responsibility -> responsibility-routed learning. Schematic, no footer."""
    import icons as I
    W_, H_ = 1000, 546
    b = ""
    # ------------------------------------------------------------------ titles
    heads = [(20, "1", RED, "A team fails. Who was responsible?", "Typed messages make every claim checkable"),
             (352, "2", GOLD, "Replay it, one agent at a time", "Oracle swapped in; same random draws"),
             (702, "3", GREEN, "Turn responsibility into learning", "Only the responsible agent is updated")]
    for x, n, col, t, sub in heads:
        b += f'<circle cx="{x + 11}" cy="24" r="12" fill="{col}"/>' + T(x + 11, 30, n, size=15, fill="#FFFFFF", weight="bold")
        b += T(x + 30, 30, t, size=16.5, anchor="start", weight="bold", fill=col) + T(x + 30, 50, sub, size=13.5, anchor="start", fill=C_SUB)
    # ------------------------------------------------------------------ stage 1: the failure
    b += R(20, 70, 316, 70, rx=12, fill="#F6F7F9", stroke=C_MUT, sw=1.3)
    for k, (lab, fn) in enumerate((("laptop", I.laptop), ("book", I.book), ("bottle", I.bottle), ("notebook", I.notebook), ("pencil case", I.pencilcase))):
        cx = 52 + k * 62
        b += fn(cx, 96, 36, GREY_D) + T(cx, 132, lab, size=13, fill=C_SUB)
    xs = {"Eye": 62, "Brain": 178, "Arm": 294}
    ny = 216
    b += arrow([(62, 144), (62, 178)], color=GREY_D, sw=2.0, free="vector")
    for k, (nm, x) in enumerate(xs.items()):
        b += _node(I, x, ny, 32, nm)
        b += T(x, ny + 52, nm, size=15, fill={"Eye": TEAL, "Brain": PURPLE, "Arm": ORANGE}[nm], weight="bold")
        b += f'<circle cx="{x + 26}" cy="{ny - 26}" r="12" fill="{RED}"/>' + T(x + 26, ny - 20, "?", size=17, fill="#FFFFFF", weight="bold")
    for (a, c, lab) in (("Eye", "Brain", "SCENE"), ("Brain", "Arm", "PLAN")):
        x1, x2 = xs[a] + 36, xs[c] - 36
        b += arrow([(x1, ny), (x2, ny)], color=GREY_D, sw=2.0, free="both") + T((x1 + x2) / 2, ny - 10, lab, size=13, fill=C_SUB, weight="bold")
    b += _stack(I, 84, 354, 92, False) + I.mark(126, 306, False, 13)
    b += T(150, 330, "A heavier item ends", size=14.5, anchor="start") + T(150, 349, "up on a lighter one:", size=14.5, anchor="start") + T(150, 368, "a rule is broken.", size=14.5, anchor="start", fill=RED, weight="bold")
    b += tag(36, 410, "one team reward: low", RED)
    b += T(36, 454, "Which agent should learn", size=15, anchor="start", weight="bold") + T(36, 474, "from this failure?", size=15, anchor="start", weight="bold")
    # ------------------------------------------------------------------ stage 2: replays
    rows = [("Eye", 150, True), ("Brain", 262, False), ("Arm", 374, False)]
    b += T(452, 90, "swap in an oracle", size=13.5, fill=C_SUB, weight="bold") + T(604, 90, "fixed?", size=13.5, fill=C_SUB, weight="bold") + T(664, 90, "gain", size=13.5, fill=C_SUB, weight="bold")
    b += L(360, 100, 680, 100, color=C_MUT, sw=1.0)
    nx = {"Eye": 392, "Brain": 452, "Arm": 512}
    for kind, y, fixed in rows:
        b += L(nx["Eye"] + 21, y, nx["Brain"] - 21, y, color=GREY, sw=2.0) + L(nx["Brain"] + 21, y, nx["Arm"] - 21, y, color=GREY, sw=2.0)
        for nm, x in nx.items():
            b += _node(I, x, y, 21, nm, oracle=(nm == kind))
        b += T(nx[kind], y - 38, "oracle", size=13, fill=GOLD, weight="bold")
        b += T(nx[kind], y + 44, kind, size=13.5, fill={"Eye": TEAL, "Brain": PURPLE, "Arm": ORANGE}[kind], weight="bold")
        b += arrow([(548, y), (576, y)], color=GREY, sw=1.8, free="both")
        b += _stack(I, 604, y + 2, 54, fixed) + I.mark(630, y - 30, fixed, 9)
        b += R(646, y - 9, 32 if fixed else 3, 18, rx=3, fill=GREEN if fixed else GREY, stroke="none", sw=0)
    b += L(644, 124, 644, 396, color=C_MUT, sw=1.0)
    b += T(350, 452, "responsibility of agent i:", size=14.5, anchor="start", weight="bold", fill=GOLD)
    b += T(350, 472, "R(replay with oracle i) - R(real)", size=14.5, anchor="start", fill=C_TXT)
    # ------------------------------------------------------------------ stage 3: routed learning
    ups = [(150, "Eye", TEAL, "Learns: keeps the reports that helped the team.", False),
           (262, "Brain", PURPLE, "Gets the failure back as a valid example: relabeled and replay-checked.", False),
           (374, "Arm", ORANGE, "No update: a perfect Arm would not have prevented this failure.", True)]
    for y, nm, col, txt, muted in ups:
        c = GREY if muted else col
        b += arrow([(686, y), (716, y)], color=c, sw=2.6 if not muted else 1.6, dash="5 4" if muted else None, free="both")
        b += _node(I, 746, y, 21, nm)
        for k, ln in enumerate(wrap(txt, 190, 14.5)):
            b += T(780, y - 12 + k * 19, ln, size=14.5, anchor="start", fill=C_SUB if muted else C_TXT, weight="bold" if (nm == "Eye") else None)
    # ------------------------------------------------------------------ loop back
    b += arrow([(840, 420), (840, 506), (170, 506), (170, 484)], color=GREEN, sw=2.6, free="both")
    b += T(505, 528, "a better team meets the next episode", size=14.5, fill=GREEN, weight="bold")
    save_svg("fig_mechanism", W_, H_, b)


def fig_overview():
    """Figure 2: the team architecture. Typed message bus as the spine; plan path, perception loop, and the responsibility chain."""
    import icons as I
    W_, H_ = 1000, 640
    b = ""

    def circ(x, y, r, col, fn, s=None, fill=None):
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill or tint(col, 0.85)}" stroke="{col}" stroke-width="2.2"/>' + fn(x, y, s or r * 1.25, col)

    # ---- the bus band (drawn first, everything attaches to it)
    b += R(250, 300, 700, 40, rx=20, fill=tint(NAVY, 0.88), stroke=NAVY, sw=2.2)
    b += T(268, 326, "typed message bus: 13 types, budgeted", size=14.5, anchor="start", fill=NAVY, weight="bold")
    # ---- perception loop (left)
    b += I.world(100, 300, 150, 74, GREY_D) + T(100, 356, "World", size=15, fill=GREY_D, weight="bold")
    b += T(100, 84, "Eye A, Eye B", size=15, fill=TEAL, weight="bold") + circ(100, 130, 34, TEAL, I.eye)
    b += arrow([(100, 262), (100, 172)], color=GREY_D, sw=2.2, free="both") + T(112, 222, "images", size=14, anchor="start", fill=GREY_D)
    b += arrow([(138, 118), (292, 118), (292, 298)], color=TEAL, sw=2.4, free="both") + T(215, 108, "SCENE", size=14, fill=TEAL, weight="bold")
    b += arrow([(318, 298), (318, 150), (140, 150)], color=PURPLE, sw=2.4, free="both") + T(230, 170, "LOOK", size=14, fill=PURPLE, weight="bold")
    # ---- decision core (top)
    b += circ(500, 52, 24, GREY_D, I.person, 40, "#F1F3F5") + T(430, 58, "Human", size=15, anchor="end", fill=GREY_D, weight="bold")
    b += arrow([(500, 80), (500, 118)], color=GREY_D, sw=2.2, free="both") + T(512, 104, "instruction", size=14, anchor="start", fill=GREY_D)
    b += arrow([(526, 52), (860, 52), (860, 298)], color=GREY_D, sw=2.2, free="both") + T(700, 42, "AMEND", size=14, fill=GREY_D, weight="bold")
    b += I.docs(380, 160, 56, BLUE) + T(380, 206, "Harness", size=15, fill=BLUE, weight="bold") + T(380, 224, "rules, skills", size=13.5, fill=C_SUB)
    b += arrow([(412, 160), (456, 160)], color=BLUE, sw=2.2, free="both")
    b += circ(500, 160, 40, PURPLE, I.plan) + T(552, 188, "Brain", size=15, anchor="start", fill=PURPLE, weight="bold")
    b += arrow([(542, 160), (620, 160)], color=PURPLE, sw=2.4, free="both") + T(581, 148, "plan", size=14, fill=PURPLE, weight="bold")
    b += I.gate(650, 160, 64, GREEN) + T(650, 120, "Verifier: dry run", size=14, fill=GREEN, weight="bold")
    b += arrow([(650, 196), (650, 288)], color=GREEN, sw=2.4, free="both") + T(662, 246, "verified plan", size=14, anchor="start", fill=GREEN)
    b += arrow([(476, 298), (476, 196)], color=NAVY, sw=2.2, free="both") + T(466, 270, "SCENE, TOUCH,", size=14, anchor="end", fill=NAVY) + T(466, 288, "CHALLENGE", size=14, anchor="end", fill=NAVY)
    b += arrow([(524, 198), (524, 298)], color=NAVY, sw=2.2, free="both") + T(534, 252, "LOOK", size=14, anchor="start", fill=NAVY)
    # ---- scheduler on the bus
    b += circ(650, 320, 30, GREEN, I.ladder, 40, "#FFFFFF") + T(692, 326, "Scheduler, ladder", size=14.5, anchor="start", fill=GREEN, weight="bold")
    # ---- arms (below)
    for x, nm in ((380, "Arm L"), (520, "Arm R")):
        b += circ(x, 450, 34, ORANGE, I.arm) + T(x, 506, nm, size=15, fill=ORANGE, weight="bold")
        b += arrow([(x - 10, 342), (x - 10, 414)], color=NAVY, sw=2.2, free="both") + arrow([(x + 14, 414), (x + 14, 342)], color=ORANGE, sw=2.2, free="both")
    b += T(360, 384, "CLAIM, LOCK", size=14, anchor="end", fill=NAVY)
    b += T(550, 372, "TOUCH,", size=14, anchor="start", fill=ORANGE) + T(550, 390, "CHALLENGE,", size=14, anchor="start", fill=ORANGE) + T(550, 408, "STATUS", size=14, anchor="start", fill=ORANGE)
    b += arrow([(338, 450), (130, 450), (130, 338)], color=ORANGE, sw=2.4, free="both") + T(240, 440, "skills", size=14, fill=ORANGE, weight="bold")
    # ---- responsibility chain (bottom), gold
    b += arrow([(70, 338), (70, 512)], color=GREY_D, sw=2.2, free="both")
    b += I.mark(70, 545, False, 15) + T(70, 584, "rules.check_final", size=14, fill=GREY_D, weight="bold")
    b += arrow([(100, 545), (326, 545)], color=GOLD, sw=2.6, free="both") + T(213, 533, "replay, oracle i swapped in", size=14, fill=GOLD, weight="bold")
    b += f'<circle cx="356" cy="545" r="28" fill="{tint(GOLD, 0.85)}" stroke="{GOLD}" stroke-width="2.2" stroke-dasharray="5 4"/>' + f'<polygon points="356,532 372,558 340,558" fill="none" stroke="{GOLD}" stroke-width="3"/>' 
    b += T(356, 592, "responsibility", size=14, fill=GOLD, weight="bold")
    b += arrow([(388, 545), (606, 545)], color=GOLD, sw=2.6, free="both") + T(497, 533, "routes learning", size=14, fill=GOLD, weight="bold")
    b += circ(650, 545, 24, TEAL, I.eye) + circ(704, 545, 24, PURPLE, I.plan)
    b += T(677, 592, "Eye and Brain learn", size=14, fill=C_TXT, weight="bold")
    # ---- legend
    for k, (lab, col) in enumerate((("perception", TEAL), ("planning", PURPLE), ("action", ORANGE), ("checking", GREEN), ("responsibility", GOLD))):
        y = 392 + k * 24
        b += R(790, y, 16, 12, rx=3, fill=tint(col, 0.5), stroke=col, sw=1.4) + T(814, y + 11, lab, size=14, anchor="start", fill=C_TXT)
    save_svg("fig_overview", W_, H_, b)


def fig_overview():
    """Figure 2 (v5): layered system figure. Knowledge -> agents -> typed interfaces -> bus -> worlds and checker -> responsibility and learning."""
    import icons as I
    W_, H_ = 1000, 944
    b = ""

    def circ(x, y, r, col, fn, s=None, fill=None):
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill or tint(col, 0.85)}" stroke="{col}" stroke-width="2.2"/>' + fn(x, y, s or r * 1.25, col)

    bands = [(8, 170, GREY, 0.95, "KNOWLEDGE AND TASK", 36, GREY_D), (178, 334, GREY, 1.0, "AGENTS", 36, GREY_D),
             (342, 500, GREY, 0.95, "TYPED INTERFACES", 36, GREY_D), (508, 668, NAVY, 0.94, "COORDINATION", 36, NAVY),
             (676, 772, GREY, 0.95, "WORLD", 36, GREY_D), (780, 936, GOLD, 0.93, "RESPONSIBILITY AND LEARNING", 112, GOLD)]
    for y0, y1, col, a_, lab, lx, lc in bands:
        if a_ < 1.0:
            b += R(24, y0, 944, y1 - y0, rx=12, fill=tint(col, a_), stroke="none", sw=0)
        b += T(lx, y0 + 20, lab, size=13.5, anchor="start", weight="bold", fill=lc)
    # ================= A: knowledge and task
    b += I.docs(490, 100, 58, BLUE) + T(452, 68, "Harness", size=15.5, anchor="end", weight="bold", fill=BLUE)
    for k, ln in enumerate(("19 markdown files: 6 rules,", "4 skills, object, agent and", "memory notes. Retrieval by", "BM25: pass 1 rules (k = 5),", "pass 2 notes (k = 3).")):
        b += T(452, 86 + k * 16, ln, size=13.5, anchor="end", fill=C_TXT)
    b += circ(565, 100, 26, GREY_D, I.person, 40, "#F1F3F5") + T(606, 68, "Human instruction", size=15.5, anchor="start", weight="bold", fill=GREY_D)
    ins = FD["mechanism"]["instruction"].split(": ", 1)[1]
    for k, ln in enumerate(wrap("Pack the bag: " + ins, 350, 13.5)[:5]):
        b += T(606, 86 + k * 16, ln, size=13.5, anchor="start", fill=C_TXT)
    # ================= B: agents
    b += T(116, 212, "Eye A, Eye B (VLMs, pixels only)", size=14.5, anchor="start", weight="bold", fill=TEAL) + circ(150, 258, 34, TEAL, I.eye)
    b += circ(530, 262, 44, PURPLE, I.plan) + T(584, 260, "Brain", size=15.5, anchor="start", weight="bold", fill=PURPLE) + T(584, 278, "(text-only LLM)", size=13.5, anchor="start", fill=PURPLE)
    b += T(835, 212, "Arm L, Arm R (blind executors)", size=14.5, weight="bold", fill=ORANGE)
    for x in (790, 880):
        b += circ(x, 258, 30, ORANGE, I.arm)
    b += L(790, 288, 790, 300, color=ORANGE, sw=2) + L(880, 288, 880, 300, color=ORANGE, sw=2) + L(790, 300, 880, 300, color=ORANGE, sw=2)
    b += arrow([(490, 130), (490, 238)], color=BLUE, sw=2.2, free="both")
    b += arrow([(565, 128), (565, 230)], color=GREY_D, sw=2.2, free="both") + T(575, 200, "instruction, AMEND", size=13.5, anchor="start", fill=GREY_D)
    b += arrow([(484, 244), (186, 244)], color=PURPLE, sw=2.2, free="both") + T(335, 234, "LOOK (via the bus)", size=13.5, fill=PURPLE, weight="bold")
    # ================= B2: typed interfaces
    sc = FD["messages"]["examples"]["SCENE"]["content"]["objects"][0]
    b += L(70, 372, 70, 492, color=TEAL, sw=3) + T(80, 384, "SCENE message (recorded)", size=13.5, anchor="start", weight="bold", fill=TEAL)
    js = ['{"name": "%s", "cls": "%s",' % (sc["name"], sc["cls"]), ' "region": "%s",' % sc["region"], ' "mass_guess": %s,' % sc["mass_guess"],
          ' "fragility_guess": %s,' % sc["fragility_guess"], ' "occluded": %s,' % str(sc["occluded"]).lower(), ' "confidence": %s}' % sc["confidence"]]
    for k, ln in enumerate(js):
        b += T(80, 404 + k * 16, ln, size=13, anchor="start", font=MONO, fill=C_TXT)
    b += L(358, 372, 358, 492, color=PURPLE, sw=3) + T(368, 384, "plan: a DAG of grasp (g) and place (p) tasks", size=13.5, anchor="start", weight="bold", fill=PURPLE)
    nodes = {"ga": (388, 428, "g"), "pa": (436, 428, "p"), "pb": (484, 428, "p"), "pc": (532, 428, "p"), "gb": (484, 474, "g"), "gc": (532, 474, "g")}
    for x, y, lab in nodes.values():
        col = PURPLE if lab == "p" else ORANGE
        b += f'<circle cx="{x}" cy="{y}" r="11" fill="{tint(col, 0.8)}" stroke="{col}" stroke-width="1.8"/>' + T(x, y + 5, lab, size=13.5, fill=col, weight="bold")
    for a_, c_ in (("ga", "pa"), ("pa", "pb"), ("pb", "pc")):
        b += arrow([(nodes[a_][0] + 13, nodes[a_][1]), (nodes[c_][0] - 13, nodes[c_][1])], color=GREY_D, sw=1.6, free="both")
    for k in ("gb", "gc"):
        b += arrow([(nodes[k][0], nodes[k][1] - 13), (nodes[k][0], 442)], color=GREY_D, sw=1.6, free="both")
    b += arrow([(546, 428), (566, 428)], color=GREEN, sw=1.6, free="both") if False else ""
    b += I.gate(604, 428, 46, GREEN) + T(604, 468, "verifier: none,", size=13.5, fill=GREEN, weight="bold") + T(604, 484, "static, or dry run", size=13.5, fill=C_SUB)
    b += circ(690, 428, 24, GREEN, I.ladder, 34, "#FFFFFF") + T(722, 424, "scheduler,", size=13.5, anchor="start", weight="bold", fill=GREEN) + T(722, 441, "bag lock", size=13.5, anchor="start", fill=C_SUB)
    b += L(872, 372, 872, 492, color=ORANGE, sw=3)
    for k, (ln, bold) in enumerate((("skills", True), ("grasp, place,", False), ("handover,", False), ("push_aside", False), ("senses", True), ("force, pose", False))):
        b += T(882, 386 + k * 17 + (6 if k >= 4 else 0), ln, size=13.5, anchor="start", weight="bold" if bold else None, fill=ORANGE if bold else C_TXT)
    # ================= C: the bus band, two rows
    by0 = 538
    b += R(40, by0, 920, 64, rx=22, fill=tint(NAVY, 0.84), stroke=NAVY, sw=2.4)
    b += T(64, by0 + 25, "typed message bus: 13 types, schema-checked, token-budgeted", size=14.5, anchor="start", weight="bold", fill=NAVY)
    groups = [("perceive", TEAL, ["SCENE", "LOOK"]), ("coordinate", NAVY, ["ABILITY", "CLAIM", "OFFER", "DECLINE", "HANDOVER", "LOCK"]),
              ("feedback", ORANGE, ["STATUS", "BLOCKED", "TOUCH", "CHALLENGE"]), ("human", GREY_D, ["AMEND"])]
    xl = 600
    for g, col, _ in groups:
        b += T(xl, by0 + 25, g, size=13.5, anchor="start", weight="bold", fill=col)
        xl += tw(g, 13.5, True) + 16
    x = 64
    for g, col, ms in groups:
        for mname in ms:
            b += T(x, by0 + 51, mname, size=13.5, anchor="start", weight="bold", fill=col)
            x += tw(mname, 13.5, True) + 15
    # connectors to the bus
    b += arrow([(240, 492), (240, 536)], color=TEAL, sw=2.4, free="both") + T(250, 520, "SCENE", size=13.5, anchor="start", fill=TEAL, weight="bold")
    b += arrow([(346, 536), (346, 272), (484, 272)], color=NAVY, sw=2.2, free="both") + T(415, 262, "SCENE, TOUCH,", size=13, fill=NAVY) + T(415, 292, "CHALLENGE", size=13, fill=NAVY)
    b += arrow([(690, 454), (690, 536)], color=GREEN, sw=2.4, free="both") + T(700, 510, "verified plan", size=13.5, anchor="start", fill=GREEN, weight="bold")
    b += arrow([(806, 302), (806, 536)], color=ORANGE, sw=2.2, free="both") + arrow([(826, 536), (826, 302)], color=NAVY, sw=2.2, free="both")
    b += T(796, 322, "CLAIM, LOCK,", size=13, anchor="end", fill=ORANGE) + T(796, 338, "STATUS, TOUCH,", size=13, anchor="end", fill=ORANGE) + T(796, 354, "CHALLENGE", size=13, anchor="end", fill=ORANGE)
    b += T(836, 470, "tasks", size=13, anchor="start", fill=NAVY, weight="bold")
    # notes below the band
    b += T(40, 632, "modes: push, delta, pull, init-only", size=14.5, anchor="start", weight="bold", fill=NAVY)
    b += T(40, 650, "per-episode token budget; over-budget messages are refused", size=13.5, anchor="start", fill=C_TXT)
    b += I.ladder(560, 642, 34, GREEN) + T(584, 632, "recovery ladder on a failed skill", size=14.5, anchor="start", weight="bold", fill=GREEN)
    b += T(584, 650, "retry, alternate parameters, retreat, reassign, replan, abort", size=13.5, anchor="start", fill=C_TXT)
    # ================= D: worlds
    dy = 8
    wx = [60, 360, 660]
    b += "".join(f'<circle cx="{wx[0] + 8 + i * 12}" cy="{724 + dy - 12 + j * 12}" r="3.4" fill="{GREEN}"/>' for i in range(3) for j in range(3))
    b += T(112, 720 + dy, "Symbolic world", size=14.5, anchor="start", weight="bold", fill=GREEN) + T(112, 738 + dy, "fast; the loop is developed here", size=13.5, anchor="start", fill=C_TXT)
    b += L(wx[1] + 4, 742 + dy, wx[1] + 36, 742 + dy, color=BLUE, sw=2.6) + L(wx[1] + 4, 742 + dy, wx[1] + 4, 710 + dy, color=BLUE, sw=2.6) + f'<circle cx="{wx[1] + 22}" cy="{724 + dy}" r="5" fill="{BLUE}"/>'
    b += T(wx[1] + 52, 720 + dy, "Pose-level simulator", size=14.5, anchor="start", weight="bold", fill=BLUE) + T(wx[1] + 52, 738 + dy, "same loop outcomes, tested", size=13.5, anchor="start", fill=C_TXT)
    b += R(wx[2] + 2, 716 + dy, 24, 24, rx=2, fill="none", stroke=ORANGE, sw=2.2) + R(wx[2] + 12, 708 + dy, 24, 24, rx=2, fill="none", stroke=ORANGE, sw=2.2)
    b += T(wx[2] + 52, 720 + dy, "Isaac Lab", size=14.5, anchor="start", weight="bold", fill=ORANGE) + T(wx[2] + 52, 738 + dy, "physics; outcomes from state change", size=13.5, anchor="start", fill=C_TXT)
    b += arrow([(24, 730), (12, 730), (12, 258), (112, 258)], color=GREY_D, sw=2.2, free="both")
    b += arrow([(914, 258), (990, 258), (990, 724), (966, 724)], color=ORANGE, sw=2.2, free="both") + T(952, 246, "skills", size=13.5, fill=ORANGE, weight="bold")
    # ================= E: responsibility and learning
    yc = 866
    b += arrow([(84, 772), (84, 838)], color=GREY_D, sw=2.2, free="both")
    b += I.mark(84, yc, False, 15) + T(84, 904, "rules.check_final", size=13.5, fill=GREY_D, weight="bold") + T(84, 921, "success or failure", size=13.5, fill=C_SUB)
    b += arrow([(108, yc), (190, yc)], color=GOLD, sw=2.4, free="both") + T(150, yc - 12, "on failure", size=13.5, fill=GOLD, weight="bold")
    for k, (nm, x_) in enumerate((("Eye", 226), ("Brain", 286), ("Arm", 346))):
        fn = {"Eye": I.eye, "Brain": I.plan, "Arm": I.arm}[nm]
        col = {"Eye": TEAL, "Brain": PURPLE, "Arm": ORANGE}[nm]
        b += (f'<circle cx="{x_}" cy="{yc}" r="26" fill="none" stroke="{GOLD}" stroke-width="2.4" stroke-dasharray="5 4"/>' if k == 0 else "") + circ(x_, yc, 19, col, fn, 25)
    b += T(286, yc + 44, "replay with an oracle for one agent", size=13.5, fill=GOLD, weight="bold") + T(286, yc + 61, "at a time; same random draws", size=13.5, fill=C_SUB)
    b += arrow([(380, yc), (448, yc)], color=GOLD, sw=2.4, free="both")
    b += f'<circle cx="480" cy="{yc}" r="28" fill="{tint(GOLD, 0.85)}" stroke="{GOLD}" stroke-width="2.2" stroke-dasharray="5 4"/>' + f'<polygon points="480,{yc - 15} 496,{yc + 11} 464,{yc + 11}" fill="none" stroke="{GOLD}" stroke-width="3"/>'
    b += T(480, yc + 52, "responsibility", size=13.5, fill=GOLD, weight="bold")
    b += arrow([(512, yc), (566, yc)], color=GOLD, sw=2.4, free="both") + L(572, yc - 46, 572, yc + 46, color=GOLD, sw=2.4)
    routes = [(yc - 46, TEAL, I.eye, "Eye: LoRA fine-tuning, then rejection sampling (K = 8) guided by credit"),
              (yc, PURPLE, I.plan, "Brain: SFT, then GRPO on the plan reward; hindsight relabeling"),
              (yc + 46, ORANGE, I.arm, "Arms: flow-matching distillation of scripted demonstrations")]
    for y, col, fn, txt in routes:
        b += arrow([(572, y), (606, y)], color=GOLD, sw=2.2, free="both") + circ(632, y, 17, col, fn, 22)
        ls = wrap(txt, 290, 13.5)
        for k, ln in enumerate(ls):
            b += T(660, y + 5 - (8 if len(ls) > 1 else 0) + k * 16, ln, size=13.5, anchor="start", fill=C_TXT)
    save_svg("fig_overview", W_, H_, b)
