"""Figures 1-6. All numbers and traces are read from generated/figdata.json and generated/fixture.json."""
from __future__ import annotations
import json
from diagram_common import *  # noqa: F401,F403
from diagram_common import PAPER_DIR

FD = json.loads((PAPER_DIR / "generated" / "figdata.json").read_text())
FX = json.loads((PAPER_DIR / "generated" / "fixture.json").read_text())
ROLE = {"Eye": TEAL, "Brain": PURPLE, "Arm": ORANGE}


def pct(rep, true):
    return (rep - true) / true * 100


# ------------------------------------------------------------------------------------------------ 1 mechanism
def fig_mechanism():
    m = FD["mechanism"]
    W_, H_ = 1000, 520
    b = ""
    items = sorted([i for i in m["items"] if not i["distractor"]], key=lambda i: -i["true_mass"])
    # --- a: Eye report vs truth
    b += panel(20, 28, "a", "What the Eye reported (kg)")
    x0, sc, y0, rh = 118, 100.0, 62, 36
    for k, it in enumerate(items):
        y = y0 + k * rh
        err = pct(it["eye_mass"], it["true_mass"])
        bad = abs(err) > 15 and it["name"] in ("pencilcase1", "notebook1")
        b += T(x0 - 8, y + 17, it["name"], size=14.5, anchor="end", fill=RED if bad else C_TXT, weight="bold" if bad else None)
        b += R(x0, y + 3, it["true_mass"] * sc, 11, rx=2, fill="#C7CDD4", stroke="none", sw=0)
        b += R(x0, y + 16, it["eye_mass"] * sc, 11, rx=2, fill=RED if bad else TEAL, stroke="none", sw=0)
        if bad:
            b += T(x0 + max(it["true_mass"], it["eye_mass"]) * sc + 6, y + 26, f"{err:+.0f}%", size=14.5, anchor="start", fill=RED, weight="bold")
    ly = y0 + len(items) * rh + 6
    b += R(20, ly, 12, 11, rx=2, fill="#C7CDD4", stroke="none", sw=0) + T(38, ly + 10, "true mass", size=14.5, anchor="start")
    b += R(130, ly, 12, 11, rx=2, fill=TEAL, stroke="none", sw=0) + T(148, ly + 10, "Eye report", size=14.5, anchor="start")
    # --- b: the plan
    b += panel(375, 28, "b", "The plan built on it")
    order = [t["obj"] for t in json.loads(m["no_fix"]["plans"][0])["tasks"] if t["skill"] == "place"] if isinstance(m["no_fix"]["plans"][0], str) \
        else [t["obj"] for t in m["no_fix"]["plans"][0]["tasks"] if t["skill"] == "place"]
    order = [o for o in order if o != "laptop1"]
    tm = {i["name"]: i["true_mass"] for i in m["items"]}
    bx, bw, bh, by0 = 395, 170, 31, 62
    top = by0 + len(order) * bh
    b += R(bx - 8, by0 - 6, bw + 16, len(order) * bh + 12, rx=8, fill="#F6F7F9", stroke=C_MUT, sw=1.2)
    for k, o in enumerate(reversed(order)):          # last placed on top
        y = by0 + k * bh
        flag = o in ("notebook1", "pencilcase1")
        b += R(bx, y + 2, bw, bh - 4, rx=4, fill=tint(RED, 0.85) if flag else "#FFFFFF", stroke=RED if flag else GREY, sw=1.5 if flag else 1.0)
        b += T(bx + 8, y + 21, o, size=14.5, anchor="start", weight="bold" if flag else None, fill=RED if flag else C_TXT)
        b += T(bx + bw - 8, y + 21, f"{tm[o]:.2f} kg", size=14.5, anchor="end", fill=RED if flag else C_SUB)
    b += T(bx + bw / 2, top + 26, "bag, main compartment (top to bottom)", size=14.5, fill=C_SUB)
    viol = m["no_fix"]["violations"][0]
    s, _ = card(595, 62, 190, "Rule broken", f"{viol['rule']}: {viol['detail']}. True masses disagree with the planner's order.", color=RED, bsize=14.5)
    b += s
    s, _ = card(595, 215, 190, "Attributed to Eye", "The loop's attribution rules charge this failure to the Eye, because the mass it reported is far from the truth. A heuristic label, not yet validated against hand labels.",
                color=TEAL, bsize=14.5)
    b += s
    b += tag(595, 395, f"score {m['no_fix']['score']:.2f}  fail", RED)
    # --- c: fixes
    b += panel(800, 28, "c", "Two fixes, same seed")
    ch = m["challenge_msgs"]
    lines = "\n".join(f"{c['content']['obj']}: {c['content']['evidence']['measured_weight']:.2f} vs {c['content']['evidence']['reported']:.2f}" for c in ch[:3])
    s, bb = card(800, 62, 180, "CHALLENGE", f"{len(ch)} arm challenges (touched vs reported kg), e.g.\n{lines}\nThen one replan.", color=ORANGE, bsize=14.5)
    b += s
    y2 = bb[1] + bb[3] + 12
    s, bb2 = card(800, y2, 180, "Probe", "Arms weigh every item before planning; measured masses override the Eye guesses.", color=BLUE, bsize=14.5)
    b += s
    y3 = bb2[1] + bb2[3] + 14
    b += tag(800, y3, f"challenge {m['challenge']['score']:.2f} ok", GREEN) + tag(800, y3 + 32, f"probe {m['probe']['score']:.2f} ok", GREEN)
    b += T(800, y3 + 78, f"rounds: {m['no_fix']['rounds']} / {m['challenge']['rounds']} / {m['probe']['rounds']}", size=14.5, anchor="start", fill=C_SUB)
    b += T(800, y3 + 96, "(no fix / challenge / probe)", size=14.5, anchor="start", fill=C_SUB)
    b += T(20, 478, f"Recorded symbolic episode (seed {m['seed']}, Eye mass noise sigma {m['mass_sigma']}, scripted Brain): FIXTURE, a mechanism illustration, not a result.",
           size=14.5, anchor="start", fill=C_SUB)
    b += T(20, 498, "Masses in panels a and b are the scene the planner actually used (first plan call); challenge values come from the arms' own touch.",
           size=14.5, anchor="start", fill=C_SUB)
    save_svg("fig_mechanism", W_, H_, b)


# ------------------------------------------------------------------------------------------------ 2 overview
def fig_overview():
    W_, H_ = 1000, 700
    cx = [20, 270, 520, 770]
    cw = 210
    r0, r1, r2, r3 = 20, 150, 310, 470
    b = ""
    HAR = card(cx[1], r0, cw, "Harness", "19 markdown files: rules, skills, object and agent notes, memory. Two-pass BM25 retrieval.", h=96, color=BLUE)
    HUM = card(cx[0], r1, cw, "Human", "Gives the instruction; may AMEND it mid-episode (via the bus).", h=110, color=GREY_D, fill="#F1F3F5")
    BRN = card(cx[1], r1, cw, "Brain", "Text-only LLM, never sees pixels. Plans a DAG of tasks; tools: dry_run, check_rule, submit_plan.", h=110, color=PURPLE)
    VER = card(cx[2], r1, cw, "Verifier", "none, static or dry run before any arm moves; up to 2 retries with reasons fed back.", h=110, color=GREEN)
    SCH = card(cx[3], r1, cw, "Scheduler + ladder", "Head task per arm; bag lock. Retry, alternate params, retreat, reassign, replan, abort.", h=110, color=GREEN)
    BUS = card(cx[1], r2, cw, "Message bus", "13 typed messages, validated, token-budgeted (push, delta, pull, init-only).", h=110, color=NAVY)
    ARM = card(cx[3], r2, cw, "Arm L, Arm R", "Blind: proprioception, force, skill command. Report TOUCH; CHALLENGE the Eye.", h=110, color=ORANGE)
    EYE = card(cx[1], r3, cw, "Eye A, Eye B", "VLMs, the only agents with pixels. Emit SCENE JSON (class, region, mass, fragility).", h=110, color=TEAL)
    PRB = card(cx[2], r3, cw, "Probe (optional)", "Arms weigh every item before planning; measured mass overrides Eye guesses.", h=88, color=BLUE, dash="6 4")
    WRD = card(cx[3], r3, cw, "World", "Symbolic, pose-level or Isaac Lab; one checker, rules.check_final.", h=110, color=GREY_D, fill="#F1F3F5")
    for s_, _ in (HAR, HUM, BRN, VER, SCH, BUS, ARM, EYE, PRB, WRD):
        b += s_
    har, hum, brn, ver, sch, bus, arm, eye, prb, wrd = (c[1] for c in (HAR, HUM, BRN, VER, SCH, BUS, ARM, EYE, PRB, WRD))
    b += connect(har, "b", brn, "t", color=BLUE)
    b += connect(hum, "r", brn, "l", color=GREY_D)
    b += connect(brn, "r", ver, "l", color=PURPLE)
    b += connect(ver, "r", sch, "l", color=GREEN)
    b += connect(hum, "b", bus, "l", color=GREY_D)
    b += connect(bus, "t", brn, "b", color=NAVY)
    b += connect(sch, "b", arm, "t", color=GREEN)
    b += connect(arm, "l", bus, "r", color=ORANGE)
    b += connect(arm, "b", wrd, "t", color=ORANGE)
    b += connect(wrd, "l", eye, "r", oa=100, ob=100, color=GREY_D)
    b += connect(eye, "t", bus, "b", color=TEAL)
    b += T(625, r2 + 46, "TOUCH, CHALLENGE", size=14.5, fill=ORANGE, weight="bold")
    b += T(72, r2 + 30, "AMEND", size=14.5, fill=GREY_D, weight="bold")
    b += T(625, r3 + 124, "images", size=14.5, fill=GREY_D)
    b += pending_tag(cx[1], 600, "Eye LoRA: pending") + pending_tag(cx[1] + 160, 600, "Brain LoRA: pending") + pending_tag(cx[1] + 330, 600, "Arm policy: pending")
    b += T(cx[0], 640, "Legend", size=14, anchor="start", weight="bold", fill=C_TXT)
    b += R(cx[0] + 70, 628, 14, 12, rx=3, fill=tint(TEAL), stroke=TEAL) + T(cx[0] + 90, 639, "Eye", size=14.5, anchor="start")
    b += R(cx[0] + 140, 628, 14, 12, rx=3, fill=tint(PURPLE), stroke=PURPLE) + T(cx[0] + 160, 639, "Brain", size=14.5, anchor="start")
    b += R(cx[0] + 220, 628, 14, 12, rx=3, fill=tint(ORANGE), stroke=ORANGE) + T(cx[0] + 240, 639, "Arm", size=14.5, anchor="start")
    b += R(cx[0] + 290, 628, 14, 12, rx=3, fill="#FFFFFF", stroke=BLUE, dash="4 3") + T(cx[0] + 310, 639, "optional", size=14.5, anchor="start")
    b += R(cx[0] + 385, 628, 14, 12, rx=3, fill=tint(RED), stroke=RED) + T(cx[0] + 405, 639, "pending", size=14.5, anchor="start")
    b += T(cx[0], 676, "Roles are separated by interface: only Eyes receive pixels, only the Brain reads text and plans, Arms are blind executors.", size=14.5, anchor="start", fill=C_SUB)
    save_svg("fig_overview", W_, H_, b)


# ------------------------------------------------------------------------------------------------ 3 messages
def fig_messages():
    W_, H_ = 1000, 760
    sch = FD["messages"]["schema"]
    b = panel(20, 26, "a", "Thirteen typed messages (required fields)")
    sender = {"ABILITY": ORANGE, "SCENE": TEAL, "CLAIM": GREY, "OFFER": GREY, "DECLINE": GREY, "STATUS": GREY, "BLOCKED": GREY,
              "TOUCH": ORANGE, "LOOK": PURPLE, "CHALLENGE": ORANGE, "HANDOVER": GREY, "LOCK": GREY, "AMEND": GREY_D}
    names = list(sch.keys())
    cw, ch, gx = 188, 70, 12
    for k, n in enumerate(names):
        r, c = divmod(k, 5)
        x, y = 20 + c * (cw + gx), 42 + r * (ch + 10)
        col = sender[n]
        b += R(x, y, cw, ch, rx=8, fill=tint(col), stroke=col, sw=1.3)
        b += T(x + 10, y + 21, n, size=14, anchor="start", fill=col, weight="bold")
        for j, ln in enumerate(wrap(", ".join(sch[n]), cw - 20, 14.5)[:2]):
            b += T(x + 10, y + 40 + j * 16, ln, size=14.5, anchor="start", fill=C_TXT)
    b += T(20, 296, "Colour = sender in the recorded episodes: Eye (teal), Arm (orange), Brain (purple), human (dark grey). Light grey: not exchanged in them.", size=14.5, anchor="start", fill=C_SUB)
    b += '<g transform="translate(0,34)">'
    # b: real JSON
    ex = FD["messages"]["examples"]["CHALLENGE"]
    b += panel(20, 308, "b", "A real CHALLENGE (recorded, seed 24)")
    js = json.dumps({"type": ex["type"], "sender": ex["sender"], "receiver": ex["receiver"], "content": ex["content"]}, separators=(",", ":"))
    lines = wrap(js.replace(",", ", ").replace(":", ": "), 430, 13, mono=True)
    ht = 20 + len(lines) * 17
    b += R(20, 322, 460, ht + 36, rx=8, fill="#F6F7F9", stroke=C_MUT)
    for j, ln in enumerate(lines):
        b += T(32, 346 + j * 17, ln, size=14.5, anchor="start", font=MONO)
    b += T(32, 346 + len(lines) * 17 + 8, f"token proxy for this message: {ex['n_tokens_proxy']} (about 4 characters per token)", size=14.5, anchor="start", fill=C_SUB)
    # bus
    yb = 322 + ht + 56
    s, _ = card(20, yb, 460, "Bus", "Validates the schema, charges tokens, enforces a per-episode budget; a message over budget is refused and the sender is told.", color=NAVY)
    b += s
    # c: communication modes (FIXTURE)
    b += panel(510, 308, "c", "Four communication modes")
    comm = FX["comm"]
    mx = max(v["tokens_per_episode"] for v in comm.values())
    for k, (mode, desc) in enumerate((("push", "every round"), ("delta", "changed objects only"), ("pull", "start and after failures"), ("init_only", "once at start"))):
        y = 332 + k * 62
        v = comm[mode]
        b += T(510, y + 16, mode.replace("_", "-"), size=14, anchor="start", weight="bold", fill=NAVY)
        b += T(510, y + 34, desc, size=14.5, anchor="start", fill=C_SUB)
        w = 280 * v["tokens_per_episode"] / mx
        b += R(660, y + 4, max(w, 2), 24, rx=3, fill=tint(BLUE, 0.35), stroke=BLUE, sw=1.0)
        b += T(660 + w + 8 if w < 190 else 660 + w - 8, y + 22, f"{v['tokens_per_episode']:,.0f}", size=14.5, anchor="start" if w < 190 else "end",
               fill=C_TXT if w < 190 else "#FFFFFF", weight="bold")
    b += tag(510, 590, "FIXTURE", ORANGE)
    b += T(598, 606, "tokens per episode, symbolic world, noisy Eyes, 80 seeds", size=14.5, anchor="start", fill=C_SUB)
    b += T(510, 640, f"Success: push {comm['push']['k']}/80, delta {comm['delta']['k']}/80, pull {comm['pull']['k']}/80, init-only {comm['init_only']['k']}/80.", size=14.5, anchor="start", fill=C_SUB)
    b += T(510, 660, "Checks that the accounting works; says nothing about trained agents.", size=14.5, anchor="start", fill=C_SUB)
    b += "</g>"
    save_svg("fig_messages", W_, H_, b)


# ------------------------------------------------------------------------------------------------ 4 harness
def fig_harness():
    h = FD["harness"]
    W_, H_ = 1000, 580
    b = panel(20, 26, "a", f"{h['n_files']} markdown files")
    cnt = {t: sum(1 for d in h["docs"] if d["type"] == t) for t in h["types"]}
    col = {"rule": BLUE, "skill": GREEN, "object": TEAL, "agent": PURPLE, "memory": ORANGE, "task": GREY_D}
    y = 44
    for t in ("rule", "skill", "object", "agent", "memory", "task"):
        ids = ", ".join(d["id"] for d in h["docs"] if d["type"] == t)
        s, bb = card(20, y, 300, f"{t} ({cnt[t]})", ids, color=col[t], bsize=14.5, tsize=14)
        b += s
        y += bb[3] + 8
    ytop = y
    # b: two-pass retrieval
    b += panel(345, 26, "b", "Two-pass BM25 retrieval")
    s1, p1 = card(345, 44, 300, "Pass 1: rules", f"Query = the instruction. Top k=5 of {cnt['rule']} rules. Instruction-required rules retrieved: {len(h['required_rules'])} of {len(h['required_rules'])} in this episode.", color=BLUE)
    s2, p2 = card(345, p1[1] + p1[3] + 26, 300, "Pass 2: notes", "Query = classes in the scene. Top k=3 object, skill and memory notes.", color=TEAL)
    s3, p3 = card(345, p2[1] + p2[3] + 26, 300, "Brain context", "Instruction + retrieved text + messages.", color=PURPLE)
    b += s1 + s2 + s3
    b += connect(p1, "b", p2, "t", color=C_SUB) + connect(p2, "b", p3, "t", color=C_SUB)
    s4, p4 = card(345, p3[1] + p3[3] + 24, 300, "Monitor write-back", "After episodes, measured success rates and failure causes are written back into the skill files.", color=GREEN, dash="6 4")
    b += s4
    # c: tokenizer fix with real scores
    b += panel(670, 26, "c", "The lunchbox tokenizer fix")
    b += T(670, 52, "Tokens of  \"lunchbox1 last\":", size=14.5, anchor="start", fill=C_TXT)
    b += T(670, 72, "before: " + " | ".join(h["tok_old"]), size=14.5, anchor="start", font=MONO, fill=RED)
    b += T(670, 92, "after:  " + " | ".join(h["tok_new"]), size=14.5, anchor="start", font=MONO, fill=GREEN)
    b += T(670, 118, "BM25 score of each rule for this instruction:", size=14.5, anchor="start", fill=C_TXT)
    old = dict(h["rule_rank_old"]); new = dict(h["rule_rank_new"])
    mx = max(list(old.values()) + list(new.values()))
    yy = 130
    need = set(h["required_rules"])
    for rid, _ in h["rule_rank_new"]:
        hl = rid == "lunchbox_last"
        b += T(670, yy + 12, rid, size=14.5, anchor="start", weight="bold" if hl else None, fill=NAVY if hl else C_TXT)
        yy += 17
        b += R(670, yy, 290 * old[rid] / mx, 9, rx=2, fill=tint(RED, 0.4), stroke="none", sw=0)
        b += R(670, yy + 11, 290 * new[rid] / mx, 9, rx=2, fill=tint(GREEN, 0.25), stroke="none", sw=0)
        b += T(670 + 290 * max(old[rid], new[rid]) / mx + 6, yy + 18, f"{old[rid]:.1f} / {new[rid]:.1f}", size=14.5, anchor="start", fill=C_SUB) if hl else ""
        yy += 28
    b += R(670, yy + 4, 12, 10, rx=2, fill=tint(RED, 0.4), stroke="none", sw=0) + T(688, yy + 13, "before", size=14.5, anchor="start")
    b += R(750, yy + 4, 12, 10, rx=2, fill=tint(GREEN, 0.25), stroke="none", sw=0) + T(768, yy + 13, "after", size=14.5, anchor="start")
    rk_old = [r for r, _ in h["rule_rank_old"]].index("lunchbox_last") + 1
    rk_new = [r for r, _ in h["rule_rank_new"]].index("lunchbox_last") + 1
    b += T(670, yy + 38, f"lunchbox_last: rank {rk_old} -> {rk_new} of {len(old)}.", size=14.5, anchor="start", fill=C_TXT, weight="bold")
    b += T(20, H_ - 36, "The pre-fix tokenizer is our reconstruction (alphanumeric runs, so lunchbox1 stays one token). At k=5 both retrieve all required rules for this", size=14.5, anchor="start", fill=C_SUB)
    b += T(20, H_ - 18, "episode; the fix widens the margin over the next rule (5.3). The earlier dropped-rule case reported by the authors is not reproduced here.", size=14.5, anchor="start", fill=C_SUB)
    save_svg("fig_harness", W_, H_, b)


# ------------------------------------------------------------------------------------------------ 5 loop
def fig_loop():
    lp = FD["loop"]
    W_, H_ = 1000, 670
    ev = lp["events"]
    st = lp["story"]
    b = panel(20, 26, "a", f"One recorded episode (seed {lp['seed']}, flaky planner, leaky Eyes)")
    lanes = [("Brain", PURPLE), ("Eyes", TEAL), ("Arm L", ORANGE), ("Arm R", ORANGE), ("Bus", NAVY)]
    lx0, lx1 = 110, 980
    ly = {n: 70 + i * 62 for i, (n, _) in enumerate(lanes)}
    for n, c in lanes:
        b += R(lx0 - 4, ly[n] - 2, lx1 - lx0 + 8, 52, rx=6, fill=tint(c, 0.95), stroke=tint(c, 0.6), sw=1.0)
        b += T(lx0 - 12, ly[n] + 28, n, size=14, anchor="end", weight="bold", fill=c)
    keep = [e for e in ev if e["ev"] == "exec" or e.get("type") in ("LOOK", "CHALLENGE", "AMEND")]
    n = len(keep)
    step = (lx1 - lx0 - 40) / max(n, 1)
    short = {"notebook": "nb", "pencilcase": "pc", "lunchbox": "lb", "bottle": "bt", "laptop": "lp", "book": "bk"}

    def abbr(o):
        for k_, v_ in short.items():
            if o.startswith(k_):
                return v_ + o[len(k_):]
        return o[:3]

    xs_ = {}
    for k, e in enumerate(keep):
        x = lx0 + 10 + k * step
        xs_[k] = x
        cxm = round(x + (step - 2) / 2, 1)
        if e["ev"] == "exec":
            lane = "Arm " + e["arm"]
            col = ORANGE if e["ok"] else RED
            b += R(x, ly[lane] + 4, step - 2, 44, rx=3, fill=tint(col, 0.8), stroke=col, sw=1.0)
            b += T(cxm, ly[lane] + 21, "g" if e["skill"] == "grasp" else "p", size=14.5, weight="bold")
            b += T(cxm, ly[lane] + 40, abbr(e["obj"]), size=14.5)
        else:
            t = e["type"]
            lane = {"LOOK": "Eyes", "CHALLENGE": "Bus", "AMEND": "Bus"}[t]
            col = {"LOOK": TEAL, "CHALLENGE": ORANGE, "AMEND": GREY_D}[t]
            b += R(x, ly[lane] + 4, step - 2, 44, rx=3, fill=tint(col, 0.75), stroke=col, sw=1.4)
            b += T(cxm, ly[lane] + 31, {"LOOK": "L", "CHALLENGE": "C", "AMEND": "A"}[t], size=14.5, weight="bold")
            if t == "CHALLENGE":
                pass
    il = [k for k, e in enumerate(keep) if e.get("type") == "LOOK"]
    ia = [k for k, e in enumerate(keep) if e.get("type") == "AMEND"]
    pm = [(0, "P")] + [(k + 1, "R") for k in il + ia if k + 1 < n]
    for k, lab in pm:
        b += R(xs_[k], ly["Brain"] + 4, step - 2, 44, rx=3, fill=tint(PURPLE, 0.78), stroke=PURPLE, sw=1.2)
        b += T(round(xs_[k] + (step - 2) / 2, 1), ly["Brain"] + 31, lab, size=14.5, weight="bold")
    yl = ly["Bus"] + 70
    b += T(20, yl, "Time order. Arms: g grasp, p place, then the item (bk book, nb notebook, pc pencilcase, bt bottle, lp laptop, lb lunchbox).", size=14.5, anchor="start", fill=C_SUB)
    b += T(20, yl + 19, "Eyes: L = LOOK. Bus: C = CHALLENGE, A = AMEND. Brain: P = first plan, R = replan after LOOK or AMEND.", size=14.5, anchor="start", fill=C_SUB)
    b += T(20, yl + 40, f"Executed tasks, {sum(1 for e in ev if e['ev'] == 'exec')}; CHALLENGE {st['challenges']}; LOOK {st['looks']}; AMEND at round {lp['amend_round']}; replans {st['replans']}; rounds {st['rounds']}; success: {'yes' if st['success'] else 'no'}.", size=14.5, anchor="start", fill=C_TXT)
    # b: stage strip
    yb = yl + 70
    b += panel(20, yb, "b", "Stages of the loop, and what this trace shows")
    rej = sum(1 for f in st["failures"] if f["cause"] == "plan_rejected")
    inf = sum(1 for f in st["failures"] if f["cause"] == "static_infeasible")
    stages = [("Probe", "optional; not used in this episode", BLUE, True),
              ("Verify + retry", f"dry-run check: {rej} drafts rejected with reasons, {inf} planning phases ended unsatisfied", GREEN, False),
              ("LOOK", f"item missing from the scene: {st['looks']} fresh observation, then replan", TEAL, False),
              ("Rounds", "head task of each arm; bag access exclusive", ORANGE, False),
              ("CHALLENGE", f"touch vs Eye mass over 20%: {st['challenges']} challenges, scene corrected", ORANGE, False),
              ("AMEND", f"instruction change at round {lp['amend_round']} forces a replan", GREY_D, False),
              ("End check", "rules.check_final; fatal failures attributed", NAVY, False)]
    sw_ = 134
    for k, (t, d, c, opt) in enumerate(stages):
        x = 20 + k * (sw_ + 7)
        s, _ = card(x, yb + 14, sw_, t, d, h=148, color=c, dash="6 4" if opt else None, tsize=16, bsize=14.5, pad=8)
        b += s
    b += T(20, H_ - 12, "FIXTURE: symbolic world, scripted Brain behind a flaky-planner wrapper, parametric Eyes. Recovery ladder: retry, alternate params, retreat, reassign, replan, abort.", size=14.5, anchor="start", fill=C_SUB)
    save_svg("fig_loop", W_, H_, b)


# ------------------------------------------------------------------------------------------------ 6 tools
def fig_tools():
    t = FD["tools"]
    W_, H_ = 1000, 560
    b = panel(20, 26, "a", "ToolBrain: draft, dry_run feedback, submit")
    chain = ["check_rule", "query_scene", "look", "dry_run", "submit_plan"]
    x = 20
    boxes = []
    for k, n in enumerate(chain):
        col = GREEN if n in ("dry_run", "submit_plan") else PURPLE
        s, bb = card(x, 44, 170, n, "", h=40, color=col, tsize=16, center=True)
        b += s
        boxes.append(bb)
        x += 190
    b += connect(boxes[3], "r", boxes[4], "l", color=C_SUB, sw=1.2)
    b += T(20, 112, f"Up to {t['max_turns_default']} turns of tool calls, in any order; the prompt asks the model to dry-run before it submits, and a bare plan reply also commits.", size=14.5, anchor="start", fill=C_SUB)
    # draft and feedback (real)
    d = t["draft_tasks"]
    dr = t["dry_run_draft"]
    b += panel(20, 150, "b", f"A real flawed draft (seed {t['seed']})")
    pl = [x_ for x_ in d if x_["skill"] == "place"]
    y = 168
    for it in pl[:6]:
        b += R(20, y, 420, 26, rx=4, fill="#FFFFFF", stroke=GREY, sw=1.0)
        b += T(30, y + 18, f"{it['id']}  arm {it['arm']}  deps: {', '.join(it['deps'])}", size=13, anchor="start", font=MONO)
        y += 30
    b += T(20, y + 14, f"... {len(d)} tasks; two placements have swapped ordering dependencies.", size=14.5, anchor="start", fill=C_SUB)
    s, bb = card(470, 168, 240, "dry_run result", "static_ok: " + str(dr["static_ok"]).lower() + "\nsuccess: " + str(dr["success"]).lower() + "\nviolations:\n  " + "; ".join(dr["violations"]), color=RED, mono=True, bsize=14.5)
    b += s
    s, bb2 = card(740, 168, 240, "After revising", "static_ok: " + str(t["dry_run_final"]["static_ok"]).lower() + "\nsuccess: " + str(t["dry_run_final"]["success"]).lower() + "\nviolations: none", color=GREEN, mono=True, bsize=14.5)
    b += s
    b += connect(bb, "r", bb2, "l", color=C_SUB, sw=1.2)
    b += T(595, bb[1] + bb[3] + 20, "feedback is the only thing the Brain sees", size=14.5, fill=C_SUB) if False else ""
    # c: verification ablation (FIXTURE)
    yc = 400
    b += panel(20, yc, "c", "Verification against a flaky planner")
    v = FX["verify"]
    rows = [("none", "verify_none"), ("static", "verify_static"), ("dry run", "verify_dry_run")]
    for k, (lab, key) in enumerate(rows):
        yy = yc + 22 + k * 34
        kk, nn = v[key]["k"], v[key]["n"]
        b += T(110, yy + 18, lab, size=14, anchor="end", weight="bold", fill=NAVY)
        b += R(120, yy + 2, 400, 24, rx=3, fill="#F1F3F5", stroke=C_MUT, sw=0.8)
        b += R(120, yy + 2, 400 * kk / nn, 24, rx=3, fill=tint(BLUE, 0.35), stroke=BLUE, sw=1.0)
        b += T(530, yy + 20, f"{kk}/{nn}", size=14, anchor="start", weight="bold")
    b += tag(660, yc + 40, "FIXTURE", ORANGE)
    b += T(660, yc + 82, "Planner replaced by a wrapper that injects", size=14.5, anchor="start", fill=C_SUB)
    b += T(660, yc + 100, "errors with probability 0.5; clean Eyes; 80 seeds.", size=14.5, anchor="start", fill=C_SUB)
    b += T(660, yc + 118, "Checks the verifier mechanism, not a model.", size=14.5, anchor="start", fill=C_SUB)
    save_svg("fig_tools", W_, H_, b)
