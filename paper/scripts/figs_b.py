"""Figures 7-12. Data from generated/figdata.json; fig_results reads a results bundle if one exists, else draws red placeholders."""
from __future__ import annotations
import json, math, os
from pathlib import Path
from diagram_common import *  # noqa: F401,F403
from diagram_common import PAPER_DIR

FD = json.loads((PAPER_DIR / "generated" / "figdata.json").read_text())


# ------------------------------------------------------------------------------------------------ 7 signals
def fig_signals():
    sg = FD["signals"]
    w = sg["weights"]
    W_, H_ = 1000, 700
    b = T(20, 26, "Learning signals and what each one trains", size=16, anchor="start", weight="bold", fill=NAVY)
    rows = [
        ("Eye (VLM)", TEAL, "Supervised fine-tuning on rendered images with SCENE JSON targets (LoRA). Then rejection sampling: K=8 reports per image, keep the best by field accuracy or by counterfactual credit."),
        ("Brain (text LLM)", PURPLE, "Supervised fine-tuning on teacher plans validated by closed-loop success, then GRPO with the plan-level reward below. Needs reward spread, so SFT must teach the format first."),
        ("Arms (blind policy)", ORANGE, "Distillation of scripted differential-IK demonstrations into a small transformer with a flow-matching action head (an MLP head is the ablation)."),
        ("Team signals", NAVY, "Counterfactual Eye credit (replace one Eye's report, replay with the same random draws) and team hindsight relabeling of finished episodes."),
    ]
    y = 44
    for name, c, desc in rows:
        s1, p1 = card(20, y, 200, name, "", h=84, color=c, tsize=16.5, center=True)
        s2, p2 = card(250, y, 540, None, desc, h=84, color=c, fill="#FFFFFF", bsize=14.5)
        b += s1 + s2 + connect(p1, "r", p2, "l", color=c, sw=1.3)
        b += pending_tag(820, y + 28, "results pending", w=160)
        y += 100
    # reward formula from code
    s, _ = card(20, y + 6, 960, "Brain plan-level reward (weights read from the code)",
                f"R = {w['format']} r_format + {w['feasible']} r_feasible + {w['rules']} r_rules + {w['length']} r_length   (+ {sg['rollout_weight']} r_rollout, optional closed-loop rollout reward)\n"
                f"r_rules = valid-item fraction from a deterministic dry run, minus {sg['r_rules_penalty']} per rule violation and per crush, floored at 0.   r_length = 1 up to {sg['length_cap']} tasks, then falls linearly to 0 at {2 * sg['length_cap']}.",
                color=PURPLE, bsize=14, mono=False)
    b += s
    y += 6 + 120
    ep = sg["episode_reward"]
    s, _ = card(20, y + 8, 960, "Episode score used for credit",
                f"R(episode) = valid-item fraction - {ep['w_viol']} per violation - {ep['w_crush']} per crushed item - {ep['w_replan']} per replan.   Credit C(i, r) = R(real) - R(Eye i replaced by r).   Oracle gain G = R(oracle) - R(real) = -C.",
                color=NAVY, bsize=14)
    b += s
    b += T(20, H_ - 14, "Formulas are definitions; no trained model has been run with them yet, so every outcome is pending.", size=14.5, anchor="start", fill=C_SUB)
    save_svg("fig_signals", W_, H_, b)


# ------------------------------------------------------------------------------------------------ 8 credit
def fig_credit():
    c = FD["credit"]
    h = FD["hindsight"]
    W_, H_ = 1000, 640
    b = panel(20, 26, "a", f"Counterfactual Eye credit (seed {c['seed']}, Eye {c['eye']})")
    base = c["base_score"]
    rows = [("real reports", base), ("Eye A empty", c["rows"]["empty"]["cf_score"]), ("Eye A degraded", c["rows"]["degraded"]["cf_score"]),
            ("oracle: all fields", c["rows"]["oracle"]["cf_score"]), ("oracle: mass only", c["rows"]["oracle_mass_guess"]["cf_score"]),
            ("oracle: class only", c["rows"]["oracle_cls"]["cf_score"]), ("oracle: fragility only", c["rows"]["oracle_fragility_guess"]["cf_score"]),
            ("oracle: region only", c["rows"]["oracle_region"]["cf_score"])]
    x0, sc = 190, 230.0
    for k, (lab, v) in enumerate(rows):
        yy = 52 + k * 34
        b += T(x0 - 8, yy + 17, lab, size=14.5, anchor="end", weight="bold" if k == 0 else None)
        b += R(x0, yy + 3, sc, 22, rx=3, fill="#F1F3F5", stroke=C_MUT, sw=0.8)
        col = GREY if k == 0 else (GREEN if v > base + 1e-9 else (RED if v < base - 1e-9 else GREY))
        b += R(x0, yy + 3, sc * v, 22, rx=3, fill=tint(col, 0.3), stroke=col, sw=1.0)
        b += T(x0 + sc + 8, yy + 20, f"{v:.3f}", size=14.5, anchor="start", weight="bold")
    xl = x0 + sc * base
    b += L(xl, 48, xl, 52 + len(rows) * 34, color=GREY_D, sw=1.2, dash="4 3")
    ys = 52 + len(rows) * 34 + 10
    s, _ = card(20, ys, 450, "Sign convention",
                f"C = R(real) - R(replaced). Empty: C = {c['rows']['empty']['credit']:+.3f}, so the Eye helped. Oracle: C = {c['rows']['oracle']['credit']:+.3f}; the oracle gain G = -C = {c['oracle_gain']:+.2f} is the score a perfect Eye would recover, all of it in the mass field "
                f"(G_mass = {c['field_gain']['mass_guess']:+.2f}, others 0). All draws are keyed (seed, arm, skill, object, attempt) so replays share common random numbers.",
                color=NAVY, bsize=14.5)
    b += s
    # b: hindsight
    b += panel(500, 26, "b", "Team hindsight relabeling (same episode)")
    ins = h["orig_instruction"].split(": ", 1)[1] if ": " in h["orig_instruction"] else h["orig_instruction"]
    s1, p1 = card(500, 44, 480, "Original instruction", ins, color=GREY_D, bsize=14.5, tsize=14)
    viol = h["orig_violations"][0]
    s2, p2 = card(500, p1[1] + p1[3] + 22, 480, "Outcome", f"fail: {viol['rule']}, {viol['detail']}", color=RED, bsize=14.5, tsize=14)
    new = h["relabeled_instruction"].split(": ", 1)[1]
    s3, p3 = card(500, p2[1] + p2[3] + 22, 480, "Relabeled to what was achieved", new + f"\n(dropped: {', '.join(h['dropped'])})", color=BLUE, bsize=14.5, tsize=14)
    s4, p4 = card(500, p3[1] + p3[3] + 22, 480, "Replay check",
                  f"Replaying the executed plan ({h['n_tasks']} tasks) from the initial state satisfies the relabeled spec: success {str(h['replay_success']).lower()}, score {h['replay_score']:.2f}. Only then is the record admitted.", color=GREEN, bsize=14.5, tsize=14)
    b += s1 + s2 + s3 + s4 + connect(p1, "b", p2, "t", color=C_SUB) + connect(p2, "b", p3, "t", color=C_SUB) + connect(p3, "b", p4, "t", color=C_SUB)
    H_ = int(max(p4[1] + p4[3], ys + 150) + 44)
    b += T(20, H_ - 14, "FIXTURE episode: symbolic world, scripted Brain, parametric Eyes; credit uses no challenge and no verification so the Eye error reaches the plan.", size=14.5, anchor="start", fill=C_SUB)
    save_svg("fig_credit", W_, H_, b)


# ------------------------------------------------------------------------------------------------ 9 bridge
def fig_bridge():
    br = FD["bridge"]
    lay = br["layout"]
    W_, H_ = 1000, 640
    b = panel(20, 26, "a", "Isaac scene, top view (Layout constants, metres)")
    sc = 300.0
    ox, oy = 250, 520           # pixel of world (0, y=-0.8) bottom-centre; x right, y up
    px = lambda x: ox + x * sc
    py = lambda y: oy - (y + 0.8) * sc * 1.0
    # table band
    ytab0, ytab1 = -0.10, 0.23
    b += R(px(-0.62), py(ytab1), 1.24 * sc, (ytab1 - ytab0) * sc, rx=6, fill="#F6F7F9", stroke=C_MUT, sw=1.2)
    for nm, x in br["zone_x"].items():
        b += T(px(x), py(ytab0) + 20, nm.replace("table_", "") + " zone", size=14.5, fill=C_SUB)
    for xb in (lay["left_max"], lay["right_min"]):
        b += L(px(xb), py(ytab1), px(xb), py(ytab0), color=C_MUT, sw=1.0, dash="4 3")
    for yr in br["row_y"]:
        for x in br["zone_x"].values():
            b += f'<circle cx="{px(x):.1f}" cy="{py(yr):.1f}" r="4" fill="{GREY}"/>'
    # bag
    bc, bh = lay["bag_center"], lay["bag_half"]
    bx0, bx1, by0, by1 = bc[0] - bh[0], bc[0] + bh[0], bc[1] - bh[1], bc[1] + bh[1]
    b += R(px(bx0), py(by1), (bx1 - bx0) * sc, (by1 - by0) * sc, rx=6, fill="#FFFFFF", stroke=NAVY, sw=1.8)
    ld, fd = lay["laptop_slot_depth"], lay["front_pocket_depth"]
    b += R(px(bx0), py(by0 + ld), (bx1 - bx0) * sc, ld * sc, rx=0, fill=tint(PURPLE, 0.85), stroke=PURPLE, sw=1.0)
    b += R(px(bx0), py(by1), (bx1 - bx0) * sc, fd * sc, rx=0, fill=tint(ORANGE, 0.85), stroke=ORANGE, sw=1.0)
    b += T(px(0), py((by0 + ld + by1 - fd) / 2) + 5, "main", size=14.5, fill=NAVY, weight="bold")
    b += T(px(bx1) + 8, py(by0 + ld / 2) + 5, "laptop slot (back)", size=14.5, anchor="start", fill=PURPLE)
    b += T(px(bx1) + 8, py(by1 - fd / 2) + 5, "front pocket", size=14.5, anchor="start", fill=ORANGE)
    b += T(px(0), py(by1) - 8, "bag", size=14.5, fill=NAVY, weight="bold")
    # arms
    for nm, (x, y, _) in br["arm_base"].items():
        b += f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="13" fill="{tint(ORANGE, 0.7)}" stroke="{ORANGE}" stroke-width="1.6"/>'
        b += T(px(x), py(y) + 5, nm, size=14.5, fill=ORANGE, weight="bold")
    # eyes
    for nm, (pos, look) in br["eye_poses"].items():
        x, y = max(min(pos[0], 0.78), -0.78), pos[1]
        b += R(px(x) - 14, py(y) - 11, 28, 22, rx=4, fill=tint(TEAL, 0.7), stroke=TEAL, sw=1.5)
        b += T(px(x), py(y) + 5, nm, size=14.5, fill=TEAL, weight="bold")
    b += T(px(-0.75) + 30, py(-0.75) + 24, "Eye cameras at z = 0.75, looking at (0, 0.15, 0)", size=14.5, anchor="start", fill=TEAL)
    # b: one checker, three worlds
    b += panel(560, 26, "b", "One success definition, three worlds")
    worlds = [("Symbolic world", "Fast; the loop is developed and tested here.", GREEN, False),
              ("Pose-level fake simulator", "Poses in, bridge maps to regions and compartments; loop outcomes identical to the symbolic world (tested).", BLUE, False),
              ("Isaac Lab", "Physics. Skill outcomes derived from state before and after.", ORANGE, True)]
    y = 44
    boxes = []
    for nm, d, c, pend in worlds:
        s, bb = card(560, y, 280, nm, d, color=c, bsize=14.5, tsize=16, h=96)
        b += s
        boxes.append(bb)
        if pend:
            b += R(855, y, 125, 96, rx=6, fill=tint(RED, 0.93), stroke=RED, sw=1.4, dash="6 4") + T(917, y + 50, "render", size=14.5, fill=RED, weight="bold") + T(917, y + 68, "pending", size=14.5, fill=RED, weight="bold")
        y += 112
    s, chk = card(560, y + 6, 420, "rules.check_final", "Every requested item packed, zero rule violations, nothing crushed. The same function scores all three worlds.", color=NAVY, bsize=14.5)
    b += s
    for bb in boxes:
        pass
    b += connect(boxes[0], "b", chk, "t", oa=40, ob=40, color=GREEN, sw=1.2) if False else ""
    b += T(560, chk[1] + chk[3] + 28, "Quaternion order: wxyz (Isaac Lab 2.x), xyzw (3.0).", size=14.5, anchor="start", fill=C_SUB)
    b += T(560, chk[1] + chk[3] + 48, "Ten skill outcomes derived from state:", size=14.5, anchor="start", fill=C_SUB)
    for j, ln in enumerate(wrap(", ".join(br["skill_outcomes"]), 420, 13, mono=False)):
        b += T(560, chk[1] + chk[3] + 66 + j * 17, ln, size=14.5, anchor="start", fill=C_TXT)
    b += T(560, H_ - 14, "The identical-outcome test validates the integration, not the physics.", size=14.5, anchor="start", fill=C_SUB)
    save_svg("fig_bridge", W_, H_, b)


# ------------------------------------------------------------------------------------------------ 10 design
def _matrix(b, x, y, title, rowl, cols, color, rowh=34, labw=330, colw=150):
    b += T(x, y, title, size=15, anchor="start", weight="bold", fill=color)
    y += 12
    for j, cl in enumerate(cols):
        b += T(x + labw + j * colw + colw / 2, y + 16, cl, size=14.5, weight="bold", fill=C_TXT)
    y += 26
    for i, rl in enumerate(rowl):
        yy = y + i * rowh
        b += R(x, yy, labw + colw * len(cols), rowh - 3, rx=4, fill="#FFFFFF", stroke=tint(color, 0.5), sw=1.0)
        b += T(x + 8, yy + 20, rl, size=14.5, anchor="start", fill=C_TXT)
        for j in range(len(cols)):
            cx = x + labw + j * colw + 8
            b += R(cx, yy + 4, colw - 16, rowh - 11, rx=3, fill=tint(RED, 0.92), stroke=RED, sw=1.0) + T(cx + (colw - 16) / 2, yy + 20, "pending", size=14.5, fill=RED, weight="bold")
    return b, y + len(rowl) * rowh


def fig_design():
    W_, H_ = 1000, 700
    b = ""
    b, y = _matrix(b, 20, 26, "Track A: real benchmark data, real released models", [
        "released VIKI-R checkpoints, upstream-style scoring", "released checkpoints, position-robust scoring, full split",
        "Eye -> Brain: text-only Brain with oracle descriptions", "Eye -> Brain: same Brain, no Eye description",
        "Eye -> Brain: same Brain, real Eye description", "matched-base control (Qwen2.5-VL-3B), if compute allows"],
        ["L1 acc", "L2 acc", "format rate", "failed items"], BLUE, labw=420, colw=135)
    b, y = _matrix(b, 20, y + 22, "Track B: the packing team (symbolic and physics worlds)", [
        "full system", "no harness", "verification: none / static / dry run", "no CHALLENGE, no probe", "probe on",
        "one Eye instead of two", "communication: push / delta / pull / init-only"],
        ["success", "plan pass", "replans", "msg tokens"], PURPLE, labw=420, colw=135)
    b += T(20, H_ - 34, "Planned design; every cell is empty until a run produces it. Track B is also split by evidence class: LLM-SYMBOLIC (served Brain on the symbolic task)", size=14.5, anchor="start", fill=C_SUB)
    b += T(20, H_ - 14, "and ISAAC (physics episodes). Fixture runs are kept out of this matrix (appendix F). Evaluation seeds < 100000.", size=14.5, anchor="start", fill=C_SUB)
    save_svg("fig_design", W_, H_, b)


# ------------------------------------------------------------------------------------------------ 11 scorer audit
def fig_scorer_audit():
    W_, H_ = 1000, 600
    b = panel(20, 26, "a", "Upstream L2 scorer (read from the code)")
    steps = [("Plan", "model output", GREY_D), ("Feasibility check", "pure Python, no simulator", BLUE),
             ("random.choice", "over each asset's candidate start positions, drawn on every call", RED),
             ("Correct if", "feasible AND len(reference) / len(plan) >= 0.99", BLUE),
             ("Reward", "0.1 format + 0.9 correctness", GREEN)]
    prev = None
    y = 44
    for t, d, c in steps:
        s, bb = card(20, y, 440, t, d, color=c, bsize=14.5, tsize=14)
        b += s
        if prev:
            b += connect(prev, "b", bb, "t", color=C_SUB, sw=1.2)
        prev = bb
        y = bb[1] + bb[3] + 16
    s, _ = card(20, y + 2, 440, "Other upstream facts", "Eval scripts read only the first 900 (L2) or 1000 (L1) rows. The L2 script loads id/val.json but writes to a directory named ood.", color=ORANGE, bsize=14.5, tsize=14)
    b += s
    # b: synthetic demo
    b += panel(500, 26, "b", "A toy case we built to test the idea")
    b += tag(500, 40, "SYNTHETIC", ORANGE)
    b += T(612, 56, "not a statement about VIKI-R's published numbers", size=14.5, anchor="start", fill=C_SUB)
    demos = [("plan valid only if the cup starts on the table", "unseeded draws", 18, 40, RED), ("same plan, all start positions", "exhaustive average", 20, 40, RED),
             ("position-agnostic plan", "unseeded draws", 40, 40, GREEN)]
    for k, (lab, sub, v, n, c) in enumerate(demos):
        yy = 92 + k * 92
        b += T(500, yy + 16, lab, size=14, anchor="start", weight="bold", fill=NAVY)
        b += T(500, yy + 34, sub, size=14.5, anchor="start", fill=C_SUB)
        b += R(500, yy + 44, 400, 24, rx=3, fill="#F1F3F5", stroke=C_MUT, sw=0.8)
        b += R(500, yy + 44, 400 * v / n, 24, rx=3, fill=tint(c, 0.3), stroke=c, sw=1.0)
        b += T(910, yy + 62, f"{v}/{n}" if k != 1 else "0.5", size=14, anchor="start", weight="bold")
    s, _ = card(500, 380, 480, "What it means", "When the checker draws the start position at random, a plan that is only valid for some positions passes or fails by luck, and an unseeded score moves between runs. Position-robust scoring (seeded or exhaustive) removes that.", color=NAVY, bsize=14.5, tsize=14)
    b += s
    b += pending_tag(20, 520, "real items exposed: pending")
    b += T(20, 566, "How many real VIKI-Bench items depend on a random start position is unknown until the inventory script has been run on the dataset.", size=14.5, anchor="start", fill=C_SUB)
    save_svg("fig_scorer_audit", W_, H_, b)


# ------------------------------------------------------------------------------------------------ 12 results (data figure)
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load_results(root: Path):
    recs = {}
    for f in sorted(root.glob("reports/**/records.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                recs.setdefault(r["config"], []).append(r)
    cred = None
    cf = root / "reports" / "credit" / "credit.json"
    if cf.exists():
        cred = json.loads(cf.read_text())
    return recs, cred


def fig_results(root: Path | None = None):
    root = root or Path(os.environ.get("RTP_RESULTS", PAPER_DIR / "results"))
    recs, cred = load_results(root) if root.exists() else ({}, None)
    W_, H_ = 1000, 520
    b = ""
    pw = 310
    xs = [20, 345, 670]
    titles = [("a", "Success, Wilson 95% interval"), ("b", "Plan pass vs success"), ("c", "Oracle gain by fault and field")]
    for (L_, t), x in zip(titles, xs):
        b += panel(x, 26, L_, t)

    def placeholder(x, msg):
        return R(x, 44, pw, 400, rx=8, fill=tint(RED, 0.94), stroke=RED, sw=1.4, dash="6 4") + T(x + pw / 2, 240, "pending", size=18, fill=RED, weight="bold") + \
            T(x + pw / 2, 266, msg, size=14.5, fill=RED)
    if recs:
        names = list(recs)[:14]
        for k, nm in enumerate(names):
            r = recs[nm]
            kk, nn = sum(1 for e in r if e["success"]), len(r)
            p, lo, hi = wilson(kk, nn)
            y = 56 + k * 28
            b += T(xs[0] + 150, y + 16, nm[:20], size=14.5, anchor="end")
            b += R(xs[0] + 158, y + 4, 140, 16, rx=2, fill="#F1F3F5", stroke=C_MUT, sw=0.6)
            b += R(xs[0] + 158, y + 4, 140 * p, 16, rx=2, fill=tint(BLUE, 0.35), stroke=BLUE, sw=0.8)
            b += L(xs[0] + 158 + 140 * lo, y + 12, xs[0] + 158 + 140 * hi, y + 12, color=NAVY, sw=1.6)
            pp = sum(1 for e in r if e["plan_pass"]) / nn
            b += R(xs[1] + 158, y + 4, 140, 16, rx=2, fill="#F1F3F5", stroke=C_MUT, sw=0.6)
            b += R(xs[1] + 158, y + 4, 140 * pp, 7, rx=1, fill=tint(PURPLE, 0.35), stroke=PURPLE, sw=0.8)
            b += R(xs[1] + 158, y + 12, 140 * p, 7, rx=1, fill=tint(GREEN, 0.35), stroke=GREEN, sw=0.8)
        b += T(xs[1] + 158, 470, "purple: plan pass; green: success", size=14.5, anchor="start", fill=C_SUB)
    else:
        b += placeholder(xs[0], "no records.jsonl found") + placeholder(xs[1], "no records.jsonl found")
    if cred:
        rows = cred if isinstance(cred, list) else cred.get("rows", [])
        for k, r in enumerate(rows[:8]):
            y = 56 + k * 44
            g = r["oracle_gain"]
            gm = g["mean"] if isinstance(g, dict) else g
            b += T(xs[2] + 100, y + 18, r["fault"][:14], size=14.5, anchor="end")
            b += R(xs[2] + 108, y + 4, 180, 18, rx=2, fill="#F1F3F5", stroke=C_MUT, sw=0.6)
            b += R(xs[2] + 108, y + 4, 180 * max(0.0, min(1.0, gm)), 18, rx=2, fill=tint(TEAL, 0.35), stroke=TEAL, sw=0.8)
    else:
        b += placeholder(xs[2], "no credit.json found")
    b += T(20, H_ - 40, "Generated by scripts/build_diagrams.py from the results bundle (reports/**/records.jsonl, reports/credit/credit.json).", size=14.5, anchor="start", fill=C_SUB)
    b += T(20, H_ - 20, "With no bundle present, every panel is a red placeholder; nothing here is simulated.", size=14.5, anchor="start", fill=C_SUB)
    save_svg("fig_results", W_, H_, b)
    return bool(recs), bool(cred)
