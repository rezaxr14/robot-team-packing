"""Generate generated/*.tex (tables and number macros) and generated/*.pending.json from recorded data.

Inputs : generated/figdata.json, generated/fixture.json (from collect_*.py), and, if present, a results bundle
         (RTP_RESULTS or paper/results): reports/**/records.jsonl, reports/credit/credit.json.
Rule   : no number is typed by hand. A cell with no data is a red \\todo{pending} and is listed in <table>.pending.json.
Run    : python scripts/build_paper.py
"""
from __future__ import annotations
import json, math, os
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
GEN = PAPER / "generated"
FD = json.loads((GEN / "figdata.json").read_text())
FX = json.loads((GEN / "fixture.json").read_text())
RES = Path(os.environ.get("RTP_RESULTS", PAPER / "results"))
PEND = r"\todo{pending}"


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def load_records():
    recs = {}
    for f in sorted(RES.glob("reports/**/records.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                recs.setdefault(r["config"], []).append(r)
    return recs


RECS = load_records() if RES.exists() else {}
pending: dict[str, list] = {}


def write_table(name, header, rows, colspec, caption_note=None, pend_rows=None):
    """rows: list of lists of cell strings. A cell equal to PEND is recorded as pending."""
    pend = []
    for r in rows:
        for j, c in enumerate(r):
            if c == PEND:
                pend.append({"row": r[0], "column": header[j]})
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / f"table_{name}.tex").write_text("\n".join(lines) + "\n")
    (GEN / f"table_{name}.pending.json").write_text(json.dumps({"table": name, "n_pending_cells": len(pend), "cells": pend}, indent=1))
    pending[name] = pend


def rate_cell(recs, key="success"):
    if not recs:
        return PEND
    k, n = sum(1 for r in recs if r[key]), len(recs)
    p, lo, hi = wilson(k, n)
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}] (n={n})"


# ------------------------------------------------------------------------------------------------ macros
m, lp, cr, hd, hr = FD["mechanism"], FD["loop"], FD["credit"], FD["hindsight"], FD["harness"]
items = {i["name"]: i for i in m["items"]}
pc, nb = items["pencilcase1"], items["notebook1"]
rk = lambda lst: [r for r, _ in lst].index("lunchbox_last") + 1
sc = lambda lst: dict(lst)["lunchbox_last"]
macros = {
    "mechSeed": m["seed"], "mechSigma": m["mass_sigma"],
    "mechNoFixScore": f"{m['no_fix']['score']:.2f}", "mechChalScore": f"{m['challenge']['score']:.2f}", "mechProbeScore": f"{m['probe']['score']:.2f}",
    "mechNChal": len(m["challenge_msgs"]), "mechPcTrue": f"{pc['true_mass']:.2f}", "mechPcEye": f"{pc['eye_mass']:.2f}",
    "mechPcErr": f"{(pc['eye_mass'] - pc['true_mass']) / pc['true_mass'] * 100:+.0f}", "mechNbTrue": f"{nb['true_mass']:.2f}",
    "mechNbEye": f"{nb['eye_mass']:.2f}", "mechNbErr": f"{(nb['eye_mass'] - nb['true_mass']) / nb['true_mass'] * 100:+.0f}",
    "mechRoundsNoFix": m["no_fix"]["rounds"], "mechRoundsChal": m["challenge"]["rounds"], "mechRoundsProbe": m["probe"]["rounds"],
    "loopSeed": lp["seed"], "loopLooks": lp["story"]["looks"], "loopChal": lp["story"]["challenges"], "loopRounds": lp["story"]["rounds"],
    "loopReplans": lp["story"]["replans"], "loopAmendRound": lp["amend_round"],
    "loopRej": sum(1 for f in lp["story"]["failures"] if f["cause"] == "plan_rejected"),
    "loopInf": sum(1 for f in lp["story"]["failures"] if f["cause"] == "static_infeasible"),
    "loopTasks": sum(1 for e in lp["events"] if e["ev"] == "exec"),
    "credBase": f"{cr['base_score']:.2f}", "credGain": f"{cr['oracle_gain']:.2f}", "credEmpty": f"{cr['rows']['empty']['cf_score']:.3f}",
    "hindTasks": hd["n_tasks"], "hindDropped": esc(", ".join(hd["dropped"])),
    "tokOld": f"{sc(hr['rule_rank_old']):.1f}", "tokNew": f"{sc(hr['rule_rank_new']):.1f}", "rankOld": rk(hr["rule_rank_old"]), "rankNew": rk(hr["rule_rank_new"]),
    "nHarnessFiles": hr["n_files"], "nMsgTypes": len(FD["messages"]["schema"]), "nTools": len(FD["tools"]["tool_names"]),
    "fxN": FX["n"], "toolSeed": FD["tools"]["seed"],
}
(GEN / "figdata_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))

# ------------------------------------------------------------------------------------------------ tab:messages
obs = {}
for e in lp["events"]:
    if e["ev"] == "msg":
        obs.setdefault(e["type"], set()).add(e["from"].split("_")[0])
rows = []
for t, fields in FD["messages"]["schema"].items():
    rows.append([r"\texttt{" + t + "}", esc(", ".join(fields)), ", ".join(sorted(obs[t])) if t in obs else "not exchanged in the recorded episode"])
write_table("messages", ["Type", "Required fields", "Sender in the recorded episode"], rows, r"lp{0.38\linewidth}p{0.3\linewidth}")

# ------------------------------------------------------------------------------------------------ tab:rules
rows = []
for d in FD["harness"]["docs"]:
    if d["type"] == "rule":
        rows.append([r"\texttt{" + esc(d["id"]) + "}", esc(d["text"])])
write_table("rules", ["Rule file", "Text given to the Brain (verbatim from the harness file)"], rows, r"lp{0.74\linewidth}")

# ------------------------------------------------------------------------------------------------ tab:harness files (appendix B)
rows = [[r"\texttt{" + esc(d["path"].replace("harness_files/", "")) + "}", d["type"], esc(d["keywords"])] for d in FD["harness"]["docs"]]
write_table("harness_files", ["File", "Type", "Retrieval keywords"], rows, r"llp{0.5\linewidth}")

# ------------------------------------------------------------------------------------------------ tab:trackA
write_table("trackA", ["Condition", "L1 accuracy", "L2 accuracy", "Format rate", "Failed items"], [
    ["Released VIKI-R checkpoints, upstream-style scoring", PEND, PEND, PEND, PEND],
    ["Released checkpoints, position-robust scoring, full split", PEND, PEND, PEND, PEND],
    ["Text-only Brain with oracle descriptions", PEND, PEND, PEND, PEND],
    ["Same Brain, no Eye description", PEND, PEND, PEND, PEND],
    ["Same Brain, real Eye description", PEND, PEND, PEND, PEND],
    ["Matched-base control (Qwen2.5-VL-3B), if compute allows", PEND, PEND, PEND, PEND]], r"p{0.40\linewidth}cccc")

# ------------------------------------------------------------------------------------------------ tab:main and tab:ablations
MAIN = [("full system", "full"), ]
ABL = [("verification: none", "verify_none"), ("verification: static", "verify_static"), ("verification: dry run", "verify_dry_run"),
       ("communication: push", "comm_push"), ("communication: delta", "comm_delta"), ("communication: pull", "comm_pull"),
       ("communication: init-only", "comm_init_only")]


def result_rows(spec):
    out = []
    for lab, key in spec:
        r = RECS.get(key)
        if r:
            n = len(r)
            out.append([lab, rate_cell(r), f"{sum(1 for x in r if x['plan_pass']) / n:.2f}", f"{sum(x['replans'] for x in r) / n:.2f}",
                        f"{sum(x['msg_tokens'] for x in r) / n:.0f}"])
        else:
            out.append([lab, PEND, PEND, PEND, PEND])
    return out


write_table("main", ["Condition (Track B, one evidence class per run)", "Success [95\\% Wilson]", "Plan pass", "Replans", "Msg tokens"],
            result_rows([("Full system, LLM-SYMBOLIC", "llm_full"), ("Full system, ISAAC", "isaac_full")]), r"p{0.36\linewidth}cccc")
write_table("ablations", ["Ablation", "Success [95\\% Wilson]", "Plan pass", "Replans", "Msg tokens"], result_rows([
    ("No harness", "llm_no_harness"), ("Verification: none", "llm_verify_none"), ("Verification: static", "llm_verify_static"),
    ("Verification: dry run", "llm_verify_dry_run"), ("No CHALLENGE, no probe", "llm_no_challenge"), ("Probe on", "llm_probe"),
    ("One Eye", "llm_one_eye"), ("Communication: push", "llm_comm_push"), ("Communication: delta", "llm_comm_delta"),
    ("Communication: pull", "llm_comm_pull"), ("Communication: init-only", "llm_comm_init_only")]), r"p{0.36\linewidth}cccc")

# ------------------------------------------------------------------------------------------------ tab:credit
cred_rows = []
cf = RES / "reports" / "credit" / "credit.json"
if cf.exists():
    for r in json.loads(cf.read_text()):
        g = r["oracle_gain"]
        cred_rows.append([esc(r["fault"]), f"{g['mean']:.3f} [{g['lo']:.3f}, {g['hi']:.3f}]"] + [f"{r['fields'].get(f, float('nan')):.3f}" for f in ("mass_guess", "cls", "fragility_guess", "region")])
if not cred_rows:
    cred_rows = [[f, PEND, PEND, PEND, PEND, PEND] for f in ("Plain SFT Eye", "Rejection-sampled Eye", "Fixed-Brain counterfactual")]
write_table("credit", ["Eye variant", "Oracle gain G", "mass", "class", "fragility", "region"], cred_rows, r"p{0.28\linewidth}ccccc")

# ------------------------------------------------------------------------------------------------ tab:attribution
write_table("attribution", ["Agent label", "Episodes labelled by hand", "Agreement with the heuristic", "Cohen's kappa"],
            [[a, PEND, PEND, PEND] for a in ("Eye", "Brain", "Arm")], r"lccc")

# ------------------------------------------------------------------------------------------------ tab:evidence (ledger)
write_table("evidence", ["Evidence class", "What it is", "Status in this version", "Where reported"], [
    ["TRACK A", "Real VIKI-Bench data, real released models", PEND, r"\cref{tab:trackA}"],
    ["ISAAC", "Physics episodes in Isaac Lab", PEND, r"\cref{tab:main}"],
    ["LLM-SYMBOLIC", "A served LLM Brain on the symbolic task", PEND, r"\cref{tab:main,tab:ablations}"],
    ["FIXTURE", "Scripted Brain, flaky-planner wrapper, parametric Eyes", "run (mechanism checks only)", r"\cref{app:fixture}"],
    ["Engineering", "Wiring tests on CPU with a tiny random model", "run (proves wiring, not quality)", r"\cref{app:env}"]], r"lp{0.3\linewidth}p{0.22\linewidth}l")

# ------------------------------------------------------------------------------------------------ tab:repro
write_table("repro", ["Item", "Value"], [
    ["Training and data seeds", r"$\geq 100000$"], ["Evaluation seeds", r"$< 100000$"],
    ["Stochastic draws", "keyed by (seed, arm, skill, object, attempt); Eye noise by (seed, eye, time)"],
    ["Python / Isaac Sim / Isaac Lab", "3.11 / 5.x / 2.3 (planned pairing)"], ["Compute", r"\todo{dual T4 on Kaggle: confirm memory plan and run times}"],
    ["Code and data release", r"\todo{repository, Hub and DOI links}"]], r"lp{0.62\linewidth}")

# ------------------------------------------------------------------------------------------------ appendix F fixture tables (FIXTURE)
v = FX["verify"]
write_table("fx_verify", ["Verification", "Successes", "Wilson 95\\%", "Retries used"], [
    [lab, f"{v[k]['k']}/{v[k]['n']}", f"[{v[k]['lo']:.2f}, {v[k]['hi']:.2f}]", str(v[k]["plan_retries"])] for lab, k in
    (("none", "verify_none"), ("static", "verify_static"), ("dry run", "verify_dry_run"), ("dry run, 0 retries", "verify_dry_run_0_retries"))], "lccc")
c = FX["comm"]
write_table("fx_comm", ["Mode", "Successes", "Message tokens per episode"], [
    [k.replace("_", "-"), f"{c[k]['k']}/{c[k]['n']}", f"{c[k]['tokens_per_episode']:,.0f}".replace(",", r"{,}")] for k in ("push", "delta", "pull", "init_only")], "lcc")
mn = FX["mass_noise"]
write_table("fx_mass", ["Condition", "Successes", "Wilson 95\\%", "Mean rounds"], [
    [lab, f"{mn[k]['k']}/{mn[k]['n']}", f"[{mn[k]['lo']:.2f}, {mn[k]['hi']:.2f}]", f"{mn[k]['mean_rounds']:.2f}"] for lab, k in
    (("probe", "probe"), ("challenge only", "challenge_only"), ("no fix", "none"))], "lccc")
lk = FX["look"]
write_table("fx_look", ["Eyes (omission fault)", "LOOK off", "LOOK on"], [
    [str(n), f"{lk[f'eyes{n}_look0']['k']}/{lk[f'eyes{n}_look0']['n']}", f"{lk[f'eyes{n}_look1']['k']}/{lk[f'eyes{n}_look1']['n']}"] for n in (1, 2)], "lcc")
write_table("fx_credit", ["Eye fault", "Mean episode score", "Oracle gain G [95\\% CI]", "mass", "class", "fragility", "region"], [
    [esc(r["fault"]), f"{r['base_score']:.3f}", f"{r['oracle_gain']['mean']:.3f} [{r['oracle_gain']['lo']:.3f}, {r['oracle_gain']['hi']:.3f}]"] +
    [f"{r['fields'].get(f, 0.0):.3f}" for f in ("mass_guess", "cls", "fragility_guess", "region")] for r in FX["credit"]], r"lccccccc"[:7])

# ------------------------------------------------------------------------------------------------ pending summary
(GEN / "pending_summary.json").write_text(json.dumps({k: len(v_) for k, v_ in pending.items()}, indent=1))
print("tables:", {k: len(v_) for k, v_ in pending.items()}, "| results bundle:", bool(RECS))
