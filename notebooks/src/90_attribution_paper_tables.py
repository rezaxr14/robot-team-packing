#!/usr/bin/env python3
"""Generate notebooks/90_attribution_paper_tables.ipynb — failure attribution + paper tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ops.nbbuild import Notebook, install_cell  # noqa: E402

nb = Notebook("90_attribution_paper_tables")

nb.md("""# 90 — failure attribution and the paper's tables

**Failure attribution is this project's contribution candidate**: when a packing episode fails, say which
layer caused it — brain, eye, arm, a rule, or the environment — and back the claim with a counterfactual.
The repository's `rtp.loop.attribute` supplies rule-based Eye/Brain/Arm labels; a fatal failure added by the
final rule check is a *rule*-stage failure, and reachability/arm-noise causes are environment/arm. These are
**heuristics for analysis, not ground truth**: their validity is exactly what a human labelling pass must
establish.

This notebook (1) shows a real attributed failure with its story and counterfactual, (2) writes a labelling
CSV, (3) runs `agreement()` on a **simulated** copy to prove the metric works, and (4) produces
`table_attribution` from the real records. Only step (4) is evidence about the system; the simulated kappa
is explicitly not a result.""")

nb.code(install_cell([]))

nb.code('''ACTIVE = "local_4050"          # local_4050 | local_4090 | kaggle_t4x2 | pro6000 | tpu_v5e8
import os, sys
from pathlib import Path

os.environ["RTP_PROFILE"] = ACTIVE
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("profile:", ACTIVE, "| repo:", ROOT)''')

nb.code('''from rtp.profiles import detect_caps, get_profile, resolve_runtime

prof = get_profile(ACTIVE)
prof.ensure_dirs()
caps = detect_caps()
attn, dtype, runtime_notes = resolve_runtime(prof, caps)
print("runs:", prof.paths.runs, "| reports:", prof.paths.reports, "| data:", prof.paths.data)
print("config hash:", prof.config_hash(), "| effective batch:", prof.effective_batch,
      "| max_len:", prof.max_len)
print("tuning:", prof.tuning, "| resolved attention:", attn, "| resolved dtype:", dtype)
for note in runtime_notes:
    print(" -", note)
print("attribution runs on CPU with the symbolic fixture; no Isaac Lab/Newton result is claimed")
if not caps.cuda:
    print("SKIPPED: no CUDA on this box — Isaac Lab/Newton episodes and a served VLM Brain are not run; "
          "attribution is demonstrated on the deterministic symbolic fixture")''')

nb.md("""## 1. A real failure, its story, and its attribution

`episode_story` renders the instruction, the outcome, the Eyes' errors against ground truth, and every
auto-attributed failure so a labeler can judge without reading code. `loop.attribute` returns Eye/Brain/Arm;
the final `check_final` adds fatal `rule_*` failures. The `stage` printed below is a presentation mapping
over those *real* fields:

| stage | cause in the data | why |
|---|---|---|
| eye | Eye agent, `rule_*` / `item_never_reported` | wrong or missing report changed a rule outcome |
| brain | Brain agent, `plan_*` / `static_infeasible` / `rule_*` | plan or packing order was wrong given the report |
| arm | Arm agent, `arm_noise` / `slip` / `crush` | blind execution failed |
| rule | fatal `rule_*` from the final checker | final-state constraint violated |
| environment | `out_of_reach` / `not_found` / `no_space` | world geometry or capacity, not a layer decision |

The mapping is shown explicitly so no inferred stage is mistaken for a field the code emitted.""")

nb.code('''from rtp.harness import Harness
from rtp.datagen import sample_world, sample_spec, default_eyes
from rtp.loop import LoopConfig, run_episode
from rtp.brain import ScriptedBrain
from rtp.attribution_eval import episode_story

h = Harness(ROOT / "harness_files")
SEED = 42                       # deterministic seed with a fatal, Eye-attributed rule failure
w = sample_world(SEED)
spec = sample_spec(w, SEED)
res = run_episode(w, default_eyes(SEED, True), ScriptedBrain(), spec, h, LoopConfig())

print(episode_story(w, spec, res))

def stage_of(f):
    c = f["cause"]
    if c in ("out_of_reach", "not_found", "no_space"):
        return "environment"
    if c in ("arm_noise", "slip", "crush"):
        return "arm"
    if c.startswith("rule_"):
        return "rule"
    return {"Eye": "eye", "Brain": "brain", "Arm": "arm"}.get(f["agent"], "brain")

print()
print("fatal failures (stage <- real agent/cause from loop.attribute + final check):")
for f in res.failures:
    if f.get("fatal"):
        print("  ", stage_of(f), "<-", f["agent"], f["cause"], "| obj:", f.get("obj"))''')

nb.md("""## 2. Counterfactual: does changing the Eye change the outcome?

`rtp.credit.eye_credit` replays the same seed with common random numbers, once with the real reports and
once with Eye A's reports replaced (`empty` = silent, `oracle` = perfect). If replacing the Eye's reports
flips failure to success, the failure is *counterfactually* attributable to that Eye's information. This is
a mechanistic check on the attribution, not human ground truth.""")

nb.code('''from rtp.credit import eye_credit

def run_fn(hook):
    w2 = sample_world(SEED)
    spec2 = sample_spec(w2, SEED)
    return run_episode(w2, default_eyes(SEED, True), ScriptedBrain(), spec2, h, LoopConfig(), eye_hook=hook)

base = run_fn(None)
print("real run: success =", base.success, "| score =", round(base.score, 3))
for mode in ("empty", "oracle"):
    c = eye_credit(run_fn, "A", mode)
    print(f"replace Eye A with {mode:6s}: base success={c['base_success']} "
          f"score={c['base']:.3f} -> cf success={c['cf_success']} score={c['cf']:.3f} "
          f"| credit={c['credit']:+.3f}")
    if not c["base_success"] and c["cf_success"]:
        print("   counterfactual: a perfect Eye A turns this failure into a success")
    else:
        print("   counterfactual: outcome unchanged; this failure is not eye-repairable alone")''')

nb.md("""## 3. Sample failed episodes for a human labelling pass

`run_config(..., keep_story=True)` attaches `episode_story` to each record. `sample_for_labeling` keeps only
failed episodes that carry a story and writes a CSV with `config, seed, auto_agents, human_agent, story`;
`human_agent` is intentionally **empty** — a person must fill it with one of `Eye | Brain | Arm | Mixed |
Unclear`. If the smoke sample happens to contain no failure, the skill-error knob `ExpConfig.flaky` is
raised and the sample is redrawn, with the change printed.""")

nb.code('''from rtp.experiments import ExpConfig, run_config
from rtp.attribution_eval import sample_for_labeling, write_label_csv, agreement

LABEL_SEEDS = tuple(range(12))
flaky = 0.5
cfg = ExpConfig("attribution_sample", noisy_eyes=True, flaky=flaky)
recs = run_config(cfg, LABEL_SEEDS, h, keep_story=True)
if not any(not r["success"] for r in recs):
    flaky = 0.8
    print("no failure in the sample -> raised the skill knob ExpConfig.flaky 0.5 ->", flaky)
    cfg = ExpConfig("attribution_sample", noisy_eyes=True, flaky=flaky)
    recs = run_config(cfg, LABEL_SEEDS, h, keep_story=True)

n_fail = sum(1 for r in recs if not r["success"])
label_rows = sample_for_labeling(recs, n=min(4, n_fail), seed=0)
label_path = prof.paths.reports / "attribution_labels.csv"
write_label_csv(label_rows, label_path)
print("label CSV:", label_path)
print("episodes:", len(recs), "| failures:", n_fail, "| rows to label:", len(label_rows))

import csv
with open(label_path) as f:
    first = list(csv.DictReader(f))[:2]
for row in first:
    print({k: (v[:90] + "..." if k == "story" and len(v) > 90 else v) for k, v in row.items()})''')

nb.md("""## 4. SIMULATED labelling — pipeline check only, **not** a human result

**This cell is explicitly a simulation.** It copies each auto label into the `human_agent` column so that
`agreement(path)` has something to score; the resulting accuracy and kappa are 1.0 **by construction** and
say nothing about attribution quality. The real CSV above still has empty `human_agent` cells and is the
file a person must fill in. Nothing from this cell may be reported as a paper number.""")

nb.code('''# SIMULATED LABELS: mirrors rtp.attribution_eval._auto_label, for a metric smoke test only.
def simulated_human(auto_agents):
    parts = [p for p in auto_agents.split("|") if p]
    return parts[0] if len(parts) == 1 else ("Mixed" if parts else "Unclear")

with open(label_path) as f:
    rows_sim = list(csv.DictReader(f))
for r in rows_sim:
    r["human_agent"] = simulated_human(r["auto_agents"])

sim_path = prof.paths.reports / "attribution_labels_simulated.csv"
with open(sim_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["config", "seed", "auto_agents", "human_agent", "story"])
    writer.writeheader()
    writer.writerows(rows_sim)

a = agreement(sim_path)
print("SIMULATED agreement:", a)
print("interpretation: accuracy =", round(a.get("accuracy", float("nan")), 3),
      "and kappa =", round(a.get("cohen_kappa", float("nan")), 3),
      "are 1.0 by construction (auto labels copied into human_agent). NOT a measured attribution result.")''')

nb.md("""## 5. `table_attribution` on the real records

`rtp.tables.table_attribution` reads the same records written by the grid run:

* **Config** — the `ExpConfig.name` that produced the records.
* **failed episodes** — records with `success == False`.
* **with Eye / with Brain / with Arm** — counts of failed episodes whose `fatal_agents` set (built from
  failures flagged `fatal` by `loop.attribute` and the final rule check) contains that agent. One episode
  can appear in more than one column because `fatal_agents` is a set; the columns are not a partition.

The table has no `rule`/`environment` columns: those stages come from the story + stage mapping in
section 1 and, for a real number, from the human labelling pass in section 3.""")

nb.code('''from rtp.tables import table_attribution

print(table_attribution(recs))
print()
print("fatal agents by record:", [(r["seed"], r["fatal_agents"]) for r in recs if r["fatal_agents"]])''')

nb.md("""## Numbers the paper still needs from a human labelling pass

1. **Cohen's κ and raw agreement** of `loop.attribute` against human `Eye | Brain | Arm | Mixed | Unclear`
   labels, computed by `agreement()` on the *filled* CSV — plus a Wilson interval on accuracy and a
   bootstrap interval on κ. The simulated κ above is a pipeline smoke test only.
2. **The full confusion matrix** (auto agent → human label), including how often rule-stage failures are
   called Brain by the auto labels and how often `Mixed` / `Unclear` is the human verdict.
3. **Per-stage prevalence with intervals** across failed episodes: the share truly attributable to eye,
   brain, arm, rule and environment, stratified by config and by held-out split, with the paired n.
4. **Inter-annotator agreement** between at least two human raters on a subset, so the human labels
   themselves have a reliability number; report disagreements and how they were resolved.
5. **Counterfactual validation rate**: for failures labelled eye, the fraction where an oracle/empty Eye
   replay (section 2) actually flips the outcome — the share with a mechanistic as well as a human basis.
6. **Provenance**: number of episodes labelled, annotator IDs, date, labeler instructions, the exact CSV
   path and the record/config hashes behind each row.

Until those exist, attribution is a rule-based analysis tool with a demonstrated workflow, not a validated
scientific claim. No Isaac Lab/Newton or hardware attribution result is present in this notebook.""")

path = nb.save()
print("wrote", path)
