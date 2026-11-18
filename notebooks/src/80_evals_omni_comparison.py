#!/usr/bin/env python3
"""Generate notebooks/80_evals_omni_comparison.ipynb — experiment grid + competitor harness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ops.nbbuild import Notebook, install_cell  # noqa: E402

nb = Notebook("80_evals_omni_comparison")

nb.md("""# 80 — the experiment grid, and comparison with other omni models

Two different things live here, and they must not be confused:

1. **Our ablation grid** (`default_grid()` / `verify_grid()`), run on the deterministic symbolic fixture with
   the reference planner. This gives real success rates, Wilson intervals and paired bootstrap differences
   for *our harness* — not for the trained VLM.
2. **An omni-model comparison table plus a plug-in harness.** Unless a competitor is actually served and run,
   every competitor row is `not run` with no benchmark numbers. The harness cell checks
   `RTP_COMPARE_URL` and prints `SKIPPED` when it is unset, so the notebook never fabricates a comparison.

The paper's publishable comparison requires identical prompts, seeds, harness, decoding and n≥30 per cell
with intervals; the small smoke grid below is deliberately short of that and says so.""")

nb.code(install_cell(["matplotlib"]))

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
print("this grid needs CPU only: it runs rtp.loop on the symbolic fixture, not Isaac Lab/Newton")''')

nb.md("""## 1. The grid and every knob

`default_grid()` is the ablation matrix; `verify_grid()` uses the `FlakyBrain` fixture to test the
plan-verification ladder. `ExpConfig` is the single place these knobs live — printing `asdict(cfg)` shows
exactly what each config changes, so a table cell can be traced back to a config.""")

nb.code('''from dataclasses import asdict

from rtp.harness import Harness
from rtp.experiments import (ExpConfig, default_grid, verify_grid, run_config,
                             write_records, read_records, summarize, paired_compare)

h = Harness(ROOT / "harness_files")
GRID = default_grid(noisy=True) + verify_grid(flaky=0.5)
print("configs:", len(GRID))
for c in GRID:
    print(f"  {c.name}: {asdict(c)}")''')

nb.md("""## 2. Run the CPU-feasible configs, summarise, compare

`run_config` produces one record per (config, seed); `write_records` persists them under
`profile.paths.reports`, and `read_records` reads that same file back so the tables and figures are built
from what was actually written. `summarize()` uses Wilson intervals for success and plan-pass;
`paired_compare("full", ablation)` pairs on the seeds shared by both configs and bootstraps the difference.

`SEEDS` below is a **small but real** smoke sample (n=8 per cell). n≥30 per cell is required before any of
these intervals belongs in the paper.""")

nb.code('''SEEDS = tuple(range(8))          # smoke sample: 8 paired seeds per config
print("seeds:", SEEDS, "| n per cell:", len(SEEDS),
      "| publishable n>=30:", len(SEEDS) >= 30)

all_records = []
for cfg in GRID:
    all_records.extend(run_config(cfg, SEEDS, h))
rec_path = prof.paths.reports / "eval_records.jsonl"
write_records(all_records, rec_path)
records = read_records(rec_path)
print("wrote and re-read:", rec_path, "| records:", len(records),
      "| configs:", len({r["config"] for r in records}))

summary = summarize(records)
for name, d in summary.items():
    sc = d["success"]
    pp = d["plan_pass"]
    print(f"{name:24s} success {sc['k']:2d}/{sc['n']:2d} = {100*sc['rate']:5.1f}% "
          f"[{100*sc['lo']:5.1f}, {100*sc['hi']:5.1f}] | "
          f"plan-pass {100*pp['rate']:5.1f}% [{100*pp['lo']:5.1f}, {100*pp['hi']:5.1f}]")''')

nb.md("""## 3. Paired differences against `full`

A positive mean diff means `full` beat the ablation on those paired seeds. The interval is a paired
bootstrap; with n=8 it is wide and is shown for pipeline validation, not as evidence.""")

nb.code('''ABLATIONS = [c.name for c in default_grid(noisy=True) if c.name != "full"]
for name in ABLATIONS:
    c = paired_compare(records, "full", name)
    if not c["n"]:
        print(f"full - {name}: no paired seeds")
        continue
    print(f"full - {name:18s} n={c['n']:2d} mean={100*c['mean_diff']:+6.1f} pp "
          f"[{100*c['lo']:+6.1f}, {100*c['hi']:+6.1f}]")
for c in verify_grid(flaky=0.5):
    s = paired_compare(records, "full", c.name)
    if s["n"]:
        print(f"full - {c.name:18s} n={s['n']:2d} mean={100*s['mean_diff']:+6.1f} pp "
              f"[{100*s['lo']:+6.1f}, {100*s['hi']:+6.1f}]")''')

nb.md("""## 4. Paper tables and figures from the same records

`table_main` and `table_ablation` are generated by `rtp.tables` from these records; no cell types a
number by hand. The two figures are written into `profile.paths.reports/figs/` by `rtp.figures` and shown
inline from the saved PNGs.""")

nb.code('''from rtp.tables import table_main, table_ablation

print(table_main(records))
print()
print(table_ablation(records, "full"))''')

nb.code('''from rtp.figures import fig_success, fig_gap
from IPython.display import Image, display

figdir = prof.paths.reports / "figs"
figdir.mkdir(parents=True, exist_ok=True)
p_success = fig_success(records, figdir / "fig_success.png")
p_gap = fig_gap(records, figdir / "fig_gap.png")
print("figures:", p_success, "|", p_gap)
display(Image(filename=p_success))
display(Image(filename=p_gap))''')

nb.md("""## 5. Omni-model comparison — honest status

The models below are the ones the plan actually names as baselines (§10: our ablations; π0/RDT/ACT/DP
fine-tuned on our own demonstrations; a frontier model as a zero-shot Brain; base Qwen3.5-4B with identical
tools). Metadata that this repo has **not** verified is labelled as such rather than guessed.

**Status semantics**: `measured here` applies only to our own harness/planner (the rows actually run in
sections 2-4). `not run` means exactly that — no competitor was served, no competitor number exists here.
The row for our trained Qwen3.5-4B VLM is also `not run`: no served checkpoint was available on this box.
`ScriptedBrain` is the deterministic reference planner used for grid mechanics; its success rates are not a
claim about the VLM's performance.""")

nb.code('''COLUMNS = ("model", "params", "vision", "license", "how it is run", "status")
U = "not verified here; check the source model card"
ROWS = [
    ("Our harness — ScriptedBrain reference", "n/a (fixture)", "n/a", "repo code",
     "rtp.loop.run_episode on the symbolic fixture; this notebook", "measured here"),
    ("Our Brain — Qwen3.5-4B (trained adapter)", "4B dense [plan §2]", "yes (VLM) [plan §2]",
     "Apache-2.0 [plan §2]", "rtp.brain.LLMBrain via rtp.llm_client, same prompt/decoding; needs served ckpt",
     "not run (no served checkpoint here)"),
    ("Qwen3.5-4B base (identical tools)", "4B dense [plan §2]", "yes (VLM) [plan §2]",
     "Apache-2.0 [plan §2]", "same prompt/tools with harness off; fine-tune not performed", "not run"),
    ("pi0", U, U, U, "fine-tune on our own sim demos (adapter not set up)", "not run"),
    ("RDT", U, U, U, "fine-tune on our own sim demos (adapter not set up)", "not run"),
    ("ACT", U, U, U, "fine-tune on our own sim demos (adapter not set up)", "not run"),
    ("DP (diffusion policy)", U, U, U, "fine-tune on our own sim demos (adapter not set up)", "not run"),
    ("DP3", U, U, U, "symbolic-skill / RoboTwin adapter (not set up)", "not run"),
    ("Frontier model as zero-shot Brain", U, U, U,
     "Kaggle-hosted teacher as Brain via rtp.llm_client (needs skill + CLI auth)", "not run"),
]
print("| " + " | ".join(COLUMNS) + " |")
print("|" + "---|" * len(COLUMNS))
for r in ROWS:
    print("| " + " | ".join(r) + " |")
print()
print("benchmark numbers reported for competitors in this notebook:", 0)''')

nb.md("""## 6. Competitor plug-in harness — clearly marked, and skipped when unconfigured

This is the only place a competitor could be run. It uses `rtp.llm_client.ChatClient`, an OpenAI-compatible
`/v1/chat/completions` client, wrapped as `rtp.brain.LLMBrain` through `client.generate`. The same
`ExpConfig("full")` knobs, the same `SEEDS`, the same `Harness` and `temperature=0.0` decoding are used so
that any numbers it printed would be comparable in form. With `RTP_COMPARE_URL` unset it prints `SKIPPED`
and reports no numbers. The optional `RTP_COMPARE_MODEL` and `RTP_COMPARE_KEY` come from the environment;
no token is printed.""")

nb.code('''COMPARE_URL = os.environ.get("RTP_COMPARE_URL")
if not COMPARE_URL:
    print("SKIPPED: RTP_COMPARE_URL is unset — no competitor was run; no comparison numbers are reported.")
else:
    from rtp.llm_client import ChatClient

    client = ChatClient(base_url=COMPARE_URL,
                        model=os.environ.get("RTP_COMPARE_MODEL", "default"),
                        api_key=os.environ.get("RTP_COMPARE_KEY", "EMPTY"))
    # same prompt (rtp.brain.build_prompt, via LLMBrain) and same decoding as our reference run
    generate = client.generate(system="You are the Brain. Output only the plan JSON.",
                               temperature=0.0, max_tokens=1024)
    competitor_cfg = ExpConfig("competitor", brain="llm", noisy_eyes=True)
    comp_records = run_config(competitor_cfg, SEEDS, h, generate=generate)
    comp_path = prof.paths.reports / "eval_records_competitor.jsonl"
    write_records(comp_records, comp_path)
    comp_summary = summarize(comp_records)
    sc = comp_summary["competitor"]["success"]
    print("competitor endpoint:", COMPARE_URL.rsplit("/", 1)[0])
    print(f"competitor success {sc['k']}/{sc['n']} = {100*sc['rate']:.1f}% "
          f"[{100*sc['lo']:.1f}, {100*sc['hi']:.1f}] (n={len(SEEDS)} smoke, not publishable)")
    print("records:", comp_path)''')

nb.md("""## What a publishable comparison requires

* **Same prompts**: the competitor must receive exactly `rtp.brain.build_prompt(ctx)` (same header, scene
  serialization, instruction text and retrieved snippets) through `LLMBrain`, not a hand-written prompt.
* **Same seeds and harness**: identical `sample_world` / `sample_spec` / `default_eyes` draws and the same
  `Harness` docs, retrieval settings and loop config; paired on the same seed list.
* **Same decoding**: temperature, top-p, max tokens and thinking-token cap fixed and recorded for every
  model (`RTP_NO_THINK=1` for Qwen-style templates if used). A different budget is a different experiment.
* **n≥30 per cell**, with Wilson intervals on success and a paired bootstrap on the per-seed difference;
  report the paired n and the interval, not just a point estimate.
* **Provenance**: config hash (`prof.config_hash()`), git SHA, model revision, prompt/decoding settings and
  the exact record file for every table cell; frozen held-out splits so no eval seed leaks into SFT data.
* **No absent results**: unrun models stay `not run`. Simulation results from the symbolic fixture must be
  labelled as such and not presented as Isaac Lab/Newton or hardware results.""")

path = nb.save()
print("wrote", path)
