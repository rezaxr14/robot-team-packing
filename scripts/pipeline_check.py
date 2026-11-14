"""Run the symbolic pipeline over a config matrix and write a CSV with Wilson CIs.

PIPELINE CHECK ONLY. The Brain here is a scripted fixture and the world is symbolic, so these numbers are
NOT evidence for any claim in the paper. Use it to confirm the plumbing (harness, comm modes, attribution,
credit) works before swapping in the real Eye/Brain/Arm models.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtp.harness import Harness
from rtp.datagen import sample_world, sample_spec, default_eyes
from rtp.brain import ScriptedBrain
from rtp.loop import LoopConfig, run_episode
from rtp.stats import summarize_binary

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
SEEDS = range(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
rows = []
for noisy in (False, True):
    for mode in ("push", "pull", "init_only"):
        for harness_on in (True, False):
            flags, toks, att = [], [], {}
            for s in SEEDS:
                w = sample_world(s)
                cfg = LoopConfig(comm_mode=mode, use_harness=harness_on, static_rule_ids=("leave_distractors",))
                r = run_episode(w, default_eyes(s, noisy), ScriptedBrain(), sample_spec(w, s), H, cfg)
                flags.append(r.success)
                toks.append(r.msg_tokens)
                for f in r.failures:
                    if f.get("fatal"):
                        att[f["agent"]] = att.get(f["agent"], 0) + 1
            sm = summarize_binary(flags)
            rows.append({"noisy_eyes": noisy, "comm": mode, "harness": harness_on, **sm,
                         "mean_msg_tokens": sum(toks) / len(toks), "fatal_by_agent": att})
out = Path("pipeline_check.csv")
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out} ({len(rows)} rows); PIPELINE CHECK ONLY, not evidence")
