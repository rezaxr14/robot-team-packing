#!/usr/bin/env python3
"""Generate notebooks/70_hindsight_memory.ipynb — hindsight relabelling + outcome-split memory RAG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ops.nbbuild import Notebook, install_cell  # noqa: E402

nb = Notebook("70_hindsight_memory")

nb.md("""# 70 — hindsight relabelling and failure/success memory RAG

A failed episode is not wasted data. **Hindsight relabelling** rewrites the instruction to describe what the
team actually achieved (which items were packed, which constraints held), keeps the executed plan as a
positive example, and validates it by **replay** before it enters SFT data.

The same episode stream is also a **memory store split by outcome**:

* **successes** teach what a working plan for this kind of scene looks like;
* **failures** carry the attributed cause, so a later Brain can retrieve "what went wrong last time" before it
  commits to a plan or while it recovers.

Retrieval quality is itself a measured quantity in this project: `summarize()` reports
`mean_retrieval_recall`, and the benchmark ablations compare harness-on against no-RAG. Everything below runs
on the deterministic **symbolic fixture** (`rtp.datagen` + `rtp.loop`), not on Isaac Lab/Newton; no GPU,
checkpoint or network access is needed, and no benchmark number is claimed for the trained VLM.""")

nb.code(install_cell(["transformers"]))

nb.code('''ACTIVE = "local_4050"          # local_4050 | local_4090 | kaggle_t4x2 | pro6000 | tpu_v5e8
import os, sys
from pathlib import Path

os.environ["RTP_PROFILE"] = ACTIVE
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
LOCAL_SEED = Path.home() / "models" / "Qwen3.5-4B"   # local tokenizer seed (as in 00_setup_cache)
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
print("runtime: symbolic fixture on CPU is sufficient; the DeltaNet/Isaac paths are not exercised here")
if not caps.cuda:
    print("SKIPPED: no CUDA on this box — the served Qwen3.5 Brain and Isaac Lab/Newton are not run; "
          "the deterministic symbolic fixture is used instead")''')

nb.md("""## 1. Produce failures with the real loop

`ScriptedBrain` is the deterministic reference planner; `default_eyes(..., noisy=True)` gives the two Eyes
blind spots and measurement noise, so some scenes fail even with a good planner. Two further knobs exist in
the code and can be raised if a scan happens to find no failure:

* `ExpConfig.p_arm_noise` → `sample_world(..., p_arm_noise=...)` → `World.p_arm_noise` (arm slip);
* `ExpConfig.flaky` → `FlakyBrain(p_err=...)` (probability the planner emits a flawed plan).

The cell scans a seed window with noisy eyes, collects the failures, and — **only if the window has none** —
raises the skill-error knob `FLAKY` from 0.0 to 0.5 and rescans. What it changed is printed, not hidden.""")

nb.code('''from collections import Counter

from rtp.harness import Harness
from rtp.datagen import sample_world, sample_spec, default_eyes
from rtp.loop import LoopConfig, run_episode
from rtp.brain import ScriptedBrain
from rtp.fixtures import FlakyBrain

h = Harness(ROOT / "harness_files")
SEED0, N_SCAN = 1000, 40
NOISY_EYES = True
ARM_NOISE = 0.03       # ExpConfig.p_arm_noise
FLAKY = 0.0            # ExpConfig.flaky -> FlakyBrain p_err

def make_brain(seed, flaky):
    brain = ScriptedBrain()
    return FlakyBrain(brain, p_err=flaky, seed=seed) if flaky else brain

def scan_failures(seed0, n, flaky, p_arm_noise):
    out = []
    for s in range(seed0, seed0 + n):
        w = sample_world(s, p_arm_noise=p_arm_noise)
        spec = sample_spec(w, s)
        res = run_episode(w, default_eyes(s, NOISY_EYES), make_brain(s, flaky), spec, h, LoopConfig())
        if not res.success:
            out.append((s, w, spec, res))
    return out

failures = scan_failures(SEED0, N_SCAN, FLAKY, ARM_NOISE)
if not failures:
    FLAKY = 0.5
    print("no failure in the sample -> raised the skill knob FLAKY 0.0 ->", FLAKY,
          "(FlakyBrain p_err, the same knob as ExpConfig.flaky) and rescanned")
    failures = scan_failures(SEED0, N_SCAN, FLAKY, ARM_NOISE)

print("episodes scanned:", N_SCAN, "| noisy eyes:", NOISY_EYES, "| p_arm_noise:", ARM_NOISE,
      "| flaky:", FLAKY)
print("failures collected:", len(failures))
print("fatal agent of each failure:",
      dict(Counter(f["agent"] for _, _, _, res in failures for f in res.failures if f.get("fatal"))))''')

nb.md("""## 2. Feed the failures to the hindsight generator

`gen_hindsight_sft` re-runs deterministic seeds with noisy eyes, calls `relabel` on every failed episode,
and then **replays** the relabelled plan with `ReplayBrain` against the rewritten spec. Only rows whose
replay succeeds are returned. Its `stats` dict is the real funnel; the conversion rates printed below are
computed from that dict, not asserted.""")

nb.code('''from rtp.datagen import gen_hindsight_sft

N_ROWS, MAX_TRIES = 3, 150
rows, stats = gen_hindsight_sft(N_ROWS, seed0=SEED0, harness=h,
                                noisy_eyes=NOISY_EYES, max_tries=MAX_TRIES)
if stats["failed"] == 0:
    print("generator saw 0 failures -> widened the search (seed0 +=", N_SCAN,
          ", max_tries doubled) and reran")
    rows, stats = gen_hindsight_sft(N_ROWS, seed0=SEED0 + N_SCAN, harness=h,
                                    noisy_eyes=NOISY_EYES, max_tries=2 * MAX_TRIES)

print("funnel: failures seen -> relabelled -> replay-validated:",
      stats["failed"], "->", stats["relabeled"], "->", stats["validated"])
print("episodes examined:", stats["episodes"], "| SFT rows kept:", len(rows))
print("relabel rate:", round(stats["relabeled"] / max(1, stats["failed"]), 3),
      "| replay-validation rate:", round(stats["validated"] / max(1, stats["relabeled"]), 3))''')

nb.md("""## 3. One relabelled example

`relabel` keeps only items that were actually placed successfully, drops constraints that did not hold in
the final world, and rebuilds a spec whose `check_final` passes. The "reason" below is the concrete diff
between the original and rewritten instruction for one scanned failure — it is printed from the real
`TaskSpec` objects, not paraphrased.""")

nb.code('''from rtp.hindsight import relabel

example = None
for s, w, spec, res in failures:
    rec = relabel(w, res, spec)
    if rec is not None:
        example = (s, spec, rec)
        break

if example is None:
    # the generator itself always emits a verified example when it returns rows
    meta = rows[0]["meta"]
    print("no scanned episode was relabellable; example from gen_hindsight_sft seed", meta["seed"])
    print("original:", meta["orig_instruction"])
    print("new     :", meta["relabeled_instruction"])
else:
    s, orig_spec, rec = example
    print("seed:", s)
    print("original spec:", orig_spec.text())
    print("new spec     :", rec.instruction)
    print("reason (real spec diff):")
    print("  items kept :", rec.spec["items"], "(only successfully placed, non-crushed items)")
    for key, label in (("heavy_bottom", "heavy_bottom"), ("laptop_slot", "laptop_slot"),
                       ("bottle_upright_away", "bottle_upright"), ("last_item", "lunchbox_last")):
        if getattr(orig_spec, key) != rec.spec[key]:
            print(f"  {label}: {getattr(orig_spec, key)!r} -> {rec.spec[key]!r} (did not hold; dropped)")
    if orig_spec.leave != rec.spec["leave"]:
        print("  leave:", orig_spec.leave, "->", rec.spec["leave"], "(untouched items only)")
    print("  replay validation is performed by gen_hindsight_sft/ReplayBrain (funnel above)")''')

nb.md("""## 4. Outcome-split episode memory, persisted

Each episode becomes one note: the instruction, its outcome, and the fatal attribution when it failed.
The store is split by id prefix (`success:<seed>` / `failure:<seed>`) and written as JSONL under
`profile.paths.data` — nothing outside the profile tree is touched.

Retrieval uses `rtp.harness.BM25(texts, ids)` directly: constructor takes the note texts and their ids,
`.rank(query)` returns `(id, score)` sorted by score. We show the top successes and top failures for a new
task, with their real BM25 scores, then inspect the notes themselves.""")

nb.code('''import json

from rtp.harness import BM25

MEM_N = 20
store, notes_by_id = [], {}
for s in range(100, 100 + MEM_N):
    w = sample_world(s, p_arm_noise=0.10)
    spec = sample_spec(w, s)
    res = run_episode(w, default_eyes(s, True), ScriptedBrain(), spec, h, LoopConfig())
    fatal = [f for f in res.failures if f.get("fatal")]
    outcome = "success" if res.success else "failure"
    cause = "; ".join(f"{f['agent']}:{f['cause']}" for f in fatal) or "all rules held; every item packed"
    note = f"{spec.text()} | outcome={outcome} | {cause}"
    rid = f"{outcome}:{s}"
    store.append({"id": rid, "outcome": outcome, "seed": s, "note": note})
    notes_by_id[rid] = note

mem_path = prof.paths.data / "episode_memory.jsonl"
with open(mem_path, "w") as f:
    for r in store:
        f.write(json.dumps(r) + "\\n")
print("memory written:", mem_path)
print("rows:", len(store), "| successes:", sum(r["outcome"] == "success" for r in store),
      "| failures:", sum(r["outcome"] == "failure" for r in store))

bm = BM25([r["note"] for r in store], [r["id"] for r in store])
query = "heavy book at bottom, laptop in laptop slot, bottle upright and away from laptop"
ranked = bm.rank(query)
K = 3
top_successes = [(i, round(sc, 3)) for i, sc in ranked if i.startswith("success:")][:K]
top_failures = [(i, round(sc, 3)) for i, sc in ranked if i.startswith("failure:")][:K]
print("query:", query)
print("top successes (id, BM25 score):", top_successes)
print("top failures  (id, BM25 score):", top_failures)
for i, sc in top_failures:
    print("  ", i, sc, "->", notes_by_id[i][-110:])''')

nb.md("""## 5. Why outcome-split memory matters

* **Benchmark**: retrieval feeds the Brain context; `summarize()` reports `mean_retrieval_recall`, and the
  `full` vs `static_prompt` / no-RAG ablations measure what retrieval is worth (see notebook 80).
* **Recovery**: when a task blocks, the loop appends a replan reason to `BrainContext.notes`; retrieved
  failure notes add the concrete cause of a past failure to that same decision context.
* **Hindsight**: failures that cannot be recovered are exactly the episodes section 1-3 relabels into
  replay-validated positives, so the failure store and the hindsight queue are the same stream.
* **What the paper measures**: retrieval *quality* — recall of the needed rule/memory doc, success delta of
  retrieval-assisted recovery, and the attribution accuracy of the failure notes (notebook 90). A BM25
  ranking on a fixture corpus is a pipeline demonstration, not a claim about production retrieval.""")

nb.md("""## 6. Retrieved failure notes change a decision

`build_prompt` renders `BrainContext.snippets` under "## Retrieved notes". Below, the top-ranked failure
notes become `Doc(type="memory")` snippets, so the Brain sees them in the same prompt as the instruction
and the Eyes' scene. The prompt is printed truncated; its length is measured with the real Qwen3.5 tokenizer
from the local cache when available, otherwise with a clearly labelled characters/4 proxy.""")

nb.code('''from rtp.harness import Doc
from rtp.brain import BrainContext, build_prompt

failure_docs = [Doc(id=i, type="memory", keywords="past failure attribution", body=notes_by_id[i])
                for i, _ in top_failures]
ctx = BrainContext(spec=sample_spec(sample_world(100), 100), scene={}, rule_ids=set(),
                   snippets=failure_docs,
                   notes=["retrieved failure memory: do not repeat the attributed failure"])
prompt = build_prompt(ctx, reason="recover from a remembered failure")
print(prompt[:1200])
print("... [truncated] ... total chars:", len(prompt))

token_count, tokenizer_src = None, None
try:
    from transformers import AutoTokenizer
    from rtp import assetcache as ac
    tok_spec = ac.model_spec("Qwen/Qwen3.5-4B", revision="851bf6e8", name="qwen35-4b")
    for d in (ac.asset_dir(tok_spec, prof), LOCAL_SEED):
        if d and Path(d).is_dir() and (Path(d) / "tokenizer.json").exists():
            tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
            token_count = len(tok(prompt)["input_ids"])
            tokenizer_src = str(d)
            break
except Exception as exc:
    print("real tokenizer unavailable:", type(exc).__name__, exc)

if token_count is None:
    token_count = len(prompt) // 4
    print("token count ~", token_count, "(PROXY: chars/4, NOT a real tokenizer measurement)")
else:
    print("real tokenizer:", tokenizer_src)
    print("prompt tokens:", token_count)''')

nb.md("""### Measured here vs unverified

**Measured here**: the failure funnel, the relabel diff, the persisted JSONL store, BM25 scores over that
store, and the prompt length (real tokenizer or labelled proxy).

**Unverified**: these are deterministic symbolic episodes with a scripted reference planner. No trained
Qwen3.5 Brain, Isaac Lab/Newton rollout, Kaggle result or human judgement is involved, and none of the
numbers above is a benchmark result for the paper.""")

path = nb.save()
print("wrote", path)
