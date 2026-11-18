#!/usr/bin/env python3
"""Build ``notebooks/10_sft_brain.ipynb`` — supervised warm-start of the Brain.

    $V/bin/python notebooks/src/10_sft_brain.py

The notebook is the paper's SFT stage: teacher-validated plans become ``{prompt, completion}`` rows and
``rtp.brain_train.run_sft`` trains LoRA/QLoRA adapters on them, resumably. Every hardware knob comes
from the profile, so the same notebook is a two-step smoke run on the 4050 and the real run on a
4090/T4 box.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.nbbuild import Notebook, install_cell  # noqa: E402

nb = Notebook("10_sft_brain")

# --------------------------------------------------------------------------------------------------
nb.md(r"""# 10 — SFT: supervised warm-start of the Brain

The Brain is the planner: it reads the instruction, the merged Eye reports and the retrieved harness
notes, and writes one plan as JSON. Before any policy-gradient work it must produce *plans that parse
at all*, because a group-relative method cannot learn from a rollout that is not JSON. SFT is that
warm start: the scripted teacher (`rtp.brain.ScriptedBrain`) solves episodes in the symbolic world,
only the episodes it actually solved become rows, and the model is trained to reproduce the teacher's
plan for the exact prompt it saw. GRPO (notebook 20) then optimises *quality* against those plans.

Why the teacher is a legitimate source here, and what it is not:

* a row's completion is a plan the closed loop *executed successfully* (`gen_brain_sft` drops failed
  episodes), so the target is not one person's opinion of a good plan;
* the teacher is a rule application, not a policy that generalises, so SFT learns the plan *shape*
  and the rule order. Nothing here claims a success-rate gain over the base model.

**The 4050 constraint is measured, not assumed (D-38).** `ops/seq_ceiling.py` ran one forward+backward
per length at batch 1 on the 6 GB card: 128 → 4.94 GB, 256 → 5.10, 384 → 5.26, **512 → 5.42 (fits)**,
768 and 1024 OOM, and 1024 still OOMs at `lora_r=8`, so the wall is the hybrid stack's per-token
activation state rather than the adapter. `local_4050.max_len` is therefore 512, and the data cells
below print what that budget does to real rows instead of assuming it is enough: a teacher plan is
longer than 512 tokens, so on this profile the trainer keeps the completion and drops the prompt. The
4050 can smoke-test the training path; it cannot train the task.

Everything below runs top-to-bottom. A step that needs hardware this box does not have prints
`SKIPPED: <reason>` and the notebook continues; no cell prints a number it did not measure.""")

# --------------------------------------------------------------------------------------------------
nb.md(r"""## Dependencies

Only the non-stdlib, non-`rtp` packages this notebook imports. The install cell checks with
`importlib.util.find_spec` and installs what is missing, so on the training box it is a no-op and on a
bare box it fails with pip's own message instead of a traceback at `import torch`.""")

nb.code(install_cell(["torch", "transformers", "peft", "datasets"]))

# --------------------------------------------------------------------------------------------------
nb.md(r"""## The one knob

`ACTIVE` is the only setting typed into this notebook. It picks a profile from `rtp.profiles`, and the
profile decides dtype, attention backend, tuning mode, batch geometry, `max_len`, LoRA targets and
every output directory. Two profiles never share a cache, adapter, checkpoint or log, which is what
makes "run this on the 4050 tonight, continue on the T4 tomorrow" safe.

The cell also fixes the working directory to the repo root: `nbrun` may start in `notebooks/`, and
`Harness("harness_files")` resolves relative to the repo.""")

nb.code(r'''ACTIVE = "local_4050"   # local_4050 | local_4090 | kaggle_t4x2 | pro6000 | tpu_v5e8
import os, sys
from pathlib import Path

os.environ["RTP_PROFILE"] = ACTIVE
# The same allocator setting ops/sft_smoke.py uses on this 6 GB card; must be set before torch loads.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("profile:", ACTIVE, "| repo:", ROOT)
print("python:", sys.executable)''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""## What this box can actually do

`resolve_runtime` does not echo the profile's wishes back at you: it compares them with
`detect_caps()` and returns the backend and dtype that will really be used, plus a note for every
place it had to disagree. A 4050 asking for bf16 + FlashAttention-2 gets them (Ada, cc 8.9); a T4
asking for the same gets fp16 + SDPA and the reason. Printing `config_hash()` here matters because
that hash keys the adapters and the dataset spec, so two runs with the same hash are the same
experiment and an edited profile is visibly a different one.""")

nb.code(r'''from rtp.profiles import detect_caps, get_profile, resolve_runtime

profile = get_profile(ACTIVE)
profile.ensure_dirs()
caps = detect_caps()
attn, dtype, notes = resolve_runtime(profile, caps)

print(f"profile {profile.name}: {profile.gpu}, {profile.vram_gb} GB x {profile.n_gpu}, tuning={profile.tuning}")
print(f"geometry: micro {profile.micro_batch} x accum {profile.grad_accum} on {profile.n_gpu} device(s)"
      f" = effective batch {profile.effective_batch}")
print(f"max_len: {profile.max_len} | lora_r: {profile.lora_r} | grad_ckpt: {profile.grad_ckpt}")
print(f"config hash: {profile.config_hash()}")
print(f"runtime on this box: attn={attn} dtype={dtype} (cuda={caps.cuda}, cc={caps.cc})")
if notes:
    for note in notes:
        print(" reason:", note)
else:
    print(" reasons: none - the profile's request is supported as written here")
print("paths:", profile.paths.as_dict())''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""## Storage: the model snapshot

Weights are resolved through the profile cache, never through a path typed into a notebook: local
cache → attached Kaggle dataset → our private Kaggle dataset → our private HF repo. `resolve` returns
a directory whose `manifest.json` matches the pinned revision, so a half-written or wrong-revision
snapshot is refused instead of loaded. The call below passes `attached="/kaggle/input"` so a Kaggle
session reads the attached dataset and never fills its ~20 GB working directory.

Kaggle TLS through the local proxy is intermittent here, so a failure to build the Kaggle client is
handled by falling back to HF only. The repo id is read from the authenticated account
(`store.hf_repo`), never hard-coded, and no token is printed. A cold cache is filled deliberately by
`00_setup_cache.ipynb`; here a miss just prints `SKIPPED` with the exception type. A remote step is
only a resume point if it contains `trainer_state.json`, so the cell probes the newest one before
using it — an adapter-only publish under the same prefix must not be mistaken for a training
checkpoint.""")

nb.code(r'''from rtp import assetcache as ac
from rtp.remotes import from_env

try:
    store = from_env()
    print("storage: Kaggle + HF | hf_repo:", store.hf_repo)
except Exception as exc:
    print(f"storage: Kaggle client unavailable ({type(exc).__name__}); retrying HF-only")
    try:
        store = from_env(with_kaggle=False)
        print("storage: HF only | hf_repo:", store.hf_repo)
    except Exception as exc2:
        store = None
        print(f"storage: none ({type(exc2).__name__}: {exc2}) - resolve/publish calls will be skipped")

MODEL_REV = "851bf6e8"       # pinned HF commit; the manifest hashes pin the bytes
spec = ac.model_spec("Qwen/Qwen3.5-4B", revision=MODEL_REV, name="qwen35-4b")

model_dir = None
if store is None:
    print("SKIPPED: model resolution - no storage client (see above)")
else:
    try:
        model_dir = store.resolve(spec, profile, attached="/kaggle/input")
        manifest = ac.read_manifest(model_dir) or {}
        print("model:", model_dir)
        print(f"  {manifest.get('source')} @ {manifest.get('revision')} | "
              f"{manifest.get('total_bytes')} bytes | deep_verified={manifest.get('deep_verified')}")
    except Exception as exc:
        print(f"SKIPPED: model not resolvable for {spec.slug}: {type(exc).__name__}: {exc}")
        print("  expected cache:", ac.asset_dir(spec, profile))

from rtp.checkpoints import fetch_latest

remote_step = None
resume_ok = False
if store is not None:
    try:
        remote_step = store.latest_checkpoint(profile.name, "sft")
        print("latest sft checkpoint:", remote_step)
        if remote_step is None:
            print("  none yet: run_sft starts from step 0")
            resume_ok = True                       # a fresh run has nothing stale to fetch
        else:
            # An adapter-only publish can share the checkpoint prefix. Probe the real bytes before
            # letting run_sft fetch it: a checkpoint without trainer_state.json cannot be resumed.
            fetched, path = fetch_latest(store, profile.name, "sft",
                                         profile.paths.ckpt / "sft" / "_probe")
            resume_ok = path is not None and (path / "trainer_state.json").is_file()
            print(f"  step {fetched} resumable (trainer_state.json present): {resume_ok}")
            if resume_ok:
                print("  run_sft fetches that step and continues at the next micro-batch")
            else:
                print("  adapter-only remote artifact: this smoke starts at step 0 instead")
    except Exception as exc:
        print(f"SKIPPED: resume point lookup: {type(exc).__name__}: {exc}")''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""## The SFT data, generated for real

`gen_brain_sft` is not a sampler of static text: for each candidate seed it builds a world and a task
spec, runs a full episode with the scripted teacher, and keeps the prompt and the teacher's first plan
**only if the episode succeeded**. Seeds that fail are skipped, so the yield (rows asked for vs seeds
tried) is a measurement of how hard the teacher's own distribution is, and the row count below is what
came out, not what was requested.

`write_jsonl` writes `{prompt, completion, meta}` into the profile's data directory: same file, same
name, byte-identical on a rerun because the seeds are fixed. One row is printed truncated, from the
real file row, not a template.""")

nb.code(r'''from rtp.datagen import gen_brain_sft, write_jsonl
from rtp.harness import Harness

MAX_ROWS = 64
SEED0 = 0
harness = Harness("harness_files")
print(f"harness: {len(harness.docs)} docs, types {sorted({d.type for d in harness.docs.values()})}")

rows = gen_brain_sft(MAX_ROWS, SEED0, harness)
DATA = profile.paths.data / "brain_sft.jsonl"
DATA.parent.mkdir(parents=True, exist_ok=True)
write_jsonl(rows, DATA)
print(f"asked for {MAX_ROWS} rows, got {len(rows)} -> {DATA} ({DATA.stat().st_size} bytes)")

if rows:
    example = rows[0]
    print("first row meta:", example["meta"])
    print("--- prompt (first 700 chars of", len(example["prompt"]), ") ---")
    print(example["prompt"][:700])
    print("--- completion (first 600 chars of", len(example["completion"]), ") ---")
    print(example["completion"][:600])
else:
    print("SKIPPED: seed window produced no successful teacher episode; no example to show")''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""### How long are those rows, against the budget

This is the cell that decides whether the 4050 can train on this data at all, so it uses the real
tokenizer when a model snapshot resolved. Without one it falls back to whitespace splitting and says
so: a word count is a lower bound on tokens for English prose and a bad one for JSON, where every
brace and quote is a token.""")

nb.code(r'''import statistics

tokenizer = None
if model_dir is not None:
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print("tokenizer:", type(tokenizer).__name__, "from", Path(model_dir).name)
    except Exception as exc:
        print(f"SKIPPED: tokenizer load: {type(exc).__name__}: {exc}")
else:
    print("SKIPPED: no resolved model - counts below are a whitespace proxy (words, not tokens)")

def count(text):
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return len(text.split())          # proxy: whitespace words, labelled as such everywhere below

unit = "tokens" if tokenizer is not None else "whitespace words (proxy)"
totals = [count(r["prompt"]) + count(r["completion"]) + 1 for r in rows]     # +1 for EOS
completions = [count(r["completion"]) + 1 for r in rows]
if totals:
    print(f"whole row, {unit}: min {min(totals)} | median {statistics.median(totals)} | max {max(totals)}")
    print(f"completion, {unit}: min {min(completions)} | median {statistics.median(completions)} | "
          f"max {max(completions)}")
    over = sum(1 for t in totals if t > profile.max_len)
    print(f"rows longer than profile.max_len={profile.max_len}: {over} of {len(totals)}")''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""### What `max_len` does to a row that is too long

`tokenize_pairs` is the function `run_sft` uses, so calling it here shows the real truncation policy
rather than describing it: the completion is kept whole when it fits, and prompt tokens are dropped
from the *left* (the tail of the prompt is what the answer conditions on). If the completion alone
fills the budget there is no room left for the prompt, and the row becomes "write this plan with no
instruction in front of it" — still a valid training signal for format, useless for instruction
following. That is why the 4050 is a smoke box and the reportable SFT runs use a profile with
headroom.""")

nb.code(r'''from rtp.trainer import tokenize_pairs

if tokenizer is None or not rows:
    print("SKIPPED: no tokenizer or no rows, so the truncation policy cannot be shown on real ids")
else:
    example = tokenize_pairs(rows[:1], tokenizer, max_len=profile.max_len)
    if not example:
        print("row 0 produced nothing to train on (empty completion after tokenisation)")
    else:
        ids = example[0]["input_ids"]
        scored = sum(1 for t in example[0]["labels"] if t != -100)
        prompt_ids = len(tokenizer(rows[0]["prompt"], add_special_tokens=False)["input_ids"])
        print(f"row 0: {prompt_ids} prompt tokens + {completions[0]} completion tokens "
              f"-> {len(ids)} kept for max_len={profile.max_len}")
        print(f"  scored (completion) tokens: {scored} | prompt tokens kept: {len(ids) - scored}")''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""## The run

`run_sft` is the whole stage in one call: fetch the newest remote checkpoint for `profile/stage`,
load the base in the profile's tuning mode (QLoRA here: 4-bit nf4 + LoRA, LoRA targets filtered to the
modules the checkpoint really has), tokenise, train, save a resumable checkpoint every `every` steps
and push it, then write the final adapters. The numbers below are small but real — `SMOKE` means
"two optimizer steps over a handful of rows", not "pretend".

`stop_after` is the session boundary: the trainer calls `on_checkpoint` after every optimizer step,
which saves/pushes on the `every` boundary and then raises `StopTraining` when `state.step >=
stop_after`; `fit` catches it, logs where it stopped (epoch and micro-batch) and returns the state.
The target step is derived from the *verified* resume point printed above, so the smoke always takes
real steps even when a previous session already pushed a checkpoint; if that remote step was only an
adapter publish, the smoke runs with `store=None` and starts at step 0 while the publish cell still
uses the real store.

If this box cannot hold the run, the exact torch exception is printed and no loss is invented. The
failure is informative on a 6 GB card: the 4-bit base plus one 512-token step is the measured ceiling,
and it is the *training* step that does not fit, not the adapter.""")

nb.code(r'''import math
import torch
from rtp.brain_train import run_sft

SMOKE = True                       # real but tiny; False = MAX_ROWS rows, 3 epochs, no step cap
TRAIN_ROWS = 32 if SMOKE else MAX_ROWS
EVERY = 50                         # checkpoint + push cadence; not reached by this smoke
BASE_STEP = remote_step if resume_ok else 0   # only a verified resume point is a base step
SPE = max(1, math.ceil(TRAIN_ROWS / max(1, profile.micro_batch * profile.grad_accum)))
EPOCHS = 1.0 if not SMOKE else (BASE_STEP + 3) / SPE   # enough window to take two NEW steps
STOP_AFTER = None if not SMOKE else BASE_STEP + 2
print("smoke plan: rows", TRAIN_ROWS, "| epochs", round(EPOCHS, 3), "| stop_after", STOP_AFTER,
      "| steps/epoch", SPE)

result = None
if not rows:
    print("SKIPPED: SFT - no data rows generated")
elif model_dir is None:
    print("SKIPPED: SFT - no resolved model snapshot (see the storage cell)")
elif not profile.supported:
    print(f"SKIPPED: SFT - profile {profile.name} is a stub (supported=False)")
elif not caps.cuda:
    print("SKIPPED: SFT - no CUDA on this box; a CPU pass over 4 B parameters is not a smoke test")
else:
    free, total = torch.cuda.mem_get_info()
    print(f"gpu free {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB before loading "
          f"({profile.tuning} base + LoRA on {profile.gpu})")
    try:
        result = run_sft(profile, model_dir=model_dir, data=rows,
                         store=store if resume_ok else None, stage="sft",
                         epochs=EPOCHS, lr=1e-4, every=EVERY, keep=2, stop_after=STOP_AFTER,
                         max_rows=TRAIN_ROWS, seed=0,
                         chunk_tokens=32 if SMOKE else None,   # smaller logits chunk on a shared GPU
                         log=print)
    except Exception as exc:                       # no CUDA memory / no checkpoint / no weights
        message = str(exc).splitlines()[0] if str(exc) else ""
        print(f"SKIPPED: SFT did not run on this box - {type(exc).__name__}: {message}")
        print("  no loss is reported because no optimizer step was taken")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"resumed_from {result.resumed_from} | checkpoints in {result.out_dir}")''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""## What came out

Only two things are worth showing: the loss history (one record per optimizer step, with the learning
rate actually used, from `TrainState.history`) and the adapters. `run_sft` already wrote the adapters
in PEFT format at `profile.paths.adapters/sft`; publishing them is a separate step because a directory
only leaves this machine after its manifest verifies, so the spec is
`ac.adapters_spec(profile, "sft", profile.config_hash())` — the hash in the slug is what stops a
4090 adapter from being mistaken for a 4050 one.""")

nb.code(r'''if result is None:
    print("SKIPPED: no training ran in this session, so there is no loss history")
    print("  an empty history is not a loss of 0 - nothing was measured")
else:
    for record in result.state.history:
        print(record)
    print(f"steps {result.state.step} | best loss {result.state.best_loss:.6f} | "
          f"epoch {result.state.epoch} | tokens seen {result.state.tokens_seen} | "
          f"resumed_from {result.resumed_from}")''')

nb.md(r"""## Publish the adapters

`store.publish` refuses an unverified directory, so the manifest is written first. The call asks for
both remotes; each available remote location is printed on success, and a network or credential
failure is caught and printed with its reason, because losing the Kaggle proxy must not lose the
run.""")

nb.code(r'''adapters = profile.paths.adapters / "sft"

if result is None:
    print("SKIPPED: adapters and publish - run_sft writes them only when it actually trains")
else:
    print("adapters:", adapters)
    print("  files:", sorted(p.name for p in adapters.iterdir())[:10])
    aspec = ac.adapters_spec(profile, "sft", profile.config_hash())
    ac.write_manifest(adapters, aspec)
    print("adapter spec:", aspec.slug, "| verify:", ac.verify_manifest(adapters, aspec) or "ok")
    if store is None:
        print("SKIPPED: publish - no storage client")
    else:
        try:
            published = store.publish(adapters, ac.adapters_spec(profile, "sft", profile.config_hash()))
            print("published:", published)
            print("remote locations:", {k: v for k, v in published.items() if k != "slug"})
        except Exception as exc:
            print(f"SKIPPED: publish failed ({type(exc).__name__}): {str(exc).splitlines()[0]}")''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""## Resume on another machine

`run_sft` is resumable by design, so "resume" is not a special code path — it is the same notebook run
again. The only state that has to travel is the newest pushed checkpoint, and it travels through our
HF repo. `run_sft` fetches it **automatically** via `rtp.checkpoints.fetch_latest` before any weight
is loaded: `store.latest_checkpoint(profile.name, "sft")` gives the step and
`store.download_checkpoint(profile.name, "sft", step, dest)` hands that exact directory to
`trainer.fit(resume_from=...)`. To inspect the resume point from a shell (no notebook needed):

```bash
$V/bin/python -c "from rtp.remotes import from_env; s = from_env(with_kaggle=False); print(s.hf_repo, s.latest_checkpoint('local_4050', 'sft'))"
```

Then rerun this notebook with `$V/bin/python ops/nbrun.py notebooks/10_sft_brain.ipynb` (`jupyter
nbconvert`'s CLI is broken in this venv; `ops/nbrun.py` is the working headless runner). Same
`ACTIVE`, same `config_hash`, so it continues the experiment rather than starting a new one.

**Ending a session on purpose** is `StopTraining` via `stop_after`: the `on_checkpoint` callback first
lets `CheckpointPolicy` save-and-push on the `every` boundary and then raises `StopTraining` when
`state.step >= stop_after`; `fit` catches it and reports `stopped on request at step N (epoch e, m
micro-batches in)`. Resume is *exact* because batch order is a pure function of `(seed, epoch)`: the
same permutation is rebuilt and the first `state.micro_in_epoch` micro-batches are skipped, and the
checkpoint carries the optimizer state, torch RNG state, learning-rate progress and history. A stop
that lands on a checkpoint step resumes to the micro-batch; a stop between steps keeps the adapters and
history but restarts from the previous pushed step — set `stop_after` to a multiple of `every` when
the intent is "end here, continue tomorrow". A failed push does not lose the run either: the local
checkpoint is written first and the push error is logged.

**The measured gate is `ops/sft_smoke.py`**, not this notebook's smoke. It proves the real-weight,
mid-epoch resume condition (G-14):

```bash
$V/bin/python ops/sft_smoke.py --profile local_4050 --rows 64 --steps 4 --stop-after 2
```

It exits 0 only when steps were actually taken, losses are finite, the checkpoint round-tripped and
the second session resumed at the pushed step, and it writes the peak VRAM and wall time to
`reports/sft_smoke.json`.""")

# --------------------------------------------------------------------------------------------------
nb.md(r"""## What this notebook proves, and what it does not

**Proves (when the training cell did not skip).** The SFT path end to end on the active profile:
teacher-validated data generated from fixed seeds, a real tokenised dataset with a printed length
distribution against the profile's measured `max_len`, a real optimizer trajectory (`TrainState.history`
with losses and the LR the trainer applied), PEFT adapters on disk under a config-hashed spec, and a
checkpoint in HF that a different machine can fetch. The chat/JSON plan format of Qwen3.5-4B is
*learnable* on this data in the sense that the loss falls over the run.

**Does not prove.** No success rate, no comparison against the base model, and no claim about the
reportable profiles: SFT on `local_4050` truncates the prompt away (the cell above measures how many
rows), so its loss is a format loss. `reports/sft_*.json` and the evaluation notebooks are where a
delta with intervals belongs. Nothing here has run on a Kaggle T4, so every `kaggle_t4x2` number is
unverified. The teacher is `ScriptedBrain`, a rule application in the symbolic world
(`rtp/world.py` + `rtp/loop.py`), not a frontier model and not Isaac Lab physics — the data says what
the teacher does, not what a good planner would do.""")

if __name__ == "__main__":
    print("wrote", nb.save())
