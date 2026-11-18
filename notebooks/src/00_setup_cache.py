#!/usr/bin/env python3
"""Generate notebooks/00_setup_cache.ipynb — run once per machine or Kaggle session."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ops.nbbuild import Notebook, install_cell  # noqa: E402

nb = Notebook("00_setup_cache")

nb.md("""# 00 — setup, credentials, and the per-profile asset cache

Run this once per machine or Kaggle session. It sets the **only** knob (`ACTIVE`), reports what the
box can actually do, and resolves the Qwen3.5-4B weights through the real order:

**local cache → attached Kaggle dataset → our private Kaggle dataset → our private HF repo.**

It then prints the resume point (the newest checkpoint in HF for this profile) and the datasets to
attach next time, so a rerun downloads nothing.

Everything lands inside `models/<profile>/`, `data/<profile>/`, `runs/<profile>/`: two profiles never
share an adapter, a checkpoint, a log or a cache.""")

nb.code(install_cell(["huggingface_hub", "kaggle", "torch"]))

nb.code('''ACTIVE = "local_4050"          # local_4050 | local_4090 | kaggle_t4x2 | pro6000 | tpu_v5e8
import os, sys
from pathlib import Path

os.environ["RTP_PROFILE"] = ACTIVE
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("profile:", ACTIVE, "| repo:", ROOT)''')

nb.code('''import platform, subprocess

KAGGLE = Path("/kaggle/input").is_dir()
print("kaggle:", KAGGLE, "| python:", platform.python_version(), "|", platform.platform())
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                      "--format=csv,noheader"], capture_output=True, text=True)
print(gpu.stdout.strip() or "(no nvidia-smi: this box will not train)")''')

nb.md("""## The profile is the single source of truth

`ACTIVE` above selects dtype, attention backend, tuning mode, batch geometry, `max_len` and every
path. Capability claims are **resolved**, not assumed: a 4050 asking for FlashAttention-2 on a Turing
card degrades to fp16 + sdpa *with a reason* rather than crashing mid-run.""")

nb.code('''from rtp.profiles import detect_caps, get_profile, resolve_runtime

prof = get_profile(ACTIVE)
prof.ensure_dirs()
caps = detect_caps()
attn, dtype, notes = resolve_runtime(prof, caps)
print("runs:", prof.paths.runs, "| data:", prof.paths.data, "| models:", prof.paths.models)
print("config hash:", prof.config_hash(), "| effective batch:", prof.effective_batch,
      "| max_len:", prof.max_len)
print("tuning:", prof.tuning, "| attention:", attn, "| dtype:", dtype)
for note in notes:
    print(" -", note)''')

nb.md("""## Credentials and the two remotes

`from_env()` reads `.env` (CRLF stripped — a token pasted with Windows line endings is otherwise
rejected as an illegal HTTP header), sets the Kaggle proxy this network requires, and builds both
clients. It never prints a token. HF-only work can skip Kaggle, whose auth through the proxy is
intermittent.""")

nb.code('''from rtp.remotes import from_env, kaggle_proxy

try:
    store = from_env()
    print("hf repo:", store.hf_repo)
    print("kaggle user:", store.kaggle_owner())
except Exception as exc:
    store = from_env(with_kaggle=False)          # HF still works without Kaggle
    print("kaggle unavailable:", type(exc).__name__, "- HF-only for now")
    print("hf repo:", store.hf_repo)
print("kaggle proxy:", kaggle_proxy())''')

nb.md("""## Resolve the base model

One call, four sources, and the result is manifest-verified before it is returned. On Kaggle the
model is **attached** as a private dataset and read from `/kaggle/input`, so it never occupies the
working directory.""")

nb.code('''from rtp import assetcache as ac

MODEL_REV = "851bf6e8"        # HF commit whose bytes the manifest hashes pin
spec = ac.model_spec("Qwen/Qwen3.5-4B", revision=MODEL_REV, name="qwen35-4b")
dest = ac.asset_dir(spec, prof)

try:
    model_dir = store.resolve(spec, prof, attached="/kaggle/input" if KAGGLE else None)
    problems = ac.verify_manifest(model_dir, spec)
    print("model:", model_dir)
    print("manifest ok:", not problems, "| total GB:", round(ac.read_manifest(model_dir).get("total_bytes", 0) / 1e9, 2))
except Exception as exc:
    model_dir = None
    print("SKIPPED: model not resolvable here:", type(exc).__name__, exc)
    print("expected cache:", dest)
    print("attach this dataset to the notebook:", ac.attach_hint(spec))''')

nb.md("""## What is already trained (the resume point)

Training pushes a checkpoint to HF every N steps and prunes older ones. Before starting any run, ask
for the newest step; `rtp.brain_train.run_sft` does this automatically.""")

nb.code('''for stage in ("sft", "grpo"):
    try:
        step = store.latest_checkpoint(prof.name, stage)
    except Exception as exc:
        step = f"unknown ({type(exc).__name__})"
    print(f"{stage:5s} latest remote step:", step)''')

nb.md("""## Offline wheelhouse (what makes a Kaggle run install in seconds)

The wheelhouse is the profile's package set resolved once for these exact tags. Attaching it means
`ops/install.sh --wheelhouse` never touches the network.""")

nb.code('''tags = ac.env_tags()
wheels = ac.wheelhouse_spec(python=tags["python"], torch=tags.get("torch", "?"),
                            cuda=tags.get("cuda", "?"), sm=tags.get("sm", "?"))
mounted = Path("/kaggle/input") / wheels.slug
if mounted.is_dir():
    print("wheelhouse attached:", mounted)
    print("install offline with: ops/install.sh --profile", ACTIVE, "--wheelhouse", mounted)
else:
    print("no wheelhouse attached; expected dataset:", wheels.slug)
    print("build it once for these tags:", tags)''')

nb.md("""## Fast-path assertion (GPU only)

Qwen3.5's gated DeltaNet layers must bind the real `causal_conv1d` + `fla` kernels; the pure-torch
fallback is correct but far slower, and every throughput number in the paper depends on this.""")

nb.code('''RUN_FASTPATH = True
if model_dir is None:
    print("SKIPPED: no model")
elif not caps.cuda:
    print("SKIPPED: no CUDA (those kernels are GPU-only)")
elif not RUN_FASTPATH:
    print("SKIPPED: RUN_FASTPATH is off")
else:
    r = subprocess.run([sys.executable, "ops/g2_fastpath.py", "--model", str(model_dir),
                        "--out", str(prof.paths.reports / "g2_fastpath.json")],
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    assert r.returncode == 0, "DeltaNet fast path is not bound; see the output above"''')

nb.md("""## What next

* `10_sft_brain` — supervised warm-start on teacher-validated plans (resumable).
* `20_grpo_brain` — GRPO with the symbolic rewards and a KL term.
* `30_eye_training` — Eyes (conventional CV): accuracy vs credit reward.
* `40_world_and_datagen` — datasets for every stage, published as private Kaggle datasets.
* `50_isaac_newton_eval` — Isaac Lab 3.0 + Newton evidence.
* `60_harness_zero_shot_cot` — the Brain's own harness, zero-shot and CoT.
* `70_hindsight_memory` — hindsight relabelling and success/failure memory RAG.
* `80_evals_omni_comparison` — experiment grid and the competitor harness.
* `90_attribution_paper_tables` — failure attribution and the paper's tables.

Anything this notebook wrote is inside the profile tree above. Nothing outside it is touched.""")

path = nb.save()
print("wrote", path)
