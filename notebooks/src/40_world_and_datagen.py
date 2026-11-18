#!/usr/bin/env python3
"""Generate notebooks/40_world_and_datagen.ipynb (see docs/NOTEBOOK_API.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ops.nbbuild import Notebook, install_cell  # noqa: E402

nb = Notebook("40_world_and_datagen")

nb.md('''# 40 — the symbolic world and every dataset the paper needs

This notebook produces the **fixtures**, not results. It samples the symbolic world, checks that a seed
reproduces the same scene, generates the five JSONL datasets the training/eval stages consume, and publishes
the directory with a content-hash revision. No GPU, Isaac Lab, teacher model or checkpoint is required here.

Only `ACTIVE` is a knob; dtype, paths, batch geometry and `max_len` come from the profile. Everything written
lands under `profile.paths.data`.
''')

nb.code(install_cell([]))

nb.code('''ACTIVE = "local_4050"          # local_4050 | local_4090 | kaggle_t4x2 | pro6000 | tpu_v5e8
import os, sys
from pathlib import Path

os.environ["RTP_PROFILE"] = ACTIVE
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("profile:", ACTIVE, "| repo:", ROOT)
''')

nb.code('''from rtp.profiles import get_profile, detect_caps, resolve_runtime

profile = get_profile(ACTIVE)
paths = profile.ensure_dirs()
attn, dtype, notes = resolve_runtime(profile, detect_caps())
print("data:", paths.data, "| reports:", paths.reports)
print("config hash:", profile.config_hash(), "| max_len:", profile.max_len,
      "| effective batch:", profile.effective_batch)
print("resolved attention:", attn, "| resolved dtype:", dtype)
for note in notes:
    print(" -", note)
''')

nb.md('''## The symbolic world is a fixture, not a physics result

`sample_world(seed)` and `sample_spec(world, seed)` build the same scene from the same seed every time. This is
the *symbolic* fixture used for rewards, the Eye/Brain/Arm loop and dataset generation — **not** a simulation
result. The rule from `rtp/world.py` is explicit: "It is NOT a physics result and must never be reported as
one." No Isaac Lab scene, renderer or controller produced the table below; the pose-level sim path is exercised
separately in `50_isaac_newton_eval`.
''')

nb.code('''from rtp.datagen import sample_world, sample_spec

SEED = 1234
w = sample_world(SEED)
spec = sample_spec(w, SEED)
print(f"scene seed={SEED}  objects={len(w.objs)}")
print(f"{'name':12s} {'class':11s} {'mass':>6s} {'fragility':>9s} {'region':13s}")
for name, o in w.objs.items():
    print(f"{name:12s} {o.cls:11s} {o.mass:6.3f} {o.fragility:9.3f} {o.region:13s}")
print("\\nTaskSpec:", spec.to_dict())
print("instruction:", spec.text())
''')

nb.code('''def scene_signature(seed):
    """Everything that defines a sampled scene, as plain comparable values."""
    world = sample_world(seed)
    task = sample_spec(world, seed)
    objects = [(o.name, o.cls, o.mass, o.fragility, o.size, o.region, o.friction, o.is_distractor)
               for o in world.objs.values()]
    return objects, task.to_dict()

a = scene_signature(SEED)
b = scene_signature(SEED)
assert a == b, "sample_world/sample_spec are not deterministic for the same seed"
print("determinism: two independent draws of seed", SEED, "are equal -> assertion passed")
c = scene_signature(SEED + 1)
print("a different seed gives a different scene:", a != c)
''')

nb.md('''## Five datasets, generated with real counts

Each generator is run here at a small but real size; nothing is stubbed.

* `gen_brain_sft` keeps only episodes the scripted teacher solved in the closed loop.
* `gen_grpo_prompts` emits `prompt` + `scene_json` + `spec_json` for the symbolic reward functions.
* `gen_eye_sft` emits SCENE-JSON targets; `gen_eye_grpo_rows` emits Eye-GRPO prompts.
* `gen_hindsight_sft` starts from **failed** episodes, relabels them and validates each by replay. Its `stats`
  is the yield funnel printed below: episodes seen -> failures -> relabeled -> replay-validated.

The `image` field and every `renders/...` path are placeholders; see the Isaac section at the end.
''')

nb.code('''import json
from rtp.harness import Harness
from rtp.datagen import (gen_brain_sft, gen_grpo_prompts, gen_eye_sft,
                         gen_eye_grpo_rows, gen_hindsight_sft)

harness = Harness(ROOT / "harness_files")
N = 48            # one row per sampled scene
EYE_N = 24        # two views (A and B) per seed -> 48 Eye rows
SEED0 = 10000
brain = gen_brain_sft(N, SEED0, harness)
grpo = gen_grpo_prompts(N, SEED0, harness)
eye = gen_eye_sft(EYE_N, SEED0)
eye_grpo = gen_eye_grpo_rows(EYE_N, SEED0)
hindsight, funnel = gen_hindsight_sft(N, SEED0 + 5000, harness)

DATASETS = {
    "brain_sft": brain,
    "grpo_prompts": grpo,
    "eye_sft": eye,
    "eye_grpo_rows": eye_grpo,
    "hindsight_sft": hindsight,
}

def preview(row, limit=260):
    s = json.dumps(row, default=str)
    return s if len(s) <= limit else s[:limit] + " ... (truncated)"

for name, rows in DATASETS.items():
    example = preview(rows[0]) if rows else "(none)"
    print(f"{name:16s} rows={len(rows):4d}  example: {example}")
print("\\nhindsight yield funnel:", funnel)
''')

nb.code('''from rtp.datagen import write_jsonl

FILE_NAMES = {}
for name, rows in DATASETS.items():
    path = paths.data / f"{name}.jsonl"
    write_jsonl(rows, path)
    FILE_NAMES[name] = f"{name}.jsonl"
    print("wrote", path, "| bytes:", path.stat().st_size)
''')

nb.md('''## Publishing: content-addressed, read-only on Kaggle

The revision is a hash of the *data bytes* (sorted file name + bytes), so a rerun with the same rows resolves
to the same slug. `ac.data_spec_kind` gives a profile-free dataset spec; `ac.write_manifest` records the exact
bytes and `store.publish` follows the storage order local -> attached Kaggle dataset -> private Kaggle dataset
-> private HF repo. Tokens are read from the environment and never printed.

On Kaggle, a dataset is **attached to the notebook read-only at `/kaggle/input`**. Training reads its data
from that mount, so generated data and downloads never fill Kaggle's small `/kaggle/working` directory; that
is why publishing is a dataset operation and not a write into the working tree. If either remote is
unconfigured or unreachable, the local files above are still valid and the failure is printed, not hidden.
''')

nb.code('''from rtp.remotes import from_env

try:
    store = from_env()
    print("hf repo:", store.hf_repo)
    print("kaggle client:", "present" if store.kaggle is not None else "absent")
except Exception as exc:
    store = None
    print("SKIPPED: remotes unavailable:", type(exc).__name__, str(exc)[:200])
    print("datasets remain local under", paths.data)
''')

nb.code('''import hashlib
from rtp import assetcache as ac

digest = hashlib.sha256()
for name in sorted(FILE_NAMES):
    blob = (paths.data / FILE_NAMES[name]).read_bytes()
    digest.update(name.encode())
    digest.update(blob)
content_hash = digest.hexdigest()

spec = ac.data_spec_kind("symbolic", "world-and-datagen",
                         revision=content_hash[:8], source=f"produced by {profile.name}")
manifest = ac.write_manifest(paths.data, spec, deep=True)
problems = ac.verify_manifest(paths.data, spec)
print("content hash:", content_hash)
print("spec slug:", spec.slug, "| manifest:", manifest.name, "| verify problems:", problems)

if store is None:
    print("SKIPPED: publish needs a remote store")
else:
    try:
        where = store.publish(paths.data, spec)
        print("kaggle:", where.get("kaggle"))
        print("hf:", where.get("hf"))
    except Exception as exc:
        print("SKIPPED: publish failed:", type(exc).__name__, str(exc)[:300])
''')

nb.md('''## What changes when Isaac Lab renders real images

The Eye datasets here contain **paths, not pixels**: `gen_eye_sft` sets `image` to
`"renders/seed<seed>_view<view>.png"`, and `gen_eye_grpo_rows` embeds the same `renders/...` placeholder in its
prompt. Those strings are a contract between generation and rendering, not observations.

To make them real:

* Render each `(seed, view)` from the *same* symbolic world state in Isaac Lab 3.0 + Newton and replace the
  `image` field with the rendered frame (or a pointer to the stored frame) that the camera actually returned.
* Check the Newton Warp camera contract (rgb `uint8` `(N,H,W,3)`, depth `float32` `(N,H,W,1)`) before accepting a
  frame; G-3 only recorded this for a cube, not for a kitchen or an eye station.
* Replace the oracle SCENE-JSON targets with object poses from the sim bridge / detector output when the two
  FR3 + PTZ-eyes scene exists, and record which seed, station, zoom and occluders produced each frame.

Until those steps are run, every Eye row remains a placeholder. No rendered-image accuracy number may be
claimed from this notebook.
''')

nb.md('''## What is and is not established here

Established by cells above: the symbolic world is seed-reproducible; the five datasets are real JSONL files with
the printed row counts and hindsight yield funnel; the published slug is pinned to the data hash. Not
established: any manipulation success, any rendered-image accuracy, and any Isaac Lab or T4 result. Those need
their own recorded runs.
''')

nb.save()
