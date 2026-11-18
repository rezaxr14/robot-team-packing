#!/usr/bin/env python3
"""Generate notebooks/50_isaac_newton_eval.ipynb (see docs/NOTEBOOK_API.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ops.nbbuild import Notebook, install_cell  # noqa: E402

nb = Notebook("50_isaac_newton_eval")

nb.md('''# 50 — Isaac Lab 3.0 + Newton evidence, and what is not measured

**G-3 is a cube-fall render/step test on the 4050, not a manipulation scene.** It proves that a Newton-only
Isaac Lab install steps physics (`newton_mjwarp`) and that the Newton Warp renderer returns rgb and depth with
the expected shapes and dtypes for a falling cube. It does **not** contain two FR3 arms, a kitchen, PTZ eyes,
a grasp, or a task.

This notebook reads the evidence that actually exists, shows how the gate is re-run in its own venv, and runs
the only episode-level evaluation available on this box: the symbolic/pose fixtures through `SimWorld` +
`FakeSim`. That result is explicitly labelled as a fixture, never as Isaac physics.
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
print("reports:", paths.reports, "| data:", paths.data)
print("config hash:", profile.config_hash(), "| max_len:", profile.max_len,
      "| effective batch:", profile.effective_batch)
print("resolved attention:", attn, "| resolved dtype:", dtype)
for note in notes:
    print(" -", note)
''')

nb.md('''## The G-3 evidence that exists

The JSON below is the recorded output of `ops/g3_sim.py`, read from disk — every value printed comes from that
file, not from this notebook. The checks are the gate's own assertions (physics stepped, rgb/depth contracts,
non-blank frame, geometry hit, frames differ). A missing report is reported as `SKIPPED`, never replaced with
an assumed value.
''')

nb.code('''import json

report_path = ROOT / "reports" / "g3_sim.json"
if not report_path.exists():
    g3 = None
    print("SKIPPED: no recorded G-3 report at", report_path)
else:
    g3 = json.loads(report_path.read_text())
    print("result:", g3.get("result"), "| physics:", g3.get("physics"),
          "| renderer:", g3.get("renderer"), "| device:", g3.get("device"))
    print("versions:", g3.get("versions"))
    print("checks:", g3.get("checks"))
    print("steps:", g3.get("steps"), "| steps/s:", g3.get("steps_per_s"),
          "| seconds:", g3.get("seconds"))
    if g3.get("error"):
        print("error:", str(g3["error"])[:300])
''')

nb.code('''from IPython.display import Image, display

png_dir = ROOT / "reports" / "g3_sim"
pngs = sorted(png_dir.glob("*.png")) if png_dir.is_dir() else []
print("recorded PNGs:", [p.name for p in pngs] or "(none)")
if pngs:
    display(Image(filename=str(pngs[0]), width=480))
else:
    print("SKIPPED: no recorded PNG to display")
''')

nb.md('''## Detect the Isaac environment without importing it here

Isaac Lab lives in a **separate venv** (`$ISAAC_PY`, default `~/rtp_work/venvs/isaac_newton/bin/python`) with
its own Python and its own torch/CUDA build. Nothing in this kernel imports `isaaclab`: a subprocess only asks
that venv whether the package is discoverable. If the venv is missing or `isaaclab` is not importable there,
the cell prints `SKIPPED` and everything else still runs on CPU.

Re-running the gate is also a subprocess for the same reason: mixing two torch builds in one interpreter is a
reliable way to get a crash that says nothing about the gate. `RUN_GATE` is off by default; the command that
would run is always printed so a re-run is an explicit, recorded action.
''')

nb.code('''import importlib.util, subprocess

ISAAC_PY = os.environ.get("ISAAC_PY") or str(Path.home() / "rtp_work" / "venvs" / "isaac_newton" / "bin" / "python")
isaac_py = Path(ISAAC_PY).expanduser()
if not isaac_py.is_file():
    ISAAC_OK = False
    print("SKIPPED: Isaac Lab venv not available (no python at", str(isaac_py), ")")
else:
    probe = subprocess.run(
        [str(isaac_py), "-c",
         "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('isaaclab') else 1)"],
        capture_output=True, text=True, timeout=120,
    )
    ISAAC_OK = probe.returncode == 0
    if ISAAC_OK:
        print("isaac python:", isaac_py,
              "| isaaclab importable in that venv (not imported in this kernel)")
    else:
        print("SKIPPED: Isaac Lab venv not available (isaaclab not importable there)")
        print("probe stderr:", probe.stderr.strip()[:300])
''')

nb.code('''RUN_GATE = False
gate_out = paths.reports / "g3_sim.json"
gate_images = paths.reports / "g3_sim"
cmd = [str(isaac_py), "ops/g3_sim.py", "--out", str(gate_out), "--images", str(gate_images)]
print("gate command:", " ".join(cmd))
if not RUN_GATE:
    print("SKIPPED: gate not re-run (RUN_GATE is False; set True to run it in the Isaac venv)")
elif not ISAAC_OK:
    print("SKIPPED: gate not re-run (Isaac Lab venv not available)")
else:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False, timeout=600)
    print("return code:", r.returncode)
    print((r.stdout or r.stderr)[-2000:])
''')

nb.md('''## The available episode-level evaluation: symbolic / pose fixtures

`rtp.simworld.SimWorld` mirrors a backend's object **poses** into the symbolic `World`, so the loop, rules and
rewards see one interface. `rtp.fakesim.FakeSim` is that backend: it decides with the symbolic world but
returns poses, grip peaks and held flags, forcing the pose -> region -> World path to run. It is an
integration-test double, **not physics**.

The success counts and Wilson intervals below are therefore **symbolic/pose fixture results, NOT Isaac
physics**. They show the loop and dataset machinery work on this CPU box; they say nothing about contacts,
grasp fidelity or an Isaac Lab scene.
''')

nb.code('''from rtp.datagen import sample_world, sample_spec, default_eyes
from rtp.fakesim import FakeSim
from rtp.simworld import SimWorld
from rtp.loop import run_episode, LoopConfig
from rtp.brain import ScriptedBrain
from rtp.harness import Harness
from rtp.stats import wilson, summarize_binary

harness = Harness(ROOT / "harness_files")
SEED0, EPISODES = 5000, 12
flags, detail = [], []
for seed in range(SEED0, SEED0 + EPISODES):
    world = SimWorld(FakeSim(sample_world), seed=seed)
    spec = sample_spec(world, seed)
    res = run_episode(world, default_eyes(seed, True), ScriptedBrain(), spec, harness, LoopConfig())
    flags.append(bool(res.success))
    detail.append((seed, bool(res.success), res.replans, res.msg_tokens))

print("backend:", type(world.backend).__name__, "| world:", type(world).__name__)
print("episodes (seed, success, replans, msg_tokens):", detail)
summary = summarize_binary(flags)
lo, hi = wilson(summary["k"], summary["n"])
print("success:", summary["k"], "/", summary["n"], "| rate:", round(summary["rate"], 3))
print("Wilson 95% interval:", (round(lo, 3), round(hi, 3)))
print("classification: symbolic/pose fixture results, NOT Isaac physics")
''')

nb.md('''## What remains for the paper — none of it may be claimed until run

* **Two FR3 + PTZ eyes scene**: Isaac Lab 3.0 + Newton with two compliant Franka Research 3 arms, Franka
  Hands and two PTZ eye models. No such scene exists; G-3 is a cube-fall test on the 4050.
* **Camera contracts**: rgb/depth shape, dtype and eye-tool contract for the kitchen and for each eye station
  (plus segmentation from our own CV, since Newton Warp renders only rgb/depth). Only the single cube camera
  contract has been recorded; no eye or manipulation camera result exists.
* **T4 numbers**: the G-3 T4 half and every `kaggle_t4x2` throughput/success figure. No T4 session has been
  recorded, so all of those numbers are unverified.

None of the above may be written into a table, abstract or claim until it has actually been run with a recorded
config hash and seed. The gate command printed above is how the first of those runs is reproduced; everything
else needs its own new scene and its own measurement.
''')

nb.save()
