#!/usr/bin/env python3
"""Build notebooks/30_eye_training.ipynb: Eye data, the three reward conditions, and credit assignment."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ops.nbbuild import Notebook, install_cell

nb = Notebook("30_eye_training")

nb.md("""# 30 — Eyes: data, the three reward conditions, and credit assignment

The Eye is a **conventional computer-vision detector**: it consumes one rendered view and emits
**SCENE JSON**, a fixed schema with one row per object (`name`, `cls`, `region`, `size`,
`mass_guess`, `fragility_guess`, `occluded`, `confidence`). No language model sits in the eye path.
The paper's training question is not "can a VLM describe a table?", it is **which reward teaches a
detector to report the details the team actually needs**.

This notebook measures, on the symbolic fixture only:

1. the real data rows (`gen_eye_sft`, `gen_eye_grpo_rows`) and their SCENE-JSON targets;
2. `r_eye_format`, `r_eye_accuracy`, `r_eye_credit` on the same completions;
3. what the three GRPO conditions (`format,accuracy` / `format,credit` / `format,accuracy,credit`)
   would assign to those completions;
4. the runtime TRL config that an Eye GRPO run would use — and why it is skipped here;
5. the viewpoint/occlusion mechanism: a blind view loses credit even when field accuracy stays
   perfect.

**Not measured here:** any metric that needs real pixels, a trained detector, or an Isaac Lab scene
with two FR3 arms and PTZ eyes. Image paths are placeholders until the renderer fills them. No Eye
GRPO run has been recorded yet.
""")

nb.code(install_cell(["torch", "trl"]))

nb.code("""ACTIVE = "local_4050"          # local_4050 | local_4090 | kaggle_t4x2 | pro6000 | tpu_v5e8
import os, sys
from pathlib import Path

os.environ["RTP_PROFILE"] = ACTIVE
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from rtp.profiles import detect_caps, get_profile, resolve_runtime

prof = get_profile(ACTIVE)
attn, dtype, notes = resolve_runtime(prof, detect_caps())
print("profile:", prof.name, "| accel:", prof.accel, "| gpu:", prof.gpu)
print("dtype:", dtype, "| attention:", attn, "| max_len:", prof.max_len)
print("micro_batch:", prof.micro_batch, "| grad_accum:", prof.grad_accum,
      "| effective batch:", prof.effective_batch)
for note in notes:
    print(" -", note)
""")

nb.md("""## What an Eye row is

`gen_eye_sft` writes the SFT pair: an `image` path plus a `target` string that is exactly the SCENE
JSON the detector must learn to emit. `gen_eye_grpo_rows` writes the GRPO prompt plus the columns
the reward adapters need (`seed`, `view`). The prompt mentions an image file, but **the file is a
placeholder**: `gen_eye_sft` has no renderer, so it records the path the Isaac Lab renderer will
write later. This notebook never reads pixels and never pretends it did.

Two views (A, B) exist because the paper's mechanism needs occlusion: each default Eye is blind to
one table region, and the other Eye is supposed to cover it.
""")

nb.code("""import json
from rtp.datagen import gen_eye_grpo_rows, gen_eye_sft, sample_world

SMOKE = 2
sft = gen_eye_sft(SMOKE, 1000)
grpo = gen_eye_grpo_rows(SMOKE, 1000)
print("SFT rows:", len(sft), "| GRPO rows:", len(grpo), "(one row per seed x view)")

row = sft[0]
target = json.loads(row["target"])
print("first SFT row image:", row["image"], "| file exists:", Path(row["image"]).is_file())
print("target rows:", len(target))
print(json.dumps(target[:2], indent=1, sort_keys=True))
print()
print("GRPO rows (the image is still a placeholder):")
for r in grpo[:2]:
    print(json.dumps(r, sort_keys=True))
print()
print("renders/ exists on this box:", Path("renders").is_dir(),
      "-> no pixels are available to this notebook")
""")

nb.md("""## The three signals

`rtp/eye_rewards.py` defines three functions over the same parsed completion:

* **format** — did the text parse as a SCENE JSON list with every required field? This is a gate,
  not a perception score.
* **accuracy** — field-level F1 against ground truth: class and region exact, mass within 15%,
  fragility within 0.15. It treats every visible field alike and penalises hallucinated objects.
* **credit** — the paper's contribution. Replay the same symbolic episode twice with common random
  numbers: once with *this* report as the Eye's output, once with the same Eye silent. Return
  `score(with report) - score(silent)`. A detail earns credit only if changing it changes the team
  outcome, so a wrong laptop class can cost more than a wrong pencil-case fragility even though
  uniform accuracy barely moves.

Credit is measured on `rtp/world.py` (a symbolic stand-in), not on Newton physics. That is why the
mechanism is testable today while the Isaac Lab result is still pending.
""")

nb.code("""import rtp.eye_rewards as er
from rtp.eye_rewards import parse_scene_completion, r_eye_accuracy, r_eye_credit, r_eye_format
from rtp.harness import Harness
from rtp.perception import default_blind, oracle_report

H = Harness("harness_files")
SEED = 3
world = sample_world(SEED)
truth = oracle_report(world, "A").content["objects"]          # full ground truth, both views see all objects here
valid = json.dumps(truth)
wrong = json.dumps([dict(o, cls="book") if o["cls"] == "laptop" else o for o in truth])
malformed = '[{"name": "laptop1", "cls": "laptop", "region": "table_right", "size": 3.0,}'  # truncated: not JSON

cases = [("valid", valid), ("wrong-field", wrong), ("malformed", malformed)]
print(f"{'completion':12s} {'parsed':>8s} {'format':>7s} {'accuracy':>9s} {'credit A':>9s} {'credit B':>9s}")
for label, comp in cases:
    parsed = parse_scene_completion(comp)
    n = "None" if parsed is None else str(len(parsed))
    acc = r_eye_accuracy(comp, world, default_blind("A"))
    cA = r_eye_credit(comp, SEED, "A", H, n_eyes=2)
    cB = r_eye_credit(comp, SEED, "B", H, n_eyes=2)
    print(f"{label:12s} {n:>8s} {r_eye_format(comp):7.2f} {acc:9.3f} {cA:9.3f} {cB:9.3f}")

assert r_eye_format(valid) > r_eye_format(malformed), "format must separate valid from malformed"
assert r_eye_format(malformed) == 0.0, "malformed output must get zero format"
assert parse_scene_completion(malformed) is None
print()
print("The wrong-field case keeps format=1 and accuracy near 1: one class label changed out of ten objects.")
print("Credit tells a different story - view A's credit flips sign between the valid and the wrong-field")
print("report, and even the valid report earns different credit from view A and view B.")
print("A malformed completion is not neutral: r_eye_credit returns -1.0 when the Eye says nothing usable.")
""")

nb.md("""## The three GRPO conditions

The paper compares three reward sets, selected by `trl_eye_reward_funcs(harness, which=(...))`:

| condition | what it optimises |
|---|---|
| `format,accuracy` | standard perception reward: parse well, match fields |
| `format,credit` | parse well, then only what changes team outcomes |
| `format,accuracy,credit` | both signals, so the comparison is not confounded by dropping accuracy |

TRL sums the per-function returns (default weight 1 each); the table below shows each function's
raw value and that sum. The functions are closures, so `__doc__` is absent on the wrappers; the
docstrings live on the underlying scorers and the factory, and are printed alongside.
""")

nb.code("""conditions = [
    ("format,accuracy", ("format", "accuracy")),
    ("format,credit", ("format", "credit")),
    ("format,accuracy,credit", ("format", "accuracy", "credit")),
]
completions = [("valid", valid), ("wrong-field", wrong), ("malformed", malformed)]
conv = [[{"role": "assistant", "content": c}] for _, c in completions]
cols = {"seed": [SEED] * len(completions), "view": ["A"] * len(completions)}

print("factory:", er.trl_eye_reward_funcs.__doc__.strip().splitlines()[0])
for f in er.trl_eye_reward_funcs(H):
    first = (f.__doc__ or "").strip().splitlines()
    print(f"  {f.__name__:12s} adapter doc: {first[0] if first else '(none; see the scorer below)'}")
for name in ("r_eye_format", "r_eye_accuracy", "r_eye_credit"):
    first = (getattr(er, name).__doc__ or "").strip().splitlines()
    print(f"  {name:14s} {first[0] if first else '(no docstring)'}")
print()

for label, which in conditions:
    funcs = er.trl_eye_reward_funcs(H, which=which)
    values = [f([], conv, **cols) for f in funcs]
    print("condition:", label, "| funcs:", [f.__name__ for f in funcs])
    header = "  " + f"{'completion':12s}" + "".join(f"{f.__name__:>14s}" for f in funcs) + f"{'SUM':>10s}"
    print(header)
    for i, (name, _) in enumerate(completions):
        per = [values[j][i] for j in range(len(funcs))]
        total = sum(per)
        print("  " + f"{name:12s}" + "".join(f"{v:14.3f}" for v in per) + f"{total:10.3f}")
    print()
""")

nb.md("""## The GRPO run that is not recorded

`train/grpo_eye.py` builds a `GRPOConfig`, wraps the reward functions, and trains an image-conditioned
policy. **No Eye GRPO run has been recorded on this repo.** The reward functions are unit-tested
(`tests/test_training_signals.py`); the trainer path is not.

The cell below does the honest work: it introspects this machine's `GRPOConfig` signature, keeps only
the kwargs this TRL version accepts, prints them, and then checks the real prerequisites. On a CPU
box, with no rendered images, or with no resolved base model it prints `SKIPPED: ...` and trains
nothing. It never invents a loss curve.
""")

nb.code("""import inspect

from rtp import assetcache as ac
from rtp.remotes import from_env

try:
    import torch
    from trl import GRPOConfig, GRPOTrainer
except Exception as exc:                       # noqa: BLE001  (import failure is the skip reason)
    GRPOConfig = GRPOTrainer = None
    print("SKIPPED: torch/trl not importable:", type(exc).__name__)

if GRPOConfig is not None:
    wanted = dict(
        output_dir=str(prof.paths.runs / "eye_grpo"),
        num_generations=4,
        max_completion_length=prof.max_len,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=prof.grad_accum,
        bf16=(prof.dtype == "bfloat16"),
        logging_steps=5,
        save_steps=200,
        report_to="none",
    )
    sig = inspect.signature(GRPOConfig.__init__)
    kept = {k: v for k, v in wanted.items() if k in sig.parameters}
    dropped = sorted(set(wanted) - set(kept))
    print("GRPOConfig parameters:", len(sig.parameters))
    print("resolved kwargs:", json.dumps(kept, indent=1, default=str))
    print("dropped (not in this TRL version):", dropped)

    model_spec = ac.model_spec("Qwen/Qwen3.5-4B", revision="851bf6e8", name="qwen35-4b")
    data_path = prof.paths.data / "eye_grpo.jsonl"
    renders = sorted(Path("renders").glob("*.png")) if Path("renders").is_dir() else []

    launch = True
    if not torch.cuda.is_available():
        print("SKIPPED: CPU only - image-conditioned GRPO needs a GPU; GRPOTrainer.train() is not called")
        launch = False
    if not renders:
        print(f"SKIPPED: 0 rendered Eye images under renders/ (dataset has {len(grpo)} placeholder paths)")
        launch = False
    if not data_path.is_file():
        print(f"SKIPPED: no GRPO data at {data_path}")
        launch = False

    model_dir = None
    if launch:
        try:
            store = from_env(with_kaggle=False)
            model_dir = store.resolve(model_spec, prof, verify="size")
            print("base model:", model_dir)
        except Exception as exc:               # noqa: BLE001  (network/token/cache miss)
            print("SKIPPED: base model could not be resolved:", type(exc).__name__)
            launch = False

    if not launch:
        print("No eye GRPO result exists; nothing above is a training result.")
    else:
        # Mirror train/grpo_eye.py: same reward funcs, config and LoRA. .train() is reached
        # only on a GPU box with rendered images and a real dataset; failures print SKIPPED.
        try:
            from datasets import load_dataset
            from peft import LoraConfig
            funcs = er.trl_eye_reward_funcs(H, which=("format", "credit"))
            ds = load_dataset("json", data_files=str(data_path))["train"]
            cfg = GRPOConfig(**kept)
            print("resolved TRL config:", cfg)
            print("reward funcs:", [f.__name__ for f in funcs], "| rows:", len(ds))
            trainer = GRPOTrainer(model=str(model_dir), reward_funcs=funcs, args=cfg, train_dataset=ds,
                                  peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear"))
            trainer.train()
        except Exception as exc:               # noqa: BLE001  (unverified multimodal path: never fabricate a result)
            print("SKIPPED: GRPOTrainer.train() could not run:", type(exc).__name__, exc)
""")

nb.md("""## Viewpoint and occlusion: why a blind view loses credit

`default_blind("A")` is `("table_right",)` and `default_blind("B")` is `("table_left",)`. An Eye with
`miss_occluded_p=1.0` simply omits every object in its blind region, so this is the cleanest possible
occlusion test: same world, same class/region labels, different fields of view.

One subtlety worth stating honestly: `oracle_report(world, "A")` and `oracle_report(world, "B")`
return the **same object list** — `oracle_report` only changes the message's view label. It does not
apply occlusion. To get a view-specific report you must construct `Eye(EyeConfig(name=view,
blind_regions=default_blind(view), miss_occluded_p=1.0))`. The cell shows both facts.

Credit is team-relative. With the default two Eyes (`n_eyes=2`), the partner may cover the blind
spot, so the credit gap can vanish. Remove the partner (`n_eyes=1`) and the occlusion cost appears:
the blind report is worth less than a complete report of the same view. This is the mechanism the
paper claims, demonstrated on the symbolic fixture rather than on pixels.
""")

nb.code("""from rtp.perception import Eye, EyeConfig

SEED_VIEW = 2
world = sample_world(SEED_VIEW)
reports, credits = {}, {}
for view in ("A", "B"):
    eye = Eye(EyeConfig(name=view, blind_regions=default_blind(view), miss_occluded_p=1.0), SEED_VIEW)
    msg = eye.report(world, 0)
    reports[view] = msg
    credits[view] = r_eye_credit(json.dumps(msg.content["objects"]), SEED_VIEW, view, H, n_eyes=2)
    absent = sorted(set(world.objs) - {o["name"] for o in msg.content["objects"]})
    print(f"view {view}: blind region {default_blind(view)} | reports {len(msg.content['objects'])} objects "
          f"| omits {absent} | credit (own view, n_eyes=2) = {credits[view]:.3f}")
print("credit differs between the two blind views:", credits["A"] != credits["B"])

ora = oracle_report(world, "A")
orb = oracle_report(world, "B")
print()
print("oracle_report dicts differ per view:", ora.content != orb.content)
print("oracle_report object lists identical:", ora.content["objects"] == orb.content["objects"])
print("oracle_report view labels:", ora.content["view"], "vs", orb.content["view"],
      "-> the label changes, occlusion does not")

full = json.dumps(ora.content["objects"])
blind_a = json.dumps(reports["A"].content["objects"])
print()
print("same view A, partner removed (n_eyes=1):")
print("  complete report credit:", round(r_eye_credit(full, SEED_VIEW, "A", H, n_eyes=1), 3))
print("  blind report credit:   ", round(r_eye_credit(blind_a, SEED_VIEW, "A", H, n_eyes=1), 3))
print("with the default partner (n_eyes=2):")
print("  complete report credit:", round(r_eye_credit(full, SEED_VIEW, "A", H, n_eyes=2), 3))
print("  blind report credit:   ", round(r_eye_credit(blind_a, SEED_VIEW, "A", H, n_eyes=2), 3))
print()
print("The partner's coverage explains why the n_eyes=2 gap is smaller: credit is assigned to the team,")
print("not to a field in isolation. That is the property the paper uses, and the reason a blind Eye can")
print("still earn credit when the other Eye sees the object.")
""")

nb.md("""## Measured vs pending

**Measured in this notebook (symbolic, CPU):**

* SCENE-JSON row shape and the placeholder status of every image path;
* parse/format/accuracy/credit on real completions, including a wrong-field case where accuracy
  stays high and credit flips negative;
* the exact values each of the three reward conditions assigns to those completions;
* the TRL `GRPOConfig` signature on this box and the filtered kwargs a run would use;
* the viewpoint mechanism: per-view occlusion removes objects, `oracle_report` does not, and credit
  separates a complete report from a blind one when the partner cannot compensate.

**Pending, and not claimed:**

* **No Eye GRPO run exists.** The cell prints `SKIPPED` because there are no rendered images (and no
  profile-local GRPO data file).
* No detector was trained or evaluated on pixels; no accuracy/credit number here is a perception
  result.
* No Isaac Lab scene with two FR3 arms and PTZ eyes exists; episodes are `rtp/world.py` symbolic
  rollouts. The paper must label them as such.
* `rtp/vlm_eye.py` is an optional **comparison baseline** that asks a VLM for the same SCENE JSON;
  it is not the Eye design and no baseline number is produced here.
""")

if __name__ == "__main__":
    print(nb.save())
