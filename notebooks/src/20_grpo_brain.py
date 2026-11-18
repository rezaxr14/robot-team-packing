#!/usr/bin/env python3
"""Build ``notebooks/20_grpo_brain.ipynb`` — GRPO on the Brain with symbolic rewards + KL.

    $V/bin/python notebooks/src/20_grpo_brain.py

The reward functions are real and are exercised on CPU here. The policy-gradient run is gated:
no GRPO result is claimed, and the notebook prints the TRL config it *would* use after filtering
kwargs against the installed TRL's real signature.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.nbbuild import Notebook, install_cell  # noqa: E402

nb = Notebook("20_grpo_brain")

# --------------------------------------------------------------------------------------------------
nb.md(r"""# 20 — GRPO: group-relative policy optimisation of the Brain

SFT (notebook 10) teaches the Brain to emit a plan that **parses**. GRPO then optimises plan
*quality*: for each prompt it samples a group of completions, scores them with the deterministic
rewards in `rtp.rewards` (format, static feasibility, rule compliance by dry run, length, and a
symbolic closed-loop rollout), and updates towards the rewards that are better than the group's
mean. A KL term against the **frozen SFT reference** is what keeps the policy near the warm start:
the plan's budget is logged and a run stops when it is exceeded (IMPLEMENTATION_PLAN §8, P8).

This notebook does three things honestly:

1. generates real GRPO prompt rows and shows the two columns the rewards actually read
   (`scene_json`, `spec_json`);
2. runs the **real** TRL reward adapters on CPU over three completions — a valid plan from
   `ScriptedBrain`, a malformed one, and a rule-violating one — and prints every numeric reward;
3. introspects this box's `GRPOConfig.__init__`, filters the requested kwargs to those TRL really
   accepts, and prints the resolved config. **No GRPO training has been run here**; the `.train()`
   call is gated and its failure path prints `SKIPPED` instead of a made-up loss curve.

No Isaac Lab/Newton physics and no frontier-model judge are involved: the reward signal is the
symbolic world (`rtp/world.py` + `rtp/loop.py`), so even the reward demo is a pipeline check, not a
paper result.""")

# --------------------------------------------------------------------------------------------------
nb.md(r"""## Dependencies

Only the non-stdlib, non-`rtp` imports this notebook needs. `trl` and `transformers` are pinned by the
venv used for this repo; the install cell is a no-op when they are present.""")

nb.code(install_cell(["torch", "transformers", "trl", "datasets", "peft"]))

# --------------------------------------------------------------------------------------------------
nb.md(r"""## The one knob, and what this box can do

As in every notebook, `ACTIVE` is the only setting typed here; the profile supplies dtype, attention,
batch geometry, `max_len` and all paths. The GRPO adapter must be trained from the **SFT adapter** of
the same profile and `config_hash`, otherwise the KL reference is a different model.""")

nb.code(r'''ACTIVE = "local_4050"   # local_4050 | local_4090 | kaggle_t4x2 | pro6000 | tpu_v5e8
import os, sys
from pathlib import Path

os.environ["RTP_PROFILE"] = ACTIVE
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("profile:", ACTIVE, "| repo:", ROOT)''')

nb.code(r'''from rtp.profiles import detect_caps, get_profile, resolve_runtime

profile = get_profile(ACTIVE)
profile.ensure_dirs()
caps = detect_caps()
attn, dtype, notes = resolve_runtime(profile, caps)

print(f"profile {profile.name}: {profile.gpu}, {profile.vram_gb} GB x {profile.n_gpu}, tuning={profile.tuning}")
print(f"config hash: {profile.config_hash()} | max_len: {profile.max_len} | "
      f"effective batch: {profile.effective_batch} | lora_r: {profile.lora_r}")
print(f"runtime: attn={attn} dtype={dtype} (cuda={caps.cuda}, cc={caps.cc})")
for note in notes:
    print(" reason:", note)
if not notes:
    print(" reasons: none - the profile's request is supported as written here")''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""## Storage: SFT reference and base model

GRPO starts from the SFT adapters, so both the pinned base snapshot and the newest SFT checkpoint
matter. Resolution order is the same as in notebook 10: local cache → attached Kaggle dataset → our
private Kaggle dataset → our private HF repo, with `attached="/kaggle/input"` for Kaggle sessions.
Kaggle auth is intermittent through this network's proxy, so a failure falls back to HF only; no repo
id is hard-coded and no token is printed.""")

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
        print(f"storage: none ({type(exc2).__name__}: {exc2})")

MODEL_REV = "851bf6e8"
spec = ac.model_spec("Qwen/Qwen3.5-4B", revision=MODEL_REV, name="qwen35-4b")
model_dir = None
if store is None:
    print("SKIPPED: model resolution - no storage client")
else:
    try:
        model_dir = store.resolve(spec, profile, attached="/kaggle/input")
        print("model:", model_dir, "| resolved from cache/attach, manifest verified")
    except Exception as exc:
        print(f"SKIPPED: model not resolvable for {spec.slug}: {type(exc).__name__}: {exc}")

sft_adapters = profile.paths.adapters / "sft"
print("SFT reference adapters:", sft_adapters, "| exists:", sft_adapters.is_dir())

if store is not None:
    try:
        print("latest grpo checkpoint:", store.latest_checkpoint(profile.name, "grpo"))
    except Exception as exc:
        print(f"SKIPPED: resume point lookup: {type(exc).__name__}: {exc}")''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""## Prompts: the two columns the rewards read

`gen_grpo_prompts` returns one row per seed with the same Brain prompt used in SFT plus two extra
columns: `scene_json` (the Eyes' merged report) and `spec_json` (serialised `TaskSpec`). The TRL
adapters in `rtp.rewards` receive these as dataset columns via `**kwargs`, parse them inside
`_unpack`, and never look at the text of the prompt. A fifth reward (closed-loop rollout) also reads
the `seed` column so it can rebuild the exact world the prompt was written for.""")

nb.code(r'''import json
from rtp.datagen import gen_grpo_prompts, write_jsonl
from rtp.harness import Harness

N_PROMPTS, SEED0 = 12, 0
harness = Harness("harness_files")
prompts = gen_grpo_prompts(N_PROMPTS, SEED0, harness)
DATA = profile.paths.data / "brain_grpo.jsonl"
DATA.parent.mkdir(parents=True, exist_ok=True)
write_jsonl(prompts, DATA)

print(f"prompts requested {N_PROMPTS}, got {len(prompts)} -> {DATA}")
if prompts:
    row = prompts[0]
    print("columns:", sorted(row.keys()))
    print("seed:", row["seed"])
    print("scene_json (first 260 chars):", row["scene_json"][:260])
    print("spec_json  (first 260 chars):", row["spec_json"][:260])
    print("reward columns present:", all(k in row for k in ("scene_json", "spec_json", "seed")))
else:
    print("SKIPPED: no GRPO prompts were generated")''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""## Real rewards on CPU: three completions

The three completions are deliberately different, and all numbers below come from calling the real
adapters returned by `trl_reward_funcs(harness=harness)`, not from a table:

* **valid** — `ScriptedBrain` plans against the same scene/spec the prompt describes (its `rule_ids`
  come from the same harness retrieval the prompt used), so the JSON parses and the rules hold;
* **malformed** — not JSON, so every reward must be 0;
* **rule-violating** — a syntactically valid plan with an extra `grasp` of a `leave` item, which
  `static_check` rejects but `r_format` cannot see. This is why format reward alone is not enough.

The format reward must be strictly higher for the valid plan than for the malformed one; the rules
reward must strictly separate the valid plan from the rule-violating one.""")

nb.code(r'''from rtp.brain import BrainContext, ScriptedBrain
from rtp.loop import LoopConfig, retrieve_docs
from rtp.rewards import trl_reward_funcs, trl_reward_weights
from rtp.rules import TaskSpec

if not prompts:
    print("SKIPPED: reward demo - no prompt row")
else:
    row = prompts[0]
    scene = json.loads(row["scene_json"])
    spec = TaskSpec.from_dict(json.loads(row["spec_json"]))
    docs = retrieve_docs(harness, LoopConfig(), spec, scene, [])
    ctx = BrainContext(spec=spec, scene=scene, rule_ids={d.id for d in docs if d.type == "rule"},
                       snippets=docs)
    valid = ScriptedBrain().plan(ctx).to_json()
    malformed = '{"tasks": [,,]}'

    bad = json.loads(valid)
    leave = spec.leave[0] if spec.leave else None
    if leave is not None:
        bad["tasks"].insert(0, {"id": "t_bad_leave", "arm": "L", "skill": "grasp", "obj": leave,
                                "params": {"force": 10.0}, "deps": []})
    else:
        for task in bad["tasks"]:
            if task["skill"] == "place":
                task["params"]["compartment"] = "nowhere"
                break
    violating = json.dumps(bad)

    completions = [valid, malformed, violating]
    labels = ["valid", "malformed", "rule-violating"]
    funcs = trl_reward_funcs(harness=harness)
    kwargs = {"scene_json": [row["scene_json"]] * 3, "spec_json": [row["spec_json"]] * 3,
              "seed": [row["seed"]] * 3}
    print("reward funcs:", [f.__name__ for f in funcs])
    print("reward weights:", trl_reward_weights(include_rollout=True), "(same order)")
    scores = {}
    for func in funcs:
        scores[func.__name__] = [round(float(v), 4) for v in func([row["prompt"]] * 3,
                                                                  completions, **kwargs)]
    header = "".join(f"{name:>17s}" for name in labels)
    print("completion      ", header)
    for name, values in scores.items():
        print(f"{name:15s}", "".join(f"{v:17.4f}" for v in values))

    fmt = scores["format_reward"]
    rules = scores["rules_reward"]
    assert fmt[0] > fmt[1], "format reward must be higher for the valid plan than the malformed one"
    assert rules[0] > rules[2], "rules reward must separate the valid plan from the rule violation"
    print(f"asserted: format valid {fmt[0]} > malformed {fmt[1]}; "
          f"rules valid {rules[0]} > violating {rules[2]}")''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""## The GRPO run that is not recorded

No GRPO run exists for this repo. TRL 1.18's `GRPOConfig` has a large, version-specific signature,
so the cell below introspects `inspect.signature(GRPOConfig.__init__)` **at runtime** and keeps only
the kwargs this installed TRL accepts; the dropped ones are printed rather than passed and silently
ignored. `beta` is the KL coefficient to the frozen SFT reference. The reward weights are the ones
printed above, in the same order as the reward functions.

`RUN_GRPO` is off by default: a group of 4 completions at this profile's 512 tokens does not fit the
6 GB 4050, and TRL 1.18 + Qwen3.5 support has not been verified here. Turning it on still requires a
resolved model, an SFT adapter and CUDA; any failure of `.train()` prints `SKIPPED` with the real
exception type. Nothing in this notebook reports a GRPO loss or a success delta.""")

nb.code(r'''import inspect
import json

RUN_GRPO = False        # deliberate: no GRPO result is claimed on this box
trainer = None          # so the resume cell still has a value if TRL cannot be imported

try:
    import torch
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
except Exception as exc:
    GRPOConfig = GRPOTrainer = Dataset = None
    print(f"SKIPPED: torch/trl/datasets not importable: {type(exc).__name__}: {exc}")

if GRPOConfig is not None:
    wanted = dict(
        output_dir=str(profile.paths.runs / "grpo"),
        beta=0.04,                              # KL to the frozen SFT reference
        num_generations=4,
        max_completion_length=profile.max_len,
        max_prompt_length=profile.max_len,      # may not exist in this TRL version; filtered below
        reward_weights=trl_reward_weights(include_rollout=True),
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=profile.grad_accum,
        bf16=(dtype == "bfloat16"),
        logging_steps=5,
        save_steps=50,
        report_to="none",
    )
    sig = inspect.signature(GRPOConfig.__init__)
    kept = {k: v for k, v in wanted.items() if k in sig.parameters}
    dropped = sorted(set(wanted) - set(kept))
    print("GRPOConfig.__init__ parameters:", len(sig.parameters))
    print("resolved config:", json.dumps(kept, indent=1, default=str))
    print("dropped (not supported by this TRL):", dropped)

    trainer = None
    if not RUN_GRPO:
        print("SKIPPED: GRPO has not been run here (RUN_GRPO=False); no loss or reward curve is claimed")
    elif not caps.cuda:
        print("SKIPPED: GRPO has not been run here - no CUDA on this box")
    elif model_dir is None:
        print("SKIPPED: GRPO has not been run here - no resolved base model")
    elif not sft_adapters.is_dir():
        print(f"SKIPPED: GRPO has not been run here - no SFT reference adapters at {sft_adapters}")
    elif not prompts:
        print("SKIPPED: GRPO has not been run here - no prompt rows")
    else:
        try:
            cfg = GRPOConfig(**kept)
            trainer = GRPOTrainer(
                model=str(sft_adapters),        # LoRA adapters over the frozen base
                reward_funcs=trl_reward_funcs(harness=harness),
                args=cfg,
                train_dataset=Dataset.from_list(prompts),
            )
            trainer.train()
        except Exception as exc:
            trainer = None
            print(f"SKIPPED: GRPO .train() failed on this box: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0]}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""## Checkpoints: push now, resume on the next machine

GRPO checkpoints are small enough to live in the HF repo (`store.push_checkpoint` writes under
`checkpoints/<profile>/grpo/step-<n>`). `rtp.checkpoints.fetch_latest` is the matching read side: it
returns the newest step and a local directory that can be handed to
`GRPOTrainer.train(resume_from_checkpoint=...)` — the TRL equivalent of `run_sft`'s automatic
`fetch_latest`. With no GRPO run in this session there is no trainer output to push, so the cell
prints `SKIPPED` and only *shows* the call it would make.""")

nb.code(r'''from rtp.checkpoints import fetch_latest

if store is None:
    print("SKIPPED: checkpoint push/fetch - no storage client")
    print("  when configured: store.push_checkpoint(root, profile.name, 'grpo', step)")
    print("  and: fetch_latest(store, profile.name, 'grpo', dest) -> (step, path)")
elif trainer is None:
    print("SKIPPED: checkpoint push - no GRPO trainer output (no .train() ran)")
    print("  when there is: store.push_checkpoint(profile.paths.runs / 'grpo', "
          "profile.name, 'grpo', trainer.state.global_step)")
else:
    out = profile.paths.runs / "grpo"
    step = int(trainer.state.global_step)
    try:
        url = store.push_checkpoint(out, profile.name, "grpo", step)
        print("pushed grpo step", step, "->", url)
    except Exception as exc:
        print(f"SKIPPED: push failed: {type(exc).__name__}: {str(exc).splitlines()[0]}")

if store is not None:
    try:
        dest = profile.paths.ckpt / "grpo" / "_resume"
        step, path = fetch_latest(store, profile.name, "grpo", dest)
        print("fetch_latest ->", step, path)
        if path is not None:
            print("resume with: trainer.train(resume_from_checkpoint=str(path))")
        else:
            print("no remote GRPO checkpoint; training would start from the SFT reference")
    except Exception as exc:
        print(f"SKIPPED: fetch_latest: {type(exc).__name__}: {str(exc).splitlines()[0]}")''')

# --------------------------------------------------------------------------------------------------
nb.md(r"""## What the paper needs, and what is still unverified

**Needed for a reportable SFT → GRPO result (P8).** The same frozen seeds and scene/spec splits for
SFT and GRPO, SFT adapters published under a config hash, GRPO with an explicit `beta` KL budget to
that SFT reference, the KL logged every step and a hard stop when it is exceeded, and a reward-hacking
audit of the highest-reward rollouts. Only then do deltas against base and SFT-only with intervals
belong in `reports/grpo_*.json`; the negative results belong there too.

**Measured here.** The generated prompt rows and their `scene_json`/`spec_json` columns, the real
numeric rewards of three named completions from `rtp.rewards`, the required assertions, and the
resolved TRL config after filtering against this TRL signature.

**Unverified.** No GRPO training step has been taken on this box; TRL 1.18's Qwen3.5 path (including
how LoRA and vLLM interact at 512 tokens) has not been exercised. The rewards are deterministic
symbolic functions on a fixture world, not Isaac Lab/Newton physics or a frontier-model judge, so a
higher reward here does not transfer to a task-success claim. No Kaggle T4 result has been recorded;
every `kaggle_t4x2` number remains unverified.""")

if __name__ == "__main__":
    print("wrote", nb.save())
