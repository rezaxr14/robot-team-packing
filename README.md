# robot_team: language-coordinated robot team (Eye / Brain / blind Arms)

Research code for the paper draft in the project Doc. Everything here runs on CPU in pure Python
(+ numpy for one test). The symbolic world is a **test fixture and reward source, not a physics result**.

## Layout
| path | what |
|---|---|
| `rtp/messages.py` | typed messages (ABILITY, SCENE, CLAIM, ..., AMEND), schema validation, token-budget bus |
| `rtp/world.py` | symbolic packing world; force-aware skills (slip/crush), bag lock, common random numbers |
| `rtp/perception.py` | Eyes with viewpoint occlusion + fault injection; counterfactual report replacement; `VLMEye` adapter |
| `rtp/plan.py` | plan DAG schema, robust JSON parsing, round scheduler (bag lock, soft ordering deps) |
| `rtp/rules.py` | `TaskSpec`, final-state rule checker, static feasibility, deterministic dry run |
| `rtp/harness.py` | .md skills/rules/notes, BM25 (+optional embedding fusion), monitor write-back |
| `rtp/brain.py` | `ScriptedBrain` (teacher/fixture), `LLMBrain` adapter + prompt builder |
| `rtp/loop.py` | closed loop: comm modes, recovery ladder, replan, amendments, failure attribution |
| `rtp/credit.py` | counterfactual Eye credit under common random numbers |
| `rtp/hindsight.py` | team hindsight relabeling + replay validation |
| `rtp/rewards.py` | episode score; plan-level GRPO rewards; TRL adapters |
| `rtp/datagen.py` | scenes, specs, Brain SFT / GRPO prompt / Eye SFT data |
| `rtp/flowmath.py`, `rtp/flowhead.py` | flow-matching math (tested) and the arm policy (torch, untested here) |
| `rtp/trackA.py` | load the VIKI-R L2 scorer without verl; seeded / exhaustive scoring |
| `rtp/llm_client.py` | stdlib OpenAI-compatible chat client (vLLM) |
| `rtp/experiments.py`, `rtp/tables.py` | config grid -> per-episode records -> markdown tables (Wilson, paired bootstrap) |
| `rtp/eye_rewards.py` | Eye rewards: format, field accuracy, and counterfactual team credit (the paper's Eye-training comparison) |
| `rtp/skills.py` | arm skill spec + executor-agnostic regression suite (symbolic executor now; sim executor later) |
| `rtp/simbridge.py` | simulator state -> regions/compartments/upright -> `World`; skill outcome classification (tested on synthetic poses) |
| `rtp/simworld.py`, `rtp/fakesim.py` | `SimBackend` interface, `SimWorld` adapter (runs the loop on any simulator), `StateDiffBackend`, pose-level `FakeSim` test double |
| `rtp/toolbrain.py` | multi-turn LLM Brain with tools (check_rule, query_scene, look, dry_run, submit_plan) |
| `rtp/trackA_pipeline.py` | Eye->Brain pipeline on VIKI L2 with oracle / no-Eye / real-Eye conditions |
| `rtp/fixtures.py` | `FlakyBrain`: imperfect-planner fixture (ghost objects, missing grasps, wrong arm, rule-breaking order, dropped items, garbage) |
| `ops/` | preflight (GPU arch detection), env installs, model downloads, vLLM helper, stage runner with logging/resume, bundler, stage plan |
| `isaac/packing_sim.py`, `isaac/bringup/` | Isaac Lab 2.3 scene + scripted IK skills implementing `SimBackend`; staged bring-up scripts 01-07 (never run) |
| `rtp/vlm_eye.py` | the real Eye: renders a view, asks a VLM for SCENE JSON, parses; failures become empty reports attributed to the Eye |
| `rtp/policy_rollout.py` | receding-horizon execution of a learned arm policy (tested on a toy arm) |
| `isaac/learned_sim.py`, `isaac/regression.py` | learned-arm simulator; regression-suite executor for the simulator |
| `rtp/eye_rft.py` | Eye rejection-sampling fine-tuning (best-of-K by accuracy or credit) + selection-disagreement measure |
| `rtp/figures.py`, `scripts/build_report.py` | paper figures; one report with an evidence ledger (TRACK A / ISAAC / LLM-SYMBOLIC / FIXTURE) and missing-input list |
| `rtp/attribution_eval.py` | hand-labelling export and agreement (accuracy, Cohen's kappa) for Eye/Brain/Arm attribution |
| `scripts/`, `train/`, `isaac/` | Stage 0 scripts, experiment grid, dataset generation, SFT/GRPO/arm-distillation, Isaac Lab skeleton |
| `RUNBOOK.md` | what to run, in order, and what to paste back |

## Status
- **Implemented and tested (173 tests, `python -m pytest tests -q`)**: everything in `rtp/` (the torch tests skip automatically where torch/TRL are absent)
  (including the Track A runner against the real upstream scorer code and prompts).
- **Run on CPU with torch/TRL/PEFT (tiny random model)**: `rtp/flowhead.py`, `train/{sft_brain,grpo_brain,distill_arm,merge_lora}.py`.
- **Implemented, NOT run** (no GPU/Isaac/HF access in the authoring sandbox): `train/grpo_eye.py`, `scripts/s0_inventory.py` (schema-tolerant, never run on real data),
  `scripts/s0_trackA_eval.py` end-to-end against a real model server (its components are tested with a fake model).
- **Not implemented**: Isaac Lab scene and skills (`isaac/team_env.py` is a partly grounded skeleton, see
  `isaac/DESIGN.md`; its sim-to-symbolic bridge is implemented in `rtp/simbridge.py`), privileged-state RL experts, distillation, online arm RL, VLM Eye fine-tuning,
  Brain SFT/GRPO runs, latent-message ablation, all paper experiments.
- No result in this repo is evidence for a paper claim. `scripts/pipeline_check.py` only checks plumbing.

## Findings so far that affect the plan
1. VIKI-R's L1/L2 scorers are offline Python (no simulator needed), so Track A needs no simulator port.
2. The upstream L2 `eval_single` draws a random start position per asset per call: on a synthetic case the
   same prediction passed 18/40 unseeded draws. Whether real VIKI-Bench items are exposed is untested
   (run `scripts/s0_inventory.py`). Report seeded or exhaustive scoring (`rtp/trackA.py`).
3. VIKI-R is built on Qwen2.5-VL (3B/7B); a same-base control is needed for a fair Track A comparison.
4. Upstream L2 "correct" means feasible AND the predicted plan is not longer than the reference
   (`len(ref)/len(pred) >= 0.99`); a valid but longer plan counts as wrong. The upstream eval scripts also truncate
   to the first 900 (L2) / 1000 (L1) rows and have an id/ood naming mismatch. Our runner reports upstream-style,
   position-robust, and full-split numbers separately.

**Architecture: `ARCHITECTURE.md`.** **GPU days: start with `RUNBOOK_4DAY.md`** (preflight, installs, gates, fallbacks, paste-back protocol); stages live in `ops/plan.py`.
See `TRACKA_CONTROL.md` for the upstream recipe, why it is not retrained here, and the fair-comparison checklist.

## Quickstart
    pip install pytest numpy
    python -m pytest tests -q
    python scripts/pipeline_check.py 50          # plumbing check only
    python -m rtp.flowhead                       # needs torch; toy sanity check of the arm head
