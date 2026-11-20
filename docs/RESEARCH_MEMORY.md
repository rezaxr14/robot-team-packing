# Research memory (persistent, living)

This is the **index and decision log** for external research that informs the new plan. It exists so
subagent findings are not lost between sessions and so nobody re-researches the same question.
**Do not duplicate the findings here**: each thread owns one file under `docs/findings/`; this file
holds pointers, the current verdict, and what we actually changed because of the research.

Rule (D-71): if a finding belongs in an existing doc (`IMPLEMENTATION_PLAN.md`,
`docs/AGENT_RUNBOOK.md`, `DECISIONS.md`, `docs/STATUS_HONEST.md`, `docs/DATAGEN_KAGGLE.md`,
`docs/SCENE_DESIGN.md`, `paper/`), **update that doc** instead of creating a new one. Only genuinely
new topics get a `docs/findings/*.md` file.

Every entry must carry: source URL + access date, `external source, not reproduced here`, and a
status (`new` / `accepted` / `rejected` / `superseded`). No measured result may be invented: we have
no trained model, no certified arm and no hardware result yet.

## Threads

| thread | owner file | status | question |
|---|---|---|---|
| A2 Newton arm API | `docs/findings/newton_arm_api.md` | accepted (committed `a371536`) | exact Isaac Lab 3.0 + Newton calls for MJCF spawn, effort mode, state reads, gravity comp, Jacobian, wrenches, substeps, contacts |
| Omni comparison models | `docs/findings/omni_models.md` | accepted in part (12 rows researched; 3 shortlisted) | which additional image+language robot models can be scored on our held-out seeds through the same bounded-impedance interface |
| Prior robot-team work | `docs/findings/prior_work.md` | accepted in part (metadata-level only) | typed messaging, verifiers, attribution, success/failure memory, safe compliance; what others did and what they do not claim |
| SFT / GRPO recipes | `docs/findings/training_recipes.md` | accepted as plan, not implemented | assistant-token loss masking, k=4/k=8 GRPO, KL to SFT, anti-reward-hacking, 6 GB/T4 constraints |
| Scene bundle + eyes | `docs/findings/scene_and_eyes.md` | accepted as plan, not implemented | hashable bundle format, multi-camera view library, eye metrics, wrench/substep pitfalls, SimReady ingestion |

### Accepted findings -> where they were applied (2026-11-20)

* **A2**: D-72; `ops/a3_arm_smoke.py` uses the native `add_mjcf()` articulation and the report marks
  the effort-mode route as not done; the per-substep wrench fix (PR #8269, backport `5608601`) is the
  contract for A4/G-7.
* **Omni models**: three feasible additions (X-VLA 0.9B, pi0-FAST, NSDL 3B) added to
  `rtp.benchmark.COMPARISON_ROSTER`, so they appear as `not run` rows in
  `reports/omni_comparison.{json,md}`; GR00T N1.6 and OpenVLA etc. kept as future work with reasons in
  the thread file.
* **Prior work**: `paper/sections/02_related.tex` now cites NSDL, InteractGen, RoboTwin 2.0,
  Who&When and difference rewards with metadata-level `verified:` notes in `paper/refs.bib`; the
  stale "LOCUS-Bench unavailable" note was replaced; all external scores stay external.
* **Training recipes**: actions recorded below for R5-R7 (no code changed yet).
* **Scene bundle / eyes**: actions recorded below for R1/R2 and A4 (no code changed yet).

### Open actions (do not re-research; implement or explicitly reject)

1. `rtp/profiles.py`: add `kl_budget`, `grpo_beta`, `grpo_num_generations`,
   `grpo_max_completion_length`; the runbook R7 references a KL budget that does not exist.
2. `rtp/sft_format.py` + `tests/test_sft_masks.py`: assistant-only loss masks over multi-turn
   tool episodes (labels=-100 on user/tool-result/image-placeholder tokens); wire through
   `rtp/trainer.py` / `train/sft_brain.py`, repeat the G-14 4050 smoke.
3. R7 GRPO smoke on Kaggle T4 only: k=4/k=8 budget-matched, `dr_grpo`, `eps_high=0.28`,
   `mask_truncated_completions=True`, KL logging and stop on budget; format validity as a hard gate.
4. `configs/views.yaml`: resolve ring R=1.6 (`docs/SCENE_DESIGN.md`) vs R=2.0 (`reports/eye_placement.json`),
   define `view_id`, JPEG encoder and bundle manifest hashing before `ops/make_scene_bundle.py`.
5. `ops/render_views.py` R1 smoke: bundle hash stable over two runs and **measured** 640x480 fps.
6. A4/G-7: wrench impulse with substeps 1 and 4, recording the every-substep backport commit.
7. Attribution/memory: per-turn inputs in `rtp/datagen_run.py`, typed `(family, cause, condition)`
   memory index in `rtp/datagen_memory.py`, separate deterministic/LLM-judge verdicts in
   `rtp/harness.py`, no-op oracle control before any credit number is believed.

## How findings become code

1. **Accepted** finding -> add a line to `DECISIONS.md` (with the source) and, if it changes work,
   edit `docs/AGENT_RUNBOOK.md` / `IMPLEMENTATION_PLAN.md` in place.
2. **Rejected** finding -> keep the one-line reason in the thread file; do not re-litigate it later.
3. **Superseded** finding -> keep the old line struck through with a pointer to the new evidence.
4. A finding that is only a claim about another system never becomes a number in our paper or
   `docs/STATUS_HONEST.md`. Our numbers come only from `reports/*.json` and committed tests.

## Update log

- 2026-11-20: file created after the Plan v3 patch; five research threads opened.
- 2026-11-20: all five threads complete. A1/A2/A3 grounded locally (SIM-POSE/SIM); external research
  was accepted only where it produced citations, roster rows or named open actions. No external
  score was promoted to a project result; no GRPO/SFT/bundle code was written from a finding yet.
