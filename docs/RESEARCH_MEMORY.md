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
| A2 Newton arm API | `docs/findings/newton_arm_api.md` | in progress | exact Isaac Lab 3.0 + Newton calls for MJCF spawn, effort mode, state reads, gravity comp, Jacobian, wrenches, substeps, contacts |
| Omni comparison models | `docs/findings/omni_models.md` | in progress | which additional image+language robot models can be scored on our held-out seeds through the same bounded-impedance interface |
| Prior robot-team work | `docs/findings/prior_work.md` | in progress | typed messaging, verifiers, attribution, success/failure memory, safe compliance; what others did and what they do not claim |
| SFT / GRPO recipes | `docs/findings/training_recipes.md` | in progress | assistant-token loss masking, k=4/k=8 GRPO, KL to SFT, anti-reward-hacking, 6 GB/T4 constraints |
| Scene bundle + eyes | `docs/findings/scene_and_eyes.md` | in progress | hashable bundle format, multi-camera view library, eye metrics, wrench/substep pitfalls, SimReady ingestion |

## How findings become code

1. **Accepted** finding -> add a line to `DECISIONS.md` (with the source) and, if it changes work,
   edit `docs/AGENT_RUNBOOK.md` / `IMPLEMENTATION_PLAN.md` in place.
2. **Rejected** finding -> keep the one-line reason in the thread file; do not re-litigate it later.
3. **Superseded** finding -> keep the old line struck through with a pointer to the new evidence.
4. A finding that is only a claim about another system never becomes a number in our paper or
   `docs/STATUS_HONEST.md`. Our numbers come only from `reports/*.json` and committed tests.

## Update log

- 2026-11-20: file created after the Plan v3 patch; five research threads opened. No finding has
  been accepted yet; A1 (FR3 kinematics, SIM-POSE) is the only new measured result this session.
