# Architecture (as implemented)

A human gives a plain-language job. Three roles act in simulation; every inter-agent exchange is a typed message.

```
                 instruction (text)                       harness: .md rules / skills / notes
                        |                                          | BM25 (rules by instruction, objects/skills by scene)
                        v                                          v
 Eye A ----SCENE----> [ Brain ] <-----retrieved text---------------+
 Eye B ----SCENE---->    |  \__ tools (ToolBrain): check_rule, query_scene, look, dry_run, submit_plan
   ^  (images only)      |
   |                     v  plan DAG (tasks: arm, skill, obj, params, deps)
   |              verify (static / dry run) --rejected--> back to the Brain with reasons (bounded retries)
   |                     |
   |                     v  scheduler: per round, head task of each arm; one bag access at a time (lock)
   |            Arm L / Arm R (blind: proprioception + contact + skill command; no image, no instruction)
   |                     |  TOUCH / BLOCKED / STATUS messages
   |                     v
   +--- LOOK <--- Brain ----- recovery ladder: retry -> alt params (from TOUCH) -> retreat+retry -> other arm -> replan -> abort
                     CHALLENGE: an arm contradicts an Eye's mass; scene corrected, placements replanned
                     probe (optional): arms weigh every item BEFORE planning; ordering uses measured masses
```

## One episode, in order (rtp/loop.py)
1. Arms send `ABILITY`. Eyes observe (`SCENE`); scenes are fused by view name (order-independent) into an object memory.
2. Optional probe phase: arms `weigh` each item; measured masses override Eye guesses.
3. Brain plans (retrieval -> prompt -> plan). Plans are verified before any arm moves; rejections return as notes.
4. Missing items trigger `LOOK` (fresh Eye observation, bounded); found items trigger a replan.
5. Rounds: scheduler picks tasks; skills run in the world; `TOUCH` feeds corrections; failures climb the ladder.
6. Mid-episode `AMEND` changes the spec and forces a replan.
7. End: `rules.check_final` is the single definition of success; every fatal failure is attributed to Eye / Brain / Arm by
   explicit rules (heuristic labels, validated against hand labels via `rtp/attribution_eval.py`).

## One definition of success, three worlds
`rules.check_final(world, spec)` scores: the symbolic world (`rtp/world.py`), pose-level fake simulator (`rtp/fakesim.py`),
and Isaac Lab (`isaac/packing_sim.py` -> `rtp/simbridge.py: world_from_sim`). `rtp/simworld.py: SimWorld` lets the same loop
run on any backend; skill outcomes are derived from state before/after (`report_from_states`), validated by the regression
suite (`rtp/skills.py`) and by running full episodes through the pose-level path.

## Learning signals
| Signal | Trains | Where |
|---|---|---|
| plan-level GRPO rewards: format, feasibility, rule compliance (dry run), length, closed-loop rollout | Brain | `rtp/rewards.py`, `train/grpo_brain.py` |
| SFT on teacher plans validated by closed-loop success | Brain | `rtp/datagen.py`, `train/sft_brain.py` |
| team hindsight relabeling (instruction rewritten to what was achieved; admitted only if replay succeeds) | Brain | `rtp/hindsight.py` |
| counterfactual Eye credit (episode score with this report vs with the Eye silent, common random numbers) | Eye | `rtp/credit.py`, `rtp/eye_rewards.py` |
| Eye rejection sampling (best-of-K by accuracy or credit) | Eye | `rtp/eye_rft.py` |
| distillation of scripted IK skills into a blind flow-matching policy | Arms | `train/distill_arm.py`, `rtp/flowhead.py` |

## Evidence levels (what each result may be used for)
TRACK A (real benchmark, real model) and ISAAC (physics) can support claims. LLM-SYMBOLIC supports claims scoped to the symbolic
task. FIXTURE results (scripted Brain, flaky-planner fixture, parametric Eyes) check mechanics only. `scripts/build_report.py`
labels every table accordingly.

## Known limits
Sim only; one task family; the symbolic checker defines success for rewards and Track B; failure attribution is rule-based;
counterfactual credit estimates a team counterfactual (the Brain adapts to the replaced report) and is not additive across
Eyes; Isaac skills are scripted, the load-cell in `weigh` is the simulator's mass, and mid-air handover is not scripted.
