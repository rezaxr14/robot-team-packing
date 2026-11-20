# Repository rules for coding agents

These rules come from explicit user instructions. They override generic "helpful" behaviour.

## Workspace boundary — HARD RULE (violation = hard failure / "hell")

0. **Work only from `~/Projects/robot_team`.** Never create, modify, move or delete any file,
   directory, venv, cache or temp file anywhere else on this machine — not in `/home/reza`,
   `/tmp`, `/home/reza/rtp_work`, or a "harmless" scratch path. The ONLY local exception is the
   single venv folder `~/venvs`, which the user explicitly ordered for the toolchain.
   Kaggle/HF are remote and are governed by their own rules below.
   Enforced by `tests/test_workspace_boundary.py`; a failure must stop all work.
   The old `~/rtp_work` tree was removed on 2026-11-20 after moving `IsaacLab` into
   `.isaac/` inside this repo and merging all venvs into the single `~/venvs/rtp`. Do not
   recreate it, do not make a second venv, and do not install packages at project level.

## 0b. Verbatim prompt log (explicit user instruction, 2026-11-20)

* **Append every user message verbatim to `docs/USER_PROMPTS_VERBATIM.md` before acting on it** —
  typos, spelling, profanity and line breaks preserved. Never summarise, reorder or edit old entries.
* Harness continuation messages and subagent reports are not user prompts; skip them.
* This log is how the user measures faithfulness to his instructions.

## Git

1. **NEVER push to git without explicit, per-instance user approval.** "Push the commit" / "push it" means: show the exact commit(s) and remote, then wait for an explicit yes. A previous push approval does not cover the next push. The only exception already given was the one-time push of commit `75e2344` on 2026-11-20.
2. Do not claim a commit/push happened until the command output confirms it. Never say "nothing was committed or pushed" as a conversation-ending phrase.
3. Do not revert or reset user commits without a direct instruction.

## Claims and evidence

4. **Never say "fixed", "done", "looks right" or "clean" until the user confirms the image/scene/result.** Independent tests and subagent checks are supporting evidence, not user acceptance.
5. Every number in a report or paper must come from a recorded command output in `reports/` or a committed test result. Mark its evidence class (`FIXTURE`, `SIM-POSE`, `SIM-ABSTRACT`, `SIM`, `MODEL`, `HARDWARE`) and never promote a weaker class to a stronger one.
6. "Spec exists", "notebook exists", "code written" are not evidence that a stage ran. Write "not run" in `docs/STATUS_HONEST.md` when it was not run.
7. Do not call any arm/config "safe" around people. D-22 assigns hardware/L0/L1 sign-off to Claude; all current control work is a simulation cross-check only.

## Documentation (explicit user rule, 2026-11-20)

8. **Do not create documentation for the sake of creating documentation.** Before creating a new doc, search `docs/`, `README.md`, `IMPLEMENTATION_PLAN.md`, `DECISIONS.md` and `docs/STATUS_HONEST.md`. If a related doc exists, **add to / modify / delete that doc** instead of making a new one that duplicates it.
9. New docs are allowed only for genuinely new information that no existing doc covers (e.g. `docs/findings/newton_arm_api.md` for A2 API discoveries). When adding one, add a pointer to it from the relevant existing doc/README index.
10. Keep `docs/STATUS_HONEST.md` authoritative for claim status; keep `docs/USER_PLAN_VERBATIM.md` authoritative for the user's plan. Do not restate their content in other docs — link to them.
11. If an existing doc becomes fully redundant, delete it and repoint references, rather than leaving two copies that drift.

## Subagents

12. When the user asks for subagents, use only **`opencode/hy4-flash-free`** (explicitly set the model parameter). Do not use deepseek or other providers for these tasks.
13. Use subagents for independent verification of visual/geometry work before presenting it, and for parallel documentation/audit tasks. Divide file ownership so subagents do not edit the same files.

## Kaggle / HF / secrets

14. Kaggle HTTP access goes through `http://localhost:10808` only (set `RTP_HTTP_PROXY`). **Never route HuggingFace or other traffic through that proxy.**
15. Tokens live in `.env` (gitignored). Never print them, never commit them, never paste them into a report, prompt or chat.
16. Never upload the 9 GB model from this machine; publish large artifacts from the Kaggle worker.

## Workflow

17. Simulation is Isaac Lab 3.0 + Newton only. Use `source ops/env.sh && $ISAAC_PY` for GPU/sim work; the training venv is `/home/reza/venvs/cu132_torch2.15_mujoco4.0_numpy2.5_matplotlib3.11/bin/python`.
18. Work in stages with user review between them. Show the exact image path / command output and wait.
19. Follow `docs/AGENT_RUNBOOK.md` in order: A1–A10 (arms first) before R4–R9. One step = one commit, with STATUS_HONEST and DECISIONS updated in the same commit.
