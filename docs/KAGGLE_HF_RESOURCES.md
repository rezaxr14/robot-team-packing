# Kaggle + HuggingFace resources (for Claude / any collaborator)

This document records every external account, dataset, model and credential the project uses, so a
collaborator knows what exists and what is still missing. No token values are stored here — only
which `.env` keys hold them.

Authoritative context: claim status `docs/STATUS_HONEST.md`; the multi-turn datagen loop and gates
`docs/DATAGEN_KAGGLE.md`; execution order `docs/AGENT_RUNBOOK.md` (arms first, A1–A10 then R4–R9).

## 1. Credentials (in `.env`, never committed)

| Key | Used for |
|---|---|
| `KAGGLE_API_TOKEN` | Kaggle CLI / `kaggle-benchmarks` (frontier AI model access + hosted tasks). |
| `HF_TOKEN` | HuggingFace dataset/model push & pull (private repos). |
| `HF_USERNAME` | HuggingFace account name. |

## 2. Network / proxy rules

- `localhost:10808` is the **Kaggle-only** HTTP/SOCKS proxy. Use it for Kaggle CLI, `kaggle-benchmarks`
  and hosted tasks.
- **Do NOT use this proxy for HuggingFace** (it causes SSL failures). HF is reached directly.

## 3. Kaggle — frontier AI models (the "Brain" candidates)

Via Kaggle Benchmarks (`kaggle-benchmarks`, CLI `kaggle benchmarks ...`). Two routes:

1. **Interactive Model Proxy** — `kaggle benchmarks init -y --env-file F` writes a short-lived
   `MODEL_PROXY_URL` / `MODEL_PROXY_API_KEY` (~1 h), then `POST {MODEL_PROXY_URL}/openapi/chat/completions`.
   - **Blocked from this machine**: `403 User location "RU" is not supported for this model/API`.
2. **Hosted tasks** — `tasks push -f ...` → `tasks run -m <one-model>` → `tasks status` →
   `tasks download`. The worker gets proxy vars injected (~12 h). This route **works** and is what
   we use. One model per run; Kaggle Secrets do not work inside tasks.

Models in the user's plan (`docs/USER_PLAN_VERBATIM.md`) and the bring-up ladder
(`docs/DATAGEN_KAGGLE.md` §5), with the CLI slugs used there (resolve against the live catalogue at
run time):
- Cheap test targets first: **Claude Haiku 5.5** (`claude-haiku-5-5`), **Luna** (`gpt-5.6-luna`).
- Then strong models: Claude Opus / Sonnet 5.5 (`claude-opus-5-5-default`, `claude-sonnet-5-5-default`),
  Gemini Argon / 3.8 (`gemini-4-argon`, `gemini-3.8-flash`), OpenAI Astra / Sol 6.2
  (`gpt-6-astra`, `gpt-6.1-sol`).
- Hosted-only roster: Haiku 5.5, Luna, Opus, Astra; the interactive key reaches a 6-vision-model
  subset. `docs/KAGGLE_MODELS.md` / `KAGGLE_SKILL.md` hold the full catalogue and calling details.

Quota is billed to the Kaggle account (`kaggle benchmarks quota`); last observed spend $0.09/day,
$0.27/month. Check before large runs.

### What has actually been run on Kaggle
- Hosted task `rtp-cot-frontier-fullplan` on `google/gemini-3.8-flash`: 8/8 real model traces, all
  `finish_reason: stop`, with reasoning prefix + parseable Plan JSON from our harness prompts.
  Files: `data/local_4050/cot_frontier_traces.jsonl`, `reports/cot_frontier.json`. These are
  verbatim transport records; the plans were **not replayed or scored** in that run.
- The earlier Qwen-based CoT teacher was **cancelled and deleted** — no Qwen-generated CoT exists.
- `ops/score_frontier.py` replays those traces on the held-out seeds into the strict prediction
  format; its earlier syntax error is fixed (D-67), but **no replay report has been produced**.
- Interactive proxy from this machine is geo-blocked; use hosted tasks.
- Scope note (two different CoT paths): `docs/STATUS_HONEST.md` / `reports/cot_trace.json` record
  **0** frontier traces for the separate `ops/cot_trace.py` path because `RTP_SERVE_URL` is unset;
  that path is not the Kaggle-benchmarks hosted-task route above. Neither path is the multi-turn
  datagen loop (`docs/DATAGEN_KAGGLE.md`), which is built/fixture-tested but not run on Kaggle.

## 4. HuggingFace — datasets (assets)

All are public NVIDIA datasets, downloaded into `assets/cache/`, with attribution in
`assets/ROBOT_ATTRIBUTION.csv` and a manifest in `assets/simready_downloaded.json`.

| Dataset | Local path | Contents | Status |
|---|---|---|---|
| `nvidia/simready-assets` | `assets/cache/nvidia__simready-assets/` | 14 SimReady packages (USD geometry + Newton/MuJoCo physics payloads). | 9/14 import cleanly (G-13 PARTIAL); 5 have orphan-joint/asset errors. Texture-free subset downloaded; real albedo maps for 6 used objects now pulled into each package's `Textures/`. |
| `nvidia/menagerie` (or local cache) | `assets/cache/menagerie/` | `franka_fr3_v2` (single arm) and `franka_fr3_duo` (two-arm MJCF). | Used for the two-arm scene; arms render, not stepped/controlled. |

Notes:
- The 5 failing SimReady packages (bottle, 2 cabinets, counter, paperbag) have unparented joints or
  unreadable layers; they need `add_articulation()` before they can be rendered/stepped.
- The 6 textured objects (mug, plate, book, notebook, pencil, dragon fruit) have real albedo maps;
  the chair has no albedo and uses an explicit color.

## 5. HuggingFace — models

| Repo | Visibility | Contents | Status |
|---|---|---|---|
| `Reza-Nadimi/robot-team-packing` | **private** | Trained Qwen3.5-4B adapter / datasets published from Kaggle. | 9.32 GB dataset `rtp-model-qwen35-4b-851bf6e8` published from Kaggle (never uploaded from this machine). |
| `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` | public | Base models for SFT / zero-shot benchmarks. | 4B SFT smoke only (loss 1.0814 → 0.7424, 2 steps). Trained Brain **not scored**. |

Other HF repos in the account are untouched and out of scope.

## 6. What is NOT done (so nobody mistakes stubs for results)

- Frontier CoT plans not replayed/scored; `ops/score_frontier.py` is fixed but has no replay report.
- Trained adapter and all omni competitors are `not run` in `reports/omni_comparison.json`.
- Multi-turn datagen is **built and fixture-tested on SIM-POSE but not run on Kaggle**
  (`docs/DATAGEN_KAGGLE.md`); no certified arm model feeds SFT yet (arms first, D-62).
- No GRPO, no M1–M8 benchmark matrix (`IMPLEMENTATION_PLAN.md` §10), no impedance safety
  validation — G-6b is a SIM-ABSTRACT supervisor on a point mass, hardware/L0/L1 sign-off is
  Claude's (D-22, D-66).
- Scene bundle, view library and GPU replay gate are not built; the assembled SimReady scene
  (`reports/scene_manifest.json`) is not stepped or rendered; no certified arm skills.

## 7. Useful commands

```bash
# Kaggle quota (needs proxy)
export HTTP_PROXY=http://localhost:10808 HTTPS_PROXY=http://localhost:10808
kaggle benchmarks quota --format json

# List available models
kaggle benchmarks tasks list-models 2>/dev/null | head

# Render the stage-1 preview (NO proxy)
source ops/env.sh && $ISAAC_PY ops/preview_scene.py --width 640 --height 480
```
