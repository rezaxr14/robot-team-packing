# Kaggle + HuggingFace resources (for Claude / any collaborator)

This document records every external account, dataset, model and credential the project uses, so a
collaborator knows what exists and what is still missing. No token values are stored here — only
which `.env` keys hold them.

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

Models available on this account (from `docs/KAGGLE_MODELS.md` / `KAGGLE_SKILL.md`, shortlist):
- Anthropic: Claude Opus, Sonnet, **Haiku 3.5 / 4.5** (Haiku is cost-effective — test target).
- Google: Gemini 3.8, 3.5, 3.5 Flash, Flash Lite, **Gemini 3.8 Flash** (used for CoT traces).
- OpenAI: Astra 6.2, Sol 6.2, **Luna** (cost-effective — test target).

Quota is billed to the Kaggle account (`kaggle benchmarks quota`); last observed spend $0.09/day,
$0.27/month. Check before large runs.

### What has actually been run on Kaggle
- Hosted task `rtp-cot-frontier-fullplan` on `google/gemini-3.8-flash`: 8/8 real model traces, all
  `finish_reason: stop`, with reasoning prefix + parseable Plan JSON from our harness prompts.
  Files: `data/local_4050/cot_frontier_traces.jsonl`, `reports/cot_frontier.json`.
- The earlier Qwen-based CoT teacher was **cancelled and deleted** — no Qwen-generated CoT exists.
- Plans were **not replayed or scored** yet. `ops/score_frontier.py` exists but still has an invalid
  placeholder import line (`from rtp.brain import ReplayBrain if False else None`) and has not been run.
- Interactive proxy from this machine is geo-blocked; use hosted tasks.

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

- Frontier CoT plans not replayed/scored; `ops/score_frontier.py` broken (invalid import line).
- Trained adapter and all omni competitors are `not run` in `reports/omni_comparison.json`.
- No multi-turn frontier data generation yet (the main user request).
- No GRPO, no full benchmark matrix, no impedance safety validation (D-22: Claude must decide).
- Scene is rendered only — not stepped, no manipulation, no arm policy.

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
