# Additional omni / VLA / multimodal robot models — research findings

**Status: research only. No model in this file has been downloaded, run, fine-tuned or scored in this repository.**
Every statement of fact below is tagged `[EXT — not reproduced here]` (primary source consulted, not independently
verified locally). Statements about our own plans are tagged `[REPO]`, and judgements are tagged `[ANALYSIS]`.
There are **no measured results** in this file; any future table cell for these models must remain `not run`
until a run card with a config hash, seed list and git SHA exists (`docs/AGENT_BUILD_SPEC.md` §0, `[REPO]`).

## 0. Method and provenance

- Access date for **all** URLs below: **2026-11-20**. `[EXT — not reproduced here]`
- The `websearch` integration returned HTTP 402 for every query on 2026-11-20; evidence was gathered by direct
  `webfetch` of primary sources: Hugging Face model cards and the HF model API, arXiv abstracts/API, GitHub
  READMEs, and the OpenVLA project page. `[TOOL — not reproduced here]`
- Context read first: `IMPLEMENTATION_PLAN.md` §10 (M1–M8), D-64 and D-68 in `DECISIONS.md`,
  `docs/AGENT_BUILD_SPEC.md` WP-I (Tiers 1–3), `reports/omni_comparison.json`, and
  `paper/sections/04_experiments.tex`. `[REPO]`
- The plan already contains: M1–M8 (base Qwen3.5-4B/9B, SFT, SFT+GRPO k=4/k=8, frontier zero-shot and in-harness,
  base 4B in harness); LeRobot VLA Tier 2 (ACT, Diffusion Policy, π0, π0.5, SmolVLA, GR00T N1.5/N1.7);
  RoboTwin 2.0 Tier 3 (ACT/DP/DP3/RDT/π0); and Track A (VIKI-R on VIKI-Bench). `[REPO]`
- "Same bounded-impedance interface" below means the planned `isaac/lerobot_env.py` path: end-effector delta
  pose + gripper at 10–30 Hz through the same clamped impedance layer used by the Brain's skills, scored with
  the same held-out seeds `< 100000` and the same `final_check`. `[REPO]`

## 1. Already-planned rows — checked for duplication, not additions

| Name / version | Key external fact (accessed 2026-11-20) | Verdict |
|---|---|---|
| NVIDIA GR00T N1.5-3B | HF repo created 2025-05-28; 3B `[EXT — not reproduced here]` | Already in D-68/WP-I Tier 2 — do not duplicate |
| NVIDIA GR00T N1.7-3B | HF repo created 2026-02-25; NVIDIA Open Model License, `Cosmos-Reason2-2B` backbone, 3B; 4 denoising steps at 11.7 Hz on H100, 7.1 Hz on L20 (external timing table) `[EXT — not reproduced here]` | Already in D-68/WP-I Tier 2 — do not duplicate |
| Physical Intelligence π0 (`lerobot/pi0_old`) | HF repo created 2025-02-03; Apache-2.0 tag; ~3B class `[EXT — not reproduced here]` | Already in Tier 2 and `omni_comparison.json` |
| Physical Intelligence π0.5 (`lerobot/pi05_base`) | Paper 2025-04-22; HF repo created 2025-09-09; Gemma licence; 3.6B F32; LeRobot card states only the flow-matching head was released (not subtask prediction, action tokenisation or RL) `[EXT — not reproduced here]` | Already in Tier 2 — do not duplicate |
| SmolVLA (`lerobot/smolvla_base`) | Paper 2025-06-02; Apache-2.0; 0.5B; card states single-GPU training and consumer-GPU/CPU deployment `[EXT — not reproduced here]` | Already in Tier 2 — do not duplicate (but it is the only planned row that comfortably fits 6 GB) |
| VIKI-R (VIKI-Bench) | arXiv 2506.09049 (v1 2025-06-10, NeurIPS 2025 D&B); VLM + CoT SFT + RL `[EXT — not reproduced here]` | Already in paper Track A — do not duplicate |

Sources (all accessed 2026-11-20):
`https://huggingface.co/api/models?search=GR00T-N1&limit=50`,
`https://huggingface.co/nvidia/GR00T-N1.7-3B`,
`https://huggingface.co/api/models?search=pi0&limit=30`,
`https://huggingface.co/lerobot/pi05_base`,
`https://huggingface.co/lerobot/smolvla_base`,
`https://arxiv.org/search/?query=VIKI-Bench&searchtype=all`. `[EXT — not reproduced here]`

## 2. Additional candidates (14 entries)

### 2.1 NVIDIA Isaac GR00T N1.6-3B

- **Name / org / date.** `nvidia/GR00T-N1.6-3B`, NVIDIA; HF repo created 2025-12-01. `[EXT — not reproduced here]`
- **Weights / licence / size.** Open, not gated; NVIDIA OneWay Noncommercial licence (model card); 3,286,608,832 BF16 parameters (≈6.6 GB weights). `[EXT — not reproduced here]`
- **Modalities / action space.** Variable number of RGB frames + language string + proprioception vector (N1.X family card); continuous action vectors decoded per embodiment by an MLP (flow-matching DiT head). `[EXT — not reproduced here]`
- **Compute at our scale.** 3B BF16 weights alone exceed the 4050's 6 GB; the published timing table covers H100/H20/RTX Pro/L40/L20/Orin and NVIDIA states its models are optimised for NVIDIA GPU systems — no 6 GB recipe is documented. `[EXT — not reproduced here]` T4 16 GB LoRA is plausible from weight size but unmeasured. `[ANALYSIS]`
- **Same interface / seeds.** In principle yes through the planned LeRobot/GR00T env adapter emitting EE deltas into the impedance layer, with the same held-out seeds; not demonstrated here. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://huggingface.co/nvidia/GR00T-N1.6-3B`; `https://huggingface.co/api/models/nvidia/GR00T-N1.6-3B`. `[EXT — not reproduced here]`
- **Verdict: KEEP as future work.** N1.7 is already in the plan and supersedes N1.6; add N1.6 only as an explicit version-sweep row if compute allows. `[ANALYSIS]`

### 2.2 Physical Intelligence / Hugging Face π0-FAST (`lerobot/pi0fast-base`)

- **Name / org / date.** π0-FAST base checkpoint, Physical Intelligence (method) / Hugging Face LeRobot (checkpoint); FAST paper 2025-01-16, HF repo created 2026-01-09. `[EXT — not reproduced here]`
- **Weights / licence / size.** Open, not gated; Gemma licence tag; 2,923,335,408 F32 parameters (~3B). `[EXT — not reproduced here]`
- **Modalities / action space.** Multi-view images, proprio/state, optional language instruction; output continuous actions **decoded from FAST action tokens** (next-token cross-entropy objective), not a flow head. `[EXT — not reproduced here]`
- **Compute at our scale.** LeRobot recipes default to bf16 and a single CUDA device; 3B is too large for 4050 6 GB training and needs T4/4090-class LoRA; inference on a 4050 is plausible but unmeasured. `[EXT + ANALYSIS]`
- **Same interface / seeds.** Yes via the planned LeRobot policy adapter (`PI0FastPolicy.select_action`) → EE delta + gripper through the impedance layer, same held-out seeds. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://huggingface.co/lerobot/pi0fast-base`; `https://arxiv.org/abs/2501.09747`. `[EXT — not reproduced here]`
- **Verdict: ADD.** It is the only open, LeRobot-native **token-action** baseline that would sit beside the planned flow-matching π0/π0.5 rows. `[ANALYSIS]`

### 2.3 X-VLA 0.9B (`lerobot/xvla-base`)

- **Name / org / date.** X-VLA-0.9B, Tsinghua/AIR and collaborators (paper) / Hugging Face LeRobot (checkpoint); paper 2025-10-11, HF repo created 2025-11-19. `[EXT — not reproduced here]`
- **Weights / licence / size.** Open, not gated; Apache-2.0; 879,738,545 F32 parameters (0.9B). `[EXT — not reproduced here]`
- **Modalities / action space.** Multi-view images, proprio/state, optional language; continuous actions via flow matching, with soft prompts for cross-embodiment control. `[EXT — not reproduced here]`
- **Compute at our scale.** <1B weights fit in 6 GB for inference and plausibly for LoRA on 4050/T4; Kaggle CPU hosted task is for our JSON-action Brain and is not a policy-inference target. `[EXT + ANALYSIS]`
- **Same interface / seeds.** Yes — LeRobot-native (`XVLAPolicy.select_action`), so it plugs into the same planned env adapter and held-out seed manifest. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://huggingface.co/lerobot/xvla-base`; `https://huggingface.co/api/models/lerobot/xvla-base`; `https://arxiv.org/abs/2510.10274`. `[EXT — not reproduced here]`
- **Verdict: ADD.** Smallest open, LeRobot-supported cross-embodiment VLA found; the only new VLA that is clearly trainable within the 6 GB/16 GB budget. `[ANALYSIS]`

### 2.4 NSDL (Neuro-Symbolic Division of Labor), 3B / 7B / 8B

- **Name / org / date.** NSDL (primary row `XinchengTan/nsdl-3b-sft-viki`), Zhejiang University / Hong Kong Polytechnic University; arXiv 2611.16671 submitted 2026-11-12, checkpoint repo created 2026-11-11. `[EXT — not reproduced here]`
- **Weights / licence / size.** Open, not gated; code Apache-2.0; 3B checkpoint derives from Qwen2.5-VL-3B and carries the Qwen Research licence (non-commercial research use); 7B/8B checkpoints derive from Apache-2.0 Qwen2.5-VL-7B / Qwen3-VL-8B. 3B checkpoint: 4,065,787,904 BF16 parameters (≈8.1 GB storage). `[EXT — not reproduced here]`
- **Modalities / action space.** Scene image + instruction; tools `detect` (YOLO evidence) and `schedule` (deterministic solver Φ); output is a plan sketch (`[GOAL]` / `[ASSIGN]` / `[SKETCH]`) expanded into a **symbolic joint plan**, not motor actions. `[EXT — not reproduced here]`
- **Compute at our scale.** 3B ≈8.1 GB weights → T4 16 GB inference/training plausible, 4050 6 GB tight; the solver itself is CPU-cheap; Kaggle CPU hosted task could in principle run the symbolic half but not the VLM. `[EXT + ANALYSIS]`
- **Same interface / seeds.** Not through the impedance layer: it is a planner, so it belongs to the symbolic/harness tier. It can be scored on the same held-out families only if its VIKI-L2 checker is made seeded and its plan adapter emits our skill JSON — its published scorer must be audited like our existing VIKI scorer. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://arxiv.org/abs/2611.16671`; `https://github.com/Xincheng-Tan/NSDL`; `https://huggingface.co/api/models/XinchengTan/nsdl-3b-sft-viki`. `[EXT — not reproduced here]`
- **Verdict: ADD (Track A extension).** It is the first open, tool-using neuro-symbolic competitor on VIKI-L2 found since VIKI-R, and its detect/schedule split is directly comparable to our Brain + verifier story. `[ANALYSIS]`

### 2.5 Gemini Robotics (1.0)

- **Name / org / date.** Gemini Robotics, Google DeepMind; arXiv 2503.20020, 2025-03-25. `[EXT — not reproduced here]`
- **Weights / licence / size.** Closed; no weights, licence or parameter count in the consulted primary source. `[EXT — not reproduced here]`
- **Modalities / action space.** Multimodal (images) + language → continuous robot actions; companion Gemini Robotics-ER does detection, pointing, trajectories, correspondences and 3D boxes. `[EXT — not reproduced here]`
- **Compute at our scale.** No local compute path; managed/proprietary access only. `[EXT + ANALYSIS]`
- **Same interface / seeds.** No — cannot be loaded into our env, so no impedance interface and no control of our held-out seeds. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://arxiv.org/abs/2503.20020`. `[EXT — not reproduced here]`
- **Verdict: REJECT.** No reproducible artefact; listing it as a VLA row would imply a result we cannot produce. `[ANALYSIS]`

### 2.6 Gemini Robotics 1.5 / Gemini Robotics-ER 1.5

- **Name / org / date.** Google DeepMind; arXiv 2510.03342, published 2025-10-02 (v3 2025-11-28). `[EXT — not reproduced here]`
- **Weights / licence / size.** Closed; no weights or size in the consulted source. `[EXT — not reproduced here]`
- **Modalities / action space.** Multi-embodiment VLA, images + language, actions interleaved with natural-language reasoning; ER 1.5 produces embodied reasoning outputs. `[EXT — not reproduced here]`
- **Compute at our scale.** No local path. `[ANALYSIS]`
- **Same interface / seeds.** No. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://arxiv.org/abs/2510.03342`. `[EXT — not reproduced here]`
- **Verdict: REJECT.** Closed; belongs to the already-planned frontier-row discussion (M6/M7) only if an accessible endpoint exists. `[ANALYSIS]`

### 2.7 Gemini Robotics On-Device (GROD)

- **Name / org / date.** Google DeepMind / Gemini Robotics team; on-device model announced in 2025; the consulted 2026 study refers to a **managed SFT API** with no access to weights, gradients or losses. `[EXT — not reproduced here]`
- **Weights / licence / size.** Closed; no public weights or parameter count. `[EXT — not reproduced here]`
- **Modalities / action space.** Images + language → robot actions; adaptation only through submitted demonstrations and a closed API. `[EXT — not reproduced here]`
- **Compute at our scale.** Cannot be trained or served locally. `[ANALYSIS]`
- **Same interface / seeds.** No — deployment and its reward loop are outside our harness. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://arxiv.org/abs/2607.29172` (CLIFT, 2026-07-31, describes GROD's managed-API regime). `[EXT — not reproduced here]`
- **Verdict: REJECT.** No weights, no controllable seeds, no local interface. `[ANALYSIS]`

### 2.8 Physical Intelligence π0.6

- **Name / org / date.** No official π0.6 paper, model card or checkpoint was found in the consulted sources. `[EXT — not reproduced here]`
- **Weights / licence / size.** Not established. Third-party HF repos with names containing `pi06` exist but are not attributable to Physical Intelligence in the consulted metadata. `[EXT — not reproduced here]`
- **Modalities / action space / compute / interface.** Not established for an official release. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://huggingface.co/api/models?search=pi06&limit=20`; arXiv API query `all:"pi0.6"` returned 0 results. `[EXT — not reproduced here]`
- **Verdict: REJECT.** Do not add a model that cannot be pinned to a primary-source artefact. `[ANALYSIS]`

### 2.9 OpenAI generalist VLA / omni robot model

- **Finding.** No OpenAI-released VLA with public or API-accessible image+language→robot-action interface was identified in the consulted primary sources. `[EXT — not reproduced here]`
- **Consequence.** There is nothing to pin a run card to; generalist image+language models are already covered as frontier Brains via `rtp/datagen_run.py` in `mode="eval"` (M6/M7). `[REPO + ANALYSIS]`
- **Verdict: REJECT (no identified artefact).** `[ANALYSIS]`

### 2.10 OpenVLA-7B

- **Name / org / date.** `openvla/openvla-7b`, Stanford/UC Berkeley/TRI et al.; paper 2024-06-10. `[EXT — not reproduced here]`
- **Weights / licence / size.** Open, not gated; MIT; 7,541,237,184 BF16 parameters, ~15 GB storage. `[EXT — not reproduced here]`
- **Modalities / action space.** Fused SigLIP+DinoV2 visual encoder and Llama-2-7B backbone; discretised **action tokens** decoded to continuous robot actions; LoRA reported as best memory/performance trade-off. `[EXT — not reproduced here]`
- **Compute at our scale.** Upstream pretraining used 64 A100 GPUs for 15 days (external project page); 7.5B cannot be trained on 4050 6 GB or usefully served there, and 16 GB T4 would need QLoRA; not LeRobot-native, so a custom adapter would be required. `[EXT + ANALYSIS]`
- **Same interface / seeds.** In principle via a custom env adapter and the same held-out seeds, but not demonstrated. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://openvla.github.io/`; `https://huggingface.co/api/models/openvla/openvla-7b`; `https://arxiv.org/abs/2406.09246`. `[EXT — not reproduced here]`
- **Verdict: KEEP as future work.** An important open token-action VLA, but infeasible at our 6 GB scale and duplicative of the π0-FAST role. `[ANALYSIS]`

### 2.11 LingBot-VLA-4B

- **Name / org / date.** `robbyant/lingbot-vla-4b`, Robbyant; paper arXiv 2601.18692, 2026-01-26 (v4 2026-06-15); separate RoboTwin-posttrained checkpoint repo created 2026-03-09. `[EXT — not reproduced here]`
- **Weights / licence / size.** Open weights, not gated; licence not stated in the consulted API tags (check model card before use); ~4B class. `[EXT — not reproduced here]`
- **Modalities / action space.** Dual-arm VLA trained on ~20,000 hours of real robot data; continuous actions. `[EXT — not reproduced here]`
- **Compute at our scale.** Upstream reports an 8-GPU codebase; no single-6 GB recipe found. `[EXT + ANALYSIS]`
- **Same interface / seeds.** Not LeRobot-native; would need a custom adapter, so same seeds are possible in principle but not plug-in. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://arxiv.org/abs/2601.18692`; `https://huggingface.co/api/models?search=LingBot-VLA&limit=10`. `[EXT — not reproduced here]`
- **Verdict: KEEP as future work.** Strong dual-arm VLA and directly relevant to RoboTwin-style rows, but too large and too custom for the current hardware budget. `[ANALYSIS]`

### 2.12 LoopedWAM (0.58B policy)

- **Name / org / date.** `anhdao69/LoopedWAM`; arXiv 2611.25595 submitted 2026-11-19; HF repo created 2026-11-19. `[EXT — not reproduced here]`
- **Weights / licence / size.** Open, Apache-2.0 tag, not gated; 0.58B policy parameters reported in the paper. `[EXT — not reproduced here]`
- **Modalities / action space.** World-action model: images + language → continuous actions plus latent future prediction; recurrent shared transformer core. `[EXT — not reproduced here]`
- **Compute at our scale.** Paper reports 7.4× lower peak GPU memory than Fast-WAM and H100 latency numbers; no 6 GB measurement. `[EXT — not reproduced here]`
- **Same interface / seeds.** Custom stack, no LeRobot adapter; same seeds only after new env work. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://arxiv.org/abs/2611.25595`; `https://huggingface.co/api/models?search=LoopedWAM&limit=10`. `[EXT — not reproduced here]`
- **Verdict: KEEP as future work.** Very promising size/efficiency, but one day old at access time and unreproduced; wait for a stable release before adding a row. `[ANALYSIS]`

### 2.13 RoboTwin-Phys (benchmark, not a model)

- **Name / org / date.** RoboTwin-Phys; arXiv 2609.26292, 2026-09-22. `[EXT — not reproduced here]`
- **Weights / licence / size.** Benchmark and 5,000+ demonstrations, not a policy; no model weights. `[EXT — not reproduced here]`
- **Modalities / action space.** Continuous physical-condition variation over 13 attributes; evaluates WAMs and VLAs on dual-arm tasks. `[EXT — not reproduced here]`
- **Compute at our scale.** SAPIEN-style simulation; heavier than our pose backend. `[EXT + ANALYSIS]`
- **Same interface / seeds.** Not a row in our matrix; would require a separate physics-diversity protocol. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://arxiv.org/abs/2609.26292`. `[EXT — not reproduced here]`
- **Verdict: KEEP as future work.** Its mass/friction/dynamics axis matches our heavy/fragile and push families, but it is a benchmark extension, not an additional comparison model. `[ANALYSIS]`

### 2.14 Qwen-CIALLO

- **Name / org / date.** Qwen-CIALLO; arXiv 2611.13732, 2026-11-10. `[EXT — not reproduced here]`
- **Weights / licence / size.** Weights were **not verified** in the consulted sources; backbone is Qwen3.5-4B (the same family as our Brain). `[EXT + ANALYSIS]`
- **Modalities / action space.** Images + language → shared behavioural-intent tokens decoded by an embodiment-conditioned action codec into robot-specific actions. `[EXT — not reproduced here]`
- **Compute at our scale.** 4B-class backbone; 4050 6 GB training would need QLoRA and a 256-token ceiling (our measured local constraint), T4 more plausible; no recipe found. `[REPO + EXT + ANALYSIS]`
- **Same interface / seeds.** Not through the impedance layer as published; needs codec integration. `[ANALYSIS]`
- **Source (accessed 2026-11-20).** `https://arxiv.org/abs/2611.13732`. `[EXT — not reproduced here]`
- **Verdict: KEEP as future work.** Conceptually close to our "Brain plans skills, arm executes bounded deltas" split, but no verified weights and no runnable adapter. `[ANALYSIS]`

## 3. Ranked shortlist of feasible additions

All three rows would be **`not run`** until a run card exists; no performance claim is made here. `[REPO + ANALYSIS]`

1. **X-VLA 0.9B** — ADD as a Tier-2 VLA row: 0.9B Apache-2.0 LeRobot-native flow-matching VLA, the only new candidate that realistically fine-tunes inside 6 GB/16 GB and uses the planned EE-delta impedance adapter and held-out seeds. `[EXT + ANALYSIS]`
2. **π0-FAST (`lerobot/pi0fast-base`)** — ADD as a Tier-2 VLA row: 3B Gemma-licensed, LeRobot-native autoregressive FAST-token VLA, giving the comparison a discrete-action baseline beside the existing flow-matching π0/π0.5 rows. `[EXT + ANALYSIS]`
3. **NSDL 3B (SFT, VIKI)** — ADD as a Track-A extension: open-weight neuro-symbolic planner with `detect`/`schedule` tools on VIKI-L2, the closest published competitor to our Brain+verifier design that can be rerun with our own seeded scorer. `[EXT + ANALYSIS]`

## 4. Which rows they complement, not duplicate

- **X-VLA 0.9B** complements the Tier-2 **ACT / Diffusion Policy / SmolVLA** rows (small policies) and adds a cross-embodiment soft-prompt architecture; it does **not** duplicate π0/π0.5/GR00T (3–4B flow-matching) or M1–M8 (symbolic Brain with tools/memory). `[ANALYSIS]`
- **π0-FAST** complements the Tier-2 **π0 and π0.5** rows (same family, different action head: FAST tokens vs flow matching) and RDT (diffusion); it does **not** duplicate SmolVLA (small flow-matching) or any M-row. `[ANALYSIS]`
- **NSDL 3B** complements paper **Track A (VIKI-R)** and matrix rows **M6/M7** (frontier zero-shot vs in-harness) on the same VIKI-L2 data; it does **not** duplicate M1–M5 (our Qwen3.5 Brain with SFT/GRPO) because it uses a different backbone and an explicit deterministic solver. `[ANALYSIS]`
- **Not added:** GR00T N1.6 (superseded by the planned N1.7), OpenVLA-7B and LingBot-VLA-4B (compute/custom-stack blockers), Gemini Robotics variants (closed), π0.6 / OpenAI VLA / Qwen-CIALLO (no pinned artefact), LoopedWAM (too new), RoboTwin-Phys (benchmark, not a model). `[ANALYSIS]`

## 5. Rules for using this file

- Do **not** convert any external claim above into a number in a table or in `paper/`; publication claims require a local run card. `[REPO]`
- Pin the exact HF commit, licence and model revision in any future run card; several licences above are restrictive (Gemma, NVIDIA non-commercial, Qwen Research). `[EXT + REPO]`
- For NSDL and any future VIKI-Bench competitor, audit the scorer for unseeded start-position draws before comparing, as done for VIKI-R in `paper/sections/04_experiments.tex` §`scorer`. `[REPO]`
- This file was created by a research subagent using `opencode/hy4-flash-free`; no other repository file was edited. `[TOOL]`
