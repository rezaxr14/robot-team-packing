# Architecture V2 — two Eyes, one vision Brain, two blind Arms

Status: **design + verified model/platform facts**. Implementation status is marked per section.
This document is the contract we build against; `DECISIONS.md` records who decided what.
Nothing here is a result. Every table cell in the paper stays `\todo` until a reportable run exists.

Everything labelled **[V]** was read out of a real file or an upstream source at the date given.
Everything labelled **[U]** is unverified and must not be quoted as fact.

---

## 1. What the system is

A kitchen packing workspace with two robot arms. One `Qwen3.5-4B` vision-language model (the
**Brain**) plans for both arms and drives both **Eyes**. The Eyes are *not* language models: they
are ceiling-mounted cameras on movable mounts running a YOLO-class detector plus classical CV
kernels. They return geometries, detections, masks and optional image tiles — never prose. The
Brain can spend part of its budget asking an Eye to move, rotate, zoom, apply a kernel, or hand
back a crop.

The Arms never see. They receive one symbolic skill at a time (`grasp`, `place`, `handover`,
`weigh`, `release`, `retreat`, `push_aside`) and report `TOUCH`/`BLOCKED` only.

The scientific claim is **attribution**: when the team fails, which component caused it
(Eye missed / Brain did not ask / Arm executed badly), and how much active eye work is worth.
See §12.

```
            harness .md  ──BM25 + dense (Qdrant)──┐
                                                 v
   Eye A ──OBS/EYE_REPLY──►  [ Brain: Qwen3.5-4B ]  ◄── instruction
   Eye B ──OBS/EYE_REPLY──►    │        ▲                 (text)
       ▲                      │        │
       └──── EYE_CMD ─────────┘        │  plan DAG (Task: arm, skill, obj, params, deps)
                                       ▼
                             verify (static | dry run) ──reject──► replan
                                       │
                                       ▼
                             scheduler (per round, one task head per arm, bag lock)
                                       │
                          Arm L / Arm R (blind: proprio + contact)
                                       │  TOUCH / BLOCKED / STATUS
                                       ▼
                             rules.check_final → success? → attribution(Eye|Brain|Arm)
```

## 2. Eyes: mount model, geometry, occlusion

Implementation target: `rtp/eyes/mount.py`, `rtp/eyes/visibility.py`. **Not implemented yet.**

Design defaults (ours, all in one dataclass in `rtp/eyes/mount.py`):

| item | default | meaning |
|---|---|---|
| ring centre | workspace counter centre | each Eye orbits a circle around the arms |
| ring radius | 1.2–1.8 m | enforced standoff; never overhead |
| height | 2.2–2.6 m | ceiling mount |
| stations | 12 per ring (`k·30°`) | discrete orbit positions; `orbit` moves ±1 or to a named station |
| off-nadir tilt | 25°–55° | skewed view, so top faces are foreshortened |
| pan range | −60°…+60° around the station azimuth | `pan` |
| tilt range | 15°…65° off-nadir | `tilt` |
| zoom | 1×–4× digital | `zoom` |
| FOV | 70° H × 50° V (configurable) | used for the visibility map |
| start station | sampled per episode, both Eyes not co-located | reproducibility via episode seed |

Occlusion: the scene contains columns, a shelf, the bag wall and (optionally) the other Eye's mount.
Visibility is computed by casting rays from the camera pose to the centres of a `8 × 6` grid over
the counter, clipped by the FOV cone and the depth/normal test. The result is
`visibility: {region_id: 0..1}` plus a list of occluders. Neither Eye sees the whole counter by
construction; the two Eyes are sampled so their union is still incomplete on a fraction of seeds
(that fraction is a knob, `union_gap`, and it is what makes "ask the other Eye" a real decision).

## 3. Eye tool surface (what an Eye can be asked to do)

One table, both Eyes. `cost` is in budget slots. **Design default: 6 slots per planning round,
24 per episode**; both are profile fields so an ablation can change them.

| tool | args | returns | cost |
|---|---|---|---|
| `orbit` | `station` \| `±1` | viewpoint id, visibility delta | 1 |
| `pan` | `deg` (clamped) | aim vector, visible mask | 1 |
| `tilt` | `deg` (clamped) | aim vector, visible mask | 1 |
| `zoom` | `1..4`, target region or object | cropped frame + scale | 1 |
| `kernel` | `sobel\|canny\|laplacian\|blur\|clahe\|hsv\|threshold` | scalar stats (edge density, contrast, hue hist) + optional tile | 1 |
| `detect` | `items\|obstacles\|all`, `min_conf` | boxes + class + conf + centroid + depth | 1 |
| `segment` | `class` | masks, area, free-space estimate for a compartment | 1 |
| `focus` | `object` | re-aim + zoom-to-fit on that object | 1 |
| `crop` | `object` \| `box` | image tile returned to the Brain | 2 |
| `snapshot` | — | full frame at current pose | 3 |
| `describe` | — | geometric features only (no language) | 1 |

Rules of the surface:
- Every call is logged with `(episode, round, eye, tool, args, cost, latency_ms, ok)`.
  This log is what lets us plot *success vs eye calls* and *did this kernel request pay for itself*.
- Calls are idempotent with respect to the plan: a failed `EYE_CMD` returns `ok=false` with a
  reason, never an exception into the planner.
- The Brain sees `budget_left` in every reply. Exhausting the budget forces it to plan with **current**
  observations; that failure mode is what the "no look" ablation measures.
- `detect`/`segment` are the Eyes' own trained products. `crop`/`snapshot` are the only paths that
  put pixels into the Brain, and they are deliberately the most expensive.

**Eye perception stack (not an LLM):** detector = `ultralytics` YOLO11n-seg (instance masks);
kernels = OpenCV. *Licence note:* `ultralytics` is **AGPL-3.0** — acceptable for research, flagged
in `DECISIONS.md`; a BSD-3 fallback (`torchvision` RetinaNet/Mask R-CNN) is a one-line swap in
`rtp/eyes/detector.py`. **[U]** detector accuracy on ceiling-skewed renders — no training run yet.
Classes: 8 packing items + 2 distractors + obstacles `{column, shelf, bag_wall, cable, wall}`.

Training data: Isaac-rendered frames with randomized station, azimuth, tilt, lighting, motion blur
and occluders; held-out **stations** (not just held-out frames) plus MuJoCo renders, so the report
can state a sim-to-sim gap rather than an in-distribution mAP.

## 4. Message schemas

Extends `rtp/messages.py` (which today carries ABILITY/SCENE/CLAIM/TOUCH/BLOCKED/STATUS/AMEND).
All messages validate on construction; malformed messages are a Brain error, not an Eye error.

```json
OBS       {"eye":"A","station":3,"pose":{"pos":[x,y,z],"quat":[w,x,y,z]},"fov":[70,50],
           "detections":[{"cls":"book","conf":0.91,"box":[u0,v0,u1,v1],"centroid_m":[x,y,0.03],
                          "depth":1.83}],
           "occluders":["column_1"],"visibility":{"r0c1":0.0,"r3c4":0.65},
           "frame_id":"A-0031","conf":0.74}
EYE_CMD   {"eye":"A","tool":"focus","args":{"object":"lunchbox"},"request_id":"q7","budget_left":4}
EYE_REPLY {"request_id":"q7","ok":true,"result":{"pose":{...},"zoom":2.0,"frame_id":"A-0032",
           "detections":[...]},"latency_ms":41,"cost":1,"frame_id":"A-0032"}
SCENE     {"objects":[{"name":"book","cls":"book","region":"mid","occluded":false,"conf":0.8,
                       "mass_guess":0.6}],"eye":"A","frame_id":"A-0031"}
```

`OBS` messages from both Eyes are fused by region+class into the Brain's object memory, exactly as
`SCENE` is fused today (`rtp/perception.py`), so ordering never matters. `EYE_REPLY` is *addressed*;
it updates the Eye's view state and the object memory, and it is the only thing that can change
`budget_left`.

## 5. Brain rules.md / skills.md

The Brain gets its own instruction files, retrieved into the prompt like every other `.md`:

- `harness_files/brain/rules.md` — hard constraints only: which arm may enter which region, bag lock
  discipline, "no plan that the dry run rejects", "never ask an Eye to do what an existing OBS already
  answers", "state the reason for every eye call", "if budget is 0, plan with what you have",
  "never emit joint values or waypoints".
- `harness_files/brain/skills.md` — the two channels in one file: the arm skills with their legal
  params (`rtp/skills.py` is the source of truth) and the eye tools with their args/costs (§3), each
  with one canonical example call and one canonical *bad* call.
- `harness_files/eyes/rules.md`, `harness_files/eyes/tools.md` — the Eyes' own spec: what each tool
  returns, what counts as failure, how to report occlusion and low confidence.

These are prompt material and the eval fixture for format rewards (GRPO), not code.

## 6. Brain → Arm contract (unchanged)

`rtp/plan.py::Task{id, arm, skill, obj, params, deps}` and `Plan.from_dict` remain the whole
contract. Legal params are declared in `rtp/skills.py`: `grasp.force` (N), `place.compartment`
(`main|front|laptop`), `place.orientation` (`upright|flat`), `handover.to` (`L|R`); `weigh`,
`release`, `retreat`, `push_aside` take none. The Brain never emits joint values, waypoints, gains
or velocities. Cartesian waypoints and joint targets are produced downstream by differential IK + PD
(`isaac/packing_sim.py::goto`, `mjsim/backend.py`).

## 7. Qwen3.5-4B — verified structure

Read from the downloaded checkpoint (`~/models/Qwen3.5-4B/config.json`,
`model.safetensors.index.json`, 2 shards, `total_size = 9 319 737 856` B ≈ 9.32 GB) and from
`transformers 5.23.0`'s `models/qwen3_5/modeling_qwen3_5.py` in the active venv.

| property | value |
|---|---|
| class | `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5` |
| text layers | 32, layout `linear, linear, linear, full` × 8 (`full_attention_interval: 4`) |
| hidden / MLP | 2560 / 9216 SiLU, dense, **no MoE** |
| full attention | 16 Q heads, **4 KV** heads, `head_dim 256`, `attn_output_gate: true`, `q_norm`/`k_norm` |
| Gated DeltaNet | 16 key / **32 value** heads, `head_dim 128`, `conv_kernel_dim 4`, `mamba_ssm_dtype: float32` |
| RoPE | partial rotary factor 0.25 (64 dims), theta 1e7, interleaved mRoPE sections `[11,11,10]` |
| vocab / embeddings | 248 320, tied |
| context | 262 144 (`max_position_embeddings`) |
| vision | depth 24, hidden 1024, 16 heads, patch 16, `spatial_merge_size 2`, `out_hidden_size 2560`, `deepstack_visual_indexes: []` |
| extra head | `mtp_num_hidden_layers: 1` (MTP head ships in the checkpoint) |
| attn backends | `_supports_flash_attn = True`, `_supports_sdpa = True` (vision tower also routes through `ALL_ATTENTION_FUNCTIONS`, so `flash_attention_2` covers image tokens too) |

Consequences that are design constraints, not opinions:

1. **DeltaNet recurrence must stay fp32** even under bf16/fp16 autocast (`mamba_ssm_dtype: float32`).
   `ops/check_model.py` will assert this on the loaded module; a mixed-dtype run must fail loudly.
2. Only 8 of 32 layers are full attention, so FlashAttention-2 applies to those 8 plus the 24 vision
   blocks, not to the 24 linear-attention layers.
3. `head_dim 256` is inside FlashAttention-2's supported range ("all head dimensions up to 256",
   backward for 256 on consumer GPUs since 2.5.5, no dropout) — **[V]** from the FA2 README.
   FA2's mainline CUDA build requires Ampere/Ada/Hopper; Turing support lives in a separate
   `flash-attention-turing` repo with a core subset — **[V]** same README. So T4 = SDPA.

### Trainable blocks (adapter coverage, verified module names)

| block type | module(s) | adapter targets | default |
|---|---|---|---|
| full attention | `Qwen3_5Attention` | `q_proj, k_proj, v_proj, o_proj` | train |
| gated DeltaNet projections | `Qwen3_5GatedDeltaNet` | `in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj` | train |
| DeltaNet state params | `A_log`, `dt_bias`, `conv1d`, gated `norm` | — | **frozen** (fp32 recurrence state) |
| MLP | `Qwen3_5MLP` | `gate_proj, up_proj, down_proj` | train |
| norms | `Qwen3_5RMSNorm`, `Qwen3_5RMSNormGated` | — | frozen |
| vision blocks | `Qwen3_5VisionAttention` (`qkv, proj`), `Qwen3_5VisionMLP` | `qkv, proj`, `linear_fc1, linear_fc2` | Eye: last-K blocks; Brain: frozen tower |
| vision merger | `Qwen3_5VisionPatchMerger` | `linear_fc1, linear_fc2` | **train** (both roles) |
| embeddings / lm_head | tied | — | frozen (optional `modules_to_save` for full tuning) |

"Train the full attention blocks and gated DeltaNet blocks and vision merger and every up/down" is
implemented as **adapter coverage over every one of those block types** (`tuning: lora|qlora`) with
`tuning: full` available where VRAM allows. It is not implemented as "unfreeze everything on a 6 GB
card", which cannot fit 4.66 B parameters plus activations.

## 8. Hardware profiles (one dict, no notebook edits)

Single source of truth: `rtp/profiles.py` (to be written). Notebooks set only

```python
ACTIVE = "kaggle_t4x2"        # or "local_4050" | "local_4090" | "pro6000" | "tpu_v5e8" | ...
```

and everything else (dtype, attention impl, tuning mode, batch, grad-accum, checkpointing, output
dirs, resume file, serving flags, eval seeds) is derived. CLI equivalent: `RTP_PROFILE=local_4050`.

| profile | accel | VRAM | tuning | dtype | attention | LoRA | batch × accum | grad ckpt | serve | class |
|---|---|---|---|---|---|---|---|---|---|---|
| `local_4050` | 1× RTX 4050 Laptop | 6 GB | **qlora** (NF4) | bf16 compute | `flash_attention_2` if importable else `sdpa` | r32 | 1 × 16 | on | 4-bit | dev only |
| `local_4090` | 1× RTX 4090 | 24 GB | lora | bf16 | `flash_attention_2` | r32 | 4 × 4 | off | bf16 | reportable |
| `kaggle_t4x2` | 2× T4 (DDP) | 2×16 GB | lora | **fp16** | `sdpa` | r32 | 2 × 8 | on | vLLM fp16, `enforce_eager`, tp=2 | reportable (headline) |
| `pro6000` | 1× RTX Pro 6000 | 96 GB | lora → full | bf16 | `flash_attention_2` | r64 | 8 × 4 | off | bf16 | reserved |
| `tpu_v5e8` / `tpu_v6e1` / `tpu_v5e1` | XLA | — | stub | bf16 | `sdpa` | — | — | — | — | **stub, `supported=False`** |

bf16 on T4: **not available** (bf16 requires Ampere+), hence fp16 + SDPA on `kaggle_t4x2` — **[V]**
FA2 README; and `torch.cuda.is_bf16_supported()` is the runtime guard.

Per-profile isolation (a profile may never write into another's tree):

```
runs/<profile>/{ckpt,logs,reports,state.json}     adapters + optimizer state + resume cursors
data/<profile>/                                   generated datasets, caches, episode captures
models/<profile>/hf                               HF_HOME pinned per profile
```

`state.json` = `{stage: {status, last_step, config_hash}}`; a stage whose `config_hash` is unchanged
is skipped, so each profile resumes its own pipeline independently and reproducibly.

## 9. Retrieval: Qdrant + BM25 hybrid

Chosen: **Qdrant** (Rust, Apache-2.0) via `qdrant-client`, hybrid retrieval fused by RRF over the
BM25 scorer already in `rtp/harness.py`.

Verified capabilities used: embedded/local mode (`QdrantClient(path=...)`, no server), named dense
vectors, **sparse vectors** (own index/storage), payload filtering — **[V]** from
`qdrant/qdrant` README and the official vectors documentation ([local mode](https://github.com/qdrant/qdrant-client#local-mode), [vectors](https://qdrant.tech/documentation/concepts/vectors/)).

Collections:
- `harness_docs` — rules/skills/notes, dense `bge-small-en-v1.5` (384-d, CPU) + sparse BM25/SPLADE.
- `episode_notes` — one point per failure/success note with `payload {profile, seed, skill, obj_cls,
  cause, kind, text}`; filters on `(skill, obj_cls, cause)` turn "what went wrong last time" into a
  retrieval query. `kind` separates positive traces from negative/counterfactual ones (D-10).

Fusion: `score = Σ 1/(k + rank_i)` over the BM25 and dense lists (RRF, already implemented), so no
score-normalisation between the two engines is needed. A hash-embedding fallback keeps tests offline.

## 10. Episode capture and RAG loop

One episode directory per run (`runs/<profile>/episodes/<ep_id>/`, or `reports/` for ad-hoc runs):

```
meta.json          profile, seed, instruction, git sha, config_hash, outcome, attribution
frames/            eye snapshots/crops actually sent to the Brain (not every frame)
eye_calls.jsonl    one line per EYE_CMD/EYE_REPLY (§3 log)
obs.jsonl          OBS per round, both Eyes
plan.json          accepted plan; rejected plans with verifier reasons
trace.jsonl        Brain prompt/response pairs, incl. Kaggle-generated CoT traces
arm.jsonl          per-skill start/end poses, params, SkillReport, TOUCH
memory.jsonl       retrieval queries + retrieved ids (so we can audit what the Brain read)
```

Failure → `episode_notes` with `kind=failure`, the attributed cause, and the shortest fix if the
hindsight replay found one. Successes feed SFT positives; failures feed recovery and hindsight
relabelling (`rtp/hindsight.py`), and negative/counterfactual rows are tagged so they never enter the
positive SFT set.

## 11. Training plan

1. **SFT** on (a) teacher plans that closed-loop succeeded, (b) Kaggle-Frontier-generated
   task-grounded CoT traces *only after* they pass the verifier and a replay (see §13), (c) eye-call
   traces from episodes that succeeded *with* eye calls.
2. **GRPO** with the existing reward decomposition (`rtp/rewards.py`: format, feasibility, dry-run
   rule compliance, length, closed-loop success) **plus** eye efficiency (`−λ·cost`), and **KL
   regularisation against the SFT adapter** (reference model = merged SFT weights, β recorded per
   profile). Reward weights live in the profile dict, not in the training script.
3. **KL anchor**: keep the SFT adapter immutable, load a frozen merged copy as `ref_model`, and log
   `kl(π‖π_sft)` per step; a run whose KL exceeds the budget stops rather than drifts.
4. **Ablations**: one-eye, static prompt, no verification, no challenge, no look (budget 0), and
   learned-eye-policy vs Brain-driven calls. All share the same eval seeds per profile.

Benchmark: the existing reproducible suite (fixed seeds, Wilson intervals, paired bootstrap) against
omni-model baselines on the same task and metrics — **[U]** exact baseline list until the model
servers exist; any omni comparison is a future run, not a claim.

## 12. What the paper claims

1. **Attribution**: agreement with hand labels (accuracy + Cohen's κ) for Eye/Brain/Arm failure
   attribution in a two-Eye, two-Arm team — the headline.
2. **Active perception is worth N calls**: success vs eye-call budget, and whether Brain-requested
   kernels/views beat a fixed best view at equal compute.
3. **Verification is the big lever**: `verify_none` vs `verify_dry_run` (fixture numbers exist:
   61 % → 100 % on the symbolic fixture; fixture only, never physics evidence).
4. **Team credit beats field accuracy** for training the Eyes (`rtp/credit.py` counterfactual credit).
5. **Block-coverage ablation**: which Qwen3.5-4B block types (full attention vs DeltaNet vs merger)
   must be adapted for tool use — adapter-coverage grid, one profile at a time.

## 13. Kaggle Frontier AI — **UNVERIFIED**

`https://www.kaggle.com/docs/frontier-ai` and `https://www.kaggle.com/docs/api` both return
**HTTP 403** to unauthenticated fetches (checked 2026-11-16). We therefore do **not** know its
endpoints, available models, quotas, rate limits or whether it accepts images.

Intended use (design, not fact): generate task-grounded CoT traces for the Brain, one trace per
(scene, instruction) with the eye-call and arm-skill reasoning made explicit, then filter by
verifier + replay before any trace enters SFT. This service is orthogonal to the hardware profiles:
it produces data for every profile, and no cloud job is launched until the user authorises
authentication. Blocked on: (a) user authorisation to authenticate, or (b) a session with access to
the docs.

## 14. Open items

| # | item | needs |
|---|---|---|
| 1 | Kaggle Frontier AI API surface/limits | authorisation or doc access (§13) |
| 2 | Claude's implementation plan | file in repo root (not present yet) |
| 3 | YOLO licence (AGPL) vs BSD fallback | user call |
| 4 | Eye ring radius/height/station counts | user call, defaults in §2 |
| 5 | Isaac/Newton on this laptop | 6 GB 4050 cannot host a full scene; Newton floor is cc 7.5 + driver R580+ (§15) |
| 6 | Isaac Sim RT-core requirement for T4 | documented limitation, T4 stays on `mjsim` |

## 15. Newton / Isaac Lab platform facts

**[V]** `isaac-sim/IsaacLab@main` ships `source/isaaclab_newton` = `isaaclab_newton 17.2.3`,
`requires-python >=3.12`, depending on `isaaclab`; `isaaclab_newton/physics/newton_manager.py`
imports `warp as wp` and `from newton import Axis, ModelBuilder, ModelFlags, State`, with
`NewtonManager.backend_name = "newton"`.

**[V]** Newton's requirements, quoted from `github.com/newton-physics/newton` README §Requirements and
the [installation guide](https://newton-physics.github.io/newton/latest/guide/installation.html):
"Since Warp 1.18, standard Warp wheels require an NVIDIA Turing GPU (compute capability 7.5+) and
driver R580+ for CUDA acceleration. On an R545–R579 driver or an older supported GPU, install with
`warp-lang==1.17.0`." Also Python 3.11+, Linux x86-64/aarch64 (Windows x86-64, macOS CPU-only), no
local CUDA toolkit needed, and MuJoCo Warp as the primary backend (Newton generalises Warp's
deprecated `warp.sim`).

Practical reading: an RTX 4050 (cc 8.9) and 4090 (cc 8.9) satisfy Newton's floor; the 4050's 6 GB is
the binding limit for scene size, not for Newton itself. A T4 is cc 7.5 — exactly on Newton's
boundary — but Isaac Sim requires an RTX-class GPU with RT cores, which a T4 lacks, so dual-T4
profiles run `mjsim` (MuJoCo CPU) for control experiments and only serve/evaluate models.

## 16. Evidence classes

| class | what it proves | where |
|---|---|---|
| `FIXTURE` | mechanics only (scripted Brain, parametric Eyes) | `rtp/world.py`, `tests/` |
| `MJSIM` | laptop physics regression (MuJoCo, gantry surrogate) | `mjsim/` |
| `ISAAC_NEWTON` | physics results, reportable | `isaac/` (not yet runnable end-to-end) |
| `LLM-SYMBOLIC` | claims scoped to the symbolic task | `rtp/loop.py` + real model server |
| `MODEL` | training/eval of Qwen3.5-4B adapters | `train/`, `runs/<profile>/` |

`mjsim` is never ISAAC evidence. Fixture results never become physics claims. Paper cells stay
`\todo` until the corresponding class has a real run.
