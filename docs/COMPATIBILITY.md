# Software compatibility and hardware profiles

This file is **authoritative for version constraints, the profile capability matrix and the
per-profile isolation tree** (`IMPLEMENTATION_PLAN.md` §8). Claim status is in
`docs/STATUS_HONEST.md`; execution order is in `docs/AGENT_RUNBOOK.md` (arms first).

## 1. Versions
| component | constraint | evidence |
|---|---|---|
| Isaac Lab | 3.0 GA (`v3.0.0`), Python 3.12, quaternion order x,y,z,w | [V:S-10, S-15] |
| Newton | pinned by Isaac Lab's own lock at the chosen tag (`newton[sim]==1.6.1`, `warp-lang==1.17.0`, `mujoco-warp~=3.12.0` at v3.0.0) via `uv sync`, never a hand-picked pip version; GPU floor cc 7.5; the locked CUDA 13 torch needs driver 580.65.06+ | [V:S-17, S-19], tag `pyproject.toml`/`uv.lock` |
| Renderers | Newton Warp (base install); OVRTX (`isaaclab_ov`, ovrtx v0.5.1) optional | [V:S-12] |
| Qwen3.5-4B | dense, Apache-2.0 | [V:S-08] |
| DeltaNet fast path | `flash-linear-attention` + `causal-conv1d` (needs `nvcc`) | [V:S-09] |
| FlashAttention-2 | Ampere/Ada/Hopper mainline; Turing only via fork; head_dim 256 supported | [R] |
| Qdrant | local mode, dense+sparse | [R]; hybrid in local mode **[G]** |
| LangGraph | version/licence pinned at install | **[G]** |
| `kaggle-benchmarks` | Apache-2.0; `MODEL_PROXY_API_KEY` outside Kaggle | [V:S-04, S-05] |
| libfranka / `franka_ros2` | version matched to FR3 firmware | **[G: P10]** |
Keep the sim environment separate from the VLM-training environment if their Torch/Warp pins conflict.

## 2. Profile capability matrix
| profile | VLM training | VLM serving | Newton sim (physics + Warp renderer) | OVRTX | role |
|---|---|---|---|---|---|
| `local_4050` (Ada 6 GB) | QLoRA NF4; FA2 if present else SDPA; fast path needs `nvcc` build; forward/backward ceiling **512 tokens**, but a real LoRA step with optimizer state fits only at **256** (D-38, D-47); 8-bit AdamW, LM-head chunk 64 | 4-bit | ✓ small env counts [G: G-3] | **[G]** | dev |
| `local_4090` (Ada 24 GB) | BF16 LoRA + FA2 | bf16 | ✓ full | **[G]** | headline |
| `kaggle_t4x2` (Turing 2×16 GB, no bf16) | fp16 LoRA, SDPA, DDP | fp16 eager | cc 7.5 meets floor; driver R580+ or Warp 1.17.0 **[G: G-3]**; no display → EGL/Rerun/recorded video | **[G]** | Kaggle headline |
| `pro6000` | LoRA → full-FT candidate, bf16+FA2 | bf16 | ✓ | **[G]** | reserved |
| `tpu_v5e8`, `tpu_v6e1`, `tpu_v5e1` | stub (`supported=False`) | — | — | — | G-12 |
Training defaults (`rtp/profiles.py`): LoRA r32 (`pro6000` r64), micro-batch × grad-accum
1×16 / 4×4 / 2×8 / 8×4 for `local_4050` / `local_4090` / `kaggle_t4x2` / `pro6000`, gradient
checkpointing on for `local_4050` and `kaggle_t4x2` only. bf16 is unavailable on T4 (Ampere+ only)
→ fp16 + SDPA. G-3 is PASS on the 4050 and a Kaggle T4, but only a cube-fall rgb/depth gate, one
device at a time (`reports/g3_sim.json`, `reports/g3_sim_t4.json`).

A GPU that trains the VLM is not automatically sized for many-env Newton training with cameras; `reports/sim_scaling_<profile>.json` (P2) is the authority.

## 3. Isolation (D-10/D-11)
```
runs/<profile>/{ckpt,logs,reports,state.json}   data/<profile>/   $MODELS/<profile>/hf
```
`state.json = {stage:{status,last_step,config_hash}}`; unchanged hash ⇒ skip; no path escapes the profile tree.
