# Software compatibility and hardware profiles

## 1. Versions
| component | constraint | evidence |
|---|---|---|
| Isaac Lab | 3.0 GA (`v3.0.0`), Python 3.12, quaternion order x,y,z,w | [V:S-10, S-15] |
| Newton | pinned by Isaac Lab's requirements at the chosen tag; Python 3.10+ standalone; GPU floor cc 7.5; Warp 1.18 needs R580+, else Warp 1.17.0 | [V:S-17, S-19] |
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
| `local_4050` (Ada 6 GB) | QLoRA NF4; FA2 if present else SDPA; fast path needs `nvcc` build | 4-bit | ✓ small env counts [G: G-3] | **[G]** | dev |
| `local_4090` (Ada 24 GB) | BF16 LoRA + FA2 | bf16 | ✓ full | **[G]** | headline |
| `kaggle_t4x2` (Turing 2×16 GB, no bf16) | fp16 LoRA, SDPA, DDP | fp16 eager | cc 7.5 meets floor; driver R580+ or Warp 1.17.0 **[G: G-3]**; no display → EGL/Rerun/recorded video | **[G]** | Kaggle headline |
| `pro6000` | LoRA → full-FT candidate, bf16+FA2 | bf16 | ✓ | **[G]** | reserved |
| `tpu_v5e8`, `tpu_v6e1`, `tpu_v5e1` | stub (`supported=False`) | — | — | — | G-12 |
A GPU that trains the VLM is not automatically sized for many-env Newton training with cameras; `reports/sim_scaling_<profile>.json` (P2) is the authority.

## 3. Isolation (D-10/D-11)
```
runs/<profile>/{ckpt,logs,reports,state.json}   data/<profile>/   models/<profile>/hf
```
`state.json = {stage:{status,last_step,config_hash}}`; unchanged hash ⇒ skip; no path escapes the profile tree.
