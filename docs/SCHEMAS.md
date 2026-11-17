# Schemas (design contracts; no code yet)

Extends `docs/ARCHITECTURE_V2.md` §3–4 and §10. Timestamps: monotonic ns + ISO wall clock. `meta` always has profile, git_sha, config_hash, seed.

## 1. Eye tool call / reply
```json
// EYE_CMD
{"request_id":"q7","eye":"A|B","tool":"orbit|pan|tilt|zoom|kernel|detect|segment|focus|crop|snapshot|describe|status|calibrate_check",
 "args":{},"reason":"<=200 chars","budget_left":4,"deadline_ms":3000}
// EYE_REPLY (failures are ok=false, never exceptions)
{"request_id":"q7","ok":true,"error":null,
 "result":{"pose":{"station":3,"pan_deg":0,"tilt_deg":0,"zoom":1.0},
           "detections":[{"cls":"book","conf":0.91,"box":[0,0,0,0],"centroid_m":[0,0,0.03],
                          "centroid_source":"plane|triangulated|depth|sim_gt","err_m_est":0.01}],
           "visibility":{"r0c1":0.0},"occluders":["column_1"],"frame_id":"A-0032"},
 "cost":1,"budget_left":3,"latency_ms":41,"motion_ms":380,"health":"ok|degraded|fault"}
```

## 2. Arm event (skill layer → Brain)
```json
{"arm":"L|R","event":"PUSHED|YIELDED|RESUMED|TOUCH|BLOCKED|FAULT","t_ns":0,
 "skill_id":"t3","e_peak_m":0.0,"f_ext_peak_n":0.0,"yield_s":0.0,"resume_err_m":0.0}
```

## 3. L1 telemetry (per arm, logged at ≥ 100 Hz, decimated from 1 kHz)
`{t_ns, state: TRACK|YIELD|RETURN|FAULT, q[7], dq[7], x_ref[7], x[7], e[6], f_ext[6], K_lin, K_rot, profile, tau_cmd[7],
  clamp_flags{force,speed,leash,torque,rate}, s_plan, v_cart, protective_stop}`

## 4. Episode directory
```
runs/<profile>/episodes/<ep_id>/
  meta.json  frames/  eye_state.jsonl  eye_calls.jsonl  obs.jsonl  brain_io.jsonl  plan.jsonl
  arm.jsonl (skill reports + arm events)  l1_telemetry.parquet  safety.jsonl  memory.jsonl  outcome.json
```
`brain_io.jsonl` stores exactly what the model emitted. Teacher reasoning text, if returned, is stored as `teacher_visible_reasoning`; no hidden reasoning is assumed.

## 5. SFT example / GRPO rollout
```json
{"example_id":"…","source":"teacher|self_rollout|hindsight","teacher":{"model":"…","tier":"strong|bulk|cheap|diversity","cost_usd":0.0},
 "images":["frames/A-0031.jpg"],"obs":{},"instruction":"…","events":[],
 "target":{"rationale":"…","tool_calls":[],"plan":{}},
 "gates":{"verifier":"pass","replay":"pass|fail_attributed","eye_efficiency":"pass","agreement":2},
 "label":"positive|negative_attributed","split":"train|val|heldout"}
{"rollout_id":"…","policy":"adapter_hash","ref":"sft_hash","group_id":"…",
 "reward":{"format":1,"feasibility":1,"rules":0.8,"success":1,"eye_cost":-0.2,"yield_compliance":1,"length":-0.05},"kl_to_ref":0.0}
```

## 6. Failure note (Qdrant `episode_notes`)
`{profile, seed, skill, obj_cls, cause: eye|brain|arm|env, kind: success|failure|counterfactual, text, fix, episode_ref}`

## 7. Cache manifest (`cache_manifest.json` in every cached Kaggle dataset)
```json
{"schema":1,"kind":"model|wheels|adapters|data","slug":"rtp-model-qwen35-4b-<rev8>",
 "source":{"type":"hf","repo":"Qwen/Qwen3.5-4B","revision":"<commit sha>"},
 "profile_tags":["local_4090","kaggle_t4x2"],"env":{"python":"3.12","torch":"x.y","cuda":"12.x","sm":"75|89"},
 "files":[{"path":"…","sha256":"…","bytes":0}],"total_bytes":0,"created":"ISO","license":"Apache-2.0","private":true}
```

## 8. L2→L1 interface (same in sim and real)
In: `{frame, x_plan(t), s, stiffness_profile, scale∈[0.5,1.5], gripper, max_force, deadline}`. Out: state, `f_ext`, clamp flags, events.
