# Honest status of the paper's headline claims

This file exists after an explicit correction: earlier status messages overstated what had been
done. Each row below links to the command/report that actually produces the evidence. A row without
a measured artifact is **NOT DONE**, even if a notebook or a spec file exists.

Updated: 2026-11-19 (will be amended as each gate actually passes).

| claim | state | evidence / what is missing |
|---|---|---|
| Qwen3.5-4B module map and DeltaNet fast path measured | **DONE** | `reports/qwen35_modules.json`, `reports/g2_fastpath.json` |
| Newton-only Isaac Lab 3.0 stepping with rgb+depth on 4050 **and** T4 | **DONE (gate only)** | `reports/g3_sim.json`, `reports/g3_sim_t4.json`; this is a falling cube, **not** a manipulation scene |
| Real 4B QLoRA SFT step + mid-epoch resume from HF | **DONE (4 steps)** | `reports/sft_smoke.json`, G-14; tiny smoke, not a reportable trained model |
| Private storage local → Kaggle → HF | **DONE** | `rtp/remotes.py`, `ops/hub.py`, private Kaggle dataset `rtp-model-qwen35-4b-851bf6e8` (9.32 GB, published from Kaggle), private HF repo `Reza-Nadimi/robot-team-packing` |
| SimReady assets from Hugging Face ingested | **IN PROGRESS** | `ops/fetch_simready.py`, `assets/selection.json` exist; catalog/fetch still running; FR3 v2 + Duo MJCF are downloaded and verified (`reports/robot_assets.json`, `reports/g5_fr3.json`: 7 and 14 joint DoF, 17.31/36.26 kg) |
| Environment built from SimReady assets (2 FR3 + PTZ eyes, kitchen) | **NOT DONE** | no `isaac/scenes/`, no asset-based scene; only a symbolic fixture and a cube-fall gate |
| Chain-of-thought traces from the harness with frontier AI | **PARTIAL** | `rtp/frontier_cot.py` harness + `ops/cot_trace.py` produced 6 real `scripted-teacher` traces on harness retrieval (`reports/cot_trace.json`); `frontier_status: unavailable: RTP_SERVE_URL is unset`, `frontier_traces: 0` |
| Proper training on that environment's data | **NOT DONE** | SFT used 64 symbolic teacher rows, not environment episodes; GRPO never run |
| Hindsight relabelling recorded from real failures | **PARTIAL** | `notebooks/70_hindsight_memory.ipynb`: 3/3 replay-validated on the symbolic fixture; `rtp.asset_datagen.gen_asset_hindsight` now does the same on asset-backed pose-level worlds (tests pass), pending a real run |
| Success/failure memory RAG | **PARTIAL** | same notebook: 20-row memory + BM25 retrieval on symbolic outcomes; not connected to a trained Brain |
| Scored on a fixed test set | **DONE for fixture only** | `rtp/benchmark.py` + `ops/score_test.py`, seeds 0-29 (training >=100000): ScriptedBrain 26/30 = 86.7% [70.3, 94.7] (`reports/test_scores.json`); **trained Brain not scored** |
| Compared with other omni models | **HARNESS ONLY** | `reports/omni_comparison.{json,md}` has a strict prediction loader, Wilson + paired bootstrap; all competitor rows and the trained-Brain row are `not run` with no numbers |
| Impedance control making arms safe around people/children | **SIM CROSS-CHECK ONLY** | `rtp/control/impedance.py` + `ops/g6_impedance.py`: step error 0.27% (limit 5%), push deflection 0.06655 m vs F/K 0.06667 m, return error 9.6e-10 m (`reports/g6_impedance.json`). **NOT** a safety certification; D-22 assigns hardware/L0/L1 sign-off to Claude; no plan-clock gate, leash, clamps, state machine or ISO limits |

## Rules for updating this file

* Add a row only when the command's output exists in `reports/` or a committed test result.
* Mark evidence class: `FIXTURE` (symbolic), `SIM` (Newton), `GATE` (engineering), `MODEL` (trained
  weights), `HARDWARE` (real robot). Never pool classes in a paper number.
* "Notebook exists" is not evidence that the stage ran. "Spec exists" is not evidence that the
  behaviour is implemented or measured.
