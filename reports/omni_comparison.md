# Omni-model comparison (held-out symbolic fixture)

This table uses the fixed test seeds and the same symbolic CPU fixture for every scored row. No trained Brain adapter was scored by this CLI; rows with no prediction file are `not run` with no numbers.

| Model | n | Success % [95% Wilson CI] | Plan-pass % | Mean msg tokens | Paired vs ours (ours − model) | Status | How run / note |
|---|---|---|---|---|---|---|---|
| Ours - ScriptedBrain (fixture) | 30 | 86.7 [70.3, 94.7] | 96.7 | 10835 | — | scored | rtp.loop.run_episode on the symbolic fixture; fixture / reference planner, NOT a trained model |
| Our Brain - Qwen3.5-4B (trained adapter) | not run | — | — | — | — | not run | rtp.brain.LLMBrain via rtp.llm_client with identical prompt/decoding; needs a served checkpoint; not run in this session: no adapter was loaded |
| Qwen3.5-4B base (identical tools) | not run | — | — | — | — | not run | same prompt/tools with harness off; fine-tune not performed |
| pi0 | not run | — | — | — | — | not run | fine-tune on our own sim demos (adapter not set up) |
| RDT | not run | — | — | — | — | not run | fine-tune on our own sim demos (adapter not set up) |
| ACT | not run | — | — | — | — | not run | fine-tune on our own sim demos (adapter not set up) |
| DP (diffusion policy) | not run | — | — | — | — | not run | fine-tune on our own sim demos (adapter not set up) |
| DP3 | not run | — | — | — | — | not run | symbolic-skill / RoboTwin adapter (not set up) |
| Frontier model as zero-shot Brain | not run | — | — | — | — | not run | Kaggle-hosted teacher as Brain via rtp.llm_client (needs skill + CLI auth) |
| X-VLA 0.9B (LeRobot) | not run | — | — | — | — | not run | 0.9B Apache-2.0 cross-embodiment flow VLA; fine-tune on our sim demos through the bounded-delta interface; external source, not reproduced here; adapter not set up |
| pi0-FAST (LeRobot) | not run | — | — | — | — | not run | 3B FAST-token VLA; discrete-action baseline beside the flow-matching pi0/pi0.5 rows; external source, not reproduced here; adapter not set up |
| NSDL 3B (VIKI-SFT) | not run | — | — | — | — | not run | open-weight neuro-symbolic planner with detect/schedule tools on VIKI-L2; rerun with our seeded scorer; external source, not reproduced here; adapter not set up |
