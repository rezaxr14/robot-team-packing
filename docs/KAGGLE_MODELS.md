# Kaggle-hosted frontier models (teachers / judges)

Decisions: D-27 (use them), D-33 (local first). You authorise access and terms. The model catalogue, the calling method and quota handling come from the Kaggle skill you will supply plus Kaggle CLI auth. This document defines what the project needs from that interface and how it is used; it does not gate any other work.

## 1. Interface the project codes against (`rtp/teacher/client.py`)
```python
class Teacher(Protocol):
    name: str; tier: Literal["strong","bulk","cheap","diversity"]
    def propose(self, scene: SceneInput) -> TeacherOutput: ...   # images + OBS + rules + tool list -> tool_calls, plan, rationale
    def judge(self, episode: EpisodeRef) -> Attribution: ...      # eye|brain|arm|env + short reason
    def usage(self) -> Usage: ...                                 # tokens, cost, remaining quota if the interface reports it
```
Implementation: `KaggleTeacher(model_id)`, written when the skill is supplied. Until then the interface, prompts, parsers, gates, budget controller and queue are built and tested against **recorded response fixtures** (test data only, never training data). Each example records `teacher.model` and `tier`.

## 2. Roles
Teacher (plans from rendered eye frames); judge (failure attribution, cross-checked against the programmatic cause); hard-case miner (arrangements from failure notes, kept only if the sim confirms the current Brain fails them); compliance-aware teacher (scenarios with `PUSHED/YIELDED/RESUMED`); agreement signal (≥ 2 teachers agree + verifier pass = high confidence).

## 3. Model pool and tiers
Taken from the skill's catalogue at integration time and tiered by **measured accepted-example yield per unit cost** on a 200-scene pilot: strong / bulk / cheap / open-weight diversity. Names seen on Kaggle's benchmark pages (Claude Opus/Sonnet/Haiku generations, Gemini 4 Argon/Carbon, Gemini 3.x Flash/Pro, Qwen, DeepSeek, gpt-oss) are candidates, not a dependency.

## 4. Budget controller
`N_traces ≈ 0.8 · remaining_budget / cost_per_accepted_trace` using whatever usage/quota the skill's interface reports; stop at 80 %; per-call log `(model, tokens, cost, accepted)`; resumable queue keyed by scene id.

## 5. Prompt contract
Inputs: rules, legal tool list, OBS JSON, eye frames, retrieved notes (ids), arm-event protocol. Output: `{tool_calls:[…], plan:{tasks:[…]}, rationale:"≤ 400 chars per decision"}`. The schema has no joint values, stiffness or speed fields.

## 6. Quality pipeline (provider-independent)
verifier gate → sim-replay gate (success / attributed failure) → eye-efficiency filter → multi-teacher agreement → diversity quotas → contamination guard (benchmark seeds excluded). Only emitted text is stored; any returned reasoning text is stored as `teacher_visible_reasoning`.

## 7. Sequencing
Non-Kaggle phases never wait on this. Order: pipeline built and fixture-tested now → you supply the skill + CLI auth → implement `KaggleTeacher`, run G-4 → generate at scale with the Kaggle-hosted models. There is no substitute data source.
