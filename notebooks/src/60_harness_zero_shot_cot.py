#!/usr/bin/env python3
"""Build notebooks/60_harness_zero_shot_cot.ipynb: Harness retrieval, zero-shot planning, CoT, tools."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ops.nbbuild import Notebook, install_cell

nb = Notebook("60_harness_zero_shot_cot")

nb.md("""# 60 — The Brain's harness: zero-shot, retrieval, CoT, and tool use

This is the **Brain** side of the paper, not the Eyes. The Brain is the text planner that receives
SCENE JSON from the Eyes and the instruction, retrieves `rules.md` / `skills.md` / object notes from
the harness, and emits a JSON plan DAG for two blind arms.

What is real here:

* `Harness("harness_files")` loads the committed `.md` documents and BM25-ranks them;
* the BM25 + dense fusion path (reciprocal rank fusion) runs with a deterministic **fake** embedder,
  so the fusion logic is testable without the Qdrant/embedding installer;
* a zero-shot episode on the symbolic fixture produces a prompt and a `ScriptedBrain` plan that
  `parse_plan` accepts;
* `ToolBrain` executes `check_rule` and `dry_run` against the symbolic world and commits a plan;
* prompt/completion token costs are counted with the real Qwen tokenizer when the model cache is
  present, else with a clearly labelled proxy.

What is **not** real here: no served model is called unless `RTP_SERVE_URL` is set. `ScriptedBrain`
is a deterministic fixture and a data-generation teacher, not evidence about an LLM. No zero-shot
LLM result and no CoT ablation result exists yet.
""")

nb.code(install_cell(["transformers"]))

nb.code("""ACTIVE = "local_4050"          # local_4050 | local_4090 | kaggle_t4x2 | pro6000 | tpu_v5e8
import os, sys
from pathlib import Path

os.environ["RTP_PROFILE"] = ACTIVE
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from rtp.profiles import detect_caps, get_profile, resolve_runtime

prof = get_profile(ACTIVE)
attn, dtype, notes = resolve_runtime(prof, detect_caps())
print("profile:", prof.name, "| accel:", prof.accel, "| gpu:", prof.gpu)
print("dtype:", dtype, "| attention:", attn, "| max_len:", prof.max_len)
for note in notes:
    print(" -", note)
""")

nb.md("""## Model resolution

The Brain notebook must not invent a model path. It resolves the Qwen3.5-4B cache through
`store.resolve(spec, profile)`, which checks the profile-local cache, then an attached Kaggle dataset,
then our private remotes. On a box with no cache/token/network the cell prints
`SKIPPED: model resolution failed: <ExceptionType>` and continues; later cells fall back to a
labelled token proxy instead of pretending a tokenizer ran.
""")

nb.code("""from rtp import assetcache as ac
from rtp.remotes import from_env

MODEL_REV = "851bf6e8"
spec = ac.model_spec("Qwen/Qwen3.5-4B", revision=MODEL_REV, name="qwen35-4b")
model_dir = None
try:
    store = from_env(with_kaggle=False)
    print("private HF repo:", store.hf_repo)
    model_dir = store.resolve(spec, prof, verify="size")
    print("model:", model_dir)
    print("files:", len(list(ac.iter_files(model_dir))))
except Exception as exc:                       # noqa: BLE001  (cache/token/network absence)
    print("SKIPPED: model resolution failed:", type(exc).__name__)
    print("No model weights are read by this notebook; token counts will use a labelled proxy.")
""")

nb.md("""## The harness documents

The harness is a directory of Markdown files with YAML front matter (`id`, `type`, `keywords`).
`Harness.reload()` parses every `*.md`, rejects duplicate ids, and builds one BM25 index over
`id + type + keywords + body`. Retrieval is therefore inspectable: we can print the document ids,
their types, and the raw BM25 scores, not just a final list.

Three real queries are used: a rule query, a skill query, and an object/constraint query. `k=6`
matches the default rule budget in `LoopConfig.k_rules`.
""")

nb.code("""from rtp.harness import Harness

H = Harness("harness_files")
print("docs:", len(H.docs))
print(sorted(H.docs))
print()

QUERIES = [
    "heavy items at the bottom of the bag",
    "grasp force slip crush",
    "laptop in the padded slot",
]
for q in QUERIES:
    ranked = H.bm25.rank(q)[:6]
    print("query:", q)
    print("  BM25:", [(doc_id, round(score, 2)) for doc_id, score in ranked])
    docs = H.retrieve(q, k=6)
    print("  retrieve(k=6):", [(d.id, d.type) for d in docs])
    print()
""")

nb.md("""## Hybrid retrieval: BM25 + dense vectors, fused by RRF

`Harness.retrieve(q, k, embed_fn=...)` ranks by BM25 and by cosine similarity of `embed_fn` vectors,
then fuses the two rankings with reciprocal rank fusion (`1/(60+rank)`). The real deployment uses an
embedding model and Qdrant; **that installer is separate**. This cell isolates the fusion logic with
a tiny deterministic fake embedder: tokens are hashed into 64 signed buckets and L2-normalised.
It is a stand-in for the real embedding model, not a claim about its quality.

RRF is rank-based, so a dense signal that disagrees with BM25 can change the order even when the
BM25 scores are unchanged. The cell prints BM25 order, dense order, and the fused order side by side.
""")

nb.code("""import hashlib, math
from rtp.harness import tokenize

DIM = 64
def fake_embed(text):
    \"\"\"Deterministic hash-of-tokens stand-in for the real embedding model.\"\"\"
    v = [0.0] * DIM
    for tok in tokenize(text):
        h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
        bucket = int.from_bytes(h[:4], "big") % DIM
        v[bucket] += 1.0 if h[4] & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]

def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))

def dense_rank(query, embed_fn):
    qv = embed_fn(query)
    sims = [(doc_id, cosine(qv, embed_fn(d.text))) for doc_id, d in H.docs.items()]
    return sorted(sims, key=lambda x: (-x[1], x[0]))

for q in QUERIES:
    plain = [d.id for d in H.retrieve(q, k=6)]
    dense = [doc_id for doc_id, _ in dense_rank(q, fake_embed)[:6]]
    hybrid = [d.id for d in H.retrieve(q, k=6, embed_fn=fake_embed)]
    print("query:", q)
    print("  BM25 :", plain)
    print("  dense:", dense)
    print("  fused:", hybrid)
    print("  RRF changed the BM25 order:", plain != hybrid)
    print()

print("The fake embedder is only a deterministic stand-in; a real Qdrant/embedding run is a separate")
print("installer and is not executed here.")
""")

nb.md("""## Zero-shot task construction

A zero-shot episode is built without any trained Brain:

1. `sample_world(seed)` and `sample_spec(world, seed)` produce the symbolic scene and instruction;
2. each default Eye reports, and `merge_scenes` unions them (non-occluded, higher confidence wins);
3. `retrieve_docs(H, LoopConfig(), spec, scene, [])` does the same two-pass retrieval the loop uses:
   rules against the instruction, object/skill/memory notes against the scene;
4. `build_prompt(BrainContext(...))` renders the exact text a model would receive.

The prompt is printed in full so the reader can inspect what is and is not in context. The scene is
symbolic (no pixels), and the instruction's object names come from the same generator.
""")

nb.code("""import json

from rtp.brain import BrainContext, ScriptedBrain, build_prompt
from rtp.datagen import default_eyes, sample_spec, sample_world
from rtp.loop import LoopConfig, retrieve_docs
from rtp.perception import merge_scenes

SEED = 0
world = sample_world(SEED)
spec = sample_spec(world, SEED)
scene = merge_scenes([e.report(world, 0) for e in default_eyes(SEED)])
docs = retrieve_docs(H, LoopConfig(), spec, scene, [])
ctx = BrainContext(spec=spec, scene=scene,
                   rule_ids={d.id for d in docs if d.type == "rule"}, snippets=docs)

print("instruction:", spec.text())
print("scene objects:", len(scene), "| retrieved:", [(d.id, d.type) for d in docs])
print("rule ids:", sorted(ctx.rule_ids))
print("prompt chars:", len(build_prompt(ctx)))
print()
print(build_prompt(ctx))
""")

nb.md("""## Zero-shot plan (scripted fixture, CPU)

`ScriptedBrain` is the deterministic reference planner used to generate SFT data and to test the
loop. It only applies a rule whose id was retrieved into its context, so it is a useful check that
retrieval produced a usable plan — **not** a zero-shot LLM result.

The cell runs it, serialises the plan with `Plan.to_json()`, asserts that `parse_plan` accepts the
round-trip, and runs the same static check and symbolic dry run the loop uses before execution.
""")

nb.code("""from rtp.plan import parse_plan
from rtp.rules import dry_run, static_check

brain = ScriptedBrain()
plan = brain.plan(ctx)
print("plan notes:", plan.notes)
print("tasks:", len(plan.tasks), "| first task:", plan.tasks[0].id if plan.tasks else None)
print("plan JSON (first 700 chars):", plan.to_json()[:700] + (" ..." if len(plan.to_json()) > 700 else ""))

reparsed = parse_plan(plan.to_json())
static_ok, static_errors = static_check(reparsed, scene, spec)
dry = dry_run(reparsed, scene, spec)
print()
print("parse_plan tasks:", len(reparsed.tasks))
print("static_check ok:", static_ok, "| errors:", static_errors[:3])
print("dry_run success:", dry["success"], "| failed_tasks:", dry["failed_tasks"][:5],
      "| missing:", dry["missing"][:5])
print("This validates the fixture's plan on the symbolic scene; it is not an LLM answer.")
""")

nb.md("""## The real-model cell (guarded)

Nothing below runs unless `RTP_SERVE_URL` points at a served OpenAI-compatible endpoint. When it is
unset the cell prints `SKIPPED: ...`; it does not substitute a fake answer. When it is set, the cell
sends the **same** prompt to the endpoint, wraps the reply in `LLMBrain`, and validates the plan with
`parse_plan`. The endpoint/model names come from `RTP_SERVE_URL` / `RTP_SERVE_MODEL`, never from a
hard-coded path in this notebook.
""")

nb.code("""import os, urllib.request

base_url = os.environ.get("RTP_SERVE_URL", "").strip()
if not base_url:
    print("SKIPPED: RTP_SERVE_URL is unset; no served model is called and no answer is fabricated")
else:
    from rtp.brain import LLMBrain
    from rtp.plan import parse_plan

    served_model = os.environ.get("RTP_SERVE_MODEL", prof.name)
    url = base_url.rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        url += "/v1/chat/completions"

    def generate(prompt):
        body = json.dumps({"model": served_model, "temperature": 0.0,
                           "max_tokens": prof.max_len,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
        return payload["choices"][0]["message"]["content"]

    answer = LLMBrain(generate).plan(ctx)
    checked = parse_plan(answer.to_json())
    print("served model:", served_model)
    print("plan tasks:", len(checked.tasks), "| parse_plan accepted:", len(checked.tasks) == len(answer.tasks))
""")

nb.md("""## Chain-of-thought: what can be measured without a model

`build_prompt(ctx, reason=...)` carries a reasoning/context field: the string is appended under the
prompt's `## Notes` section. That is a **prompt-side** hook. The paper's CoT ablation also needs a
model to generate two completions — one with a reasoning prefix, one without — and then to execute
both plans, so that success/quality can be compared.

This notebook can only measure the accounting: prompt length, completion length, and token cost with
the real Qwen tokenizer (or a labelled proxy). The reasoning prefix below is a deterministic fixture
derived from the task; it is **not** a sampled model output. No CoT success result is claimed.
""")

nb.code("""reason = ("Heavy items first (sort by estimated mass); laptop in the laptop slot; "
          "bottle upright in a different compartment; lunchbox last; leave mug and plate untouched.")
prompt_plain = build_prompt(ctx)
prompt_reasoned = build_prompt(ctx, reason=reason)

tokenizer = None
if model_dir is not None:
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        print("tokenizer:", type(tokenizer).__name__, "| vocab:", tokenizer.vocab_size)
    except Exception as exc:                   # noqa: BLE001  (tokenizer files missing)
        print("tokenizer unavailable:", type(exc).__name__, "-> using a labelled proxy")

def count_tokens(text):
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(1, round(len(text) / 3.5))      # labelled proxy: ~3.5 chars per token

print("counter:", "real Qwen tokenizer" if tokenizer is not None else "proxy: chars/3.5")
print("reason present in reasoned prompt:", reason in prompt_reasoned)
print("prompt tokens: plain =", count_tokens(prompt_plain), "| with reason =", count_tokens(prompt_reasoned))

plan_json = plan.to_json()
cot_prefix = "Reasoning: " + reason + "\\n"
completions = [("plan only", plan_json), ("reasoning + plan", cot_prefix + plan_json)]
for label, text in completions:
    print(f"{label:18s} chars={len(text):5d} tokens={count_tokens(text):5d}")
base_tokens = count_tokens(plan_json)
cot_tokens = count_tokens(cot_prefix + plan_json)
print("CoT prefix cost:", cot_tokens - base_tokens, "tokens; delta prompt cost:",
      count_tokens(prompt_reasoned) - count_tokens(prompt_plain), "tokens")
print("This is token accounting only: the prefix is a fixture, and no model was asked to reason.")
""")

nb.md("""## Tool use: `ToolBrain`

`ToolBrain` is the multi-turn Brain. Each model turn is exactly one JSON object: a tool call or the
plan itself. The tools are `check_rule`, `query_scene`, `look`, `dry_run`, and `submit_plan`. Only the
committed plan reaches the arms; drafts and dry runs cost tokens, not robot time.

To exercise the tool loop on CPU without a model, the cell supplies a scripted chat fixture that
calls `check_rule`, then `dry_run` on the same plan the ScriptedBrain produced, then `submit_plan`.
The results printed are the real return values from `rtp/toolbrain.py` and `rtp/rules.py`; the chat
fixture is labelled, and it is not a model's tool-use policy.
""")

nb.code("""from rtp.toolbrain import TOOL_HEADER, ToolBrain

print(TOOL_HEADER)
print()

draft = ScriptedBrain().plan(ctx)
plan_dict = json.loads(draft.to_json())
steps = [
    {"tool": "check_rule", "args": {"rule_id": "heavy_bottom"}},
    {"tool": "dry_run", "args": {"plan": plan_dict}},
    {"tool": "submit_plan", "args": {"plan": plan_dict}},
]
cursor = {"i": 0}

def scripted_chat(messages):
    last = messages[-1]["content"]
    if last.startswith("TOOL RESULT:"):
        print("real tool result:", last[len("TOOL RESULT: "):][:260])
    out = json.dumps(steps[cursor["i"]])
    cursor["i"] += 1
    return out

tb = ToolBrain(scripted_chat, harness=H)
committed = tb.plan(ctx)
print()
print("tool calls:", tb.totals["tool_calls"], "| turns:", tb.totals["turns"],
      "| errors:", tb.totals["errors"], "| plans:", tb.totals["plans"])
print("committed tasks:", len(committed.tasks), "| parse_plan accepted:",
      len(parse_plan(committed.to_json()).tasks) == len(committed.tasks))
print("The chat policy is a fixture; the tool outputs above are real.")
""")

nb.md("""## Measured vs pending

**Measured on this CPU box (symbolic):**

* the harness documents, their types, and real BM25 scores for three queries;
* the hybrid path: BM25 order, fake-dense order, and RRF-fused order, with at least one query whose
  order changes when the dense ranking is fused in;
* a full zero-shot prompt, its retrieved rules, and a `ScriptedBrain` plan that round-trips through
  `parse_plan` and passes the static/dry-run checks;
* token costs for a plain prompt/plan and for a reasoning-augmented prompt/plan, using the real
  tokenizer when the cache is available;
* two real `ToolBrain` tool calls (`check_rule`, `dry_run`) and a committed plan.

**Pending, not claimed:**

* no served model was called (`RTP_SERVE_URL` unset) and no zero-shot LLM success number exists;
* the CoT ablation needs a model to generate both completions and a loop to execute both; only the
  token accounting is measured here;
* the fake embedder is not the real embedding model, and the Qdrant installer/run is separate;
* the scene/plans are symbolic `rtp/world.py` fixtures, not Isaac Lab or physics results.
* `ScriptedBrain` is a deterministic teacher/fixture; its plans are not evidence about an LLM Brain.
""")

if __name__ == "__main__":
    print(nb.save())
