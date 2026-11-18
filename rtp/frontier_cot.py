"""Chain-of-thought trace harness for the Brain.

This module separates two things that are easy to confuse:

* ``run_cot`` sends a prompt to a *real* OpenAI-compatible endpoint (via
  ``rtp.llm_client.ChatClient``, configured by ``RTP_SERVE_URL`` / ``RTP_MODEL``, or an injected
  client) and records what the model actually returned. If no client is configured it raises
  ``NoFrontierClient``: it never fabricates a completion.
* ``scripted_cot`` produces a *real* trace from ``ScriptedBrain`` -- a deterministic teacher and
  test fixture, tagged ``source="scripted-teacher"``. These traces are genuine artifacts of our
  harness (real retrieval, real prompt, real ScriptedBrain plan), but they are NOT evidence about
  a frontier LLM.

Both paths write the same record schema so downstream tooling can count them separately.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .brain import BrainContext, ScriptedBrain, build_prompt
from .llm_client import ChatClient
from .plan import PlanError, extract_json

SCHEMA_VERSION = 1
FRONTIER_SOURCE = "frontier"
SCRIPTED_SOURCE = "scripted-teacher"

# Rules / skills snippets from harness_files/brain/*.md are always included: they are the design
# contract the Brain is supposed to follow, and they are not in the loop's two-pass retrieval
# (which filters for types "rule" and "object"/"skill"/"memory").
BRAIN_DOCS = ("brain_rules", "brain_skills")
RULE_TYPES = ("rule", "rules")
OTHER_TYPES = ("object", "skill", "skills", "memory", "agent", "task")

STYLES = {
    "brief": (
        "## Chain-of-thought request\n"
        "Override the earlier 'Output ONLY a JSON object' instruction for this run. "
        "Before the plan, write a SHORT reasoning prefix (2-4 sentences) naming the rules you "
        "applied and any uncertainty from the scene. Then give the final plan as one JSON object. "
        "Use exactly this format:\n"
        "Reasoning: <2-4 sentences>\n"
        'Plan: {"tasks":[...],"notes":"..."}'
    ),
    "full": (
        "## Chain-of-thought request\n"
        "Override the earlier 'Output ONLY a JSON object' instruction for this run. "
        "First write a step-by-step reasoning prefix: list the relevant retrieved rules, order "
        "the items, assign arms, and name any uncertainty; then give the final plan as one JSON "
        "object. Use exactly this format:\n"
        "Reasoning: <numbered steps>\n"
        'Plan: {"tasks":[...],"notes":"..."}'
    ),
}


class NoFrontierClient(RuntimeError):
    """Raised when a frontier completion would require inventing one."""


@dataclass
class CoTRequest:
    prompt: str
    doc_ids: list
    docs: list
    ctx: BrainContext


def char_proxy_tokens(text: str) -> int:
    """Labelled fallback counter (~3.5 chars/token). Not a real tokenizer."""
    return max(1, round(len(text) / 3.5)) if text else 0


def _counter(token_counter):
    fn = token_counter or char_proxy_tokens
    label = getattr(fn, "__name__", None) or type(fn).__name__
    return fn, ("char-proxy(chars/3.5)" if token_counter is None else label)


def cot_request(harness, spec, scene, *, k: int = 6, style: str = "brief", embed_fn=None,
                notes=(), reason: str = "") -> CoTRequest:
    """Build a CoT prompt from real harness retrieval + ``rtp.brain.build_prompt``.

    Two retrieval passes mirror ``rtp.loop.retrieve_docs``: rules against the instruction,
    objects/skills/memory/agents against the scene. The brain_rules/brain_skills design-contract
    snippets are added explicitly. ``Harness.retrieve`` is the only retrieval entry point, so an
    optional ``embed_fn`` exercises the same BM25+RRF path as the rest of the harness.
    """
    if harness is None:
        raise ValueError("a Harness is required for CoT retrieval")
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {sorted(STYLES)}")
    notes = list(notes)
    q_rules = spec.text()
    q_other = " ".join(o.get("cls", "") for o in scene.values()) + " " + " ".join(notes)

    docs, seen = [], set()

    def add(rows):
        for d in rows:
            if d.id not in seen:
                seen.add(d.id)
                docs.append(d)

    add(harness.retrieve(q_rules, k, types=RULE_TYPES, embed_fn=embed_fn))
    add(harness.retrieve(q_other, k, types=OTHER_TYPES, embed_fn=embed_fn))
    for rid in BRAIN_DOCS:
        if rid in harness.docs:
            add([harness.get(rid)])

    ctx = BrainContext(spec=spec, scene=scene,
                       rule_ids={d.id for d in docs if d.type in RULE_TYPES},
                       snippets=docs, notes=notes)
    prompt = build_prompt(ctx, reason=reason) + "\n" + STYLES[style]
    return CoTRequest(prompt=prompt, doc_ids=[d.id for d in docs], docs=docs, ctx=ctx)


def build_cot_prompt(harness, spec, scene, *, k: int = 6, style: str = "brief", embed_fn=None,
                     notes=(), reason: str = "") -> str:
    """The exact text to send to a model (or to the scripted teacher)."""
    return cot_request(harness, spec, scene, k=k, style=style, embed_fn=embed_fn,
                       notes=notes, reason=reason).prompt


def frontier_status() -> tuple[str, str | None]:
    """(status, model). status is 'configured' or 'unavailable: <reason>'."""
    url = os.environ.get("RTP_SERVE_URL", "").strip()
    if not url:
        return "unavailable: RTP_SERVE_URL is unset", None
    model = os.environ.get("RTP_MODEL") or os.environ.get("RTP_SERVE_MODEL") or "default"
    return "configured", model


def _configured_client(model: str | None = None) -> ChatClient:
    status, env_model = frontier_status()
    if status != "configured":
        raise NoFrontierClient(
            "No frontier client configured. Set RTP_SERVE_URL to an OpenAI-compatible /v1 base "
            "URL (and optionally RTP_MODEL), or pass client=... to run_cot. No completion will "
            "be fabricated. (" + status + ")"
        )
    url = os.environ["RTP_SERVE_URL"].strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")].rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    api_key = os.environ.get("RTP_API_KEY", "EMPTY") or "EMPTY"
    return ChatClient(base_url=url, model=model or env_model, api_key=api_key)


def _call_client(client, prompt: str, *, max_tokens: int, system: str | None = None) -> str:
    """Accept a ChatClient, an OpenAI-SDK-style client, or a plain prompt->str callable."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    if hasattr(client, "chat"):
        chat = client.chat
        completions = getattr(chat, "completions", None)
        if completions is not None and hasattr(completions, "create"):
            resp = completions.create(model=getattr(client, "_cot_model", "default"),
                                      messages=messages, temperature=0.0, max_tokens=max_tokens)
            return resp.choices[0].message.content or ""
        return chat(messages, temperature=0.0, max_tokens=max_tokens)
    if callable(client):
        return client(prompt)
    raise NoFrontierClient("client must expose chat(messages, ...) or be a prompt->str callable")


_REASON = re.compile(r"(?is)^\s*Reasoning\s*:\s*(.*?)(?=\n\s*Plan\s*:|\s\{)")


def split_completion(text: str) -> tuple[str, str]:
    """Return (reasoning prefix, plan JSON) from a model completion. Tolerant of fences/no prefix."""
    if not isinstance(text, str):
        return "", ""
    m = _REASON.match(text)
    reasoning = m.group(1).strip() if m else ""
    obj = None
    try:
        obj = extract_json(text, prefer_keys=("tasks",))
    except PlanError:
        obj = None
    return reasoning, (json.dumps(obj, separators=(",", ":")) if obj is not None else "")


def _base_record(req: CoTRequest, *, source: str, style: str, seed, model, token_counter):
    count, label = _counter(token_counter)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "is_frontier": source == FRONTIER_SOURCE,
        "style": style,
        "seed": seed,
        "model": model,
        "prompt": req.prompt,
        "prompt_tokens": count(req.prompt),
        "retrieved_doc_ids": list(req.doc_ids),
        "token_counter": label,
        "completion": "",
        "reasoning": "",
        "plan_json": "",
        "completion_tokens": 0,
        "latency_ms": 0.0,
        "error": None,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def run_cot(harness, spec, scene, *, k: int = 6, style: str = "brief", client=None,
            embed_fn=None, seed=None, notes=(), reason: str = "", model: str | None = None,
            max_tokens: int = 1200, token_counter=None) -> dict:
    """Call a real frontier endpoint and record the trace.

    Missing client -> ``NoFrontierClient`` (never an invented completion). A client that fails at
    call time is recorded with ``error`` set, so failures are visible in the trace file too.
    """
    if client is None:
        client = _configured_client(model)
    req = cot_request(harness, spec, scene, k=k, style=style, embed_fn=embed_fn,
                      notes=notes, reason=reason)
    _, env_model = frontier_status()
    rec = _base_record(req, source=FRONTIER_SOURCE, style=style, seed=seed,
                       model=model or getattr(client, "model", None) or env_model or "default",
                       token_counter=token_counter)
    count, _ = _counter(token_counter)
    t0 = time.perf_counter()
    try:
        completion = _call_client(client, rec["prompt"], max_tokens=max_tokens)
        rec["completion"] = completion
        rec["reasoning"], rec["plan_json"] = split_completion(completion)
        rec["completion_tokens"] = count(completion)
    except NoFrontierClient:
        raise
    except Exception as exc:  # noqa: BLE001 -- the error itself is the recorded result
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return rec


def _scripted_reasoning(ctx: BrainContext, plan) -> str:
    """Derive the teacher's rationale from the actual plan and retrieved rules (not a fixture)."""
    rules = sorted(ctx.rule_ids & {"heavy_bottom", "laptop_slot", "bottle_upright",
                                   "lunchbox_last", "leave_distractors", "bag_exclusive"})
    places = [t for t in plan.tasks if t.skill == "place"]
    grasps = [t for t in plan.tasks if t.skill == "grasp"]
    bits = ["retrieved rules: " + (", ".join(rules) if rules else "none")]
    if "heavy_bottom" in ctx.rule_ids:
        bits.append("pack heaviest-estimated items first")
    if "laptop_slot" in ctx.rule_ids:
        slots = sorted({t.params.get("compartment") for t in places if str(t.obj).startswith("laptop")})
        bits.append("laptop to " + ("/".join(s for s in slots if s) or "configured slot"))
    if "bottle_upright" in ctx.rule_ids:
        bits.append("keep bottles upright")
    if "lunchbox_last" in ctx.rule_ids and ctx.spec.last_item:
        bits.append(f"{ctx.spec.last_item} last")
    bits.append(f"emitted {len(grasps)} grasps and {len(places)} places in dependency order")
    if ctx.spec.leave:
        bits.append("left " + ", ".join(ctx.spec.leave) + " untouched")
    return "; ".join(bits) + "."


def scripted_cot(harness, spec, scene, *, k: int = 6, style: str = "brief", embed_fn=None,
                 seed=None, notes=(), reason: str = "", token_counter=None) -> dict:
    """A REAL trace from ScriptedBrain, explicitly marked as a scripted teacher, not an LLM."""
    req = cot_request(harness, spec, scene, k=k, style=style, embed_fn=embed_fn,
                      notes=notes, reason=reason)
    rec = _base_record(req, source=SCRIPTED_SOURCE, style=style, seed=seed,
                       model="ScriptedBrain (deterministic teacher; not a frontier model)",
                       token_counter=token_counter)
    count, _ = _counter(token_counter)
    t0 = time.perf_counter()
    plan = ScriptedBrain().plan(req.ctx, reason=reason)
    reasoning = _scripted_reasoning(req.ctx, plan)
    completion = f"Reasoning: {reasoning}\nPlan: {plan.to_json()}"
    rec["completion"] = completion
    rec["reasoning"], rec["plan_json"] = split_completion(completion)
    rec["completion_tokens"] = count(completion)
    rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return rec


def write_traces(records, path) -> Path:
    """Write records as JSONL (one trace per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def read_traces(path) -> list:
    """Read JSONL traces; a missing file means no traces, not an error."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out
