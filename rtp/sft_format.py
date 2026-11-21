"""R5: render curated records as chat-JSONL SFT rows (build spec §2.5).

One episode becomes one conversation: a ``system`` message (rules/skills digest), a ``user``
message (scene text + image refs), then the emitted assistant JSON actions with any recorded
tool results as intervening ``user`` messages. Hidden provider reasoning is never available and
never copied: only emitted text (``turn["completion"]``) becomes an assistant message.

Every row carries ``meta = {seed, arm_model, schema, provenance_hash}``. The hash is a full
sha256 over the canonical JSON of ``{messages, seed, arm_model, schema}`` (sorted keys, compact
separators, UTF-8), so identical inputs give byte-identical rows and any edit to a row is
detected by :func:`verify_row`.

Refusal is fail-closed. The formatter has no list of certified arm models of its own, so it
demands the ``curation`` proof block that :func:`rtp.curate.curate` attaches to each kept
record, and rechecks replay verification, certification, model/seed namespace and proof/record
identity. A row is never emitted for an unverified or uncertified record.

Evidence class: FIXTURE (formatting/bookkeeping only; no dataset is claimed here).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from . import seeds as seed_rules

SCHEMA = 1

#: Kept deliberately short and stable: it is a digest, not a copy of the rule/skill documents.
SYSTEM_PROMPT = (
    "You are the RTP Brain. Command the two arms and two eyes through typed JSON actions. "
    "Each turn emits exactly one JSON object: a tool call or submit_plan, with a short visible "
    "rationale. Use only the documented skills and respect the packing rules."
)


class UnverifiedRecordError(ValueError):
    """A record lacks the replay/certification proof required to become a training row."""


def canonical_hash(obj: Any) -> str:
    """Full sha256 over canonical JSON (sorted keys, no whitespace, UTF-8, no ASCII escaping)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provenance_hash(messages: list, seed: int, arm_model: str, schema: int = SCHEMA) -> str:
    """Stable hash of everything in a row except the hash itself."""
    return canonical_hash({"messages": messages, "seed": int(seed),
                           "arm_model": arm_model, "schema": schema})


def _content(value: Any) -> str:
    """Render a message/tool payload as deterministic text without inventing content."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_messages(rec: dict) -> list:
    """Build the system/user/assistant conversation from an eligible record."""
    system = rec.get("system")
    if not isinstance(system, str) or not system:
        system = SYSTEM_PROMPT

    parts = []
    instruction = rec.get("instruction")
    scene_text = rec.get("scene_text")
    if isinstance(instruction, str) and instruction:
        parts.append(instruction)
    if isinstance(scene_text, str) and scene_text and scene_text != instruction:
        parts.append(scene_text)
    image_refs = rec.get("image_refs") or []
    if image_refs:
        parts.append("Images: " + ", ".join(str(p) for p in image_refs))
    user = "\n".join(parts)
    messages: list[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": user}]

    turns = [t for t in (rec.get("turns") or []) if isinstance(t, dict)]
    tool_results = rec.get("tool_results") or []
    if turns:
        for i, turn in enumerate(turns):
            if "completion" not in turn:
                continue
            messages.append({"role": "assistant", "content": _content(turn["completion"])})
            # The tool result of the last turn is the episode outcome, not a prompt for a
            # further assistant action, so no user message follows the final completion.
            if i == len(turns) - 1:
                break
            result = turn.get("tool_result")
            if result is None and i < len(tool_results):
                result = tool_results[i]
            if result is not None:
                messages.append({"role": "user", "content": _content(result)})
    else:
        for plan in (rec.get("plans") or []):          # fallback for plan-only fixtures
            messages.append({"role": "assistant", "content": _content(plan)})

    if not any(m["role"] == "assistant" for m in messages):
        raise ValueError("record has no assistant completions or plans to train on")
    return messages


def _require_eligible(rec: Any) -> None:
    """Raise unless ``rec`` carries a valid replay + certification + namespace proof."""
    if not isinstance(rec, dict):
        raise UnverifiedRecordError("record is not a dict")
    v = rec.get("verification")
    if not isinstance(v, dict) or v.get("ok") is not True:
        raise UnverifiedRecordError("replay verification is missing or not ok")

    proof = rec.get("curation")
    if not isinstance(proof, dict):
        raise UnverifiedRecordError("record was not curated (no curation proof)")
    if proof.get("schema") != SCHEMA:
        raise UnverifiedRecordError("curation proof schema mismatch")
    if proof.get("verified") is not True:
        raise UnverifiedRecordError("curation proof does not assert replay verification")
    if proof.get("arm_certified") is not True:
        raise UnverifiedRecordError("curation proof does not assert a certified arm model")
    if not proof.get("arm_model") or proof.get("arm_model") != rec.get("arm_model"):
        raise UnverifiedRecordError("curation proof arm_model does not match the record")

    seed = rec.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or not seed_rules.is_train(seed):
        raise UnverifiedRecordError(f"seed {seed!r} is not in the training namespace")
    if rec.get("mode") not in (None, "datagen"):
        raise UnverifiedRecordError(f"mode {rec.get('mode')!r} is not a datagen record")


def to_sft_rows(records: Iterable[dict]) -> list:
    """Emit one canonical chat row per curated record; refuse unverified/uncertified records."""
    rows = []
    for rec in records:
        _require_eligible(rec)
        messages = build_messages(rec)
        seed, arm = rec["seed"], rec["arm_model"]
        meta = {"seed": int(seed), "arm_model": arm, "schema": SCHEMA}
        meta["provenance_hash"] = provenance_hash(messages, seed, arm, SCHEMA)
        rows.append({"messages": messages, "meta": meta})
    return rows


def verify_row(row: Any) -> list:
    """Return a list of tampered fields (empty means the row is intact)."""
    problems: list[str] = []
    if not isinstance(row, dict):
        return ["row"]
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return ["meta"]
    if meta.get("schema") != SCHEMA:
        problems.append("meta.schema")
    expected = provenance_hash(row.get("messages"), meta.get("seed"),
                               meta.get("arm_model"), SCHEMA)
    if meta.get("provenance_hash") != expected:
        problems.append("meta.provenance_hash")
    return problems
