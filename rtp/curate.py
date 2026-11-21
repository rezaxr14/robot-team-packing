"""R5: curate replay-verified datagen records into an SFT-eligible corpus.

This is **FIXTURE** code: it filters and labels records that an ingest step has already
replay-verified. It does not run a simulator, generate episodes or build a dataset, and no
dataset is claimed until real certified skill tables and certified-arm episodes exist.

Input records are schema-1 ``rtp.datagen_run`` records as accepted by
``rtp.datagen_ingest.ingest`` (each carrying ``verification = {"ok": True, ...}``).

Rules (the runbook R5 / build-spec WP-G exclusion contract):

* keep verified successes;
* keep verified failures that carry an attributed cause (as negatives);
* keep failure -> successful-replan episodes (most valuable, ranked first);
* drop duplicate plan patterns (stable hash of the first plan's task pattern);
* cap each model's share (``max_per_model``);
* **exclude** any record whose ``arm_model`` is not in the certified manifest, whose seed is
  below ``rtp.seeds.TRAIN_MIN`` (100000) or is in ``test_seeds``, or whose mode/verification/
  outcome make it ineligible.

Nothing is silently dropped: every excluded record appears in ``manifest["excluded"]`` with a
reason from :data:`REASONS`, and ``n_input == n_kept + n_excluded`` always holds. Each kept
record gets a ``curation`` proof block that ``rtp.sft_format.to_sft_rows`` checks before it
emits a training row (the formatter has no independent list of certified models).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Iterable

from . import seeds as seed_rules

MANIFEST_SCHEMA = 1

#: The only reasons a record may be excluded with. Closed vocabulary: a new reason must be added
#: here (and tested) rather than invented at a call site.
REASONS = (
    "bad_record",               # not a dict, or missing/invalid model, seed or arm_model
    "uncertified_arm_model",    # arm_model not in the certified manifest
    "test_seed",                # seed < 100000 or explicitly held out
    "wrong_mode",               # not a datagen record (eval never feeds training)
    "error_record",             # outcome == "error" or no evaluated result
    "unverified",               # no well-formed replay verification
    "replay_failed",            # verification.ok is False (replay mismatch / exhausted)
    "unattributed_failure",     # failed but no failure carries an attributed cause
    "duplicate_plan",           # plan-pattern hash already kept
    "per_model_cap",            # model already at max_per_model
)

CATEGORY_SUCCESS = "success"
CATEGORY_FAILURE = "failure"
CATEGORY_REPLAN = "replan"
_PRIORITY = {CATEGORY_REPLAN: 0, CATEGORY_SUCCESS: 1, CATEGORY_FAILURE: 2}


def canonical_hash(obj: Any) -> str:
    """Full sha256 over canonical JSON (sorted keys, no whitespace, UTF-8, no ASCII escaping)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def attributed_failures(rec: dict) -> list:
    """Failures with a usable attributed cause.

    A failure is attributed when it has a non-empty ``cause`` other than ``"unknown"``. The
    optional ``agent``/``obj`` fields are recorded but not required: the producer of the record
    is ``rtp.loop.attribute``, which always supplies them in real data.
    """
    res = rec.get("result")
    if not isinstance(res, dict):
        return []
    out = []
    for f in res.get("failures") or []:
        if not isinstance(f, dict):
            continue
        cause = f.get("cause")
        if isinstance(cause, str) and cause.strip() and cause.strip().lower() != "unknown":
            out.append(f)
    return out


def plan_pattern_hash(rec: dict) -> str:
    """Stable hash of the first plan's pattern.

    Mirrors ``rtp.datagen_memory.notes_from_records``: a plan is canonicalised to its list of
    ``[arm, skill, obj]`` tasks so key order / whitespace in the JSON string cannot create a
    false duplicate. Records without plans fall back to their emitted assistant completions.
    """
    sources = rec.get("plans")
    if not isinstance(sources, list) or not sources:
        sources = [t.get("completion") for t in (rec.get("turns") or [])
                   if isinstance(t, dict) and "completion" in t]
    first = sources[0] if sources else None
    if isinstance(first, str):
        try:
            obj = json.loads(first)
        except (ValueError, TypeError):
            obj = first
    else:
        obj = first
    if isinstance(obj, dict) and isinstance(obj.get("tasks"), list):
        pattern: Any = [[t.get("arm"), t.get("skill"), t.get("obj")] if isinstance(t, dict) else t
                        for t in obj["tasks"]]
    else:
        pattern = obj
    return canonical_hash({"pattern": pattern})


def _category(rec: dict) -> str | None:
    """Return the keep category, or ``None`` when the record has no eligible content."""
    res = rec.get("result")
    if not isinstance(res, dict):
        return None
    fails = attributed_failures(rec)
    replans = res.get("replans") or 0
    if res.get("success") is True:
        try:
            did_replan = int(replans) > 0
        except (TypeError, ValueError):
            did_replan = False
        return CATEGORY_REPLAN if (did_replan and fails) else CATEGORY_SUCCESS
    return CATEGORY_FAILURE if fails else None


def _arm_models_set(arm_models: Iterable[str] | None) -> set:
    if arm_models is None:
        return set()
    if isinstance(arm_models, str):                       # a single model, not a bag of characters
        return {arm_models}
    return {str(m) for m in arm_models}


def _heldout_set(test_seeds: Iterable[int] | None) -> set:
    out = set()
    for s in test_seeds or ():
        try:
            out.add(int(s))
        except (TypeError, ValueError):
            continue
    return out


def _exclusion_reason(rec: Any, certified: set, heldout: set) -> tuple[str | None, dict]:
    """First failing eligibility check (checks are ordered so reasons are reproducible)."""
    if not isinstance(rec, dict):
        return "bad_record", {"type": type(rec).__name__}
    model, arm, seed = rec.get("model"), rec.get("arm_model"), rec.get("seed")
    if not isinstance(model, str) or not model:
        return "bad_record", {"field": "model", "value": model}
    if not isinstance(arm, str) or not arm:
        return "bad_record", {"field": "arm_model", "value": arm}
    if not isinstance(seed, int) or isinstance(seed, bool):
        return "bad_record", {"field": "seed", "value": seed}
    if arm not in certified:
        return "uncertified_arm_model", {"arm_model": arm}
    if not seed_rules.is_train(seed) or seed in heldout:
        return "test_seed", {"seed": seed, "heldout": seed in heldout}
    if rec.get("mode") not in (None, "datagen"):
        return "wrong_mode", {"mode": rec.get("mode")}
    if rec.get("outcome") == "error" or "result" not in rec:
        return "error_record", {"outcome": rec.get("outcome")}
    v = rec.get("verification")
    if not isinstance(v, dict) or not isinstance(v.get("ok"), bool):
        return "unverified", {"ok": v.get("ok") if isinstance(v, dict) else None}
    if v["ok"] is False:
        return "replay_failed", {"reason": v.get("reason"),
                                 "diff_keys": sorted(v.get("diff", {}))
                                 if isinstance(v.get("diff"), dict) else []}
    return None, {}


def _excluded_entry(index: int, rec: Any, reason: str, detail: dict) -> dict:
    """A JSON-serialisable audit row: every dropped record is accounted for."""
    get = rec.get if isinstance(rec, dict) else (lambda _k: None)
    return {"index": index, "seed": get("seed"), "model": get("model"),
            "arm_model": get("arm_model"), "reason": reason, "detail": detail}


def curate(records: Iterable[dict], *, arm_models: Iterable[str], test_seeds: Iterable[int] = (),
           max_per_model: int | None = None, dedupe: bool = True) -> dict:
    """Filter and label records; return a manifest (including the kept records).

    Parameters
    ----------
    records:
        Replay-verified schema-1 records. Consumed once; never modified in place.
    arm_models:
        The certified arm models (keys of ``skill_tables/manifest.json``'s ``arm_models``).
        An empty set excludes everything; this fails closed until A10 certification exists.
    test_seeds:
        Explicit held-out seeds (e.g. the seeds of ``data/heldout_manifest.json``). Seeds
        below ``rtp.seeds.TRAIN_MIN`` are excluded regardless of this argument.
    max_per_model:
        Maximum kept records per ``model``. ``None`` means no cap; 0 keeps nothing.
    dedupe:
        Drop later records sharing a plan-pattern hash (highest-value category wins).

    Returns
    -------
    dict with ``records`` (kept, each with a ``curation`` proof), ``n_input``/``n_kept``/
    ``n_excluded``, ``kept_by_category``, ``kept_by_model``, ``excluded_by_reason`` (also
    aliased as ``reasons``), the full ``excluded`` audit list, the applied filters and a
    ``sha256`` manifest hash over the kept records' identities.
    """
    certified = _arm_models_set(arm_models)
    heldout = _heldout_set(test_seeds)
    if max_per_model is not None:
        max_per_model = int(max_per_model)
        if max_per_model < 0:
            raise ValueError(f"max_per_model must be >= 0, got {max_per_model}")

    excluded: list[dict] = []
    eligible: list[tuple[int, dict, str]] = []
    for i, rec in enumerate(records):
        reason, detail = _exclusion_reason(rec, certified, heldout)
        category = None if reason else _category(rec)
        if not reason and category is None:
            reason, detail = "unattributed_failure", {"failures": len(attributed_failures(rec))}
        if reason:
            excluded.append(_excluded_entry(i, rec, reason, detail))
            continue
        eligible.append((i, rec, category))

    # Rank replan pairs above plain successes above plain failures; stable on input order.
    eligible.sort(key=lambda item: (_PRIORITY[item[2]], item[0]))

    seen_patterns: set[str] = set()
    deduped: list[tuple[int, dict, str, str]] = []
    for i, rec, category in eligible:
        pattern_hash = plan_pattern_hash(rec)
        if dedupe and pattern_hash in seen_patterns:
            excluded.append(_excluded_entry(i, rec, "duplicate_plan",
                                            {"plan_pattern_hash": pattern_hash}))
            continue
        seen_patterns.add(pattern_hash)
        deduped.append((i, rec, category, pattern_hash))

    per_model: Counter = Counter()
    kept: list[dict] = []
    for i, rec, category, pattern_hash in deduped:
        model = rec["model"]
        if max_per_model is not None and per_model[model] >= max_per_model:
            excluded.append(_excluded_entry(i, rec, "per_model_cap",
                                            {"model": model, "max_per_model": max_per_model}))
            continue
        per_model[model] += 1
        annotated = dict(rec)
        annotated["curation"] = {
            "schema": MANIFEST_SCHEMA,
            "verified": True,
            "arm_certified": True,
            "arm_model": rec["arm_model"],
            "category": category,
            "plan_pattern_hash": pattern_hash,
        }
        kept.append(annotated)

    by_category: Counter = Counter(r["curation"]["category"] for r in kept)
    by_model: dict = defaultdict(Counter)
    for r in kept:
        by_model[r["model"]]["total"] += 1
        by_model[r["model"]][r["curation"]["category"]] += 1
    reasons: Counter = Counter(e["reason"] for e in excluded)

    assert all(r in REASONS for r in reasons), sorted(set(reasons) - set(REASONS))
    identities = [{"seed": r["seed"], "model": r["model"], "arm_model": r["arm_model"],
                   "category": r["curation"]["category"],
                   "plan_pattern_hash": r["curation"]["plan_pattern_hash"]} for r in kept]

    return {
        "schema": MANIFEST_SCHEMA,
        "evidence_class": "FIXTURE",
        "n_input": len(kept) + len(excluded),
        "n_kept": len(kept),
        "n_excluded": len(excluded),
        "kept_by_category": {k: by_category.get(k, 0) for k in
                             (CATEGORY_REPLAN, CATEGORY_SUCCESS, CATEGORY_FAILURE)},
        "kept_by_model": {m: dict(c) for m, c in sorted(by_model.items())},
        "excluded_by_reason": dict(sorted(reasons.items())),
        "reasons": dict(sorted(reasons.items())),
        "excluded": excluded,
        "records": kept,
        "arm_models": sorted(certified),
        "test_seeds": sorted(heldout),
        "max_per_model": max_per_model,
        "dedupe": bool(dedupe),
        "sha256": canonical_hash(identities),
    }
