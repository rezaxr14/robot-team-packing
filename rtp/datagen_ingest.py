"""Ingest, VERIFY and ledger the records downloaded from hosted Kaggle runs; rebuild memory; repeat.

Per record (``rtp.datagen_run`` schema 1):

0. (module doc) see verify_record
1. identity/namespace checks: schema, kind, mode/seed namespace (datagen >= 100000, eval < 100000), model recorded;
2. **replay verification**: the recorded plans are replayed in order through ``run_episode`` on the same seeded world
   and the recomputed outcome (success, validity counts, violations) is compared with the recorded one. A mismatch is
   not silently accepted or dropped: the record is marked ``replay_mismatch`` with the differing keys;
3. quarantine rules: error records, mismatches and parse-only failures never feed memory or SFT.

``ingest`` returns accepted/quarantined records and a per-model ledger entry (successes, failures by cause, cost,
tokens, finish reasons) appended to ``ledger.jsonl``; ``next_memory`` rebuilds the success/failure notes for the next
iteration from ALL accepted datagen records so far (never from eval records).
"""
from __future__ import annotations

import json
import dataclasses
from collections import Counter, defaultdict
from pathlib import Path

from .benchmark import TRAIN_SEED_BASE, make_cases
from .brain import LLMBrain
from .datagen_memory import notes_from_records
from .harness import Harness
from .kaggle_chat import request_hash
from .loop import run_episode
from .toolbrain import ToolBrain

COMPARE = ("success", "plan_pass", "n_valid", "n_items", "replans", "violations", "missing")


class ReplayExhausted(RuntimeError):
    """The replay asked for more model turns than were recorded."""


class RecordedChat:
    """A ``chat`` that returns the recorded completions in order (no network). Counts request-hash drift."""

    def __init__(self, turns):
        self.turns, self.i, self.hash_drift = list(turns), 0, 0

    def __call__(self, messages, **_kw) -> str:
        if self.i >= len(self.turns):
            raise ReplayExhausted(f"replay needed turn {self.i} but only {len(self.turns)} were recorded")
        t = self.turns[self.i]
        self.i += 1
        if request_hash(messages) != t["request_hash"]:
            self.hash_drift += 1                                  # prompts/memory differ from generation time
        return t["completion"]


def verify_record(rec: dict, harness_root) -> dict:
    """Replay the recorded model turns through the SAME code path (ToolBrain/LLMBrain + run_episode) on the same
    seeded world and compare the recomputed outcome with the recorded one. Looks, dry-runs and RNG draws repeat exactly,
    so any outcome difference means tampering or code drift; changed prompts are reported as ``prompt_drift`` only."""
    if rec.get("outcome") == "error":
        return {"ok": False, "reason": "error_record", "diff": {}}
    if rec.get("schema") != 1 or rec.get("kind") != "multiturn_episode":
        return {"ok": False, "reason": "bad_schema", "diff": {}}
    seed, mode = rec.get("seed"), rec.get("mode")
    if mode == "datagen" and not (isinstance(seed, int) and seed >= TRAIN_SEED_BASE):
        return {"ok": False, "reason": "seed_namespace", "diff": {"seed": seed}}
    if mode == "eval" and not (isinstance(seed, int) and seed < TRAIN_SEED_BASE):
        return {"ok": False, "reason": "seed_namespace", "diff": {"seed": seed}}
    if not rec.get("model") or not rec.get("turns"):
        return {"ok": False, "reason": "missing_identity_or_turns", "diff": {}}
    _s, world, spec, eyes, lc = make_cases([seed])[0]
    chat = RecordedChat(rec["turns"])
    harness = Harness(harness_root)
    if rec.get("condition") == "zero_shot":
        lc = dataclasses.replace(lc, use_harness=False, verify="none", max_plan_retries=0, allow_look=False)
        brain = LLMBrain(lambda prompt: chat([{"role": "user", "content": prompt}]))
    else:
        brain = ToolBrain(chat, harness, max_turns=int(rec.get("max_turns", 8)))
    try:
        res = run_episode(world, eyes, brain, spec, harness, lc)
    except ReplayExhausted as e:
        return {"ok": False, "reason": "replay_exhausted", "diff": {"detail": str(e)}}
    again = {"success": bool(res.success), "plan_pass": bool(res.plan_pass), "n_valid": res.n_valid, "n_items": res.n_items,
             "replans": res.replans, "violations": list(res.violations), "missing": list(res.missing)}
    diff = {k: {"recorded": rec["result"].get(k), "replayed": again[k]} for k in COMPARE if rec["result"].get(k) != again[k]}
    if chat.i != len(rec["turns"]) and not diff:
        diff["turns_consumed"] = {"recorded": len(rec["turns"]), "replayed": chat.i}
    return {"ok": not diff, "reason": None if not diff else "replay_mismatch", "diff": diff, "replayed": True,
            "prompt_drift": chat.hash_drift}


def read_records(path) -> list:
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def ingest(episodes_path, harness_root, *, check_identity: dict | None = None) -> dict:
    """Verify every record of a downloaded shard. ``check_identity`` may pin e.g. {"model": ..., "iteration": ...}."""
    accepted, quarantined = [], []
    for rec in read_records(episodes_path):
        v = verify_record(rec, harness_root)
        if v["ok"] and check_identity:
            bad = {k: rec.get(k) for k, want in check_identity.items() if rec.get(k) != want}
            if bad:
                v = {"ok": False, "reason": "identity_mismatch", "diff": bad}
        (accepted if v["ok"] else quarantined).append({**rec, "verification": v} if v["ok"] else {"seed": rec.get("seed"), "model": rec.get("model"), "reason": v["reason"], "diff": v["diff"]})
    return {"accepted": accepted, "quarantined": quarantined}


def ledger_entry(accepted: list, quarantined: list, *, shard: str, iteration: int) -> dict:
    by_model: dict = defaultdict(lambda: {"episodes": 0, "success": 0, "causes": Counter(), "cost_usd": 0.0, "cost_known": True,
                                           "prompt_tokens": 0, "completion_tokens": 0, "finish_reasons": Counter()})
    for r in accepted:
        m = by_model[r["model"]]
        m["episodes"] += 1
        m["success"] += int(r["result"]["success"])
        for f in r["result"]["failures"]:
            m["causes"][f.get("cause", "unknown")] += 1
        u = r["usage"]
        m["prompt_tokens"] += u["prompt_tokens"]; m["completion_tokens"] += u["completion_tokens"]
        if u["cost_usd"] is None:
            m["cost_known"] = False
        else:
            m["cost_usd"] += u["cost_usd"]
        m["finish_reasons"].update(x for x in u["finish_reasons"] if x)
    out = {"shard": shard, "iteration": iteration, "accepted": len(accepted), "quarantined": len(quarantined),
           "quarantine_reasons": dict(Counter(q["reason"] for q in quarantined)), "models": {}}
    for k, m in by_model.items():
        out["models"][k] = {**m, "causes": dict(m["causes"]), "finish_reasons": dict(m["finish_reasons"]),
                            "cost_usd": round(m["cost_usd"], 6) if m["cost_known"] else None,
                            "success_rate": round(m["success"] / m["episodes"], 4) if m["episodes"] else None}
        del out["models"][k]["cost_known"]
    return out


def next_memory(all_accepted: list) -> list:
    """Notes for the next iteration: datagen records only (eval records must never feed memory)."""
    return notes_from_records([r for r in all_accepted if r.get("mode") == "datagen"])


def sft_eligible(accepted: list, certified_arm_models: set) -> list:
    """Records allowed into the SFT corpus: datagen mode, verified, and produced under a CERTIFIED arm-outcome model.
    Bring-up records made with the stand-in ``sim_pose_v0`` arm model are kept for harness/CoT debugging only."""
    return [r for r in accepted if r.get("mode") == "datagen" and r.get("arm_model") in certified_arm_models]


def append_ledger(path, entry: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
