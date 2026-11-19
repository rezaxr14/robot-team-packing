"""Multi-turn frontier-model episodes on the harness (data generation AND benchmark rows).

One episode = one seeded world/spec/eyes from ``rtp.benchmark.make_cases`` driven by a chat model
through ``rtp.loop.run_episode``:

* ``condition="harness"``  -> ``ToolBrain`` (multi-turn: check_rule / query_scene / look / dry_run /
  submit_plan) with the retrieved rules, skills and the success/failure memory docs in the prompt.
  Every turn is journaled (``rtp.kaggle_chat.JournaledChat``).
* ``condition="zero_shot"`` -> the same model, one prompt, no harness, no verification retries.
  (Rows "frontier zero-shot" and "base Qwen3.5-4B zero-shot" of the benchmark matrix.)

``mode="datagen"`` accepts only training-namespace seeds (>= 100000); ``mode="eval"`` accepts only
held-out seeds (< 100000) and must never feed memory or training. Every record carries model,
decoding actually used, all turns, tool calls, the plan(s), the evaluated outcome and usage, so
what failed and what succeeded is attributable to a model and an iteration.

``run_shard`` follows the pause/resume rules of KAGGLE_SKILL.md section 8.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from .benchmark import TRAIN_SEED_BASE, assert_no_train_overlap, make_cases
from .brain import LLMBrain
from .harness import Harness
from .kaggle_chat import PAUSE_KINDS, JournaledChat, ProxyError
from .loop import run_episode
from .toolbrain import ToolBrain

SCHEMA = 1
CONDITIONS = ("harness", "zero_shot")
MODES = ("datagen", "eval")


def _check_seed(seed: int, mode: str) -> None:
    if mode == "datagen":
        if seed < TRAIN_SEED_BASE:
            raise AssertionError(f"datagen seed {seed} is in the held-out namespace (< {TRAIN_SEED_BASE})")
    elif mode == "eval":
        assert_no_train_overlap([seed])
    else:
        raise ValueError(f"mode must be one of {MODES}")


def run_one(seed: int, *, transport, harness_root, condition: str = "harness", mode: str = "datagen",
            max_turns: int = 8, journal_dir=None, shard: str = "", iteration: int = 0,
            model_label: str = "", arm_model: str = "sim_pose_v0") -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    _check_seed(seed, mode)
    _s, world, spec, eyes, lc = make_cases([seed])[0]
    jp = Path(journal_dir) / f"{condition}_{seed}.jsonl" if journal_dir else None
    chat = JournaledChat(transport, jp)
    harness = Harness(harness_root)
    if condition == "harness":
        brain = ToolBrain(chat, harness, max_turns=max_turns)
    else:
        brain = LLMBrain(lambda prompt: chat([{"role": "user", "content": prompt}]))
        lc = dataclasses.replace(lc, use_harness=False, verify="none", max_plan_retries=0, allow_look=False)
    t0 = time.perf_counter()
    res = run_episode(world, eyes, brain, spec, harness, lc)
    return {
        "schema": SCHEMA, "kind": "multiturn_episode", "mode": mode, "condition": condition,
        "shard": shard, "iteration": iteration, "max_turns": max_turns, "arm_model": arm_model, "model": transport.model, "model_label": model_label or transport.model,
        "seed": seed, "instruction": spec.text(),
        "decoding": chat.turns[-1]["request_params"] if chat.turns else dict(getattr(transport, "params", {})),
        "turns": chat.turns,
        "tool_calls": dict(getattr(brain, "totals", {}).get("tool_calls", {})) if condition == "harness" else {},
        "plans": list(res.plans),
        "result": {"success": bool(res.success), "score": float(res.score), "plan_pass": bool(res.plan_pass),
                   "n_valid": res.n_valid, "n_items": res.n_items, "violations": list(res.violations),
                   "missing": list(res.missing), "failures": list(res.failures), "replans": res.replans,
                   "rounds": res.rounds, "looks": res.looks, "challenges": res.challenges,
                   "plan_retries": res.plan_retries, "msg_tokens": res.msg_tokens},
        "usage": chat.totals(), "wall_s": round(time.perf_counter() - t0, 3),
    }


def _done_seeds(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(line)["seed"] for line in path.read_text().splitlines() if line.strip()}


def run_shard(*, seeds, transport, harness_root, out_dir, condition: str = "harness", mode: str = "datagen",
              max_turns: int = 8, shard: str = "shard", iteration: int = 0, model_label: str = "",
              max_consecutive_errors: int = 3, backoffs=(30, 60, 120), sleep=time.sleep, reserve_ok=None,
              arm_model: str = "sim_pose_v0") -> dict:
    """Run seeds one by one; append finished episodes; write a manifest that diagnoses any stop."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ep_path, journal_dir = out / "episodes.jsonl", out / "journal"
    done = _done_seeds(ep_path)
    status, reason, consecutive, n_new, errors = "ok", None, 0, 0, []
    for seed in seeds:
        if seed in done:
            continue
        if reserve_ok is not None and not reserve_ok():
            status, reason = "paused", "quota reserve reached"
            break
        attempt = 0
        while True:
            try:
                rec = run_one(seed, transport=transport, harness_root=harness_root, condition=condition, mode=mode,
                              max_turns=max_turns, journal_dir=journal_dir, shard=shard, iteration=iteration,
                              model_label=model_label, arm_model=arm_model)
                with ep_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_new += 1
                consecutive = 0
                break
            except ProxyError as e:
                if e.kind in PAUSE_KINDS:
                    status, reason = "paused", f"{e.kind}: {e}"
                elif e.kind == "rate_limit" and attempt < len(backoffs):
                    sleep(backoffs[attempt])
                    attempt += 1
                    continue
                elif e.kind == "rate_limit":
                    status, reason = "paused", f"rate_limit after {len(backoffs)} back-offs: {e}"
                else:                                         # evidence, not silence
                    errors.append({"seed": seed, "error": f"{type(e).__name__}: {e}"})
                    with ep_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"schema": SCHEMA, "kind": "multiturn_episode", "outcome": "error", "seed": seed,
                                            "model": transport.model, "condition": condition, "mode": mode,
                                            "shard": shard, "iteration": iteration, "error": str(e)[:500]}) + "\n")
                    consecutive += 1
                    if consecutive >= max_consecutive_errors:
                        status, reason = "error", f"{consecutive} consecutive errors (last: {e})"
                break
        if status != "ok":
            break
    recs = [json.loads(line) for line in ep_path.read_text().splitlines() if line.strip()] if ep_path.exists() else []
    good = [r for r in recs if r.get("outcome") != "error"]
    manifest = {"status": status, "reason": reason, "model": transport.model, "arm_model": arm_model, "condition": condition, "mode": mode,
                "shard": shard, "iteration": iteration, "requested": len(list(seeds)), "recorded": len(recs),
                "new_this_run": n_new, "success": sum(1 for r in good if r["result"]["success"]),
                "errors": errors,
                "cost_usd": (sum(r["usage"]["cost_usd"] for r in good)
                             if good and all(r["usage"]["cost_usd"] is not None for r in good) else None),
                "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in good),
                "completion_tokens": sum(r["usage"]["completion_tokens"] for r in good),
                "decoding": next((r["decoding"] for r in good), None)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
