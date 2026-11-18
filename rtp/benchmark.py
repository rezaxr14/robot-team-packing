"""Held-out test set, Brain scoring and an honest omni-model comparison.

Seed namespace (this is the convention already used by ``rtp.experiments`` and the repo's
dataset scripts; it is fixed here so the split can be verified, not guessed):

* **training / data seeds: >= 100000.** ``gen_brain_sft(n, seed0=100000, ...)`` and its
  siblings in ``rtp.experiments`` walk ``seed0, seed0 + 1, ...``; the dataset builder
  offsets the other datasets to ``seed0 + 300000`` (hindsight), ``+500000`` (GRPO),
  ``+700000`` (eye GRPO) and ``+900000`` (eye SFT), and ``ops/sft_smoke.py`` defaults
  to ``--seed 100000``. Any seed ``>= 100000`` may have been seen by training.
* **evaluation seeds: 0 .. 99999 (< 100000).** The CPU fixture grid and notebook 80 use
  0..79 and 0..7. ``test_seeds(n)`` allocates the first ``n`` of them (``0..n-1``) to
  this fixed held-out test set; ``test_seeds(n, split="train")`` allocates the first
  ``n`` training seeds (``100000..``). ``test_seeds`` asserts the boundary, so an
  evaluation seed can never silently be a training seed.

The same seeds drive world / spec / eye draws through ``rtp.datagen`` (see
``make_cases``), so every model in ``compare`` sees identical episodes and paired
bootstrap on shared seeds is meaningful.

Everything here runs on the **symbolic CPU fixture**. A score from ``score_brain`` is a
*fixture* score unless a trained/served Brain was actually passed in; ``compare``
records the ``kind`` of each row and the CLI states explicitly that no trained adapter
was scored. No Isaac Lab / Newton or hardware claim is made by this module.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .brain import ScriptedBrain
from .datagen import default_eyes, sample_spec, sample_world
from .experiments import ExpConfig
from .loop import LoopConfig, run_episode
from .plan import Plan, PlanError
from .stats import paired_bootstrap_diff, summarize_binary, wilson

TRAIN_SEED_BASE = 100000   # smallest training/data seed; evaluation seeds must be below this
TEST_SEED_BASE = 0         # held-out test seeds are 0, 1, ... n-1

# Models named by the plan (§10) and by notebook 80. A row here with no prediction file
# is reported as `not run` with **no numbers** — it is never filled with an estimate.
COMPARISON_ROSTER = (
    ("Our Brain - Qwen3.5-4B (trained adapter)",
     "rtp.brain.LLMBrain via rtp.llm_client with identical prompt/decoding; needs a served checkpoint",
     "not run in this session: no adapter was loaded"),
    ("Qwen3.5-4B base (identical tools)",
     "same prompt/tools with harness off; fine-tune not performed", ""),
    ("pi0", "fine-tune on our own sim demos (adapter not set up)", ""),
    ("RDT", "fine-tune on our own sim demos (adapter not set up)", ""),
    ("ACT", "fine-tune on our own sim demos (adapter not set up)", ""),
    ("DP (diffusion policy)", "fine-tune on our own sim demos (adapter not set up)", ""),
    ("DP3", "symbolic-skill / RoboTwin adapter (not set up)", ""),
    ("Frontier model as zero-shot Brain",
     "Kaggle-hosted teacher as Brain via rtp.llm_client (needs skill + CLI auth)", ""),
)

PREDICTION_REQUIRED = ("model", "seed", "success", "plan")


# --------------------------------------------------------------------------------------
# fixed split
# --------------------------------------------------------------------------------------
def assert_no_train_overlap(seeds) -> tuple[int, ...]:
    """Refuse any seed in the training namespace (>= 100000)."""
    out = []
    for s in seeds:
        if isinstance(s, bool) or not isinstance(s, int):
            raise ValueError(f"seed must be an int, got {s!r}")
        if s >= TRAIN_SEED_BASE:
            raise AssertionError(f"eval seed {s} is in the training namespace (>={TRAIN_SEED_BASE})")
        if s < 0:
            raise ValueError(f"eval seed must be >= 0, got {s}")
        out.append(s)
    return tuple(out)


def test_seeds(n: int, split: str = "test") -> tuple[int, ...]:
    """Fixed, reproducible seed list for the held-out test set (or the train counterpart).

    ``split="test"`` -> ``0 .. n-1``. ``split="train"`` -> ``100000 .. 100000+n-1``.
    The two are disjoint by construction; the test split is additionally asserted to
    contain no training-namespace seed.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")
    if split == "test":
        seeds = tuple(range(TEST_SEED_BASE, TEST_SEED_BASE + n))
        assert_no_train_overlap(seeds)
    elif split == "train":
        seeds = tuple(range(TRAIN_SEED_BASE, TRAIN_SEED_BASE + n))
        assert not (set(seeds) & set(test_seeds(max(n, 1))))
    else:
        raise ValueError(f"unknown split {split!r}; expected 'test' or 'train'")
    return seeds


def _loop_config(cfg) -> LoopConfig:
    """Accept an ExpConfig (rtp.experiments) or a LoopConfig (rtp.loop)."""
    if cfg is None:
        cfg = ExpConfig("ours", noisy_eyes=True)
    if isinstance(cfg, LoopConfig):
        return cfg
    return LoopConfig(
        comm_mode=cfg.comm, use_harness=cfg.harness, static_rule_ids=tuple(cfg.static_rules),
        max_replans=cfg.max_replans, msg_budget=cfg.msg_budget, k_rules=cfg.k_rules,
        k_other=cfg.k_other, allow_look=cfg.allow_look, allow_challenge=cfg.allow_challenge,
        verify=cfg.verify, max_plan_retries=cfg.max_plan_retries, probe=cfg.probe,
    )


def _gen_params(cfg):
    if cfg is None:
        cfg = ExpConfig("ours", noisy_eyes=True)
    if isinstance(cfg, LoopConfig):
        return 0.03, False, 2
    return cfg.p_arm_noise, cfg.noisy_eyes, cfg.n_eyes


def make_cases(seeds, cfg=None) -> list:
    """Materialise ``(seed, world, spec, eyes, loop_config)`` from the fixed seed list.

    Worlds, specs and Eyes come only from ``rtp.datagen``; two calls with the same
    seeds and config give byte-identical episodes.
    """
    p_arm_noise, noisy_eyes, n_eyes = _gen_params(cfg)
    lc = _loop_config(cfg)
    cases = []
    for s in seeds:
        w = sample_world(s, p_arm_noise=p_arm_noise)
        spec = sample_spec(w, s)
        eyes = default_eyes(s, noisy_eyes)[:n_eyes]
        cases.append((s, w, spec, eyes, lc))
    return cases


# --------------------------------------------------------------------------------------
# scoring our Brain
# --------------------------------------------------------------------------------------
def score_brain(brain, seeds, harness, cfg=None) -> dict:
    """Score a Brain on the held-out fixture seeds.

    Returns one record per seed plus Wilson intervals from ``rtp.stats``. Records carry
    success, plan_pass, msg_tokens and the loop's heuristic Eye/Brain/Arm attribution
    (a label for analysis, not ground-truth blame). ``kind`` is ``"fixture"`` for a
    ScriptedBrain and ``"unverified"`` for anything else until a trained adapter is
    explicitly loaded by the caller.
    """
    seeds = assert_no_train_overlap(seeds)
    if not seeds:
        raise ValueError("score_brain needs at least one held-out seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("score_brain seeds must be unique")
    kind = "fixture" if isinstance(brain, ScriptedBrain) else "unverified"
    kind = getattr(brain, "benchmark_kind", kind)
    model = getattr(brain, "name", None) or type(brain).__name__

    records = []
    for s, w, spec, eyes, lc in make_cases(seeds, cfg):
        res = run_episode(w, eyes, brain, spec, harness, lc)
        fatal = [f for f in res.failures if f.get("fatal")]
        records.append({
            "model": model, "kind": kind, "seed": s,
            "success": bool(res.success), "score": float(res.score),
            "plan_pass": bool(res.plan_pass), "msg_tokens": int(res.msg_tokens),
            "n_valid": res.n_valid, "n_items": res.n_items, "replans": res.replans,
            "rounds": res.rounds,
            "attribution": {
                "available": True,                       # rtp.loop.attribute labels every failure
                "fatal_agents": sorted({f["agent"] for f in fatal}),
                "all_agents": sorted({f["agent"] for f in res.failures}),
                "heuristic": True,
            },
        })

    succ = summarize_binary([r["success"] for r in records])
    plan = summarize_binary([r["plan_pass"] for r in records])
    return {
        "model": model,
        "kind": kind,
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else (
            asdict(ExpConfig("ours", noisy_eyes=True))),
        "seeds": list(seeds),
        "records": records,
        "success": succ,
        "plan_pass": plan,
        # explicit Wilson intervals straight from rtp.stats (summarize_binary uses them too)
        "wilson": {"success": wilson(succ["k"], succ["n"]),
                   "plan_pass": wilson(plan["k"], plan["n"])},
        "mean_msg_tokens": sum(r["msg_tokens"] for r in records) / len(records),
    }


# --------------------------------------------------------------------------------------
# competitor prediction files
# --------------------------------------------------------------------------------------
def _load_one(path: Path) -> list:
    rows = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: invalid JSON: {e}") from e
            if not isinstance(r, dict):
                raise ValueError(f"{where}: row must be a JSON object")
            missing = [k for k in PREDICTION_REQUIRED if k not in r]
            if missing:
                raise ValueError(f"{where}: missing required field(s): {sorted(missing)}")
            if not isinstance(r["model"], str) or not r["model"].strip():
                raise ValueError(f"{where}: 'model' must be a non-empty string")
            if isinstance(r["seed"], bool) or not isinstance(r["seed"], int):
                raise ValueError(f"{where}: 'seed' must be an int")
            if not isinstance(r["success"], bool):
                raise ValueError(f"{where}: 'success' must be a bool (not 0/1)")
            plan = r["plan"]
            if isinstance(plan, str):
                try:
                    plan = json.loads(plan)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{where}: 'plan' is not valid JSON: {e}") from e
            if not isinstance(plan, dict) or not isinstance(plan.get("tasks"), list):
                raise ValueError(f"{where}: 'plan' must be an object with a 'tasks' list")
            try:
                Plan.from_dict(plan)                     # same strict plan schema as our Brain
            except PlanError as e:
                raise ValueError(f"{where}: 'plan' failed schema validation: {e}") from e
            if "reasoning" in r and r["reasoning"] is not None and not isinstance(r["reasoning"], str):
                raise ValueError(f"{where}: 'reasoning' must be a string when present")
            key = (r["model"].strip(), r["seed"])
            if key in seen:
                raise ValueError(f"{where}: duplicate (model, seed) = {key}")
            seen.add(key)
            rows.append({"model": r["model"].strip(), "seed": r["seed"],
                         "success": r["success"], "plan": plan,
                         "reasoning": r.get("reasoning"), "kind": "prediction"})
    return rows


def load_predictions(path) -> list:
    """Load competitor result file(s) as records; refuse malformed or missing rows.

    ``path`` may be a JSONL file or a directory of ``*.jsonl`` files. Required keys are
    ``model`` (str), ``seed`` (int), ``success`` (bool) and ``plan`` (an object with a
    valid ``tasks`` list, JSON-encoded string or dict). ``reasoning`` is optional.
    Unknown extra keys are ignored. Any bad row raises ``ValueError`` with file and line
    number: a partial or malformed file is never partially scored.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
        rows = []
        seen = set()
        for fp in files:
            for r in _load_one(fp):
                key = (r["model"], r["seed"])
                if key in seen:
                    raise ValueError(f"duplicate (model, seed) = {key} across {fp.name}")
                seen.add(key)
                rows.append(r)
        return rows
    if not p.exists():
        raise FileNotFoundError(f"prediction file not found: {p}")
    return _load_one(p)


# --------------------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------------------
def _binary(rows, key):
    if not all(key in r for r in rows):
        return None
    return summarize_binary([bool(r[key]) for r in rows])


def compare(our_records, competitor_records, roster=COMPARISON_ROSTER) -> dict:
    """Per-model comparison on the shared held-out seeds.

    Direction of the paired bootstrap is ``ours - competitor`` (positive = ours higher
    success), computed over the intersection of seeds. Models with no records are
    emitted as ``not run`` with every numeric field ``None`` — no invented numbers.
    """
    our_records = list(our_records)
    competitor_records = list(competitor_records)
    groups = {}
    for r in competitor_records:
        groups.setdefault(r["model"], []).append(r)

    ours_model = our_records[0]["model"] if our_records else "ours"
    ours_by_seed = {r["seed"]: bool(r["success"]) for r in our_records}

    def scored_row(name, rows, how_run="", note="", kind=None, is_ours=False):
        by_seed = {r["seed"]: bool(r["success"]) for r in rows}
        shared = sorted(set(ours_by_seed) & set(by_seed))
        paired = None
        if shared and not is_ours:                        # a row paired with itself is not evidence
            m, lo, hi = paired_bootstrap_diff(
                [ours_by_seed[s] for s in shared], [by_seed[s] for s in shared])
            paired = {"n": len(shared), "mean_diff": m, "lo": lo, "hi": hi}
        tokens = [r["msg_tokens"] for r in rows if "msg_tokens" in r]
        return {
            "model": name, "status": "scored", "n": len(rows),
            "success": summarize_binary([bool(r["success"]) for r in rows]),
            "plan_pass": _binary(rows, "plan_pass"),
            "mean_msg_tokens": sum(tokens) / len(tokens) if tokens else None,
            "kind": kind or (rows[0].get("kind") if rows else None),
            "paired_vs_ours": paired, "how_run": how_run, "note": note,
        }

    def not_run_row(name, how_run="", note=""):
        return {
            "model": name, "status": "not run", "n": None, "success": None,
            "plan_pass": None, "mean_msg_tokens": None, "kind": None,
            "paired_vs_ours": None, "how_run": how_run, "note": note,
        }

    models = []
    if our_records:
        models.append(scored_row(ours_model, our_records,
                                 how_run="rtp.loop.run_episode on the symbolic fixture",
                                 note="fixture / reference planner, NOT a trained model"
                                 if our_records[0].get("kind") == "fixture" else "",
                                 is_ours=True))
    else:
        models.append(not_run_row(ours_model, note="no scores were produced"))

    listed = set()
    for entry in roster:
        name, how_run = entry[0], entry[1]
        note = entry[2] if len(entry) > 2 else ""
        listed.add(name)
        if name == ours_model:
            continue
        models.append(scored_row(name, groups[name], how_run, note) if name in groups
                      else not_run_row(name, how_run, note))
    for name, rows in sorted(groups.items()):
        if name not in listed and name != ours_model:
            models.append(scored_row(name, rows, how_run="prediction file", note="model not in roster"))

    return {
        "ours_model": ours_model,
        "direction": "ours - competitor (positive = ours higher success)",
        "split": "held-out test (seeds < 100000)",
        "models": models,
    }
