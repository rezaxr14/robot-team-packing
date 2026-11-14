"""Rewards.

episode_reward: scalar score of a full episode (used for evaluation and counterfactual credit).
plan-level rewards: cheap symbolic signals for GRPO on the Brain (format, feasibility, rule
compliance via deterministic dry run, plan length). A stochastic rollout reward is provided
for the subset of steps that should use the (still symbolic) closed loop; physics-in-the-loop
reward belongs to Isaac Lab and is NOT implemented here.
"""
from __future__ import annotations
import json

from .plan import parse_plan, PlanError
from .rules import TaskSpec, static_check, dry_run


def episode_reward(res: dict, lam_msg: float = 0.0, w_viol: float = 0.15, w_crush: float = 0.3,
                   w_replan: float = 0.01) -> float:
    """res: dict with n_valid, n_items, violations(list), crushed(list), replans, msg_tokens."""
    n = max(1, res["n_items"])
    return (res["n_valid"] / n
            - w_viol * len(res["violations"])
            - w_crush * len(res["crushed"])
            - w_replan * res.get("replans", 0)
            - lam_msg * res.get("msg_tokens", 0) / 1000.0)


def _text(c) -> str:
    # TRL passes either a string or [{"role":..., "content":...}]
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return c if isinstance(c, str) else ""


def r_format(completion) -> float:
    try:
        parse_plan(_text(completion))
        return 1.0
    except PlanError:
        return 0.0


def r_feasible(completion, scene: dict, spec: TaskSpec) -> float:
    try:
        plan = parse_plan(_text(completion))
    except PlanError:
        return 0.0
    ok, errs = static_check(plan, scene, spec)
    return 1.0 if ok else max(0.0, 1.0 - len(errs) / max(1, len(plan.tasks)))


def r_rules(completion, scene: dict, spec: TaskSpec) -> float:
    """Deterministic dry-run: fraction of spec items validly packed minus rule violations."""
    try:
        plan = parse_plan(_text(completion))
    except PlanError:
        return 0.0
    r = dry_run(plan, scene, spec)
    base = r["n_valid"] / max(1, r["n_items"])
    return max(0.0, base - 0.25 * len(r["violations"]) - 0.25 * len(r["crushed"]))


def r_length(completion, cap: int = 24) -> float:
    try:
        n = len(parse_plan(_text(completion)).tasks)
    except PlanError:
        return 0.0
    return 1.0 if n <= cap else max(0.0, 1.0 - (n - cap) / cap)


def r_rollout(completion, seed: int, harness, noisy_eyes: bool = True, cfg=None) -> float:
    """Closed-loop reward: execute the plan (no replanning, no verification) in the symbolic world seeded like the prompt
    and return the episode score clipped to [0, 1]. Uses the same Eye noise draws as the prompt's scene (common random
    numbers), so the plan is judged on the world it was written for."""
    from .datagen import sample_world, sample_spec, default_eyes
    from .hindsight import ReplayBrain
    from .loop import LoopConfig, run_episode
    try:
        plan = parse_plan(_text(completion))
    except PlanError:
        return 0.0
    w = sample_world(seed)
    cfg = cfg or LoopConfig(verify="none", max_replans=0, allow_look=False, allow_challenge=False)
    res = run_episode(w, default_eyes(seed, noisy_eyes), ReplayBrain(plan), sample_spec(w, seed), harness, cfg)
    return max(0.0, min(1.0, res.score))


DEFAULT_WEIGHTS = {"format": 0.1, "feasible": 0.2, "rules": 0.6, "length": 0.1}
ROLLOUT_WEIGHT = 0.5


def combined(completion, scene, spec, weights=None) -> float:
    w = weights or DEFAULT_WEIGHTS
    return (w["format"] * r_format(completion) + w["feasible"] * r_feasible(completion, scene, spec)
            + w["rules"] * r_rules(completion, scene, spec) + w["length"] * r_length(completion))


# ---- TRL GRPOTrainer adapters: f(prompts, completions, **dataset_columns) -> list[float] ----
# Dataset columns required: scene_json (str), spec_json (str).

def _unpack(kwargs, i):
    scene = json.loads(kwargs["scene_json"][i])
    spec = TaskSpec.from_dict(json.loads(kwargs["spec_json"][i]))
    return scene, spec


def trl_reward_weights(include_rollout: bool = False) -> list:
    """Weights in the same order as trl_reward_funcs(...). Pass as GRPOConfig(reward_weights=...)."""
    w = DEFAULT_WEIGHTS
    base = [w["format"], w["feasible"], w["rules"], w["length"]]
    return base + [ROLLOUT_WEIGHT] if include_rollout else base


def trl_reward_funcs(weights=None, harness=None):
    """TRL adapters. With a harness, a fifth reward (closed-loop rollout) is appended; needs the `seed` column."""
    def format_reward(prompts, completions, **kw):
        return [r_format(c) for c in completions]

    def feasible_reward(prompts, completions, **kw):
        return [r_feasible(c, *_unpack(kw, i)) for i, c in enumerate(completions)]

    def rules_reward(prompts, completions, **kw):
        return [r_rules(c, *_unpack(kw, i)) for i, c in enumerate(completions)]

    def length_reward(prompts, completions, **kw):
        return [r_length(c) for c in completions]

    def rollout_reward(prompts, completions, **kw):
        return [r_rollout(c, int(s), harness) for c, s in zip(completions, kw["seed"])]

    funcs = [format_reward, feasible_reward, rules_reward, length_reward]
    if harness is not None:
        funcs.append(rollout_reward)
    for f in funcs:
        f.__name__ = f.__name__
    return funcs
