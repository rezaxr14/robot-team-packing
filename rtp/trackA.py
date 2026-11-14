"""Track A helper: load the VIKI-R L2 symbolic scorer WITHOUT importing verl (no torch/ray needed),
and measure/neutralise its init-position randomness.

The upstream `eval_single` draws `random.choice(positions)` for every asset on every call, so a
prediction's score can change between runs whenever an asset has more than one candidate start
position. `score_all_positions` enumerates the candidates (deterministic, exhaustive) and
`score_seeded` fixes the draw. Always report which one was used.
"""
from __future__ import annotations
import importlib.util
import itertools
import random
import shutil
import sys
import tempfile
from pathlib import Path

_CACHE = {}


def load_scorer(repo: str):
    """repo: path to a clone of MARS-EAI/VIKI-R. Returns module with eval_single."""
    if repo in _CACHE:
        return _CACHE[repo]
    src = Path(repo) / "verl/verl/utils/reward_score/utils/eval"
    tmp = Path(tempfile.mkdtemp()) / "vikieval"
    shutil.copytree(src, tmp)
    f = tmp / "eval_viki_2.py"
    txt = f.read_text().replace("from verl.utils.reward_score.utils.eval.eval import Eval", "from .eval import Eval")
    f.write_text(txt)
    (tmp / "__init__.py").write_text("")
    spec = importlib.util.spec_from_file_location("vikieval", tmp / "__init__.py", submodule_search_locations=[str(tmp)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["vikieval"] = pkg
    spec.loader.exec_module(pkg)
    mod = importlib.import_module("vikieval.eval_viki_2")
    _CACHE[repo] = mod
    return mod


def _movable(gt: dict):
    return {a: p for a, p in gt["init_pos"].items() if not (a.startswith("R") and a[1:].isdigit())}


def score_seeded(mod, pred, gt, seed: int) -> bool:
    random.seed(seed)
    return bool(mod.eval_single(pred, gt))


def score_all_positions(mod, pred, gt, max_combos: int = 5000) -> dict:
    """Exhaustively score over every combination of candidate init positions.
    Returns fraction of combinations that succeed and whether the outcome is position-dependent."""
    mv = _movable(gt)
    names = list(mv)
    combos = list(itertools.product(*[mv[n] for n in names]))
    if len(combos) > max_combos:
        rng = random.Random(0)
        combos = rng.sample(combos, max_combos)
    wins = 0
    for combo in combos:
        g = dict(gt)
        g["init_pos"] = {**gt["init_pos"], **{n: [p] for n, p in zip(names, combo)}}
        wins += bool(mod.eval_single(pred, g))
    return {"frac_success": wins / len(combos), "n_combos": len(combos),
            "position_dependent": 0 < wins < len(combos)}


def nondeterminism_rate(mod, preds_gts, n_draws: int = 20, seed: int = 0) -> dict:
    """Share of (pred, gt) items whose verdict changes across repeated unseeded draws."""
    rng_state = random.getstate()
    random.seed(seed)
    flips = 0
    for pred, gt in preds_gts:
        outs = {bool(mod.eval_single(pred, gt)) for _ in range(n_draws)}
        flips += len(outs) > 1
    random.setstate(rng_state)
    return {"items": len(preds_gts), "unstable": flips, "unstable_rate": flips / max(1, len(preds_gts))}


# ------------------------------------------------------------------------------------------------
# Track A evaluation runner pieces: upstream scorer modules, prompt extraction, per-item scoring
# ------------------------------------------------------------------------------------------------
import ast
import base64
import concurrent.futures
import re
import types
from dataclasses import dataclass

from .stats import wilson


@dataclass
class Upstream:
    viki1: types.ModuleType
    viki2: types.ModuleType
    evalmod: types.ModuleType      # eval_viki_2 (has eval_single)
    prompts: dict


def _load_reward_module(repo: str, name: str):
    p = Path(repo) / "verl/verl/utils/reward_score" / f"{name}.py"
    txt = p.read_text().replace("from verl.utils.reward_score.utils.eval.eval_viki_2 import eval_single",
                                "from vikieval.eval_viki_2 import eval_single")
    try:
        from mathruler.grader import grade_answer  # noqa: F401
    except Exception:   # mathruler or its deps (pylatexenc) missing: viki_1's string-match fallback is disabled
        txt = txt.replace("from mathruler.grader import extract_boxed_content, grade_answer",
                          "def extract_boxed_content(*a, **k):\n    return ''\ndef grade_answer(*a, **k):\n    return False")
    mod = types.ModuleType(name)
    exec(compile(txt, str(p), "exec"), mod.__dict__)
    return mod


def extract_prompts(repo: str) -> dict:
    """Pull the upstream prompt templates out of eval/VIKI-L{1,2}/qwen.py via the AST (no copying)."""
    out = {}
    l2 = ast.parse((Path(repo) / "eval/VIKI-L2/qwen.py").read_text())
    for n in l2.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and \
                n.targets[0].id in ("ROBOT_DESCRIPTION", "ACTION_DESCRIPTION", "AGENT_AVAIL_ACTIONS"):
            out[n.targets[0].id] = ast.literal_eval(n.value)
    out["l2_template"] = _fn_string(l2, "generate_cot", "instruction_following")
    l1 = ast.parse((Path(repo) / "eval/VIKI-L1/qwen.py").read_text())
    out["l1_template"] = _fn_string(l1, "generate_cot", "instruction_following")
    out["l1_robot_set"] = _fn_string(l1, "generate_cot", "robot_set")
    return out


def _fn_string(tree, fn_name, var):
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    for n in fn.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == var:
            return ast.literal_eval(n.value)
    raise KeyError(f"{var} not found in {fn_name}")


def load_upstream(repo: str) -> Upstream:
    evalmod = load_scorer(repo)
    return Upstream(_load_reward_module(repo, "viki_1"), _load_reward_module(repo, "viki_2"), evalmod,
                    extract_prompts(repo))


def to_native(obj):
    """numpy/parquet types -> plain Python (same role as upstream convert_arrays_to_native)."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        if isinstance(obj, np.ndarray):
            return [to_native(x) for x in obj.tolist()]
        if isinstance(obj, np.generic):
            return obj.item()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    return obj


def build_messages(level: int, row: dict, up: Upstream) -> tuple:
    """Return (messages, gt, task_description). Mirrors upstream generate_cot message construction."""
    task = row["prompt"][1]["content"].replace("<image>", "").strip()
    img = base64.b64encode(row["images"][0]["bytes"]).decode()
    user = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
            {"type": "text", "text": f"Task Description: {task}\n"}]
    gt = row["reward_model"]["ground_truth"]
    P = up.prompts
    if level == 2:
        gt = to_native(gt)
        robots = gt.get("robots", {})
        rl = list(robots.values())
        system = P["l2_template"].format(
            ACTION_DESCRIPTION=P["ACTION_DESCRIPTION"], robots=robots,
            available_actions={r: P["AGENT_AVAIL_ACTIONS"].get(r, []) for r in rl},
            available_robots={r: P["ROBOT_DESCRIPTION"].get(r, "") for r in rl})
    else:
        system = P["l1_template"].format(robot_set=P["l1_robot_set"])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], gt, task


def parse_l2_commands(response: str, up: Upstream):
    m = re.search(r"<answer>(.*?)</answer>", response or "", re.DOTALL)
    if not m:
        return None
    try:
        pred = ast.literal_eval(m.group(1).strip())
    except Exception:
        return None
    return up.viki2.transform_actions(pred) or None


def score_l2(response: str, gt: dict, up: Upstream, idx: int = 0) -> dict:
    """Per-item L2 scoring. `*_draw` mimics upstream (one random start-position draw, seeded by idx so it
    is repeatable); `*_all` requires success under EVERY candidate start-position combination."""
    fmt = up.viki2.format_reward(response or "")
    cmds = parse_l2_commands(response, up)
    r = {"format": fmt, "parsed": cmds is not None, "feasible_draw": False, "feasible_all": False,
         "feasible_any": False, "feasible_frac": 0.0, "position_dependent": False, "length_ok": False,
         "n_combos": 0}
    if cmds is not None:
        random.seed(idx)
        r["feasible_draw"] = bool(up.evalmod.eval_single(cmds, gt))
        ex = score_all_positions(up.evalmod, cmds, gt)
        r.update(feasible_frac=ex["frac_success"], feasible_all=ex["frac_success"] == 1.0,
                 feasible_any=ex["frac_success"] > 0, position_dependent=ex["position_dependent"],
                 n_combos=ex["n_combos"])
        r["length_ok"] = len(gt["time_steps"]) / len(cmds) >= 0.99
    r["strict_draw"] = r["feasible_draw"] and r["length_ok"]       # upstream definition (one draw)
    r["strict_all"] = r["feasible_all"] and r["length_ok"]         # position-robust
    r["upstream_score"] = 0.1 * fmt + 0.9 * float(r["strict_draw"])
    return r


def score_l1(response: str, gt_str: str, up: Upstream) -> dict:
    s = up.viki1.compute_score(response or "", gt_str)
    return {"format": up.viki1.format_reward(response or ""), "correct": s > 0.1, "upstream_score": s}


def evaluate_rows(rows: list, level: int, up: Upstream, chat_fn, workers: int = 8, limit=None) -> list:
    """chat_fn(messages) -> str. Errors are kept as items with `error` set, never dropped."""
    rows = rows[:limit] if limit else rows

    def one(i):
        try:
            msgs, gt, task = build_messages(level, rows[i], up)
            resp = chat_fn(msgs)
            sc = score_l2(resp, gt, up, i) if level == 2 else score_l1(resp, gt, up)
            tid = gt.get("task_id", f"task_{i}") if isinstance(gt, dict) else f"task_{i}"
            return {"idx": i, "task_id": tid, "task": task, "response": resp, **sc}
        except Exception as e:                                           # noqa: BLE001
            return {"idx": i, "error": f"{type(e).__name__}: {e}"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        items = list(ex.map(one, range(len(rows))))
    return items


def summarize_items(items: list, level: int) -> dict:
    ok = [x for x in items if "error" not in x]
    n = len(ok)

    def rate(key, subset=None):
        s = ok if subset is None else subset
        k = sum(bool(x.get(key)) for x in s)
        lo, hi = wilson(k, len(s))
        return {"k": k, "n": len(s), "rate": k / len(s) if s else float("nan"), "lo": lo, "hi": hi}

    out = {"level": level, "n_items": len(items), "n_scored": n, "n_errors": len(items) - n,
           "format": rate("format")}
    if level == 2:
        out.update({m: rate(m) for m in ("strict_draw", "strict_all", "feasible_draw", "feasible_all",
                                         "position_dependent", "length_ok")})
        feas = [x for x in ok if x.get("feasible_any")]
        out["among_feasible_any_but_too_long"] = rate("length_ok", feas) if feas else None
        out["items_position_dependent"] = sum(bool(x.get("position_dependent")) for x in ok)
    else:
        out["correct"] = rate("correct")
    return out
