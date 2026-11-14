"""Closed-loop team episode: Eyes -> Brain plan -> blind Arms execute -> recovery ladder -> replan.

Recovery ladder per task: retry -> alternate params (from TOUCH) -> recovery (retreat, retry)
-> reassign to the other arm -> replan -> abort. Causes that retrying cannot fix skip ahead.
Every failure is attributed to Eye / Brain / Arm by rule-based heuristics (documented in
`attribute`); these labels are heuristics for analysis, not ground-truth blame.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import copy

from .messages import Message, MessageBus
from .world import World, required_force, crush_limit
from .plan import Plan, Task, PlanError, next_round, ARMS
from .rules import TaskSpec, RULE_IDS, amend, check_final, static_check, dry_run
from .perception import merge_scenes
from .brain import BrainContext
from .rewards import episode_reward

DETERMINISTIC_CAUSES = {"not_found", "out_of_reach", "wrong_slot", "no_space", "overload",
                        "held_by_other", "bad_compartment", "not_holding", "hand_full", "unknown_skill"}
BRAIN_CAUSES = {"not_found", "wrong_slot", "no_space", "locked", "overload", "held_by_other",
                "bad_compartment", "not_holding", "hand_full", "unknown_skill"}


@dataclass
class LoopConfig:
    comm_mode: str = "push"            # push (every round) | delta (push, changed objects only) | pull | init_only
    verify: str = "static"             # plan verification before execution: none | static | dry_run
    max_plan_retries: int = 2          # rejected plans are sent back to the Brain with the reasons
    allow_look: bool = True            # Brain LOOK requests for items missing from the scene
    max_looks: int = 2
    allow_challenge: bool = True       # arms CHALLENGE an Eye's mass when TOUCH disagrees
    challenge_tol: float = 0.2
    max_challenge_replans: int = 1
    probe: bool = False                # weigh every item with the arms BEFORE planning (needs the heavy_bottom rule)
    use_harness: bool = True
    static_rule_ids: tuple = RULE_IDS
    k_rules: int = 5                   # rule docs retrieved against the instruction text
    k_other: int = 3                   # object/skill/memory docs retrieved against the scene
    max_replans: int = 2
    max_rounds: int = 80
    msg_budget: int = 10 ** 9
    lam_msg: float = 0.0


@dataclass
class EpisodeResult:
    success: bool
    score: float
    n_valid: int
    n_items: int
    violations: list
    crushed: list
    missing: list
    retrieval_recall: float
    plan_pass: bool            # first plan passes the deterministic symbolic dry run on the Eye scene
    scene_final: dict
    replans: int
    rounds: int
    msg_tokens: int
    n_messages: int
    failures: list
    executed: list
    plans: list
    spec_final: dict
    looks: int = 0
    challenges: int = 0
    plan_retries: int = 0
    probes: int = 0

    def as_reward_dict(self) -> dict:
        return {"n_valid": self.n_valid, "n_items": self.n_items, "violations": self.violations,
                "crushed": self.crushed, "replans": self.replans, "msg_tokens": self.msg_tokens}


def retrieve_docs(harness, cfg, spec, scene, notes):
    """Two-pass retrieval: rules match the instruction; objects/skills match the scene and notes."""
    if harness is None:
        return []
    if not cfg.use_harness:
        return [harness.get(i) for i in cfg.static_rule_ids if i in harness.docs]
    rules = harness.retrieve(spec.text(), cfg.k_rules, types=("rule",))
    q2 = " ".join(o["cls"] for o in scene.values()) + " " + " ".join(notes)
    other = harness.retrieve(q2, cfg.k_other, types=("object", "skill", "memory"))
    return rules + other


def attribute(world: World, scene: dict, t: Task, res) -> dict:
    cause = res.cause
    o_true = world.objs.get(t.obj)
    o_rep = scene.get(t.obj)
    if cause == "arm_noise":
        who = "Arm"
    elif cause == "out_of_reach":
        who = "Eye" if (o_rep and o_true and o_rep["region"] != o_true.region) else "Brain"
    elif cause in ("slip", "crush"):
        who = "Arm"
        if o_rep and o_true:
            if abs(o_rep["mass_guess"] - o_true.mass) / max(0.05, o_true.mass) > 0.15 or \
               abs(o_rep["fragility_guess"] - o_true.fragility) > 0.15:
                who = "Eye"
    elif cause in BRAIN_CAUSES:
        who = "Eye" if (o_rep and o_true and o_rep["cls"] != o_true.cls) else "Brain"
    else:
        who = "Arm"
    return {"agent": who, "cause": cause, "task": t.id, "obj": t.obj}


def _alt_params(t: Task, res, scene: dict) -> dict:
    p = dict(t.params)
    if t.skill == "grasp":
        f = float(p.get("force", 10.0))
        if res.touch and "measured_weight" in res.touch and res.cause != "crush" and t.obj in scene:
            mw = res.touch["measured_weight"]
            need = 4.0 + 6.0 * mw
            crush = 30.0 - 22.0 * scene[t.obj]["fragility_guess"]
            if res.cause == "slip":
                f = min(max(need * 1.1, f + 0.5, f * 1.1), crush - 1.0)
                f = max(f, need * 1.02) if need * 1.02 < crush else (need + crush) / 2.0
            else:
                f = min(need * 1.25, crush - 2.0)
        elif res.cause == "slip":
            f *= 1.15
        elif res.cause == "crush":
            f *= 0.7
        p["force"] = round(f, 2)
    return p


def run_episode(world: World, eyes, brain, spec: TaskSpec, harness, cfg: LoopConfig = None,
                amendment=None, eye_hook=None, monitor=None) -> EpisodeResult:
    cfg = cfg or LoopConfig()
    bus = MessageBus(cfg.msg_budget)
    spec = copy.deepcopy(spec)
    state = {"scene": {}, "packed": set(), "notes": [], "replans": 0, "last_sent": {}, "corrections": {},
             "looks": 0, "challenges": 0, "challenge_replans": 0, "plan_retries": 0,
             "probes": 0, "probe_rounds": 0}
    failures, executed, plans = [], [], []

    for arm in ARMS:
        bus.send(Message("ABILITY", f"arm_{arm}", "brain",
                         {"zones": sorted(world.zones[arm]), "payload": world.payload[arm],
                          "force_limit": world.force_limit[arm], "state": "idle"}))

    def _changed(prev, cur) -> bool:
        return (prev["cls"] != cur["cls"] or prev["region"] != cur["region"] or prev["occluded"] != cur["occluded"]
                or abs(prev["mass_guess"] - cur["mass_guess"]) / max(0.05, prev["mass_guess"]) > 0.15
                or abs(prev["fragility_guess"] - cur["fragility_guess"]) > 0.15)

    def observe(t_offset: int = 0):
        msgs = []
        for e in eyes:
            m = e.report(world, world.t + t_offset)
            if eye_hook:
                m = eye_hook(e.name, m, world)
            if cfg.comm_mode == "delta" and e.name in state["last_sent"]:
                last = state["last_sent"][e.name]
                changed = [o for o in m.content["objects"] if o["name"] not in last or _changed(last[o["name"]], o)]
                if changed:
                    bus.send(Message("SCENE", m.sender, m.receiver, {**m.content, "objects": changed}, "delta", m.t))
            else:
                bus.send(m)
            state["last_sent"][e.name] = {o["name"]: o for o in m.content["objects"]}
            msgs.append(m)                    # delta is lossless: the Brain reconstructs the full report
        state["scene"] = {**state["scene"], **merge_scenes(msgs)}      # object memory: unseen objects keep last report
        for n, mw in state["corrections"].items():
            if n in state["scene"]:
                state["scene"][n] = {**state["scene"][n], "mass_guess": mw, "confidence": 1.0}

    def do_look(names):
        """Tool-initiated LOOK: a fresh observation; returns the requested objects now in the scene."""
        if not cfg.allow_look or state["looks"] >= cfg.max_looks:
            return {}
        state["looks"] += 1
        bus.send(Message("LOOK", "brain", "all", {"query": list(names)}))
        observe(t_offset=10000 * state["looks"] + 500)
        return {n: state["scene"][n] for n in names if n in state["scene"]}

    def make_ctx() -> BrainContext:
        scene = state["scene"]
        docs = retrieve_docs(harness, cfg, spec, scene, state["notes"])
        rule_ids = {d.id for d in docs if d.type == "rule"}
        return BrainContext(spec=spec, scene=scene, rule_ids=rule_ids, snippets=docs,
                            notes=list(state["notes"]), packed=set(state["packed"]),
                            holding=dict(world.holding),
                            abilities={a: {"zones": sorted(world.zones[a])} for a in ARMS},
                            look_fn=do_look)

    def get_plan(reason="", replan=False):
        """Ask the Brain for a plan; verify it before it reaches the arms (cfg.verify); send rejections back."""
        feedback = []
        for attempt in range(cfg.max_plan_retries + 1 if cfg.verify != "none" else 1):
            ctx = make_ctx()
            ctx.notes = ctx.notes + feedback
            need = set(spec.rule_ids()) - {"bag_exclusive"}
            if attempt == 0:
                state.setdefault("recall", []).append(len(need & ctx.rule_ids) / max(1, len(need)))
            try:
                plan = brain.replan(ctx, reason) if replan else brain.plan(ctx, reason)
            except PlanError as e:
                failures.append({"agent": "Brain", "cause": "plan_parse", "task": "-", "obj": None, "detail": str(e)})
                feedback = [f"PLAN REJECTED: your reply was not a valid plan ({e})"]
                state["plan_retries"] += attempt < cfg.max_plan_retries and cfg.verify != "none"
                if cfg.verify == "none" or attempt >= cfg.max_plan_retries:
                    return None
                continue
            ok, errs = static_check(plan, state["scene"], spec)
            problems = list(errs)
            dr = None
            if ok and (cfg.verify == "dry_run" or "plan_pass" not in state):
                dr = dry_run(plan, state["scene"], spec)
            if "plan_pass" not in state:                       # one-shot quality of the FIRST plan, before any retry
                state["plan_pass"] = bool(ok and dr["success"])
            if cfg.verify == "dry_run" and dr is not None:
                problems += [f"{v['rule']}: {v['detail']}" for v in dr["violations"]]
                problems += [f"item not packed: {m}" for m in dr["missing"]]
            if problems and cfg.verify != "none" and attempt < cfg.max_plan_retries:
                failures.append({"agent": "Brain", "cause": "plan_rejected", "task": "-", "obj": None,
                                 "detail": "; ".join(problems[:3]), "fatal": False})
                feedback = ["PLAN REJECTED (attempt %d): " % (attempt + 1) + "; ".join(problems[:5])]
                state["plan_retries"] += 1
                continue
            if errs:
                failures.append({"agent": "Brain", "cause": "static_infeasible", "task": "-", "obj": None,
                                 "detail": "; ".join(errs[:3])})
            plans.append(plan.to_json())
            return plan
        return None

    observe()

    def run_probe_phase():
        """Each arm weighs the items in its zone (centre items go to the less busy arm) before any plan is made. The
        measured masses override the Eyes' guesses (corrections), so ordering rules use load-cell values."""
        queue = {"L": [], "R": []}
        for n in spec.items:
            reg = state["scene"].get(n, {}).get("region")
            if reg is None:
                continue
            arm = "L" if reg == "table_left" else "R" if reg == "table_right" else min(queue, key=lambda a: (len(queue[a]), a))
            if reg.startswith("table"):
                queue[arm].append(n)
        used = 0
        while any(queue.values()):
            for arm in ARMS:
                if not queue[arm]:
                    continue
                n = queue[arm].pop(0)
                for attempt in range(2):
                    res = world.execute(arm, "weigh", n, {})
                    if monitor:
                        monitor.record("weigh", world.objs[n].cls if n in world.objs else "", res.ok, res.cause)
                    if res.touch:
                        bus.send(Message("TOUCH", f"arm_{arm}", "brain", res.touch))
                    if res.ok:
                        state["corrections"][n] = res.touch["measured_weight"]
                        state["probes"] += 1
                        break
                    failures.append({**attribute(world, state["scene"], Task(f"w_{n}", arm, "weigh", n), res), "fatal": False})
            used += 1
        state["probe_rounds"] = used
        for n, mw in state["corrections"].items():
            if n in state["scene"]:
                state["scene"][n] = {**state["scene"][n], "mass_guess": mw, "confidence": 1.0}

    if cfg.probe and "heavy_bottom" in spec.rule_ids():
        run_probe_phase()
    plan = get_plan()
    while (cfg.allow_look and plan is not None and getattr(brain, "requests", None)
           and state["looks"] < cfg.max_looks):
        state["looks"] += 1
        wanted = list(brain.requests)
        bus.send(Message("LOOK", "brain", "all", {"query": wanted}))
        observe(t_offset=10000 * state["looks"])               # a fresh look: new draws for the Eyes
        if any(n in state["scene"] for n in wanted):
            plan = get_plan("look found: " + ",".join(n for n in wanted if n in state["scene"]))
            if plan is None:
                break
    tasks: list = list(plan.tasks) if plan else []
    status = {t.id: "pending" for t in tasks}
    attempts: dict = {}
    amended = False
    rounds = 0

    def do_replan(reason):
        nonlocal tasks, status, plan
        state["replans"] += 1
        if cfg.comm_mode in ("pull", "push", "delta"):
            observe()
        state["notes"].append(reason)
        new = get_plan(reason, replan=True)
        keep = [t for t in tasks if status[t.id] == "done"]
        prefix = f"r{state['replans']}_"
        newtasks = []
        if new:
            for t in new.tasks:
                newtasks.append(Task(prefix + t.id, t.arm, t.skill, t.obj, dict(t.params),
                                     [prefix + d for d in t.deps]))
        tasks = keep + newtasks
        status = {t.id: ("done" if t in keep else "pending") for t in tasks}
        attempts.clear()

    while rounds < cfg.max_rounds:
        if amendment and not amended and rounds == amendment[0]:
            amended = True
            spec = amend(spec, **amendment[1])
            bus.send(Message("AMEND", "human", "brain", {"text": spec.text()}))
            do_replan("amendment: " + spec.text())
        chosen, dead = next_round(tasks, status)
        if not chosen:
            break
        replan_needed = None
        challenge_replan = False
        for t in chosen:
            res = world.execute(t.arm, t.skill, t.obj, t.params)
            if res.touch:
                bus.send(Message("TOUCH", f"arm_{t.arm}", "brain", res.touch))
            if monitor:
                cls = world.objs[t.obj].cls if t.obj in world.objs else ""
                monitor.record(t.skill, cls, res.ok, res.cause)
            executed.append({"task": asdict(t), "ok": res.ok, "cause": res.cause, "round": rounds})
            if res.ok:
                status[t.id] = "done"
                if t.skill == "place":
                    state["packed"].add(t.obj)
                if t.skill in ("grasp", "weigh") and cfg.allow_challenge and res.touch and t.obj in state["scene"]:
                    mw, rep = res.touch["measured_weight"], state["scene"][t.obj]["mass_guess"]
                    if abs(mw - rep) / max(mw, 1e-6) > cfg.challenge_tol:     # the blind arm contradicts the Eye
                        bus.send(Message("CHALLENGE", f"arm_{t.arm}", "brain",
                                         {"obj": t.obj, "evidence": {"measured_weight": mw, "reported": rep}}))
                        state["corrections"][t.obj] = mw
                        state["scene"][t.obj] = {**state["scene"][t.obj], "mass_guess": mw, "confidence": 1.0}
                        state["challenges"] += 1
                        challenge_replan = True
                continue
            failures.append({**attribute(world, state["scene"], t, res), "fatal": False})
            bus.send(Message("BLOCKED", f"arm_{t.arm}", "brain", {"task": t.id, "reason": res.cause}))
            n = attempts[t.id] = attempts.get(t.id, 0) + 1
            if res.cause in DETERMINISTIC_CAUSES and n < 3:
                n = attempts[t.id] = 3
            if cfg.comm_mode == "pull":
                observe()
            if n == 1:
                pass                                              # plain retry
            elif n == 2:
                t.params = _alt_params(t, res, state["scene"])    # alternate parameters from TOUCH
            elif n == 3:
                world.execute(t.arm, "retreat", None, {})         # recovery, then retry
                t.params = _alt_params(t, res, state["scene"])
            elif n == 4:
                other = "R" if t.arm == "L" else "L"
                if t.skill == "grasp" and t.obj in world.objs and world.reachable(other, t.obj):
                    t.arm = other
                    for u in tasks:                               # keep the matching place on the same arm
                        if u.obj == t.obj and u.skill == "place" and status[u.id] == "pending":
                            u.arm = other
                else:
                    attempts[t.id] = 5
                    replan_needed = f"task {t.id} failed: {res.cause}"
            else:
                replan_needed = f"task {t.id} failed: {res.cause}"
            if replan_needed:
                break
        if (challenge_replan and not replan_needed and "heavy_bottom" in spec.rule_ids()
                and state["challenge_replans"] < cfg.max_challenge_replans
                and any(status[u.id] == "pending" and u.skill == "place" for u in tasks)):
            state["challenge_replans"] += 1
            do_replan("challenge: measured weight differs from the Eye's estimate")
        if replan_needed:
            if state["replans"] - state["challenge_replans"] < cfg.max_replans:
                do_replan(replan_needed)
            else:
                for t in chosen:
                    if status[t.id] == "pending" and attempts.get(t.id, 0) >= 5:
                        status[t.id] = "dead"
                        failures[-1]["fatal"] = True
        if cfg.comm_mode in ("push", "delta"):
            observe()
        rounds += 1

    final = check_final(world, spec)
    scene = state["scene"]
    for n in final["missing"]:
        if n in world.objs and n not in scene:
            failures.append({"agent": "Eye", "cause": "item_never_reported", "task": "-", "obj": n, "fatal": True})
        elif not any(f.get("obj") == n and f.get("fatal") for f in failures):
            failures.append({"agent": "Brain", "cause": "item_not_packed", "task": "-", "obj": n, "fatal": True})
    for v in final["violations"]:
        who = "Brain"
        relevant = [n for n in world.packed() if world.objs[n].cls in ("laptop", "bottle", "lunchbox")]
        if v["rule"] in ("laptop_slot", "bottle_upright", "lunchbox_last") and any(
                n in scene and scene[n]["cls"] != world.objs[n].cls for n in relevant):
            who = "Eye"
        if v["rule"] == "heavy_bottom":
            names = [n for n in world.bag["main"]]
            if any(n in scene and abs(scene[n]["mass_guess"] - world.objs[n].mass) / max(0.05, world.objs[n].mass) > 0.15
                   for n in names):
                who = "Eye"
        failures.append({"agent": who, "cause": "rule_" + v["rule"], "task": "-", "obj": None,
                         "detail": v["detail"], "fatal": True})

    rd = {"n_valid": final["n_valid"], "n_items": final["n_items"], "violations": final["violations"],
          "crushed": final["crushed"], "replans": state["replans"], "msg_tokens": bus.tokens_used}
    return EpisodeResult(success=final["success"], score=episode_reward(rd, lam_msg=cfg.lam_msg),
                         n_valid=final["n_valid"], n_items=final["n_items"], violations=final["violations"],
                         crushed=final["crushed"], missing=final["missing"],
                         retrieval_recall=state["recall"][0] if state.get("recall") else 1.0,
                         plan_pass=state.get("plan_pass", False), scene_final=dict(state["scene"]), replans=state["replans"],
                         rounds=rounds + state["probe_rounds"], msg_tokens=bus.tokens_used, n_messages=len(bus.log),
                         failures=failures, executed=executed, plans=plans, spec_final=spec.to_dict(),
                         looks=state["looks"], challenges=state["challenges"], plan_retries=state["plan_retries"],
                         probes=state["probes"])
