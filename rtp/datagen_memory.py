"""Success/failure memory that is rebuilt between data-generation iterations.

After every shard the evaluated records are turned into retrievable ``type: memory`` notes
(``harness_files/memory/mem_*.md``): one note per (cause, object class, agent) with a count and a
fix hint from a *static* table, and a few recent distinct successful plan patterns. The next
iteration's Brain retrieves them through the ordinary ``Harness`` (BM25 / hybrid). Every note records
which model/seed episodes produced it (``memory_manifest.json``). Nothing is written outside the
given harness root.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

FIX = {  # static hints distilled from harness_files/rules and skills; not model output
    "slip": "raise grasp force toward need = 4 + 6*mass (use weigh first) but stay below the crush limit 30 - 22*fragility",
    "crush": "lower grasp force below the crush limit 30 - 22*fragility; fragile items (lunchbox) need gentle force",
    "out_of_reach": "give the task to the arm whose region covers the object, or hand the item over first",
    "wrong_slot": "laptop only in the laptop slot; bottle upright and in a different compartment from the laptop",
    "no_space": "check compartment capacity; place heavy items first, light ones on top",
    "overload": "do not place more than the compartment can carry; pack heavy items at the bottom",
    "held_by_other": "only one arm may hold an item; place or release before the other arm grasps",
    "hand_full": "an arm holding an item must place or release it before grasping another",
    "not_holding": "every place must depend on its grasp by the same arm",
    "bad_compartment": "use only main, front or laptop as compartment names",
    "locked": "only one arm at a time at the bag; order the places with deps",
    "not_found": "look for the object before planning it; do not plan items that were never seen",
    "unknown_skill": "use only grasp, place, release, retreat, push_aside, handover, weigh",
    "plan_parse": "reply with exactly one JSON object (a tool call or the plan)",
    "plan_rejected": "run dry_run on the draft and fix the reported violations before submit_plan",
    "static_infeasible": "run dry_run on the draft and fix the reported errors before submit_plan",
    "arm_noise": "arm execution noise: retry the task; do not change the plan",
}


def obj_class(name) -> str:
    return re.sub(r"\d+$", "", str(name or "")) or "item"


def notes_from_records(records, *, max_success: int = 12) -> list:
    """Aggregate evaluated episode records into memory notes (deterministic order)."""
    fails: dict = {}
    succ: list = []
    for rec in records:
        if rec.get("outcome") == "error" or "result" not in rec:
            continue
        src = {"model": rec.get("model"), "seed": rec.get("seed")}
        for f in rec["result"].get("failures", []):
            key = (f.get("cause", "unknown"), obj_class(f.get("obj")), f.get("agent", "?"))
            e = fails.setdefault(key, {"n": 0, "src": []})
            e["n"] += 1
            if len(e["src"]) < 5:
                e["src"].append(src)
        if rec["result"].get("success") and rec.get("plans"):
            try:
                tasks = json.loads(rec["plans"][0])["tasks"]
            except (ValueError, KeyError, TypeError):
                continue
            pattern = " -> ".join(f"{t.get('arm')}:{t.get('skill')}({t.get('obj')})" for t in tasks[:10])
            succ.append({"instruction": rec.get("instruction", ""), "pattern": pattern, "src": src})
    notes = []
    for (cause, cls, agent), e in sorted(fails.items()):
        notes.append({"kind": "failure", "id": "mem_f_" + re.sub(r"\W+", "_", f"{cause}_{cls}_{agent}").lower(),
                      "keywords": f"failure {cause} {cls} {agent}",
                      "text": f"Past failure ({e['n']}x): {agent} failed with '{cause}' on {cls}. Fix: {FIX.get(cause, 'inspect the tool result and replan with the smallest change')}.",
                      "src": e["src"]})
    seen = set()
    for s in succ:
        h = hashlib.sha1(s["pattern"].encode()).hexdigest()[:8]
        if h in seen or len(seen) >= max_success:
            continue
        seen.add(h)
        notes.append({"kind": "success", "id": "mem_s_" + h, "keywords": "success plan pattern " + s["instruction"][:80],
                      "text": f"A plan that succeeded for '{s['instruction'][:160]}': {s['pattern']}.", "src": [s["src"]]})
    return notes


def write_memory_docs(notes, harness_root, iteration: int) -> list:
    """Replace ``<root>/memory/mem_*.md`` with the notes; returns the written ids."""
    d = Path(harness_root) / "memory"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("mem_*.md"):
        old.unlink()
    ids = []
    for n in notes:
        (d / f"{n['id']}.md").write_text(
            f"---\nid: {n['id']}\ntype: memory\nkeywords: {n['keywords']}\n---\n{n['text']}\n", encoding="utf-8")
        ids.append(n["id"])
    manifest = {"iteration": iteration, "n_notes": len(notes), "kinds": dict(Counter(n["kind"] for n in notes)),
                "notes": [{"id": n["id"], "kind": n["kind"], "src": n["src"]} for n in notes]}
    (d / "memory_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return ids
