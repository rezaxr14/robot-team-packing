#!/usr/bin/env python3
"""Eye (camera) placement study for the kitchen island: coverage and occlusion by ray casting.

Evidence class DESIGN: analytic geometry with ASSUMED room dimensions (marked below) -- replace ``ROOM`` with
measured values and re-run. For every candidate ring (radius R, height H) and each of 12 stations it computes which
points of the workspace a camera looking at the table centre can see (FOV + occlusion by wall cabinets, a column, a
tall unit and BOTH ARMS in four poses). It then picks the best pair of eyes, the minimal station set, and writes
``reports/eye_placement.json`` (also the lookup table for the eye tool 'which station sees this cell').

    python ops/design_eyes.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLE_Z = 0.90
HFOV, VFOV = np.radians(70.0), np.radians(50.0)          # zoom 1 (repo default [R])
ROOM = {                                                  # ASSUMED [D] -- replace with measurements (wall_back_arms)
    "island": {"x": (-0.9, 0.9), "y": (-0.45, 0.45), "z_top": TABLE_Z},
    "arm_base_xy": [(-0.35, 0.40), (0.35, 0.40)],         # FR3 Duo-like spacing 0.70 m [G: bracket datasheet]
    "room_bounds": {"x": (-2.4, 2.4), "y": (-2.2, 1.55), "z": (0.0, 2.6)},   # cameras must be inside the room
    "obstacles": {                                        # AABBs (min xyz, max xyz)
        "wall_cabinets": ((-1.8, 1.25, 1.45), (1.8, 1.60, 2.15)),
        "column": ((0.95, -0.65, 0.0), (1.25, -0.35, 2.6)),
        "tall_unit": ((-1.9, 0.30, 0.0), (-1.3, 0.95, 2.1)),
    },
}


_BASE = ROOM
VARIANTS = {
    "wall_back_arms": _BASE,
    "wall_end_arms": {**_BASE, "arm_base_xy": [(-0.85, 0.0), (0.85, 0.0)]},
    "free_island_back_arms": {**_BASE,
        "room_bounds": {"x": (-3.0, 3.0), "y": (-2.6, 2.55), "z": (0.0, 2.6)},
        "obstacles": {"wall_cabinets": ((-3.0, 2.25, 1.45), (3.0, 2.60, 2.15)),
                      "column": ((0.95, -0.65, 0.0), (1.25, -0.35, 2.6)),
                      "tall_unit": ((-2.9, 0.30, 0.0), (-2.3, 0.95, 2.1))}},
}
VARIANTS["free_island_end_arms"] = {**VARIANTS["free_island_back_arms"], "arm_base_xy": [(-0.85, 0.0), (0.85, 0.0)]}
for _sx in (0.35, 0.55, 0.70):                             # spacing sweep on the free-standing island
    VARIANTS[f"free_back_s{int(_sx*100)}"] = {**VARIANTS["free_island_back_arms"], "arm_base_xy": [(-_sx, 0.40), (_sx, 0.40)]}
    VARIANTS[f"free_side_s{int(_sx*100)}"] = {**VARIANTS["free_island_back_arms"], "arm_base_xy": [(-_sx, 0.0), (_sx, 0.0)]}
SHOULDER_UP, REACH_PRACTICAL = 0.333, 0.80                  # FR3 shoulder height above base; 0.855 m nominal reach, 0.80 used


def reach_stats(pts):
    """Fraction of workspace points inside each arm's practical reach sphere (shoulder-centred)."""
    ok = []
    for bx, by in ROOM["arm_base_xy"]:
        d = np.linalg.norm(pts - np.array([bx, by, TABLE_Z + SHOULDER_UP]), axis=1)
        ok.append(d <= REACH_PRACTICAL)
    return {"any": float((ok[0] | ok[1]).mean()), "both": float((ok[0] & ok[1]).mean()),
            "left": float(ok[0].mean()), "right": float(ok[1].mean())}


def _box(c, h):
    c = np.asarray(c, float)
    return (tuple(c - np.asarray(h)), tuple(c + np.asarray(h)))


def arm_boxes(base_xy, reach: bool):
    bx, by = base_xy
    base = np.array([bx, by, TABLE_Z])
    if not reach:                                          # folded "ready" pose: column + forearm toward the centre
        tip = np.array([bx * 0.6, by - 0.20, TABLE_Z + 0.65])
    else:                                                  # reaching over the table centre at 0.30 m above the top
        tip = np.array([0.0, -0.10, TABLE_Z + 0.30])
    mid = (base + np.array([0, 0, 0.55]) + tip) / 2
    return [_box(base + np.array([0, 0, 0.30]), (0.08, 0.08, 0.30)),
            _box(mid, tuple(np.maximum(np.abs(tip - mid), 0.06) + 0.05)),
            _box(tip, (0.07, 0.07, 0.07))]


def workspace_points():
    xs, ys = np.arange(-0.80, 0.801, 0.05), np.arange(-0.35, 0.351, 0.05)
    return np.array([[x, y, TABLE_Z + h] for x in xs for y in ys for h in (0.05, 0.20)])   # low items and tall bottles


def visible(cam, target, pts, boxes):
    f = target - cam; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    v = pts - cam
    z = v @ f
    ok = (z > 0) & (np.abs(np.arctan2(v @ r, z)) <= HFOV / 2) & (np.abs(np.arctan2(v @ u, z)) <= VFOV / 2)
    for lo, hi in boxes:                                   # slab test, segment cam -> point
        lo, hi = np.asarray(lo), np.asarray(hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - cam) / v, (hi - cam) / v
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin < 1.0)
        # parallel rays inside the slab give nan -> treated as no constraint by nanmax/nanmin
        ok &= ~hit
    return ok


def free_mask(pts, boxes):
    """Points not inside any obstacle volume for this pose (an object cannot be there, so it is not 'unseen')."""
    m = np.ones(len(pts), bool)
    for lo, hi in boxes:
        m &= ~np.all((pts >= np.asarray(lo)) & (pts <= np.asarray(hi)), axis=1)
    return m


def inside(p, boxes, pad=0.10):
    return any(np.all(p >= np.asarray(lo) - pad) and np.all(p <= np.asarray(hi) + pad) for lo, hi in boxes)


def poses():
    base = list(ROOM["obstacles"].values())
    for rl, rr in itertools.product((False, True), repeat=2):
        arms = arm_boxes(ROOM["arm_base_xy"][0], rl) + arm_boxes(ROOM["arm_base_xy"][1], rr)
        yield f"L{'reach' if rl else 'home'}_R{'reach' if rr else 'home'}", base + arms


def study(R, H, n=12):
    pts, target = workspace_points(), np.array([0.0, 0.0, TABLE_Z])
    static = list(ROOM["obstacles"].values())
    stations = []
    for k in range(n):
        th = 2 * np.pi * k / n
        cam = np.array([R * np.cos(th), R * np.sin(th), H])
        rb = ROOM["room_bounds"]
        in_room = rb["x"][0] <= cam[0] <= rb["x"][1] and rb["y"][0] <= cam[1] <= rb["y"][1] and cam[2] <= rb["z"][1] - 0.05
        stations.append({"k": k, "az_deg": round(float(np.degrees(th)), 1), "cam": cam, "valid": bool(in_room and not inside(cam, static))})
    vis, free = {}, {}                                     # (pose, k) -> bool array; pose -> free-space mask
    for pname, boxes in poses():
        free[pname] = free_mask(pts, boxes)
        for s in stations:
            vis[(pname, s["k"])] = (visible(s["cam"], target, pts, boxes) if s["valid"] else np.zeros(len(pts), bool)) & free[pname]
    return pts, stations, vis, free


def summarize(R, H):
    pts, st, vis, free = study(R, H)
    pn = sorted({p for p, _ in vis})
    frac = lambda m, p: float(m.sum() / free[p].sum())     # fraction of the points an object could occupy
    per_station = {s["k"]: min(frac(vis[(p, s["k"])], p) for p in pn) for s in st}
    best = None
    for a, b in itertools.combinations([s["k"] for s in st if s["valid"]], 2):
        sep = abs(((st[a]["az_deg"] - st[b]["az_deg"]) + 180) % 360 - 180)
        if sep < 60:
            continue
        union = min(frac(vis[(p, a)] | vis[(p, b)], p) for p in pn)
        both = min(frac(vis[(p, a)] & vis[(p, b)], p) for p in pn)
        cand = (round(union, 4), round(both, 4), a, b, sep)
        if best is None or cand[:2] > best[:2]:
            best = cand
    return pts, st, vis, free, pn, per_station, best


def run_variant(name):
    global ROOM
    ROOM = VARIANTS[name]
    results = []
    for R in (1.2, 1.4, 1.6, 1.8, 2.0):
        for H in (2.2, 2.4, 2.5):
            _pts, st, _vis, _free, _pn, ps, best = summarize(R, H)
            results.append({"R": R, "H": H, "valid_stations": sum(s["valid"] for s in st), "best_station_cov_min": round(float(max(ps.values())), 4),
                            "pair": None if best is None else {"stations": [best[2], best[3]], "sep_deg": best[4],
                                                               "union_cov_min_over_arm_poses": float(best[0]), "both_see_min": float(best[1])}})
    results.sort(key=lambda r: (-(r["pair"]["union_cov_min_over_arm_poses"] if r["pair"] else 0),
                                -(r["pair"]["both_see_min"] if r["pair"] else 0), r["R"]))
    top = results[0]
    pts, st, vis, free, pn, ps, best = summarize(top["R"], top["H"])
    covered = np.zeros((len(pn), len(pts)), bool)
    chosen = []
    while True:                                            # greedy minimal station set (worst case over arm poses)
        gain, pick = 0, None
        for s in st:
            if not s["valid"] or s["k"] in chosen:
                continue
            g = sum(((covered[i] | vis[(p, s["k"])]).sum() - covered[i].sum()) for i, p in enumerate(pn))
            if g > gain:
                gain, pick = g, s["k"]
        if pick is None:
            break
        chosen.append(pick)
        for i, p in enumerate(pn):
            covered[i] |= vis[(p, pick)]
        if min(float(covered[i].sum() / free[p].sum()) for i, p in enumerate(pn)) >= 0.99:
            break
    worst_cov = min(float(covered[i].sum() / free[p].sum()) for i, p in enumerate(pn))
    cell_count = np.min([np.where(free[p], sum(vis[(p, s["k"])].astype(int) for s in st), 99) for p in pn], axis=0)
    rep = {"evidence_class": "DESIGN (assumed room dimensions; replace ROOM with measurements)",
           "assumptions": {"room": {k: (v if not isinstance(v, dict) else v) for k, v in ROOM.items()}, "hfov_deg": 70, "vfov_deg": 50,
                           "zoom": 1, "stations": 12, "look_at": [0, 0, TABLE_Z], "workspace_points": int(len(pts)),
                           "arm_poses": pn, "heights_sampled_m_above_top": [0.05, 0.20]},
           "ranked_configs": results,
           "recommended": {"R": top["R"], "H": top["H"], "eye_A_B_stations": top["pair"],
                           "station_table": [{"k": s["k"], "az_deg": s["az_deg"], "cam_xyz": [round(float(c), 3) for c in s["cam"]],
                                              "valid": s["valid"], "coverage_worst_arm_pose": round(float(ps[s["k"]]), 4)} for s in st],
                           "minimal_station_set_for_99pct": chosen, "minimal_set_worst_case_coverage": round(float(worst_cov), 4)},
           "dead_zone": ({"n_cells": int((cell_count < 1).sum()),
                          "bbox_xy": [float(pts[cell_count < 1][:, 0].min()), float(pts[cell_count < 1][:, 0].max()),
                                      float(pts[cell_count < 1][:, 1].min()), float(pts[cell_count < 1][:, 1].max())],
                          "per_y_row": {f"{y:.2f}": int(((cell_count < 1) & (np.isclose(pts[:, 1], y))).sum()) for y in np.unique(pts[:, 1].round(2)) if ((cell_count < 1) & (np.isclose(pts[:, 1], y))).any()}}
                         if (cell_count < 1).any() else None),
           "cells": {"n": int(len(pts)), "min_stations_seeing_a_cell_worst_pose": int(cell_count.min()),
                     "cells_seen_by_fewer_than_2_stations": int((cell_count < 2).sum()), "cells_never_seen": int((cell_count < 1).sum())}}
    rep["variant"] = name
    rep["reach"] = reach_stats(pts)
    return rep


def main() -> int:
    reps = {v: run_variant(v) for v in VARIANTS}
    rows = {}
    for v, r in reps.items():
        pr = r["recommended"]["eye_A_B_stations"]
        rows[v] = {"best_R_H": [r["recommended"]["R"], r["recommended"]["H"]], "pair": pr,
                   "minimal_station_set_for_99pct": r["recommended"]["minimal_station_set_for_99pct"],
                   "minimal_set_worst_case_coverage": r["recommended"]["minimal_set_worst_case_coverage"],
                   "cells_never_seen_worst_pose": r["cells"]["cells_never_seen"], "reach": r["reach"], "dead_zone": r["dead_zone"]}
    # rule: both arms must reach (almost) everything between them and share a handover zone, then maximise camera coverage
    ok = {v: c for v, c in rows.items() if c["reach"]["any"] >= 0.95 and c["reach"]["both"] >= 0.30}
    pool = ok or rows
    best = max(pool, key=lambda v: (pool[v]["pair"]["union_cov_min_over_arm_poses"], pool[v]["pair"]["both_see_min"]))
    out = {"evidence_class": "DESIGN (assumed room dimensions; replace VARIANTS/ROOM with measurements)",
           "selection_rule": "reach_any >= 0.95 and reach_both >= 0.30 (handover zone), then maximise worst-case two-eye coverage",
           "rule_satisfied_by": sorted(ok), "chosen_layout": best, "layout_comparison": rows,
           "detail": reps[best]}
    path = ROOT / "reports" / "eye_placement.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{'layout':26s} {'R':>4s} {'H':>4s} {'union':>6s} {'both':>6s} {'reachAny':>8s} {'reachBoth':>9s} {'neverSeen':>9s}")
    for v, c in rows.items():
        pr = c["pair"]
        print(f"{v:26s} {c['best_R_H'][0]:4.1f} {c['best_R_H'][1]:4.1f} {pr['union_cov_min_over_arm_poses']:6.3f} {pr['both_see_min']:6.3f} "
              f"{c['reach']['any']:8.3f} {c['reach']['both']:9.3f} {c['cells_never_seen_worst_pose']:9d}")
    print("chosen:", best, "| minimal station set:", rows[best]["minimal_station_set_for_99pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
