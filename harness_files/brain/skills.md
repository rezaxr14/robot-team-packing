---
id: brain_skills
type: skills
status: draft-v0 (arm params must match rtp/skills.py; eye tools match docs/SCHEMAS.md)
keywords: skills tools eye arm grasp place handover orbit pan tilt zoom kernel detect events
---
# Brain skills and tools

## A. Arm skills (blind arms; compliant L1 underneath)
| skill | params | notes |
|---|---|---|
| `grasp` | `force` (N, bounded) | approach + close; TOUCH/BLOCKED |
| `place` | `compartment` (`main|front|laptop`), `orientation` (`upright|flat`) | |
| `handover` | `to` (`L|R`) | both arms; softest stiffness profile |
| `weigh` | — | joint-load estimate |
| `release` | — | |
| `retreat` | — | safe pose |
| `push_aside` | — | clears a blocker |
Stiffness profile is chosen by the skill layer from an allow-list; it is never a Brain parameter.
Good: `{"id":"t3","arm":"R","skill":"grasp","obj":"bottle","params":{"force":12},"deps":["t2"]}`
Rejected: `{"skill":"grasp","params":{"joint_angles":[0.1,0.2]}}`, or any `stiffness`/`speed` field.

## B. Arm events you receive
`PUSHED` (human displaced the arm), `YIELDED` (plan paused), `RESUMED` (back on plan; `resume_err_m`), `TOUCH`, `BLOCKED`, `FAULT`. You react by waiting or replanning after `RESUMED`; you do not command the arm during `YIELDED`.

## C. Eye tools (cost in budget slots)
| tool | args | returns | cost |
|---|---|---|---|
| `orbit` | `station` or `±1` | viewpoint id, visibility delta | 1 |
| `pan` / `tilt` | `deg` | aim, visible mask | 1 |
| `zoom` | `1..4`, region/object | crop + scale | 1 |
| `kernel` | `sobel|canny|laplacian|blur|clahe|hsv|threshold` | scalar stats (+tile) | 1 |
| `detect` | `items|obstacles|all`, `min_conf` | boxes, class, conf, centroid, `centroid_source` | 1 |
| `segment` | `class` | masks, area, free space | 1 |
| `focus` | `object` | re-aim + zoom-to-fit | 1 |
| `crop` | `object|box` | image tile | 2 |
| `snapshot` | — | full frame | 3 |
| `describe` | — | geometric features | 1 |
| `status` | — | pose, health, latency | 0 |
Good: `{"eye":"B","tool":"orbit","args":{"station":7},"reason":"column hides the bottle from A"}`
Bad: `{"eye":"A","tool":"snapshot","args":{},"reason":""}` (no reason; cheaper tool existed).

## D. Reading replies
`ok=false` → read `error`, do not repeat the identical call. `centroid_source` tells how far to trust a position (`sim_gt` appears only in simulation). `budget_left` is authoritative.
