# Scene design: kitchen island, two arms, two eyes (evidence class DESIGN)

Directive (`docs/USER_PLAN_VERBATIM.md`): the actual 3D scene is a kitchen with 2 eyes, 2 arms and the VLM brain, a proper kitchen top and camera placements according to what we build.
All room dimensions below are **assumed [D]** until measured; `ops/design_eyes.py` re-runs on measured values (`ROOM`/`VARIANTS`) and rewrites `reports/eye_placement.json`.

## 1. Decision (D-65, from the study below)
* **Kitchen top**: free-standing island 1.8 x 0.9 m, top at 0.90 m; wall cabinets (z 1.45-2.15 m), one full-height column and one tall unit as occluders; SimReady kitchen assets for fixtures (`docs/ASSETS.md`).
* **Arms**: both on the island **centre line**, bases at x = +-0.35 m (0.70 m apart, FR3 Duo-like bracket; bracket datasheet is a gate), not at the back edge and not at the ends.
* **Eyes**: ring R = 1.6 m, H = 2.2 m, 12 stations; the two eyes start **diametrically opposed, perpendicular to the line joining the arm bases** (stations 3 and 9), then move on the ring through the tool interface.
* **Placement rule**: arms must reach every workspace point (reach-any >= 0.95) with a shared handover zone (reach-both >= 0.30); among layouts satisfying it, maximise worst-case two-eye coverage over four arm poses (home/reach x left/right).

## 2. Layout comparison (workspace 1.6 x 0.7 m, two heights, worst case over arm poses)
| layout | best R / H (m) | 2-eye worst-case coverage | both see | reach any | reach both | cells never seen |
|---|---|---|---|---|---|---|
| `wall_back_arms` | 2.0 / 2.2 | 0.840 | 0.392 | 0.928 | 0.312 | 131 |
| `wall_end_arms` | 2.0 / 2.2 | 0.932 | 0.183 | 0.859 | 0.000 | 64 |
| `free_island_back_arms` | 2.0 / 2.2 | 0.840 | 0.392 | 0.928 | 0.312 | 116 |
| `free_island_end_arms` | 2.0 / 2.2 | 0.965 | 0.320 | 0.859 | 0.000 | 32 |
| `free_back_s35` | 2.0 / 2.2 | 0.840 | 0.392 | 0.928 | 0.312 | 116 |
| `free_side_s35` | 2.0 / 2.2 | 0.985 | 0.531 | 1.000 | 0.465 | 12 |
| `free_back_s55` | 2.0 / 2.2 | 0.744 | 0.100 | 0.914 | 0.120 | 188 |
| `free_side_s55` | 2.0 / 2.2 | 0.976 | 0.284 | 1.000 | 0.222 | 20 |
| `free_back_s70` | 2.0 / 2.2 | 0.691 | 0.036 | 0.840 | 0.020 | 242 |
| `free_side_s70` | 2.0 / 2.2 | 0.971 | 0.233 | 0.996 | 0.044 | 26 |

Reading: arms at the back edge (`*_back_*`) leave 12-19 % of the table unseen by any station because the arms hide the centre-back; arms at the island ends see well but cannot cooperate (reach-both 0); centre-line arms at +-0.35 m satisfy the rule with 98.5 % coverage.
Only `free_side_s35` satisfies the rule among the sweep (free_side_s35).

## 3. Ring size for the chosen layout (H = 2.2 m)
| ring R (m) | H (m) | valid stations | best pair | coverage | both see |
|---|---|---|---|---|---|
| 2.0 | 2.2 | 12 | [3, 9] (180 deg apart) | 0.985 | 0.531 |
| 1.8 | 2.2 | 12 | [2, 8] (180 deg apart) | 0.984 | 0.416 |
| 1.6 | 2.2 | 12 | [3, 9] (180 deg apart) | 0.981 | 0.476 |
| 1.4 | 2.2 | 11 | [3, 9] (180 deg apart) | 0.967 | 0.436 |
A 1.6 m ring loses < 0.5 point of coverage versus 2.0 m, so small kitchens are fine.

## 4. Consequences
* Dead zone (worst pose, both arms over the centre): 12 of 990 sample cells around the centre line; scene generators avoid placing objects there when both arms are extended, and the Brain is told when a cell is unobservable.
* Eye tool design: a **view library** of 12 stations x zoom {1,2,4} x pan/tilt offsets rendered by Newton Warp (rgb+depth) feeds the hosted episodes (`docs/DATAGEN_KAGGLE.md`); minimal station set for 99 % worst-case coverage: [3, 9, 8, 10, 1, 5].
* Not covered: zoomed FOV, real PTZ repeatability, lighting, the real room; the study uses a 70 x 50 deg FOV at zoom 1 and box obstacles.
