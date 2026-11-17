# Assets: SimReady-first digital twin (decision D-34)

Labels as in `IMPLEMENTATION_PLAN.md`. Sources and licences verified in S-32…S-35.

## 1. Sources
| use | source | licence / access | notes |
|---|---|---|---|
| general objects, furniture, appliances | `nvidia/simready-assets` (HF) — 2,677 USD assets with physics, ~490 GB | CC-BY-4.0 | pull only chosen packages |
| object packages, materials, HDRIs | `imagineio/PhysicalAI-SimReady-Assets` — public, ungated; each package has `com.nvidia.simready.packaging.json` (owner + licence) | per-package licence in that file | USD + textures + MDL + physics; usable by URL or local download |
| kitchen-specific assets | `imagine-io/PhysicalAI-Simready-Kitchen-Assets` | CC-BY-4.0 | contents to be inventoried by `ops/fetch_simready.py --catalog` |
| kitchen scenes for variation / unseen-layout splits | `imagineio/PhysicalAI-SimReady-Homes` — ~319 furnished, collided objects per scene; kitchen parameters (shape, cabinet style, countertop, oven housing…) | gated; non-commercial research and evaluation | scenes resolve assets by `https://` at load time → localise before use |
| real scanned objects | `ALLSIDES/simready-scanned-objects` — 69 scanned real objects with `collision_mode` options | CC-BY-4.0 | use when we own the real object, or as nearest match |
| robots | Franka official FR3 / FR3 Duo / accessory models (URDF, USD, MJCF); MuJoCo Menagerie `franka_fr3`, `franka_fr3_v2`, `franka_fr3_duo` | Franka terms; Menagerie Apache-2.0 | G-5 |
| fixtures that do not exist (ring rail, carriages, mounts, frame) | own CAD → URDF/USD | ours | the only authored assets |
Not used: warehouse/industrial packs (wrong domain).

## 2. Newton compatibility
Newton imports URDF, MJCF and USD; its USD importer uses **schema resolvers** to bridge solver-specific USD schemas (PhysX, MuJoCo) [V:S-18]. SimReady packages author their physics with PhysX-style schemas and a `collision_mode`; whether each package behaves under Newton's MuJoCo-Warp solver is decided per asset by G-13, not assumed.

## 3. Ingestion pipeline (`ops/fetch_simready.py`, runs locally; Kaggle cache only after CLI auth)
1. **Catalogue**: fetch packaging JSON/metadata only → `assets/catalog.parquet` (id, name, class, bbox, mass, licence, source repo, revision).
2. **Select** by task taxonomy (§4) and filters **[D]**: graspable items mass ≤ 1.5 kg and smallest width ≤ the Franka Hand's opening **[G: read hand datasheet]**; rigid only; licence permits our use.
3. **Fetch** only selected packages at a **pinned revision** (`hf download --include …`), sha256 recorded; scenes from Homes are localised (rewrite `https://` references to local paths).
4. **Attribute**: generate `assets/ATTRIBUTION.md` from every package's licence/owner (CC-BY requires credit).
5. **Import** into Newton (`add_usd` with resolvers). Graspables: convex-decomposed collision; furniture/appliances: static bodies with simplified collision; articulated doors/drawers keep authored joints if Newton imports them, otherwise static in v1 **[G]**.
6. **Validate** (`assets/validation/<id>.json`): mass vs package metadata (≤ 5 %), drop test, slide test, friction sanity, scripted Franka Hand grasp, Newton Warp render check (textures/materials visible).
7. **Real-twin override**: for each real object in the lab, measure size, mass, COM, friction; pick the nearest SimReady/scanned asset and override mass/dims with measurements; log the residual.
8. **Freeze**: scene manifests list asset id + revision + sha; benchmark scenes are immutable.

## 4. Object taxonomy for tasks
utensils (spoon, fork, spatula, ladle — blunt only in shared mode) · containers (bowls, mugs, boxes, lunchboxes, jars) · bottles and cans · plates and trays · cutting board · packaging (bags, cartons, pouches) · small appliances handled whole · fixtures (cabinets, counters, columns, shelves, appliances as static obstacles). Knives and other sharp tools are excluded from shared-workspace tasks.

## 5. Kitchen composition
Base room from measured dimensions (`configs/kitchen_default.yaml` until measured). Fixtures, appliances, counters, cabinets and columns from the SimReady kitchen sets; the table carrying the arms is our CAD/standard SimReady table sized to the real one. A column and a cabinet are placed to occlude parts of the table so eye viewpoints matter. Unseen-kitchen generalisation split: layouts from SimReady Homes never used for training.

## 6. Gate G-13 (acceptance)
≥ 20 objects spanning ≥ 6 taxonomy classes and 3 furniture pieces import into Newton; masses within 5 % of metadata; 10 s drop test stable (no explosion/penetration); scripted Franka Hand grasp succeeds ≥ 80 % on ≥ 10 objects; textures visible in the Newton Warp renderer. Failures are recorded per asset and the asset is excluded or its collision mode changed.
