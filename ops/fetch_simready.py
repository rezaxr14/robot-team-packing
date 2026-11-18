#!/usr/bin/env python3
"""Obtain real SimReady assets from Hugging Face and record an auditable trail.

    python ops/fetch_simready.py --catalog              # query HF, write assets/catalog.{json,csv}
    python ops/fetch_simready.py --catalog --from-cache # re-derive CSV/fields from existing catalog.json
    python ops/fetch_simready.py --fetch                # download only assets/selection.json packages
    python ops/fetch_simready.py --fetch --from-cache   # NO network: build the manifest from files already on disk
    python ops/fetch_simready.py --verify               # check files/sizes/sha256 against the fetch manifest

Design rules:
  * HF_TOKEN is read from the repo's .env (or the environment) programmatically and is
    NEVER printed or written to any artefact.
  * --catalog enumerates every package in the four registered datasets; per-package licence
    is read from com.nvidia.simready.packaging.json / .metadata/LICENSE.txt only for the
    curated selection (other packages carry the repo-level default, flagged unverified).
  * --fetch is an explicit allow-list: a package absent from assets/selection.json cannot be
    downloaded. A global byte budget prevents ever pulling the ~490 GB full dataset.
  * assets/ATTRIBUTION.md and reports/simready_ingest.json are GENERATED, never hand-edited.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
CACHE = ASSETS / "cache"
REPORTS = ROOT / "reports"
ENV_PATH = ROOT / ".env"
SELECTION_PATH = ASSETS / "selection.json"
CATALOG_JSON = ASSETS / "catalog.json"
CATALOG_CSV = ASSETS / "catalog.csv"
ATTRIBUTION_PATH = ASSETS / "ATTRIBUTION.md"
FETCH_MANIFEST = ASSETS / "fetch_manifest.json"
INGEST_REPORT = REPORTS / "simready_ingest.json"

# Registered sources (docs/ASSETS.md 1, SOURCE_REGISTER S-35). layout controls how a
# package id is derived from a relative file path in that repo.
REPOS: list[dict[str, str]] = [
    {"repo_id": "nvidia/simready-assets", "layout": "top",
     "default_license": "CC-BY-4.0",
     "source_url": "https://huggingface.co/datasets/nvidia/simready-assets"},
    {"repo_id": "imagine-io/PhysicalAI-Simready-Kitchen-Assets", "layout": "kitchen",
     "default_license": "CC-BY-4.0",
     "source_url": "https://huggingface.co/datasets/imagine-io/PhysicalAI-Simready-Kitchen-Assets"},
    {"repo_id": "imagineio/PhysicalAI-SimReady-Assets", "layout": "imagineio",
     "default_license": "per-package (com.nvidia.simready.packaging.json)",
     "source_url": "https://huggingface.co/datasets/imagineio/PhysicalAI-SimReady-Assets"},
    {"repo_id": "ALLSIDES/simready-scanned-objects", "layout": "top",
     "default_license": "CC-BY-4.0",
     "source_url": "https://huggingface.co/datasets/ALLSIDES/simready-scanned-objects"},
]
REPO_BY_ID = {r["repo_id"]: r for r in REPOS}

FURNITURE_CLASSES = {"chair", "cabinet", "shelf", "counter", "table"}

# Class guesses from package names. Order matters (first match wins). These are heuristics
# for the catalog only; the authoritative task class is assets/selection.json.
CLASS_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("notebook", re.compile(r"notebook", re.I)),
    ("book", re.compile(r"book", re.I)),
    ("pencilcase", re.compile(r"pencil|stationery", re.I)),
    ("bottle", re.compile(r"bottle|beverage", re.I)),
    ("cup/mug", re.compile(r"mug|cup", re.I)),
    ("plate", re.compile(r"plate|dish", re.I)),
    ("lunchbox/food container", re.compile(r"lunchbox|storage_box|food_container|tupper", re.I)),
    ("bag", re.compile(r"bag|paperbag|backpack", re.I)),
    ("computer", re.compile(r"computer|laptop|monitor", re.I)),
    ("chair", re.compile(r"chair", re.I)),
    ("cabinet", re.compile(r"cabinet", re.I)),
    ("shelf", re.compile(r"shelf|rack", re.I)),
    ("counter", re.compile(r"counter", re.I)),
    ("table", re.compile(r"table", re.I)),
]

META_RE = re.compile(r"(^|/)\.metadata/|com\.nvidia\.simready\.packaging\.json$")
USD_RE = re.compile(
    r"simready_usd/(sm_[^/]*\.(usd|usda)"
    r"|payloads/[^/]*\.(usd|usda)"
    r"|payloads/(MuJoCo|Newton|PhysX|Physics)/[^/]*\.(usd|usda)"
    r"|runnables/[^/]*\.(usd|usda)"
    r"|materials/[^/]*\.mdl"
    r"|source_assets/[^/]*\.mdl"
    r"|\.simready/[^/]*\.json)$"
)
THUMB_RE = re.compile(r"simready_usd/\.thumbs/256x256/[^/]*\.png$")
TEX_RE = re.compile(r"simready_usd/(Textures|source_assets)/[^/]*\.png$")


class UnselectedPackage(RuntimeError):
    """Raised when --fetch is asked for a package absent from the allow-list."""


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- credentials
def load_token(path: Path = ENV_PATH) -> str | None:
    """Read HF_TOKEN from .env / environment without ever emitting its value."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "HF_TOKEN":
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ["HF_TOKEN"] = value
                return value
    return None


class HubClient:
    """Thin injectable wrapper around huggingface_hub (faked in offline tests)."""

    def __init__(self, token: str | None = None, api: Any = None, downloader: Any = None):
        import huggingface_hub as hf  # local import: module stays importable without it

        self.token = token if token is not None else load_token()
        self.api = api if api is not None else hf.HfApi(token=self.token)
        self._download = downloader if downloader is not None else hf.hf_hub_download

    def list_files(self, repo_id: str) -> list[str]:
        return list(self.api.list_repo_files(repo_id, repo_type="dataset"))

    def list_tree(self, repo_id: str, path: str) -> list[Any]:
        return list(self.api.list_repo_tree(repo_id, repo_type="dataset",
                                            path_in_repo=path, recursive=True, expand=True))

    def download(self, repo_id: str, filename: str, local_dir: Path) -> Path:
        self._download(repo_id=repo_id, filename=filename, repo_type="dataset",
                       local_dir=str(local_dir), token=self.token)
        return local_dir / filename


def repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


# --------------------------------------------------------------------------- pure helpers
def package_of(path: str, layout: str) -> str | None:
    """Derive a package id from a repo-relative file path, or None for non-package files."""
    parts = path.split("/")
    if not parts or parts[0].startswith(".") or len(parts) < 2:
        return None
    if layout == "top":
        return parts[0]
    if layout == "kitchen":
        if parts[0] == "packages" and len(parts) >= 4 and parts[1] == "Models":
            return "/".join(parts[:4])
        return "/".join(parts[:2])
    if layout == "imagineio":
        if parts[0] == "assets":
            return "/".join(parts[:3]) if len(parts) >= 3 else None
        if parts[0] == "simready_assets":
            return "/".join(parts[:2])
        return "/".join(parts[:2])
    return parts[0]


def discover_packages(files: Iterable[str], layout: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in files:
        pkg = package_of(f, layout)
        if pkg:
            out.setdefault(pkg, []).append(f)
    return out


def class_guess(package_id: str) -> str:
    for name, pattern in CLASS_RULES:
        if pattern.search(package_id):
            return name
    return "other"


def is_furniture_class(task_class: str) -> bool:
    return task_class in FURNITURE_CLASSES


def extract_license_from_text(packaging_text: str | None, license_text: str | None) -> tuple[str, str, bool]:
    """Return (licence, source, verified) from packaging.json and/or LICENSE.txt text."""
    if packaging_text:
        try:
            obj = json.loads(packaging_text)
            lic = obj.get("license")
            if isinstance(lic, str) and lic.strip():
                return lic.strip(), "com.nvidia.simready.packaging.json", True
        except (ValueError, AttributeError):
            pass
    if license_text:
        for line in license_text.splitlines():
            s = line.strip()
            if re.search(r"creative commons|CC[- ]BY|license", s, re.I) and len(s) < 200:
                return s[:160], ".metadata/LICENSE.txt", True
    return "unknown", "none", False


def load_selection(path: Path = SELECTION_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("packages"), list) or not data["packages"]:
        raise ValueError("selection.json has no packages")
    for item in data["packages"]:
        for key in ("repo_id", "package_id", "task_class", "match"):
            if key not in item:
                raise ValueError(f"selection entry missing {key}: {item}")
        if item["match"] not in ("exact", "nearest"):
            raise ValueError(f"bad match value: {item['match']}")
    return data


def selection_ids(selection: dict[str, Any]) -> set[tuple[str, str]]:
    return {(p["repo_id"], p["package_id"]) for p in selection["packages"]}


def assert_selected(selection: dict[str, Any], requested: Iterable[tuple[str, str]]) -> None:
    """Refuse to fetch anything outside the curated allow-list."""
    allowed = selection_ids(selection)
    bad = [f"{repo}::{pkg}" for repo, pkg in requested if (repo, pkg) not in allowed]
    if bad:
        raise UnselectedPackage("not in assets/selection.json: " + ", ".join(sorted(bad)))


def choose_files(items: Iterable[Any], max_file_bytes: int = 45_000_000,
                 max_textures: int = 2, max_texture_bytes: int = 1_500_000,
                 max_thumbs: int = 1) -> list[tuple[str, int]]:
    """Pick a small, auditable subset of a package's files.

    ``items`` are objects with ``.path`` and ``.size`` (size None for folders), as returned
    by HfApi.list_repo_tree(..., expand=True). The returned pairs are (path, size).
    """
    files = [(it.path, int(it.size)) for it in items
             if getattr(it, "size", None) is not None and getattr(it, "path", None)]
    chosen: list[tuple[str, int]] = []
    thumbs = 0
    for path, size in files:
        if size > max_file_bytes:
            continue
        if META_RE.search(path) or USD_RE.search(path):
            chosen.append((path, size))
        elif THUMB_RE.search(path) and thumbs < max_thumbs:
            chosen.append((path, size))
            thumbs += 1
    # small texture maps only, deterministic order, so downloads are reproducible
    tex = sorted((p, s) for p, s in files
                 if TEX_RE.search(p) and s <= max_texture_bytes and (p, s) not in chosen)
    chosen.extend(tex[:max_textures])
    return chosen


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- catalog
def fetch_package_license(hub: HubClient, repo_id: str, package: str) -> tuple[str, str, bool]:
    """Read the licence of one package from its packaging json / LICENSE.txt."""
    meta_dir = CACHE / "_meta" / repo_slug(repo_id)
    packaging = license_txt = None
    for name in ("com.nvidia.simready.packaging.json", ".metadata/LICENSE.txt"):
        rel = f"{package}/{name}"
        try:
            local = hub.download(repo_id, rel, meta_dir)
            text = local.read_text(encoding="utf-8", errors="replace")
            if name.endswith("packaging.json"):
                packaging = text
            else:
                license_txt = text
        except Exception:
            continue
    return extract_license_from_text(packaging, license_txt)


def enrich_catalog_with_selection(rows: list[dict[str, Any]], selection: dict[str, Any]
                                  ) -> list[dict[str, Any]]:
    """Attach the authoritative task class / match / selection flag to catalog rows."""
    sel = {(p["repo_id"], p["package_id"]): p for p in selection["packages"]}
    for row in rows:
        info = sel.get((row["repo"], row["package"]))
        row["selected"] = info is not None
        row["task_class"] = info.get("task_class", "") if info else ""
        row["match"] = info.get("match", "") if info else ""
    return rows


def build_catalog(hub: HubClient, selection: dict[str, Any] | None = None,
                  repos: list[dict[str, str]] = REPOS) -> list[dict[str, Any]]:
    """Enumerate all packages in the registered repos and build catalog rows."""
    verify_licenses = selection_ids(selection) if selection else set()
    rows: list[dict[str, Any]] = []
    for repo in repos:
        rid = repo["repo_id"]
        files = hub.list_files(rid)
        for package, paths in sorted(discover_packages(files, repo["layout"]).items()):
            cls = class_guess(package)
            if (rid, package) in verify_licenses:
                lic, source, verified = fetch_package_license(hub, rid, package)
            else:
                lic, source, verified = repo["default_license"], "repo default (not per-package read)", False
            rows.append({
                "repo": rid,
                "package": package,
                "class_guess": cls,
                "is_furniture": is_furniture_class(cls),
                "license": lic,
                "license_source": source,
                "license_verified": verified,
                "file_count": len(paths),
                "usd_count": sum(1 for p in paths if p.lower().endswith((".usd", ".usda"))),
                "local_path": str(Path("assets/cache") / repo_slug(rid) / package),
                "source_url": repo["source_url"],
            })
    if selection:
        enrich_catalog_with_selection(rows, selection)
    return rows


def write_catalog(rows: list[dict[str, Any]], json_path: Path = CATALOG_JSON,
                  csv_path: Path = CATALOG_CSV) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "generated_utc": utc_now(), "count": len(rows), "entries": rows}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = ["repo", "package", "class_guess", "selected", "task_class", "match",
              "is_furniture", "license", "license_source", "license_verified",
              "file_count", "usd_count", "local_path", "source_url"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


# --------------------------------------------------------------------------- fetch
def fetch_packages(hub: HubClient, selection: dict[str, Any],
                   packages: Iterable[tuple[str, str]] | None = None,
                   local_root: Path = CACHE, max_bytes: int = 200_000_000) -> dict[str, Any]:
    """Download the curated subset. Every requested package must be in the allow-list."""
    targets = list(packages) if packages is not None else [
        (p["repo_id"], p["package_id"]) for p in selection["packages"]]
    assert_selected(selection, targets)
    budget = int(selection.get("byte_budget", max_bytes))
    budget = min(budget, max_bytes)
    manifest: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    downloaded = 0
    for repo_id, package in targets:
        local_dir = local_root / repo_slug(repo_id)
        tree = hub.list_tree(repo_id, package)
        for path, size in choose_files(tree):
            if downloaded + size > budget:
                skipped.append({"repo": repo_id, "package": package, "path": path,
                                "size": size, "reason": "byte budget"})
                continue
            target = hub.download(repo_id, path, local_dir)
            actual = target.stat().st_size if target.exists() else 0
            downloaded += actual
            try:
                local_path = str(target.relative_to(ROOT))
            except ValueError:  # tests use a temporary cache outside the repo
                local_path = str(target)
            manifest.append({"repo": repo_id, "package": package, "path": path,
                             "size": actual, "sha256": sha256_file(target),
                             "local_path": local_path})
    result = {"schema": 1, "generated_utc": utc_now(), "budget_bytes": budget,
              "bytes_downloaded": downloaded, "files": manifest, "skipped": skipped}
    FETCH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    FETCH_MANIFEST.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def scan_cache(selection: dict[str, Any], local_root: Path = CACHE,
               packages: Iterable[tuple[str, str]] | None = None) -> dict[str, Any]:
    """NO-NETWORK: build a manifest from bytes already present under assets/cache/.

    Selected packages with no files on disk are recorded as pending, never as downloaded.
    """
    targets = list(packages) if packages is not None else [
        (p["repo_id"], p["package_id"]) for p in selection["packages"]]
    assert_selected(selection, targets)
    files: list[dict[str, Any]] = []
    pending: list[dict[str, str]] = []
    for repo_id, package in targets:
        pkg_dir = local_root / repo_slug(repo_id) / package
        found = sorted(f for f in pkg_dir.rglob("*") if f.is_file()
                       and ".cache/huggingface" not in str(f)) if pkg_dir.exists() else []
        if not found:
            pending.append({"repo": repo_id, "package": package, "reason": "no files in local cache"})
            continue
        for target in found:
            try:
                local_path = str(target.relative_to(ROOT))
            except ValueError:
                local_path = str(target)
            files.append({"repo": repo_id, "package": package,
                          "path": str(target.relative_to(pkg_dir)),
                          "size": target.stat().st_size,
                          "sha256": sha256_file(target),
                          "local_path": local_path})
    result = {"schema": 1, "generated_utc": utc_now(),
              "method": "cache-scan (offline, no downloads)",
              "budget_bytes": int(selection.get("byte_budget", 200_000_000)),
              "bytes_downloaded": sum(f["size"] for f in files),
              "files": files, "skipped": [], "pending": pending}
    FETCH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    FETCH_MANIFEST.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def cache_inventory(local_root: Path = CACHE) -> dict[str, dict[str, Any]]:
    """Inventory everything under assets/cache/ so foreign bytes are never misattributed."""
    out: dict[str, dict[str, Any]] = {}
    if not local_root.exists():
        return out
    for child in sorted(local_root.iterdir()):
        if not child.is_dir():
            continue
        items = [f for f in child.rglob("*") if f.is_file()]
        out[child.name] = {"files": len(items), "bytes": sum(f.stat().st_size for f in items)}
    return out


# --------------------------------------------------------------------------- generated reports
def _catalog_lookup(rows: list[dict[str, Any]] | dict[tuple[str, str], dict[str, Any]] | None = None
                    ) -> dict[tuple[str, str], dict[str, Any]]:
    if rows is None:
        try:
            rows = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))["entries"]
        except (OSError, ValueError, KeyError):
            rows = []
    if isinstance(rows, dict):
        return rows
    return {(r["repo"], r["package"]): r for r in rows}


def _package_stats(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for f in manifest["files"]:
        key = (f["repo"], f["package"])
        s = stats.setdefault(key, {"files": 0, "bytes": 0})
        s["files"] += 1
        s["bytes"] += f["size"]
    return stats


def write_attribution(manifest: dict[str, Any], selection: dict[str, Any],
                      catalog: dict[tuple[str, str], dict[str, Any]] | None = None,
                      path: Path = ATTRIBUTION_PATH) -> str:
    """Generate ATTRIBUTION.md from the fetch manifest + catalog + selection (never by hand)."""
    lookup = _catalog_lookup(catalog)
    sel = {(p["repo_id"], p["package_id"]): p for p in selection["packages"]}
    stats = _package_stats(manifest)
    pending_ids = {(p["repo"], p["package"]) for p in manifest.get("pending", [])}
    lines = [
        "# SimReady asset attribution (generated)",
        "",
        f"Generated at {utc_now()} by `ops/fetch_simready.py "
        f"{'--fetch --from-cache' if manifest.get('method', '').startswith('cache-scan') else '--fetch'}` "
        "from `assets/fetch_manifest.json`, `assets/catalog.json` and `assets/selection.json`. "
        "Do not edit by hand; re-run the tool instead.",
        "",
        f"Method: `{manifest.get('method', 'hf_hub_download')}`. "
        f"Bytes recorded as downloaded: **{manifest['bytes_downloaded']}**.",
        "",
        "Licences are read from each package's `com.nvidia.simready.packaging.json` or "
        "`.metadata/LICENSE.txt` where available; otherwise the repo-level default is shown "
        "and flagged. CC-BY-4.0 requires credit: source repositories are listed below.",
        "",
        "| Package | Task class | Match | Status | Source repo | Licence | Licence source | Files | Bytes | Local path |",
        "|---|---|---|---|---|---|---|---:|---:|---|",
    ]
    for key in sorted(sel):
        repo_id, package = key
        info = sel[key]
        s = stats.get(key, {"files": 0, "bytes": 0})
        cat = lookup.get(key, {})
        lic = cat.get("license", "unknown")
        src = cat.get("license_source", "unknown")
        status = "pending" if key in pending_ids or not s["files"] else "downloaded"
        lines.append(
            f"| `{package}` | {info.get('task_class', cat.get('class_guess', '?'))} "
            f"| {info.get('match', 'n/a')} | {status} "
            f"| [{repo_id}]({cat.get('source_url', 'https://huggingface.co/datasets/' + repo_id)}) "
            f"| {lic} | {src} | {s['files']} | {s['bytes']} | `{cat.get('local_path', 'assets/cache')}` |"
        )
    pending = manifest.get("pending", [])
    if pending:
        lines += ["", f"## Pending ({len(pending)} packages, NOT downloaded)", ""]
        for item in pending:
            lines.append(f"- `{item['package']}` ({item['repo']}): {item['reason']}")
        lines += ["", "These packages are curated and catalogued but no bytes exist in `assets/cache/` for them; "
                  "their rows above show 0 files / 0 bytes. They are **not** reported as ingested.", ""]
    nearest = [(key, p) for key, p in sel.items() if p.get("match") == "nearest"]
    if nearest:
        lines += ["", "## Nearest-match notices (not exact class matches)", ""]
        for (repo_id, package), p in sorted(nearest):
            lines.append(f"- `{package}` → **{p['task_class']}**: {p.get('nearest_note', p['why'])}")
    lines += ["", "## Source repositories", ""]
    for repo in REPOS:
        lines.append(f"- [{repo['repo_id']}]({repo['source_url']}) — default licence: {repo['default_license']}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def write_ingest_report(manifest: dict[str, Any], selection: dict[str, Any],
                        catalog: dict[tuple[str, str], dict[str, Any]] | None = None,
                        path: Path = INGEST_REPORT) -> dict[str, Any]:
    lookup = _catalog_lookup(catalog)
    sel = {(p["repo_id"], p["package_id"]): p for p in selection["packages"]}
    stats = _package_stats(manifest)
    classes: dict[str, dict[str, Any]] = {}
    for key, s in stats.items():
        p = sel.get(key, {})
        cls = p.get("task_class", class_guess(key[1]))
        c = classes.setdefault(cls, {"packages": 0, "files": 0, "bytes": 0,
                                     "exact": 0, "nearest": 0,
                                     "is_furniture": bool(p.get("is_furniture"))})
        c["packages"] += 1
        c["files"] += s["files"]
        c["bytes"] += s["bytes"]
        c[p.get("match", "nearest")] += 1
    n_obj = sum(v["packages"] for v in classes.values() if not v["is_furniture"])
    n_cls = len([k for k, v in classes.items() if not v["is_furniture"]])
    n_furn = sum(v["packages"] for v in classes.values() if v["is_furniture"])
    report = {
        "schema": 1,
        "generated_utc": utc_now(),
        "tool": "ops/fetch_simready.py",
        "method": manifest.get("method", "hf_hub_download"),
        "selection": str(SELECTION_PATH.relative_to(ROOT)),
        "totals": {
            "packages": len(stats),
            "files": len(manifest["files"]),
            "bytes": manifest["bytes_downloaded"],
            "budget_bytes": manifest["budget_bytes"],
            "repos": sorted({k[0] for k in stats}),
            "selected_packages": len(sel),
            "pending_packages": len(manifest.get("pending", [])),
        },
        "coverage": {
            "object_packages": n_obj,
            "object_classes": n_cls,
            "furniture_pieces": n_furn,
            "meets_20_objects_6_classes_3_furniture": n_obj >= 20 and n_cls >= 6 and n_furn >= 3,
        },
        "per_class": {k: classes[k] for k in sorted(classes)},
        "pending": manifest.get("pending", []),
        "skipped_files": manifest.get("skipped", []),
        "cache_inventory": cache_inventory(),
        "notes": [
            "Per-package licence verified from packaging.json/LICENSE.txt for selected packages only.",
            "Large albedo/normal textures excluded; one thumbnail and <=2 small textures per package kept.",
            "Nearest-match classes are recorded explicitly in assets/selection.json and assets/ATTRIBUTION.md.",
            "cache_inventory: ~196 MB under cache/menagerie is Franka robot meshes (G-5, a different deliverable), NOT SimReady. Only cache/nvidia__simready-assets/<package> counts as SimReady bytes.",
            "A cache scan is a point-in-time snapshot: files written to assets/cache after this run are not in the manifest; re-run --fetch --from-cache to refresh.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def verify_download(manifest_path: Path = FETCH_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    total = 0
    for f in manifest["files"]:
        p = Path(f["local_path"])
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            problems.append(f"missing {f['local_path']}")
            continue
        size = p.stat().st_size
        total += size
        if size != f["size"]:
            problems.append(f"size mismatch {f['local_path']}: {size} != {f['size']}")
        if sha256_file(p) != f["sha256"]:
            problems.append(f"sha256 mismatch {f['local_path']}")
    return {"files": len(manifest["files"]), "bytes_present": total,
            "bytes_manifest": manifest["bytes_downloaded"], "problems": problems}


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--catalog", action="store_true", help="query HF and write assets/catalog.{json,csv}")
    mode.add_argument("--fetch", action="store_true", help="download the curated packages in assets/selection.json")
    mode.add_argument("--verify", action="store_true", help="check files/sizes/sha256 against the fetch manifest")
    ap.add_argument("--selection", type=Path, default=SELECTION_PATH)
    ap.add_argument("--from-cache", action="store_true",
                    help="NO network: --catalog re-derives outputs from catalog.json; --fetch scans files already on disk")
    ap.add_argument("--max-bytes", type=int, default=200_000_000, help="hard download ceiling")
    ap.add_argument("--package", action="append", default=[],
                    help="restrict --fetch to these package ids (must be in the selection)")
    args = ap.parse_args(argv)

    try:
        if args.verify:
            result = verify_download()
            print(json.dumps(result, indent=2))
            return 1 if result["problems"] else 0

        selection = load_selection(args.selection)
        # Validate the explicit --package allow-list before any HF client/token is created:
        # a refused request must never touch the network or read credentials.
        targets = None
        if args.fetch and args.package:
            targets = [(p["repo_id"], p["package_id"]) for p in selection["packages"]
                       if p["package_id"] in args.package]
            assert_selected(selection, targets)
            if not targets:
                raise UnselectedPackage("none of the requested packages are in the selection")

        hub = HubClient()

        if args.catalog:
            if args.from_cache:
                rows = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))["entries"]
                rows = enrich_catalog_with_selection(rows, selection)
            else:
                rows = build_catalog(hub, selection=selection)
            write_catalog(rows)
            print(json.dumps({"catalog_entries": len(rows),
                              "selected_entries": sum(1 for x in rows if x.get("selected")),
                              "from_cache": args.from_cache,
                              "per_repo": {r["repo_id"]: sum(1 for x in rows if x["repo"] == r["repo_id"])
                                           for r in REPOS if any(x["repo"] == r["repo_id"] for x in rows)},
                              "json": str(CATALOG_JSON.relative_to(ROOT)),
                              "csv": str(CATALOG_CSV.relative_to(ROOT))}, indent=2))
            return 0

        # --fetch
        manifest = (scan_cache(selection, packages=targets) if args.from_cache
                    else fetch_packages(hub, selection, targets, max_bytes=args.max_bytes))
        report = write_ingest_report(manifest, selection)
        write_attribution(manifest, selection)
        print(json.dumps({"method": report["method"],
                          "packages": report["totals"]["packages"],
                          "files": report["totals"]["files"],
                          "bytes": report["totals"]["bytes"],
                          "pending": report["totals"]["pending_packages"],
                          "coverage": report["coverage"],
                          "attribution": str(ATTRIBUTION_PATH.relative_to(ROOT)),
                          "report": str(INGEST_REPORT.relative_to(ROOT))}, indent=2))
        return 0
    except UnselectedPackage as exc:
        print(f"refusing to fetch: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # network/auth errors etc. - never include token in message
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
