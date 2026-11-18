"""Per-profile asset cache: attach → local cache → download, with a manifest checked on every load.

Phase 1 of the plan (``IMPLEMENTATION_PLAN.md`` §11). One API used by every notebook and by
``ops/install.sh``:

    from rtp import assetcache
    from rtp.profiles import get_profile

    prof = get_profile()                                    # ACTIVE / $RTP_PROFILE
    spec = assetcache.model_spec("Qwen/Qwen3.5-4B", revision=REV, name="qwen35-4b")
    d = assetcache.resolve(spec, prof, attached="/kaggle/input", fetch=download_fn)

Resolution order, as the plan requires: **attached Kaggle dataset → local cache → download →
(optional) private-dataset export**. Whatever the source, the returned path is inside the profile
tree; a read-only attach is materialised into the cache first, so later stages never read from
``/kaggle/input`` and two profiles never share a byte.

The manifest (``manifest.json``) records source, revision, byte sizes, sha256 per file and the
runtime tags (python/torch/cuda/sm) that make a wheelhouse valid. Loads check sizes always, and
sha256 when the manifest says the tree was not deep-verified or when ``verify="sha256"`` is asked
for; a mismatch is refused rather than silently used. Nothing here talks to Kaggle: the exporter is
a callable you supply, so G-11 can be enabled the moment CLI auth exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .profiles import Profile

MANIFEST = "manifest.json"
SCHEMA = 1
CHUNK = 4 * 1024 * 1024
_SLUG_BAD = re.compile(r"[^a-z0-9._-]+")

#: ``resolve(verify=...)`` modes. ``size`` is the cheap every-load check; ``sha256`` is the deep one.
VERIFY_MODES = ("size", "sha256")


class AssetError(RuntimeError):
    """Base class for cache failures that should stop a stage."""


class PathEscape(AssetError):
    """A path would leave the profile tree."""


class ManifestError(AssetError):
    """The cached bytes do not match the manifest."""

    def __init__(self, message: str, problems: Sequence[str] = ()):
        super().__init__(message)
        self.problems = list(problems)


class AssetUnavailable(AssetError):
    """Neither an attach, nor the cache, nor a downloader can produce the asset."""


class ExportNotConfigured(AssetError):
    """Private-dataset export needs Kaggle CLI auth (gate G-11); non-blocking for every other phase."""


# --------------------------------------------------------------------------------------------------
# specs and slugs
# --------------------------------------------------------------------------------------------------
def slugify(text: str) -> str:
    """Kaggle dataset ids are lowercase alnum with ``.``/``_``/``-``; anything else becomes ``-``."""
    return _SLUG_BAD.sub("-", text.strip().lower()).strip("-") or "asset"


@dataclass(frozen=True)
class FileSpec:
    path: str                  # relative to the asset directory
    size: int | None = None    # bytes, when known
    sha256: str | None = None  # hex, when known


@dataclass(frozen=True)
class AssetSpec:
    """One cached thing under one slug. ``kind`` decides where it lives and how it is named."""

    kind: str                  # model | wheelhouse | adapters | data
    slug: str
    source: str = ""           # HF repo id, URL, or a note for locally built assets
    revision: str = ""         # commit sha / version that pins the bytes
    files: tuple[FileSpec, ...] = ()
    tags: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def rev8(self) -> str:
        return self.revision[:8] if self.revision else "local"


def model_spec(source: str, *, revision: str, name: str, files: Iterable[FileSpec] = (),
               notes: str = "") -> AssetSpec:
    """``rtp-model-<name>-<rev8>`` (plan §11)."""
    return AssetSpec(kind="model", slug=f"rtp-model-{slugify(name)}-{revision[:8] or 'local'}",
                     source=source, revision=revision, files=tuple(files), notes=notes)


def wheelhouse_spec(*, python: str, torch: str, cuda: str, sm: str,
                    files: Iterable[FileSpec] = ()) -> AssetSpec:
    """``rtp-wheels-py<v>-torch<v>-cu<v>-sm<cc>`` (plan §11)."""
    return AssetSpec(
        kind="wheelhouse",
        slug=f"rtp-wheels-py{slugify(python)}-torch{slugify(torch)}-cu{slugify(cuda)}-sm{slugify(sm)}",
        source="locally built", revision=f"{python}/{torch}/{cuda}/{sm}",
        files=tuple(files), tags={"python": python, "torch": torch, "cuda": cuda, "sm": sm},
    )


def adapters_spec(profile: Profile, stage: str, config_hash: str) -> AssetSpec:
    """``rtp-adapters-<profile>-<stage>-<hash8>`` (plan §11)."""
    return AssetSpec(kind="adapters", slug=f"rtp-adapters-{slugify(profile.name)}-{slugify(stage)}-{config_hash[:8]}",
                     source=f"produced by {profile.name}:{stage}", revision=config_hash,
                     tags={"profile": profile.name, "stage": stage})


def data_spec(profile: Profile, kind: str, content_hash: str) -> AssetSpec:
    """``rtp-data-<profile>-<kind>-<hash8>`` (plan §11)."""
    return AssetSpec(kind="data", slug=f"rtp-data-{slugify(profile.name)}-{slugify(kind)}-{content_hash[:8]}",
                     source=f"produced by {profile.name}:{kind}", revision=content_hash,
                     tags={"profile": profile.name, "kind": kind})


# --------------------------------------------------------------------------------------------------
# where things live, and the path-escape guard
# --------------------------------------------------------------------------------------------------
def cache_root(profile: Profile) -> Path:
    """Everything cached for one profile, inside ``models/<profile>/`` (never a shared cache)."""
    return profile.paths.models / "cache"


def guard_path(path: Path | str, root: Path | str) -> Path:
    """Return ``path`` resolved, or raise :class:`PathEscape` if it is not under ``root``.

    ``Path.resolve()`` follows symlinks, so a link that points outside the tree is caught here
    rather than after a multi-GB write.
    """
    root_r = Path(root).expanduser().resolve()
    p = Path(path).expanduser().resolve()
    if p != root_r and root_r not in p.parents:
        raise PathEscape(f"{p} is outside {root_r}")
    return p


def asset_dir(spec: AssetSpec, profile: Profile) -> Path:
    """``<profile>/models/cache/<kind>/<slug>``, guarded against escapes from the slug itself."""
    return guard_path(cache_root(profile) / spec.kind / spec.slug, profile.paths.models)


# --------------------------------------------------------------------------------------------------
# hashing and manifests
# --------------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    """Every regular file under ``root``, sorted, with the manifest itself excluded."""
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.name != MANIFEST:
            yield p


def env_tags() -> dict:
    """Runtime facts a wheelhouse must match. Missing libraries are recorded as ``?``, not guessed."""
    tags = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    try:
        import torch
        tags["torch"] = torch.__version__.split("+")[0]
        if torch.cuda.is_available():
            tags["cuda"] = str(torch.version.cuda or "?")
            cc = torch.cuda.get_device_capability(0)
            tags["sm"] = f"{cc[0]}{cc[1]}"
    except Exception:                                        # noqa: BLE001
        tags.setdefault("torch", "?")
    return tags


def build_manifest(root: Path, spec: AssetSpec, *, deep: bool = True) -> dict:
    """Describe the bytes under ``root``; ``deep`` hashes every file (needed for a first write)."""
    files = []
    total = 0
    for p in iter_files(root):
        size = p.stat().st_size
        total += size
        files.append({"path": p.relative_to(root).as_posix(), "size": size,
                      "sha256": sha256_file(p) if deep else None})
    return {
        "schema": SCHEMA, "slug": spec.slug, "kind": spec.kind, "source": spec.source,
        "revision": spec.revision, "tags": dict(spec.tags), "notes": spec.notes,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deep_verified": bool(deep), "total_bytes": total, "files": files,
    }


def write_manifest(root: Path, spec: AssetSpec, *, deep: bool = True) -> Path:
    manifest = build_manifest(root, spec, deep=deep)
    path = Path(root) / MANIFEST
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def read_manifest(root: Path) -> dict | None:
    path = Path(root) / MANIFEST
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def verify_manifest(root: Path, spec: AssetSpec | None = None, *, verify: str = "size") -> list[str]:
    """Problems with ``root`` as a cache of ``spec``; an empty list means it may be used.

    Always checks identity (slug/kind/source/revision), every recorded file's existence and size,
    and anything ``spec.files`` declares. ``verify="sha256"`` also re-hashes every file.
    """
    if verify not in VERIFY_MODES:
        raise ValueError(f"verify must be one of {VERIFY_MODES}, not {verify!r}")
    root = Path(root)
    manifest = read_manifest(root)
    if manifest is None:
        return [f"{root}: missing or unreadable {MANIFEST}"]
    problems = []

    if spec is not None:
        for key in ("slug", "kind"):
            if manifest.get(key) != getattr(spec, key):
                problems.append(f"{key}: manifest has {manifest.get(key)!r}, expected {getattr(spec, key)!r}")
        for key in ("source", "revision"):
            expected = getattr(spec, key)
            if expected and manifest.get(key) != expected:
                problems.append(f"{key}: manifest has {manifest.get(key)!r}, expected {expected!r}")
        for k, v in spec.tags.items():
            got = (manifest.get("tags") or {}).get(k)
            if got != v:
                problems.append(f"tag {k}: manifest has {got!r}, expected {v!r}")

    recorded = {f.get("path"): f for f in manifest.get("files", [])}
    for rel, rec in recorded.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing file {rel}")
            continue
        if rec.get("size") is not None and path.stat().st_size != rec["size"]:
            problems.append(f"size mismatch {rel}: {path.stat().st_size} != {rec['size']}")
            continue
        if verify == "sha256" and rec.get("sha256") and sha256_file(path) != rec["sha256"]:
            problems.append(f"sha256 mismatch {rel}")

    for f in (spec.files if spec else ()):
        rec = recorded.get(f.path)
        if rec is None:
            problems.append(f"expected file {f.path} is not in the manifest")
            continue
        if f.size is not None and rec.get("size") != f.size:
            problems.append(f"size mismatch {f.path}: manifest {rec.get('size')} != spec {f.size}")

    return problems


# --------------------------------------------------------------------------------------------------
# materialising a read-only attach, and fetching
# --------------------------------------------------------------------------------------------------
def _discard(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _copy_tree(src: Path, dest: Path) -> None:
    """Hardlink when possible (same FS, read-only attach) and copy otherwise; never move."""
    dest.mkdir(parents=True, exist_ok=True)
    for p in iter_files(src):
        rel = p.relative_to(src)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(p, out)
        except OSError:
            shutil.copy2(p, out)
    manifest = Path(src) / MANIFEST
    if manifest.exists():
        shutil.copy2(manifest, dest / MANIFEST)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stale_partials(dest: Path) -> list[Path]:
    """``<dest>.partial-<pid>`` dirs left by a dead process (its own pid is never removed)."""
    out = []
    for p in dest.parent.glob(f"{dest.name}.partial-*"):
        try:
            pid = int(p.name.rsplit("-", 1)[-1])
        except ValueError:
            out.append(p)
            continue
        if pid != os.getpid() and not _pid_alive(pid):
            out.append(p)
    return out


def _install_atomic(dest: Path, producer: Callable[[Path], None]) -> None:
    """Run ``producer`` in a sibling dir, verify it, then rename over ``dest``.

    A killed run leaves ``<dest>.partial-<pid>`` behind; the next call removes partials whose pid
    is gone, so resume is simply "call resolve again" (the plan's kill/resume criterion).
    """
    dest = Path(dest)
    for stale in _stale_partials(dest):
        _discard(stale)
    partial = dest.with_name(f"{dest.name}.partial-{os.getpid()}")
    _discard(partial)
    partial.mkdir(parents=True, exist_ok=True)
    try:
        producer(partial)
    except BaseException:
        _discard(partial)                    # a failed fetch never leaves a half cache that looks valid
        raise
    if dest.exists():
        _discard(dest)
    partial.replace(dest)


# --------------------------------------------------------------------------------------------------
# resolve
# --------------------------------------------------------------------------------------------------
def _candidates(attached: Path | str | Sequence[Path | str] | None) -> list[Path]:
    if attached is None:
        return []
    items = [attached] if isinstance(attached, (str, Path)) else list(attached)
    out = []
    for item in items:
        item = Path(item)
        if item.is_dir():
            out.append(item)
            # /kaggle/input/<dataset-slug>/ is the usual layout; also accept the parent dir.
            out.extend(sorted(p for p in item.iterdir() if p.is_dir() and not p.name.startswith(".")))
    return out


def resolve(spec: AssetSpec, profile: Profile, *, attached=None,
            fetch: Callable[[Path], None] | None = None, verify: str = "size",
            export: Callable[[Path, AssetSpec], object] | None = None,
            dry_run: bool = False, log: Callable[[str], None] | None = print) -> Path:
    """Return a verified cache directory for ``spec`` inside ``profile``'s tree.

    ``fetch(dest)`` must fill ``dest`` from the network and write a manifest (use
    :func:`write_manifest`); it only runs when neither an attach nor the cache has the asset.
    ``attached`` is a path, a list of paths, or a Kaggle ``/kaggle/input`` root.
    """
    say = log or (lambda _msg: None)
    dest = asset_dir(spec, profile)

    if dry_run:
        state = "hit" if not verify_manifest(dest, spec, verify=verify) else "miss"
        say(f"[dry-run] {spec.slug}: cache {state} at {dest}")
        return dest

    problems = verify_manifest(dest, spec, verify=verify)
    if not problems:
        say(f"{spec.slug}: cache hit ({dest})")
        return dest
    if dest.exists():
        say(f"{spec.slug}: cache invalid ({problems[0]}); rebuilding")

    for cand in _candidates(attached):
        cand_problems = verify_manifest(cand, spec, verify=verify)
        if cand_problems:
            continue
        say(f"{spec.slug}: materialising from attach {cand}")
        _install_atomic(dest, lambda tmp, src=cand: _copy_tree(src, tmp))
        _refuse(dest, spec, verify)
        _maybe_export(dest, spec, export, say)
        return dest

    if fetch is None:
        raise AssetUnavailable(
            f"{spec.slug}: not in the cache, no matching attach, and no fetch() supplied "
            f"(cache dir {dest}; attach a Kaggle dataset or pass fetch=...)")

    say(f"{spec.slug}: fetching into {dest}")
    _install_atomic(dest, lambda tmp: _fetch_and_manifest(tmp, spec, fetch))
    _refuse(dest, spec, verify)
    _maybe_export(dest, spec, export, say)
    return dest


def _fetch_and_manifest(tmp: Path, spec: AssetSpec, fetch: Callable[[Path], None]) -> None:
    fetch(tmp)
    if read_manifest(tmp) is None:
        write_manifest(tmp, spec)


def _refuse(dest: Path, spec: AssetSpec, verify: str) -> None:
    problems = verify_manifest(dest, spec, verify=verify)
    if problems:
        _discard(dest)                      # never leave bytes that look cached but are not
        raise ManifestError(f"{spec.slug}: cache verification failed", problems)


def _maybe_export(dest: Path, spec: AssetSpec, export, say) -> None:
    if export is None:
        return
    try:
        export(dest, spec)
    except ExportNotConfigured as exc:
        say(f"{spec.slug}: not exported ({exc})")
    else:
        say(f"{spec.slug}: exported; attach {spec.slug} next time")


def seed_from_dir(src: Path | str, spec: AssetSpec, profile: Profile, *,
                  verify: str = "sha256") -> Path:
    """Cache an already-downloaded directory (hardlink/copy, no network), then verify it.

    Used for a local, hash-checked download (``ops/download_model.sh``) so the notebooks never
    re-fetch 9 GB: the bytes are linked into the profile cache and a manifest is written from them.
    """
    src = Path(src)
    if not src.is_dir():
        raise AssetUnavailable(f"seed source {src} is not a directory")
    dest = asset_dir(spec, profile)
    if not verify_manifest(dest, spec, verify=verify):
        return dest

    def producer(tmp: Path) -> None:
        _copy_tree(src, tmp)
        write_manifest(tmp, spec)

    _install_atomic(dest, producer)
    _refuse(dest, spec, verify)
    return dest


def attach_hint(spec: AssetSpec, owner: str | None = None) -> str:
    """The one-line instruction a notebook prints so the next run downloads nothing."""
    name = f"{owner}/{spec.slug}" if owner else spec.slug
    return f"attach Kaggle dataset {name} next time"


# --------------------------------------------------------------------------------------------------
# private-dataset export (interface now, upload when CLI auth exists: G-11)
# --------------------------------------------------------------------------------------------------
def kaggle_owner(env: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env
    return env.get("KAGGLE_USERNAME") or None


def kaggle_metadata(spec: AssetSpec, owner: str, *, title: str | None = None) -> dict:
    """``dataset-metadata.json`` for a **private** dataset holding one asset (plan §11)."""
    return {"title": title or spec.slug, "id": f"{owner}/{spec.slug}",
            "licenses": [{"name": "other"}], "isPrivate": True}


def write_export_metadata(root: Path, spec: AssetSpec, owner: str) -> Path:
    path = guard_path(Path(root) / "dataset-metadata.json", root)
    path.write_text(json.dumps(kaggle_metadata(spec, owner), indent=2) + "\n")
    return path


def export_private(root: Path, spec: AssetSpec, *, upload: Callable[[Path], object] | None = None,
                   owner: str | None = None) -> Path:
    """Write the private-dataset metadata and (optionally) upload ``root``.

    Without ``upload`` or ``$KAGGLE_USERNAME`` this raises :class:`ExportNotConfigured`: the plan
    makes export/attach gate G-11, which waits for CLI auth and is non-blocking for every phase.
    """
    owner = owner or kaggle_owner()
    if not owner:
        raise ExportNotConfigured("set $KAGGLE_USERNAME (Kaggle CLI auth) to export private datasets (G-11)")
    meta = write_export_metadata(root, spec, owner)
    if upload is None:
        raise ExportNotConfigured(f"metadata written to {meta}; pass upload=... to publish {owner}/{spec.slug} (G-11)")
    upload(Path(root))
    return meta


# --------------------------------------------------------------------------------------------------
# scanning a local directory into a spec (used to seed the cache from a verified download)
# --------------------------------------------------------------------------------------------------
def spec_from_dir(root: Path, *, kind: str, name: str, revision: str = "", source: str = "") -> AssetSpec:
    """A spec whose files+sizes+sha256 describe ``root`` exactly (hashes every file once)."""
    root = Path(root)
    files = tuple(FileSpec(path=p.relative_to(root).as_posix(), size=p.stat().st_size, sha256=sha256_file(p))
                  for p in iter_files(root))
    slug = (f"rtp-model-{slugify(name)}-{revision[:8] or 'local'}" if kind == "model"
            else f"rtp-{slugify(kind)}-{slugify(name)}-{(revision or 'local')[:8]}")
    return AssetSpec(kind=kind, slug=slug, source=source, revision=revision, files=files)


if __name__ == "__main__":                   # python -m rtp.assetcache <spec-name>
    from .profiles import get_profile

    prof = get_profile()
    print(json.dumps({"profile": prof.name, "cache_root": str(cache_root(prof)),
                      "env_tags": env_tags()}, indent=2))
