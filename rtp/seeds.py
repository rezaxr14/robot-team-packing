"""Seed namespaces (RUNBOOK rule 8): training seeds >= 100000, held-out test seeds < 100000.

The split exists so that no evaluation seed can leak into training data, memory or prompts.
This module is the single place that defines the boundary; every generator, curator and
evaluation script should call :func:`assert_train` / :func:`assert_test` instead of comparing
literals. ``ops/audit_seeds.py`` scans an existing corpus (jsonl/json/npz manifests) and reports
violations without deleting anything.

Evidence class: FIXTURE (pure rule bookkeeping). No claim about a model is made here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

TRAIN_MIN = 100_000
TEST_MAX = TRAIN_MIN - 1

TRAIN = "train"
TEST = "test"


class SeedNamespaceError(ValueError):
    """A seed is in the wrong namespace for the artefact being written or read."""


def namespace(seed: int) -> str:
    """Return ``"train"`` for seeds >= 100000 and ``"test"`` for seeds < 100000."""
    return TRAIN if int(seed) >= TRAIN_MIN else TEST


def is_train(seed: int) -> bool:
    return namespace(seed) == TRAIN


def is_test(seed: int) -> bool:
    return namespace(seed) == TEST


def assert_train(seeds: Iterable[int], *, where: str = "dataset") -> None:
    """Raise if any seed belongs to the held-out test namespace."""
    bad = sorted({int(s) for s in seeds if not is_train(s)})
    if bad:
        raise SeedNamespaceError(
            f"{where}: test seeds {bad[:8]} (< {TRAIN_MIN}) may not be used for training, memory or prompts"
        )


def assert_test(seeds: Iterable[int], *, where: str = "evaluation") -> None:
    """Raise if any seed belongs to the training namespace."""
    bad = sorted({int(s) for s in seeds if not is_test(s)})
    if bad:
        raise SeedNamespaceError(
            f"{where}: training seeds {bad[:8]} (>= {TRAIN_MIN}) may not be used for held-out evaluation"
        )


def split_bounds(kind: str, n: int, start: int | None = None) -> tuple[int, int]:
    """Return ``(seed0, seed1)`` for ``n`` seeds of ``kind``, staying inside that namespace.

    ``start`` defaults to 100000 for training and 0 for test. Raises rather than crossing the
    boundary, so a generator can never silently spill into the other namespace.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if kind == TRAIN:
        seed0 = TRAIN_MIN if start is None else int(start)
        if seed0 < TRAIN_MIN:
            raise SeedNamespaceError(f"train split starts at {seed0} < {TRAIN_MIN}")
        return seed0, seed0 + n
    if kind == TEST:
        seed0 = 0 if start is None else int(start)
        seed1 = seed0 + n
        if seed1 > TRAIN_MIN:
            raise SeedNamespaceError(f"test split {seed0}..{seed1} crosses into training seeds")
        return seed0, seed1
    raise ValueError(f"kind must be 'train' or 'test', got {kind!r}")


def iter_seed_values(obj: Any, path: str = "") -> Iterator[tuple[str, int]]:
    """Yield ``(json_path, value)`` for every integer stored under a key named ``seed``."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            if k == "seed" and isinstance(v, int) and not isinstance(v, bool):
                yield here, v
            else:
                yield from iter_seed_values(v, here)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_seed_values(v, f"{path}[{i}]")


def _records_from(path: Path) -> Iterator[Any]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    elif suffix == ".json":
        try:
            yield json.loads(path.read_text())
        except json.JSONDecodeError:
            return


def audit_paths(paths: Iterable[Path]) -> list[dict]:
    """Report every seed found outside the training namespace in the given corpus files.

    Files are read only; nothing is modified or deleted. Each finding is
    ``{"file", "path", "seed", "namespace"}``.
    """
    findings: list[dict] = []
    for p in paths:
        p = Path(p)
        for i, rec in enumerate(_records_from(p)):
            for where, seed in iter_seed_values(rec):
                if not is_train(seed):
                    findings.append({"file": str(p), "record": i, "path": where,
                                     "seed": seed, "namespace": namespace(seed)})
    return findings
