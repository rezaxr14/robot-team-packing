"""Statistics: Wilson intervals for proportions, bootstrap CIs, paired bootstrap difference."""
from __future__ import annotations
import math
import random


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(xs, stat=lambda v: sum(v) / len(v), n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    xs = list(xs)
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = sorted(stat([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n_boot))
    return (vals[int(alpha / 2 * n_boot)], vals[int((1 - alpha / 2) * n_boot) - 1])


def paired_bootstrap_diff(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """CI for mean(a-b) over paired seeds. Returns (mean_diff, lo, hi)."""
    assert len(a) == len(b)
    d = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
    return (sum(d) / len(d), lo, hi)


def summarize_binary(flags):
    k, n = sum(bool(f) for f in flags), len(flags)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}
