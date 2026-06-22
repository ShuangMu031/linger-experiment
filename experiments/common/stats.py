"""Statistics for the benchmark - standard library + a little math (no scipy).

Wilson CI, two-proportion z-test, Holm-Bonferroni, seed-level paired bootstrap.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

# Four-system naming lives in experiments.common.naming (single source of truth).
# Import PAPER / to_paper from there; this module is statistics only.

Z_95 = 1.96


def wilson_ci(k: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(center - half, 3), round(center + half, 3))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return round(z, 2), math.erfc(abs(z) / math.sqrt(2))


def holm_bonferroni(pairs: List[Tuple[str, float]]) -> List[Tuple[str, float, float, bool]]:
    ordered = sorted(pairs, key=lambda x: x[1])
    m = len(ordered)
    out, running_max = [], 0.0
    for i, (label, p) in enumerate(ordered):
        p_adj = min(1.0, max(running_max, (m - i) * p))
        running_max = p_adj
        out.append((label, round(p, 4), round(p_adj, 4), p_adj < 0.05))
    return out


def mean_sd(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return (round(mean, 3), 0.0)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return (round(mean, 3), round(math.sqrt(var), 3))


def paired_bootstrap_diff(a: List[float], b: List[float], n_boot: int = 10000,
                          seed: int = 0) -> Tuple[float, float]:
    if not a or not b or len(a) != len(b):
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    diffs.sort()
    return (round(diffs[int(0.025 * n_boot)], 3), round(diffs[int(0.975 * n_boot)], 3))


if __name__ == "__main__":
    print("wilson_ci(73,100) =", wilson_ci(73, 100))
    print("mean_sd =", mean_sd([0.70, 0.72, 0.68, 0.74, 0.71]))
    print("bootstrap =", paired_bootstrap_diff([0.7, 0.72, 0.68, 0.74, 0.71], [0.1, 0.12, 0.08, 0.14, 0.11]))
    print("stats self-test done")
