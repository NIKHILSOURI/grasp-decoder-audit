"""Across-subject statistics: paired Wilcoxon signed-rank tests and bootstrap CIs."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(x, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    g = np.random.default_rng(seed)
    means = x[g.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_wilcoxon(a, b) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired per-subject scores (a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if len(d) < 3 or np.allclose(d, 0):
        return {"n": int(len(d)), "W": float("nan"), "p": float("nan"), "mean_diff": float(np.mean(d) if len(d) else np.nan)}
    res = stats.wilcoxon(a[ok], b[ok], zero_method="wilcox", alternative="two-sided")
    lo, hi = bootstrap_ci(d)
    return {"n": int(len(d)), "W": float(res.statistic), "p": float(res.pvalue), "mean_diff": float(d.mean()),
            "ci_lo": lo, "ci_hi": hi}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
