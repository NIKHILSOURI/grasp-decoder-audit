"""Regression metrics: R^2 (main), Pearson r (for comparison with prior work), RMSE (target units)."""

from __future__ import annotations

import numpy as np


def r2(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
    ss_res = ((y - p) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def pearson(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
    yc, pc = y - y.mean(), p - p.mean()
    den = np.sqrt((yc ** 2).sum() * (pc ** 2).sum())
    return float((yc * pc).sum() / den) if den > 0 else float("nan")


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(((np.asarray(y, float) - np.asarray(p, float)) ** 2).mean()))


def all_metrics(y, p) -> dict:
    return {"r2": r2(y, p), "r": pearson(y, p), "rmse": rmse(y, p)}
