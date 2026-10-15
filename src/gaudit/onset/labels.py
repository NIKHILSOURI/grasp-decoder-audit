"""Grasp-onset labels and evaluation intervals on the continuous series clock (seconds).

For trial k of a series (events from AllLifts, re-referenced to the HS clock):
  rest      R_k = [HandStop_{k-1} + 0.5, LEDon_k - 0.5]   (hand at rest, no cue yet)
  positive  P_k = [HandStart_k - 0.3, HandStart_k]        ("grasp starts within 300 ms")
  hold      H_k = [LiftOff_k + 0.5, LEDoff_k]             (object held still)
  evaluate  E_k = [start of R_k (or LEDon_k - 0.5), HandStart_k + 0.2]
Training uses windows ending in P (label 1) or R/H (label 0); the rest are ignored.
For the first trial of a series, R starts 2 s into the stream (causal-filter warm-up).
Deviation from the build spec ("> 1000 ms before LED-on"): inter-trial rests are often
shorter than that, so we keep a 0.5 s margin on both sides instead (docs/DECISIONS.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WARMUP_S = 2.0


@dataclass
class TrialIntervals:
    trial_id: np.ndarray
    series: np.ndarray
    led: np.ndarray
    hs: np.ndarray
    rest: np.ndarray  # (T, 2), NaN when empty
    pos: np.ndarray  # (T, 2)
    hold: np.ndarray  # (T, 2)
    evalw: np.ndarray  # (T, 2)


def intervals(stream: dict) -> TrialIntervals:
    names = [str(x) for x in stream["event_names"]]
    ev = stream["events_abs"]
    e = {n: ev[:, names.index(n)] for n in names}
    series = stream["series"]
    T = len(series)
    prev_stop = np.full(T, np.nan)
    for i in range(T):
        prev_stop[i] = e["tHandStop"][i - 1] if i > 0 and series[i - 1] == series[i] else WARMUP_S - 0.5
    rest = np.stack([prev_stop + 0.5, e["LEDOn"] - 0.5], 1)
    rest[rest[:, 1] <= rest[:, 0]] = np.nan
    pos = np.stack([e["tHandStart"] - 0.3, e["tHandStart"]], 1)
    hold = np.stack([e["tLiftOff"] + 0.5, e["LEDOff"]], 1)
    hold[hold[:, 1] <= hold[:, 0]] = np.nan
    start = np.where(np.isnan(rest[:, 0]), e["LEDOn"] - 0.5, rest[:, 0])
    evalw = np.stack([start, e["tHandStart"] + 0.2], 1)
    return TrialIntervals(stream["trial_id"], series, e["LEDOn"], e["tHandStart"], rest, pos, hold, evalw)


def grid(lo: float, hi: float, step: float) -> np.ndarray:
    if not np.isfinite(lo) or hi < lo:
        return np.empty(0)
    return np.round(np.arange(lo, hi + 1e-9, step), 2)


def training_windows(iv: TrialIntervals, rows: np.ndarray, step: float = 0.05):
    """(series, end_time, label, trial_id) for training trials `rows`."""
    S, t, y, tid = [], [], [], []
    for i in rows:
        for arr, lab in ((iv.pos, 1), (iv.rest, 0), (iv.hold, 0)):
            g = grid(arr[i, 0], arr[i, 1], step)
            S.append(np.full(g.size, iv.series[i]))
            t.append(g)
            y.append(np.full(g.size, lab))
            tid.append(np.full(g.size, iv.trial_id[i]))
    return np.concatenate(S), np.concatenate(t), np.concatenate(y), np.concatenate(tid)
