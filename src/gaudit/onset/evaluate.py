"""Online evaluation of a causal onset detector over held-out trials.

Per threshold: detection rate (an event between LED-on and HandStart + 200 ms), median
lead time of the first such event relative to HandStart (negative = before movement),
median latency from LED-on, and false alarms per minute of rest.
"""

from __future__ import annotations

import numpy as np

from .detector import detection_events
from .labels import TrialIntervals, grid

STEP = 0.02  # 20 ms online stride
THRESHOLDS = np.round(np.concatenate([np.linspace(0.05, 0.9, 18), np.linspace(0.91, 0.995, 18)]), 4)


def eval_grid(iv: TrialIntervals, rows: np.ndarray):
    """(series, time, trial_row) of every online step over the evaluation intervals."""
    S, t, r = [], [], []
    for i in rows:
        g = grid(iv.evalw[i, 0], iv.evalw[i, 1], STEP)
        S.append(np.full(g.size, iv.series[i]))
        t.append(g)
        r.append(np.full(g.size, i))
    return np.concatenate(S), np.concatenate(t), np.concatenate(r)


def score(iv: TrialIntervals, trial_rows: np.ndarray, t: np.ndarray, p: np.ndarray,
          thresholds=THRESHOLDS) -> tuple[list[dict], dict]:
    """Returns per-threshold summary rows and per-trial first-detection times per threshold."""
    by_trial = [(i, t[trial_rows == i], p[trial_rows == i]) for i in np.unique(trial_rows)]
    rest_s = sum(max(0.0, iv.rest[i, 1] - iv.rest[i, 0]) for i, _, _ in by_trial if np.isfinite(iv.rest[i, 0]))
    out, first = [], {}
    for thr in thresholds:
        n_fa, leads, lats, hit = 0, [], [], 0
        fd = np.full(len(by_trial), np.nan)
        for k, (i, tt, pp) in enumerate(by_trial):
            ev_t = tt[detection_events(pp, thr)]
            if np.isfinite(iv.rest[i, 0]):
                n_fa += int(((ev_t >= iv.rest[i, 0]) & (ev_t <= iv.rest[i, 1])).sum())
            win = ev_t[(ev_t >= iv.led[i]) & (ev_t <= iv.hs[i] + 0.2)]
            if win.size:
                hit += 1
                leads.append(win[0] - iv.hs[i])
                lats.append(win[0] - iv.led[i])
                fd[k] = win[0]
        out.append({"threshold": float(thr), "det_rate": hit / len(by_trial),
                    "median_lead_ms": 1000 * float(np.median(leads)) if leads else np.nan,
                    "median_latency_led_ms": 1000 * float(np.median(lats)) if lats else np.nan,
                    "fa_per_min": n_fa / (rest_s / 60) if rest_s > 0 else np.nan,
                    "n_trials": len(by_trial), "rest_min": rest_s / 60})
        first[float(thr)] = fd
    first["rows"] = np.array([i for i, _, _ in by_trial])
    return out, first


def pick_threshold(val_rows: list[dict], max_fa: float = 1.0) -> float:
    """Operating point chosen on validation data: lowest threshold with FA/min <= max_fa."""
    ok = [r["threshold"] for r in val_rows if np.isfinite(r["fa_per_min"]) and r["fa_per_min"] <= max_fa]
    return min(ok) if ok else max(r["threshold"] for r in val_rows)


def cue_locking(iv: TrialIntervals, rows: np.ndarray, t_det: np.ndarray) -> dict:
    """Slope of (detection - LED) on reaction time (HandStart - LED) across detected trials.
    ~1: detector tracks movement onset; ~0: detector tracks the visual cue."""
    ok = np.isfinite(t_det)
    if ok.sum() < 5:
        return {"slope": np.nan, "r": np.nan, "n": int(ok.sum())}
    rt = (iv.hs - iv.led)[rows][ok]
    lat = t_det[ok] - iv.led[rows][ok]
    slope = float(np.polyfit(rt, lat, 1)[0])
    return {"slope": slope, "r": float(np.corrcoef(rt, lat)[0, 1]), "n": int(ok.sum())}
