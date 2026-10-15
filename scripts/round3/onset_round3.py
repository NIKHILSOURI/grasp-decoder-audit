"""Round-3 onset analyses (A1, A3 onset events, B3 controls, B6 variability, C1 figure data).

Re-implements E4 (all channels) with the same folds, seeds, features, classifiers and validation-threshold rule, and
records EVERY test trial at EVERY threshold. Detector variants:
  lda, riemann                       E4 detectors (must reproduce Table III)
  lda_timematched, riemann_timematched  B3(a): negatives = windows at the same time after the cue as the positives,
                                     in trials whose hand start is > 300 ms later (fallback rule recorded per fold)
  cue_only                           B3(b): logistic regression on time since the last cue only (no EEG)
  lda_plus_cue                       B3(c): LDA features + time since the last cue
Cue timer (A1): fires at LED + 100 ms.

    python scripts/round3/onset_round3.py  ->  results/round3/onset_trials.csv.gz, onset_folds.csv, cue_timer_trials.csv
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from gaudit.data.cache import load_stream
from gaudit.experiments import fold_seed
from gaudit.onset import detector as odet
from gaudit.onset.evaluate import THRESHOLDS, eval_grid, pick_threshold, score
from gaudit.onset.labels import grid, intervals, training_windows
from gaudit.utils.seed import rng

warnings.filterwarnings("ignore")
S = list(range(1, 13))
OUT = "results/round3/"
DETECTORS = ["lda", "riemann", "lda_timematched", "riemann_timematched", "cue_only", "lda_plus_cue"]

streams = {s: load_stream("data/cache", s) for s in S}
ivs = {s: intervals(streams[s]) for s in S}
LEDS = {s: {int(sr): np.sort(ivs[s].led[ivs[s].series == sr]) for sr in np.unique(ivs[s].series)} for s in S}


def cue_age(s, ser, t):
    """Time since the most recent cue (LED-on) at or before t in the same series (causal)."""
    out = np.empty(len(t))
    for sr in np.unique(ser):
        m = ser == sr
        leds = LEDS[s][int(sr)]
        j = np.searchsorted(leds, t[m], side="right") - 1
        out[m] = np.where(j >= 0, t[m] - leds[np.clip(j, 0, None)], t[m])
    return out


def feats(s, ser, t, kind):
    if kind == "lda":
        return odet.features(streams[s], ser, t, "lda")
    if kind == "riemann":
        return odet.features(streams[s], ser, t, "riemann")
    if kind == "cue":
        return cue_age(s, ser, t)[:, None].astype(np.float32)
    if kind == "lda+cue":
        return np.hstack([odet.features(streams[s], ser, t, "lda"), cue_age(s, ser, t)[:, None]]).astype(np.float32)
    raise ValueError(kind)


def timematched(iv, rows, tau_rng, fallback):
    """Positives as usual; negatives at the positives' times after the cue but > 300 ms before hand start."""
    S_, t_, y_ = [], [], []
    for i in rows:
        g = grid(iv.pos[i, 0], iv.pos[i, 1], 0.05)
        S_.append(np.full(g.size, iv.series[i]))
        t_.append(g)
        y_.append(np.ones(g.size, int))
        if fallback:
            neg = grid(iv.led[i], iv.led[i] + 0.2, 0.05) if (iv.hs[i] - iv.led[i]) > 0.5 else np.empty(0)
        else:
            neg = grid(iv.led[i] + max(tau_rng[0], 0.0), iv.led[i] + tau_rng[1], 0.05)
            neg = neg[neg < iv.hs[i] - 0.3 - 1e-9]
        S_.append(np.full(neg.size, iv.series[i]))
        t_.append(neg)
        y_.append(np.zeros(neg.size, int))
    return np.concatenate(S_), np.concatenate(t_), np.concatenate(y_)


def labelled(s, rows, det, tm):
    iv = ivs[s]
    if det.endswith("timematched"):
        return timematched(iv, rows, tm["tau"], tm["fallback"])
    ser, t, y, _ = training_windows(iv, rows)
    return ser, t, y


def kind_of(det):
    return {"lda": "lda", "riemann": "riemann", "lda_timematched": "lda", "riemann_timematched": "riemann",
            "cue_only": "cue", "lda_plus_cue": "lda+cue"}[det]


class LogReg:
    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.m = LogisticRegression(class_weight="balanced", max_iter=2000).fit(self.sc.transform(X), y)
        return self

    def proba(self, X):
        return self.m.predict_proba(self.sc.transform(X))[:, 1]


def make_clf(det, fold):
    if det in ("lda", "lda_timematched", "lda_plus_cue"):
        return odet.OnsetLDA()
    if det in ("riemann", "riemann_timematched"):
        return odet.OnsetRiemann(seed=fold_seed(0, fold))
    return LogReg()


def per_trial(iv, trial_rows, t, p):
    """Per test trial and threshold: first in-window detection, strict/loose hit, false alarms in rest."""
    out = []
    for i in np.unique(trial_rows):
        m = trial_rows == i
        tt, pp = t[m], p[m]
        rest = iv.rest[i]
        rest_s = max(0.0, rest[1] - rest[0]) if np.isfinite(rest[0]) else 0.0
        for thr in THRESHOLDS:
            ev = tt[odet.detection_events(pp, thr)]
            fa = int(((ev >= rest[0]) & (ev <= rest[1])).sum()) if np.isfinite(rest[0]) else 0
            win = ev[(ev >= iv.led[i]) & (ev <= iv.hs[i] + 0.2)]
            fd = float(win[0]) if win.size else np.nan
            out.append((int(iv.trial_id[i]), float(thr), fd, int(win.size > 0 and fd < iv.hs[i]),
                        int(win.size > 0), fa, rest_s, float(iv.led[i]), float(iv.hs[i])))
    return out


def folds_for(scheme):
    fl = []
    if scheme == "trialwise":
        for s in S:
            for rep in range(3):
                idx = rng(0, 9, s, rep).permutation(np.arange(len(ivs[s].trial_id)))
                n = len(idx)
                ntr, nte = int(round(0.7 * n)), int(round(0.2 * n))
                fl.append((f"S{s}/r{rep}", {s: idx[:ntr]}, {s: idx[ntr + nte:]}, {s: idx[ntr:ntr + nte]}))
    else:
        for s in S:
            rest = [x for x in S if x != s]
            va = rng(0, 4, s).choice(rest, 2, replace=False)
            tr = [x for x in rest if x not in va]
            alli = {x: np.arange(len(ivs[x].trial_id)) for x in S}
            fl.append((f"S{s}", {x: alli[x] for x in tr}, {x: alli[x] for x in va}, {s: alli[s]}))
    return fl


EVAL_CACHE: dict = {}


def eval_feats(s, rows, kind):
    key = (s, kind, rows.tobytes())
    if key not in EVAL_CACHE:
        ser, t, r = eval_grid(ivs[s], rows)
        EVAL_CACHE[key] = (feats(s, ser, t, kind), t, r)
        if len(EVAL_CACHE) > 400:
            EVAL_CACHE.pop(next(iter(EVAL_CACHE)))
    return EVAL_CACHE[key]


def main():
    t0 = time.time()
    trials, folds_out = [], []
    for scheme in ("trialwise", "subjectwise"):
        fl = folds_for(scheme)
        for det in DETECTORS:
            kind = kind_of(det)
            for fold, TR, VA, TE in fl:
                ids = {k: set(np.concatenate([ivs[s].trial_id[r] for s, r in D.items()]))
                       for k, D in (("tr", TR), ("va", VA), ("te", TE))}
                assert not (ids["tr"] & ids["te"]) and not (ids["tr"] & ids["va"]) and not (ids["va"] & ids["te"])
                tm = {}
                if det.endswith("timematched"):
                    taus = np.concatenate([ivs[s].pos[r].ravel() - np.repeat(ivs[s].led[r], 2) for s, r in TR.items()])
                    tm = {"tau": (float(taus.min()), float(taus.max())), "fallback": False}
                    n_neg = sum(int((labelled(s, r, det, tm)[2] == 0).sum()) for s, r in TR.items())
                    if n_neg < 20:
                        tm["fallback"] = True
                Xy = [labelled(s, r, det, tm) for s, r in TR.items()]
                X = np.concatenate([feats(s, ser, t, kind) for (s, _), (ser, t, _) in zip(TR.items(), Xy)])
                y = np.concatenate([yy for _, _, yy in Xy])
                clf = make_clf(det, fold).fit(X, y)
                vrows = []
                for s, r in VA.items():
                    Xv, tv, rv = eval_feats(s, r, kind)
                    vr, _ = score(ivs[s], rv, tv, clf.proba(Xv))
                    vrows.append(pd.DataFrame(vr))
                thr = pick_threshold(pd.concat(vrows).groupby("threshold", as_index=False).mean().to_dict("records"), 1.0)
                for s, r in TE.items():
                    ser, t, yl = labelled(s, r, det, tm)
                    auc = float(roc_auc_score(yl, clf.proba(feats(s, ser, t, kind)))) if len(set(yl)) == 2 else np.nan
                    Xt, tt, rt = eval_feats(s, r, kind)
                    p = clf.proba(Xt)
                    res, _ = score(ivs[s], rt, tt, p)
                    op = next(x for x in res if x["threshold"] == thr)
                    folds_out.append({"detector": det, "scheme": scheme, "fold": fold, "subject": s, "auc": auc,
                                      "operating_threshold": thr, "det_rate_loose": op["det_rate"],
                                      "fa_per_min": op["fa_per_min"], "median_lead_ms": op["median_lead_ms"],
                                      "neg_rule": ("fallback: 0-200 ms after LED, RT > 500 ms" if tm.get("fallback")
                                                   else "time-matched" if tm else "rest + hold (E4)"),
                                      "n_train_pos": int((y == 1).sum()), "n_train_neg": int((y == 0).sum())})
                    for row in per_trial(ivs[s], rt, tt, p):
                        trials.append((det, scheme, fold, s, *row, thr))
            print(f"{scheme} {det} done ({time.time() - t0:.0f}s)", flush=True)
    cols = ["detector", "scheme", "fold", "subject", "trial_id", "threshold", "first_detection_time", "strict_pre",
            "loose_hit", "n_false_alarms", "rest_seconds", "led_on", "hand_start", "operating_threshold"]
    pd.DataFrame(trials, columns=cols).to_csv(OUT + "onset_trials.csv.gz", index=False)
    pd.DataFrame(folds_out).to_csv(OUT + "onset_folds.csv", index=False)
    # A1: cue timer on the same test trials
    cue = []
    for scheme in ("trialwise", "subjectwise"):
        for fold, _, _, TE in folds_for(scheme):
            for s, r in TE.items():
                iv = ivs[s]
                for i in r:
                    fire = iv.led[i] + 0.1
                    cue.append({"scheme": scheme, "fold": fold, "subject": s, "trial_id": int(iv.trial_id[i]),
                                "led_on": iv.led[i], "hand_start": iv.hs[i], "rt_s": iv.hs[i] - iv.led[i],
                                "strict_pre": int(iv.hs[i] > fire), "loose_hit": int(fire <= iv.hs[i] + 0.2),
                                "lead_ms": (fire - iv.hs[i]) * 1000})
    pd.DataFrame(cue).to_csv(OUT + "cue_timer_trials.csv", index=False)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
