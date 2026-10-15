"""Round-4 onset analyses: B1(c) Riemannian + cue age; B2 time-matched control bookkeeping, RT-matched and
class-balanced variants. Reuses the round-3 machinery (same folds, seeds, features, validation threshold rule).

    python scripts/round4/onset_round4.py  ->  results/round4/onset4_trials.csv.gz, onset4_folds.csv,
                                                b2_counts.csv, b2_rt.csv
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round3"))
import onset_round3 as o3  # noqa: E402

from gaudit.experiments import fold_seed  # noqa: E402
from gaudit.features.covariance import TangentSpace  # noqa: E402
from gaudit.onset import detector as odet  # noqa: E402
from gaudit.onset.evaluate import eval_grid, pick_threshold, score  # noqa: E402
from gaudit.utils.seed import rng  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path("results/round4")
OUT.mkdir(parents=True, exist_ok=True)
ivs = o3.ivs


def tm_rows(s, rows, tau):
    """Time-matched labelled windows with the contributing trial index (positives and negatives)."""
    iv = ivs[s]
    out = []
    for i in rows:
        g = o3.grid(iv.pos[i, 0], iv.pos[i, 1], 0.05)
        out += [(iv.series[i], t, 1, i) for t in g]
        neg = o3.grid(iv.led[i] + max(tau[0], 0.0), iv.led[i] + tau[1], 0.05)
        neg = neg[neg < iv.hs[i] - 0.3 - 1e-9]
        out += [(iv.series[i], t, 0, i) for t in neg]
    a = np.array(out, dtype=float).reshape(-1, 4)
    return a[:, 0].astype(int), a[:, 1], a[:, 2].astype(int), a[:, 3].astype(int)


def rt_matched(s, rows, tau, seed):
    """Positives and negatives from the SAME trials (those with >= 1 time-matched negative), equal numbers per trial
    and class (random subsampling), so both classes have identical reaction-time distributions."""
    ser, t, y, tr = tm_rows(s, rows, tau)
    g = rng(seed, 77, s)
    keep = np.zeros(len(y), bool)
    for i in np.unique(tr[y == 0]):
        p, n = np.flatnonzero((tr == i) & (y == 1)), np.flatnonzero((tr == i) & (y == 0))
        k = min(len(p), len(n))
        keep[g.choice(p, k, replace=False)] = True
        keep[g.choice(n, k, replace=False)] = True
    return ser[keep], t[keep], y[keep]


def balanced(s, rows, tau, seed):
    """Time-matched windows with the majority class randomly subsampled to the minority count."""
    ser, t, y, _ = tm_rows(s, rows, tau)
    g = rng(seed, 78, s)
    p, n = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = min(len(p), len(n))
    keep = np.sort(np.concatenate([g.choice(p, k, replace=False), g.choice(n, k, replace=False)]))
    return ser[keep], t[keep], y[keep]


class RiemannCue:
    """Tangent space of OAS covariances (reference fit on training windows) + time since cue -> standardize ->
    logistic regression (C = 0.1, class-balanced), i.e. the E4 Riemannian detector with one extra feature."""

    def __init__(self, seed):
        self.seed = seed

    def _z(self, cov, cue):
        return np.hstack([self.ts.transform(cov), cue[:, None]])

    def fit(self, cov, cue, y):
        self.ts = TangentSpace(seed=self.seed).fit(cov)
        Z = self._z(cov, cue)
        self.sc = StandardScaler().fit(Z)
        self.m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(self.sc.transform(Z), y)
        return self

    def proba(self, cov, cue):
        return self.m.predict_proba(self.sc.transform(self._z(cov, cue)))[:, 1]


def tau_range(TR):
    taus = np.concatenate([ivs[s].pos[r].ravel() - np.repeat(ivs[s].led[r], 2) for s, r in TR.items()])
    return float(taus.min()), float(taus.max())


def main():
    t0 = time.time()
    trials, folds_out, counts, rts = [], [], [], []
    for scheme in ("trialwise", "subjectwise"):
        fl = o3.folds_for(scheme)
        # ---------------- B2 (a)(b): bookkeeping of the time-matched training sets (no training needed)
        for fold, TR, _VA, TE in fl:
            tau = tau_range(TR)
            for s, r in TR.items():
                _, _, y, tr = tm_rows(s, r, tau)
                rt = ivs[s].hs - ivs[s].led
                pos_tr, neg_tr = np.unique(tr[y == 1]), np.unique(tr[y == 0])
                counts.append({"scheme": scheme, "fold": fold, "subject": s, "role": "train", "n_pos": int((y == 1).sum()),
                               "n_neg": int((y == 0).sum()), "n_trials": len(r), "n_trials_with_neg": len(neg_tr)})
                rts.append({"scheme": scheme, "fold": fold, "subject": s,
                            "rt_pos_median": np.median(rt[pos_tr]), "rt_pos_q25": np.percentile(rt[pos_tr], 25),
                            "rt_pos_q75": np.percentile(rt[pos_tr], 75),
                            "rt_neg_median": np.median(rt[neg_tr]) if len(neg_tr) else np.nan,
                            "rt_neg_q25": np.percentile(rt[neg_tr], 25) if len(neg_tr) else np.nan,
                            "rt_neg_q75": np.percentile(rt[neg_tr], 75) if len(neg_tr) else np.nan,
                            "tau_min": tau[0], "tau_max": tau[1]})
            for s, r in TE.items():
                _, _, y, _ = tm_rows(s, r, tau)
                _, te_t, _ = eval_grid(ivs[s], r)
                counts.append({"scheme": scheme, "fold": fold, "subject": s, "role": "test", "n_pos": int((y == 1).sum()),
                               "n_neg": int((y == 0).sum()), "n_trials": len(r), "n_eval_windows": len(te_t)})
        # ---------------- B1(c), B2(c)(d): retrain and evaluate online
        for det in ("riemann_plus_cue", "lda_tm_rtmatched", "lda_tm_balanced"):
            for fold, TR, VA, TE in fl:
                tau = tau_range(TR)
                fs = fold_seed(0, fold)
                if det == "riemann_plus_cue":
                    parts = []
                    for s, r in TR.items():
                        ser, t, y, _ = o3.training_windows(ivs[s], r)
                        parts.append((o3.feats(s, ser, t, "riemann"), o3.cue_age(s, ser, t), y))
                    clf = RiemannCue(fs).fit(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                                             np.concatenate([p[2] for p in parts]))

                    def prob(s, ser, t, cov=None):
                        cov = o3.feats(s, ser, t, "riemann") if cov is None else cov
                        return clf.proba(cov, o3.cue_age(s, ser, t))  # noqa: B023 (used within this iteration)
                else:
                    build = rt_matched if det == "lda_tm_rtmatched" else balanced
                    parts = [build(s, r, tau, fs) for s, r in TR.items()]
                    X = np.concatenate([o3.feats(s, ser, t, "lda") for (s, _), (ser, t, _) in zip(TR.items(), parts)])
                    y = np.concatenate([p[2] for p in parts])
                    clf = odet.OnsetLDA().fit(X, y)

                    def prob(s, ser, t, cov=None):
                        return clf.proba(o3.feats(s, ser, t, "lda"))  # noqa: B023
                vrows = []
                for s, r in VA.items():
                    ser, tv, rv = eval_grid(ivs[s], r)
                    vr, _ = score(ivs[s], rv, tv, prob(s, ser, tv))
                    vrows.append(pd.DataFrame(vr))
                thr = pick_threshold(pd.concat(vrows).groupby("threshold", as_index=False).mean().to_dict("records"), 1.0)
                for s, r in TE.items():
                    if det == "riemann_plus_cue":
                        ser, t, yl, _ = o3.training_windows(ivs[s], r)
                    else:
                        ser, t, yl, _ = tm_rows(s, r, tau)
                    auc = float(roc_auc_score(yl, prob(s, ser, t))) if len(set(yl)) == 2 else np.nan
                    ser_e, tt, rt_ = eval_grid(ivs[s], r)
                    p = prob(s, ser_e, tt)
                    res, _ = score(ivs[s], rt_, tt, p)
                    op = next(x for x in res if x["threshold"] == thr)
                    folds_out.append({"detector": det, "scheme": scheme, "fold": fold, "subject": s, "auc": auc,
                                      "operating_threshold": thr, "det_rate_loose": op["det_rate"],
                                      "fa_per_min": op["fa_per_min"], "n_train": int(len(y)) if det != "riemann_plus_cue" else np.nan,
                                      "n_train_pos": int((y == 1).sum()) if det != "riemann_plus_cue" else np.nan})
                    for row in o3.per_trial(ivs[s], rt_, tt, p):
                        trials.append((det, scheme, fold, s, *row, thr))
            print(f"{scheme} {det} done ({time.time() - t0:.0f}s)", flush=True)
    cols = ["detector", "scheme", "fold", "subject", "trial_id", "threshold", "first_detection_time", "strict_pre",
            "loose_hit", "n_false_alarms", "rest_seconds", "led_on", "hand_start", "operating_threshold"]
    pd.DataFrame(trials, columns=cols).to_csv(OUT / "onset4_trials.csv.gz", index=False)
    pd.DataFrame(folds_out).to_csv(OUT / "onset4_folds.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "b2_counts.csv", index=False)
    pd.DataFrame(rts).to_csv(OUT / "b2_rt.csv", index=False)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
