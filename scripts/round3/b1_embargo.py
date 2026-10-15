"""B1: force blocked CV with embargoes 0.49/0.5/1.0/2.2/3.3 s and a train-size-matched control.

Same blocked splits as E6 (seed 0, repeat 0). Embargo e drops every training AND validation window whose end lies
within e seconds (strictly less) of a test-window end in the same trial (trials are disjoint segments and filters are
applied per trial, so windows of different trials never share raw samples). Matched control: the UNPURGED training
set randomly subsampled (seed 0) to the size of the 0.5-s purged training set; validation unpurged, as in E6.

    python scripts/round3/b1_embargo.py      ->  results/round3/b1_embargo.csv, b1_embargo_summary.csv
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from gaudit.data.splits import blocked, split_masks
from gaudit.data.targets import TargetScaler
from gaudit.eval.metrics import r2
from gaudit.eval.stats import bootstrap_ci
from gaudit.experiments import fold_seed
from gaudit.pipeline import Data, Pool, build_model, model_spec
from gaudit.utils.seed import rng, seed_everything

warnings.filterwarnings("ignore")
S = list(range(1, 13))
BLOCKS = [0.25, 1.0, 4.0]
EMBARGOES = [0.49, 0.5, 1.0, 2.2, 3.3]
MODELS = ["mlp", "ridge", "riemann"]
MLP_KW = {"max_epochs": 200, "patience": 10, "batch": 512}
MIN_TRAIN = 300  # below this a fit is reported as not possible
OUT = "results/round3/"


def nearest(sorted_e, e):
    j = np.searchsorted(sorted_e, e)
    lo = np.abs(e - sorted_e[np.clip(j - 1, 0, sorted_e.size - 1)])
    hi = np.abs(sorted_e[np.clip(j, 0, sorted_e.size - 1)] - e)
    return np.minimum(lo, hi)


def purge(w, mask, te, emb_samples):
    keep = mask.copy()
    for tid in np.unique(w.trial_id[te]):
        te_e = np.sort(w.end[te & (w.trial_id == tid)])
        ii = np.flatnonzero(keep & (w.trial_id == tid))
        if ii.size:
            keep[ii[nearest(te_e, w.end[ii]) < emb_samples]] = False
    return keep


def shared(w, te, other):
    n = 0
    for tid in np.unique(w.trial_id[te]):
        o = np.sort(w.end[other & (w.trial_id == tid)])
        if o.size:
            n += int((nearest(o, w.end[te & (w.trial_id == tid)]) <= 49).sum())
    return n


def fit_eval(pool, tr, va, te, model, seed):
    X = pool.feature(model_spec(model).feature)
    y = pool.y(["force"])
    ts = TargetScaler().fit(y[tr])
    m = build_model(model, seed, MLP_KW if model == "mlp" else None)
    m.fit(X[tr], ts.transform(y[tr]), X[va], ts.transform(y[va]))
    p = ts.inverse(m.predict(X[te]).reshape(int(te.sum()), -1))
    return r2(y[te][:, 0], p[:, 0])


def main():
    t0 = time.time()
    data = Data("data/cache", S)
    pool = Pool(data, S, "force", 50, 5)
    w = pool.w
    for m in MODELS:
        pool.feature(model_spec(m).feature)
    rows = []
    for b in BLOCKS:
        for s in S:
            sp = blocked(w, s, 0, b, 0)
            tr, va, te = split_masks(w, sp)
            variants = []
            n05 = None
            for e in EMBARGOES:
                ktr, kva = purge(w, tr, te, int(round(e * 100))), purge(w, va, te, int(round(e * 100)))
                if e == 0.5:
                    n05 = int(ktr.sum())
                variants.append((f"embargo_{e:g}", e, ktr, kva))
            sub = np.flatnonzero(tr)
            pick = rng(0, 31, s, int(b * 100)).choice(sub, size=n05, replace=False)
            mtr = np.zeros_like(tr)
            mtr[pick] = True
            variants.append(("matched_to_0.5", np.nan, mtr, va))
            for name, e, ktr, kva in variants:
                sh = shared(w, te, ktr | kva)
                for model in MODELS:
                    seed_everything(fold_seed(0, "force", model, name, sp.fold))
                    ok = ktr.sum() >= MIN_TRAIN and kva.sum() >= 50
                    score = fit_eval(pool, ktr, kva, te, model, fold_seed(0, model, sp.fold)) if ok else np.nan
                    rows.append({"block_s": b, "variant": name, "embargo_s": e, "subject": s, "model": model,
                                 "r2": score, "n_train": int(ktr.sum()), "n_val": int(kva.sum()),
                                 "n_test": int(te.sum()), "n_train_unpurged": int(tr.sum()),
                                 "test_windows_sharing_samples": sh})
            print(f"block {b} subject {s} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT + "b1_embargo.csv", index=False)
    summ = []
    for (b, v, m), g in df.groupby(["block_s", "variant", "model"], sort=False):
        x = g.r2.dropna().values
        lo, hi = bootstrap_ci(x) if len(x) > 1 else (np.nan, np.nan)
        summ.append({"block_s": b, "variant": v, "model": m, "r2_mean": x.mean() if len(x) else np.nan, "ci_lo": lo,
                     "ci_hi": hi, "n_participants": len(x), "n_train_mean": g.n_train.mean(),
                     "n_train_unpurged_mean": g.n_train_unpurged.mean(), "n_test_mean": g.n_test.mean(),
                     "shared_total": int(g.test_windows_sharing_samples.sum())})
    pd.DataFrame(summ).to_csv(OUT + "b1_embargo_summary.csv", index=False)
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
