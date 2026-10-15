"""B2 (filtering sensitivity) and B7 (MLP with 1-s windows).

B2a ridge trial-wise: per-trial (paper) vs whole-series filtering, per participant.
B2b series-wise splits with whole-series filtering (leak-free: filters never cross a held-out series),
    ridge / MLP / Riemannian; template unchanged (no EEG).
B2c per-trial filtering with a 0.3-3 Hz band instead of 0.1-3 Hz, ridge trial-wise.
B7  MLP with 1.0-s windows (100 samples), trial-wise and shuffled.
Whole-series and 0.3-Hz signals are written to data/cache_round3/ (new files; the paper cache is untouched).

    python scripts/round3/b2_b7_filtering_mlp.py
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import wilcoxon

from gaudit.data.load_wayeeggal import load_subject
from gaudit.data.preprocess import DECIM, FS_RAW, _fir_bb, _sos, car
from gaudit.data.splits import make_splits
from gaudit.data.targets import TARGETS
from gaudit.eval.stats import bootstrap_ci
from gaudit.experiments import fold_seed, load_e1
from gaudit.pipeline import Data, Pool, run_fold
from gaudit.utils.seed import seed_everything

warnings.filterwarnings("ignore")
S = list(range(1, 13))
OUT = "results/round3/"
C3 = Path("data/cache_round3")
MLP_KW = {"max_epochs": 200, "patience": 10, "batch": 512}


def build_alt_cache(data):
    C3.mkdir(parents=True, exist_ok=True)
    h, sos_lf, sos_lf03 = _fir_bb(), _sos(0.1, 3.0, 2), _sos(0.3, 3.0, 2)
    for s in S:
        f = C3 / f"P{s}.npz"
        if f.exists():
            continue
        sd = data.sd[s]
        pos = {int(t): i for i, t in enumerate(sd.trial_id)}
        n_bb = np.empty_like(sd.bb)
        n_lf = np.empty_like(sd.lf)
        n_lf03 = np.empty_like(sd.lf)
        ser_list, _ = load_subject("data/raw", s)
        for sr in ser_list:
            x = car(sr.eeg.astype(np.float64))
            xp = np.pad(x, ((len(h), len(h)), (0, 0)), mode="reflect")
            bb = signal.fftconvolve(xp, h[:, None], mode="same", axes=0)[len(h):-len(h)]
            lf = signal.sosfiltfilt(sos_lf, x - x.mean(0), axis=0, padtype="even")
            for t in sr.trials:
                i = pos[t.trial_id]
                off, n = int(sd.offset[i]), int(sd.length[i])
                idx = t.abs_start + np.arange(n) * DECIM
                n_bb[off:off + n] = bb[idx]
                n_lf[off:off + n] = lf[idx]
                # per-trial 0.3-3 Hz, otherwise identical to the paper's offline_lowfreq
                xt = car(t.eeg.astype(np.float64))
                xt = xt - xt[: int(0.5 * FS_RAW)].mean(axis=0)
                n_lf03[off:off + n] = signal.sosfiltfilt(sos_lf03, xt, axis=0, padtype="even")[::DECIM][:n]
        np.savez(f, bb_series=n_bb.astype(np.float32), lf_series=n_lf.astype(np.float32), lf_03=n_lf03.astype(np.float32))
        print(f"cache P{s}", flush=True)


def variant(data, which):
    sd2 = {}
    for s in S:
        z = np.load(C3 / f"P{s}.npz")
        sd = data.sd[s]
        if which == "series":
            sd2[s] = dataclasses.replace(sd, bb=z["bb_series"], lf=z["lf_series"])
        elif which == "lf03":
            sd2[s] = dataclasses.replace(sd, lf=z["lf_03"])
    d = Data.__new__(Data)
    d.subjects, d.sd, d.seg0 = S, sd2, data.seg0
    return d


def run(data, ds, model, scheme, win=50, n_rep=5):
    pool = Pool(data, S, ds, win, 5)
    rows = []
    for sp in make_splits(pool.w, scheme, S, 0, n_repeats=n_rep, n_series_folds=3):
        seed_everything(fold_seed(0, ds, model, scheme, sp.fold))
        r, _ = run_fold(pool, sp, model, TARGETS[ds], fold_seed(0, model, sp.fold),
                        model_kw=MLP_KW if model == "mlp" else None)
        rows += r
    df = pd.DataFrame(rows)
    return df.groupby(["subject", "target"]).r2.mean().groupby("subject").mean()  # folds, then x/y/z


def paper(e1, ds, model, scheme):
    g = e1[(e1.dataset == ds) & (e1.model == model) & (e1.scheme == scheme)]
    return g.groupby(["subject", "target"]).r2.mean().groupby("subject").mean()


def compare(a, b):
    d = (b - a).values
    lo, hi = bootstrap_ci(d)
    p = wilcoxon(b.values, a.values).pvalue if not np.allclose(d, 0) else np.nan
    return {"paper_mean": a.mean(), "variant_mean": b.mean(), "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
            "variant_ci": bootstrap_ci(b.values), "wilcoxon_p": p, "n_up": int((d > 0).sum())}


def main():
    t0 = time.time()
    data = Data("data/cache", S)
    e1, _ = load_e1()
    build_alt_cache(data)
    ser = variant(data, "series")
    lf03 = variant(data, "lf03")
    summ, per = [], []
    for ds in ("force", "kin"):
        # B2a
        a, b = paper(e1, ds, "ridge", "trialwise"), run(ser, ds, "ridge", "trialwise")
        summ.append({"part": "B2a", "dataset": ds, "model": "ridge", "scheme": "trialwise",
                     "variant": "whole-series filtering", **compare(a, b)})
        per += [{"part": "B2a", "dataset": ds, "subject": s, "per_trial": a[s], "whole_series": b[s],
                 "diff": b[s] - a[s]} for s in S]
        # B2c
        c = run(lf03, ds, "ridge", "trialwise")
        summ.append({"part": "B2c", "dataset": ds, "model": "ridge", "scheme": "trialwise",
                     "variant": "per-trial 0.3-3 Hz", **compare(a, c)})
        per += [{"part": "B2c", "dataset": ds, "subject": s, "per_trial": a[s], "per_trial_0.3Hz": c[s],
                 "diff": c[s] - a[s]} for s in S]
        # B2b
        for model in ("ridge", "mlp", "riemann", "template"):
            a = paper(e1, ds, model, "serieswise")
            b = a if model == "template" else run(ser, ds, model, "serieswise")
            summ.append({"part": "B2b", "dataset": ds, "model": model, "scheme": "serieswise",
                         "variant": "whole-series filtering" if model != "template" else "unchanged (no EEG)",
                         **compare(a, b)})
        print(f"B2 {ds} done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(per).to_csv(OUT + "b2_per_participant.csv", index=False)
    # B7
    for ds in ("force", "kin"):
        for scheme in ("trialwise", "shuffled"):
            a, b = paper(e1, ds, "mlp", scheme), run(data, ds, "mlp", scheme, win=100)
            summ.append({"part": "B7", "dataset": ds, "model": "mlp", "scheme": scheme, "variant": "1.0-s windows",
                         **compare(a, b)})
        print(f"B7 {ds} done ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(summ)
    out.to_csv(OUT + "b2_b7_summary.csv", index=False)
    print(out.round(4).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
