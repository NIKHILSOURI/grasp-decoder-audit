"""B6: MLP with multitaper band power (0.5-s windows) instead of Welch, force, trial-wise and shuffled.

mne.time_frequency.psd_array_multitaper(bandwidth=4 Hz): with T = 0.5 s, NW = W*T/2 = 1, so MNE uses one DPSS taper
(2NW - 1 = 1). Same channels, bands (4-8, 8-13, 13-30, 30-40 Hz), log10 mean band power, MLP, splits and seeds as E1.

    python scripts/round4/b6_multitaper.py -> results/round4/b6_multitaper.csv
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_multitaper
from scipy.stats import wilcoxon

from gaudit.data.splits import make_splits
from gaudit.eval.stats import bootstrap_ci
from gaudit.experiments import fold_seed, load_e1
from gaudit.features.bandpower import BANDS, PREMOTOR_PARIETAL
from gaudit.features.lagged import raw_windows
from gaudit.pipeline import Data, Pool, run_fold
from gaudit.utils.seed import seed_everything

warnings.filterwarnings("ignore")
S = list(range(1, 13))


def mt_bandpower(win, chunk=20000):
    out = []
    for i in range(0, len(win), chunk):
        psd, f = psd_array_multitaper(win[i:i + chunk].astype(np.float64), sfreq=100, fmin=1, fmax=45, bandwidth=4.0,
                                      adaptive=False, normalization="full", verbose=False)
        bp = [psd[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS.values()]
        out.append(np.log10(np.stack(bp, axis=1) + 1e-12).reshape(len(psd), -1))
    return np.concatenate(out).astype(np.float32)


def main():
    data = Data("data/cache", S)
    pool = Pool(data, S, "force", 50, 5)
    feats = []
    for s in S:
        sd = data.sd[s]
        pp = [sd.ch_names.index(c) for c in PREMOTOR_PARIETAL]
        feats.append(mt_bandpower(raw_windows(sd.bb, pool.w.end[pool.rows[s]], 50, channels=pp)))
        print(f"features P{s}", flush=True)
    pool._feat[("mt_bandpower", ())] = np.concatenate(feats)
    e1, _ = load_e1()
    rows = []
    for scheme in ("trialwise", "shuffled"):
        res = []
        for sp in make_splits(pool.w, scheme, S, 0, n_repeats=5):
            seed_everything(fold_seed(0, "force", "mlp", scheme, sp.fold))
            r, _ = run_fold(pool, sp, "mlp", ["force"], fold_seed(0, "mlp", sp.fold),
                            model_kw={"max_epochs": 200, "patience": 10, "batch": 512}, feature_kind="mt_bandpower")
            res += r
        mt = pd.DataFrame(res).groupby("subject").r2.mean()
        welch = e1[(e1.dataset == "force") & (e1.model == "mlp") & (e1.scheme == scheme)].groupby("subject").r2.mean()
        d = (mt - welch).values
        rows.append({"scheme": scheme, "welch_r2": welch.mean(), "multitaper_r2": mt.mean(),
                     "multitaper_ci": bootstrap_ci(mt.values), "diff": d.mean(), "diff_ci": bootstrap_ci(d),
                     "wilcoxon_p": wilcoxon(mt.values, welch.values).pvalue, "n_up": int((d > 0).sum())})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/round4/b6_multitaper.csv", index=False)


if __name__ == "__main__":
    main()
