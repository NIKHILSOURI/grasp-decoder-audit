"""Round-2 verification (no paper result is changed; new analyses are sensitivity checks only).

    python scripts/verify_round2.py  [--skip-optional]
Writes results/verification/round2_*.csv and prints a report.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
import warnings

import numpy as np
import pandas as pd
from scipy import signal

from gaudit.data.load_wayeeggal import load_subject
from gaudit.data.preprocess import DECIM, _fir_bb, _sos, car, online_filters
from gaudit.data.splits import blocked, make_splits, split_masks
from gaudit.data.targets import TargetScaler
from gaudit.eval.metrics import r2
from gaudit.eval.stats import bootstrap_ci
from gaudit.experiments import latest_run, load_e1
from gaudit.onset.evaluate import eval_grid
from gaudit.onset.labels import intervals, training_windows
from gaudit.pipeline import ARTIFACT_CHANNELS, Data, Pool, build_model, model_spec, run_fold
from gaudit.utils.seed import rng, seed_everything

warnings.filterwarnings("ignore")
S = list(range(1, 13))
OUT = "results/verification/"


def hdr(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100, flush=True)


def fit_eval(pool, tr, va, te, model, targets, seed=0, kw=None, drop=()):
    """Same steps as pipeline.run_fold, but with explicit masks (for purged splits)."""
    X = pool.feature(model_spec(model).feature, drop=drop)
    y = pool.y(targets)
    ts = TargetScaler().fit(y[tr])
    m = build_model(model, seed, kw)
    m.fit(X[tr], ts.transform(y[tr]), X[va], ts.transform(y[va]))
    pred = ts.inverse(m.predict(X[te]).reshape(int(te.sum()), -1))
    return [r2(y[te][:, j], pred[:, j]) for j in range(len(targets))]


def main(optional: bool):
    data = Data("data/cache", S)
    e1, _ = load_e1()

    # ------------------------------------------------------------------------------------ R1
    hdr("R1  segment lengths at 500 Hz (exact, from the loader)")
    lens = {"all": [], "force": []}
    series_cache = {}
    for s in S:
        ser, _ = load_subject("data/raw", s)
        series_cache[s] = ser
        for sr in ser:
            for t in sr.trials:
                lens["all"].append(t.n)
                if t.block_type == 1:
                    lens["force"].append(t.n)
    for k, v in lens.items():
        v = np.array(v)
        print(f"{k:5s}: n={len(v)}  min {v.min()} samples ({v.min() / 500:.2f} s), median {np.median(v):.0f} "
              f"({np.median(v) / 500:.2f} s), max {v.max()} ({v.max() / 500:.2f} s); shorter than the 1,651-sample "
              f"reflect pad: {(v - 1 < 1651).sum()}")

    if optional:
        hdr("R1 optional: filter each whole series once, then cut trials (ridge trial-wise; template unchanged)")
        t0 = time.time()
        h = _fir_bb()
        sos_lf = _sos(0.1, 3.0, 2)
        alt = {}
        for s in S:
            sd = data.sd[s]
            bb_parts, lf_parts = [], []
            for sr in series_cache[s]:
                x = car(sr.eeg.astype(np.float64))
                xp = np.pad(x, ((len(h), len(h)), (0, 0)), mode="reflect")
                bb = signal.fftconvolve(xp, h[:, None], mode="same", axes=0)[len(h):-len(h)]
                lf = signal.sosfiltfilt(sos_lf, x - x.mean(0), axis=0, padtype="even")
                for t in sr.trials:
                    n100 = int(sd.length[list(sd.trial_id).index(t.trial_id)])
                    idx = t.abs_start + np.arange(n100) * DECIM
                    bb_parts.append(bb[idx])
                    lf_parts.append(lf[idx])
            alt[s] = dataclasses.replace(sd, bb=np.concatenate(bb_parts).astype(np.float32),
                                         lf=np.concatenate(lf_parts).astype(np.float32))
            assert alt[s].bb.shape == sd.bb.shape
        d2 = Data.__new__(Data)
        d2.subjects, d2.sd, d2.seg0 = S, alt, data.seg0
        res = []
        for ds, tg in (("force", ["force"]), ("kin", ["pos_x", "pos_y", "pos_z"])):
            for label, dd in (("per-trial (paper)", data), ("whole-series", d2)):
                pool = Pool(dd, S, ds, 50, 5)
                rows = []
                for sp in make_splits(pool.w, "trialwise", S, 0, n_repeats=5):
                    r, _ = run_fold(pool, sp, "ridge", tg, 0)
                    rows += r
                df = pd.DataFrame(rows).groupby(["subject", "target"]).r2.mean().groupby("subject").mean()
                res.append({"dataset": ds, "filtering": label, "ridge_r2": df.mean(), "per_subject": df.values})
        rr = pd.DataFrame(res)
        for ds in ("force", "kin"):
            a = rr[(rr.dataset == ds) & (rr.filtering == "per-trial (paper)")].iloc[0]
            b = rr[(rr.dataset == ds) & (rr.filtering == "whole-series")].iloc[0]
            dlt = b.per_subject - a.per_subject
            print(f"{ds}: ridge trial-wise R2 per-trial {a.ridge_r2:.4f} -> whole-series {b.ridge_r2:.4f} "
                  f"(change {dlt.mean():+.4f}, 95% CI {np.round(bootstrap_ci(dlt), 4)})")
        rr.drop(columns="per_subject").to_csv(OUT + "round2_r1_series_filtering.csv", index=False)
        print(f"({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------------------------ R3
    hdr("R3  Table II gains: 95% bootstrap CI over participants (10,000 resamples, seed 0)")
    e7 = pd.read_csv(latest_run("e7_residual") / "per_subject.csv")
    k = e7[e7.dataset == "kin"].groupby(["model", "scheme", "subject"], as_index=False)[["r2_template", "r2_combined"]].mean()
    both = pd.concat([e7[e7.dataset == "force"].assign(target="force"), k.assign(dataset="kin", target="position")])
    out = []
    for (t, m, sc), g in both.groupby(["target", "model", "scheme"]):
        dlt = (g.r2_combined - g.r2_template).values
        lo, hi = bootstrap_ci(dlt)
        out.append({"target": t, "model": m, "scheme": sc, "gain": dlt.mean(), "ci_lo": lo, "ci_hi": hi,
                    "n_better": int((dlt > 0).sum())})
    r3 = pd.DataFrame(out)
    print(r3.round(4).to_string(index=False))
    r3.to_csv(OUT + "round2_r3_tableII_ci.csv", index=False)

    # ------------------------------------------------------------------------------------ R5
    hdr("R5  causal filters: group delay (numerical phase derivative) and onset-window checks")
    f = np.array([0.5, 1.0, 2.0, 10.0])
    for name, sos in (("0.1-3 Hz Butterworth (order 2 design)", _sos(0.1, 3.0, 2)), ("1-40 Hz Butterworth (order 4 design)", _sos(1.0, 40.0, 4))):
        ff = np.linspace(0.05, 20, 40000)
        w, hresp = signal.sosfreqz(sos, worN=ff, fs=500)
        ph = np.unwrap(np.angle(hresp))
        gd = -np.gradient(ph, 2 * np.pi * w)
        print(name, "group delay:", {float(x): round(float(np.interp(x, w, gd)) * 1000, 1) for x in f}, "ms")
    # impulse check: peak delay of a smoothed impulse response
    x = np.zeros(20000)
    x[5000] = 1
    _ = online_filters
    min_end, overl, tot = [], 0, 0
    for s in S:
        from gaudit.data.cache import load_stream
        st = load_stream("data/cache", s)
        iv = intervals(st)
        allrows = np.arange(len(iv.trial_id))
        Str, ttr, _, _ = training_windows(iv, allrows)
        Sev, tev, _ = eval_grid(iv, allrows)
        min_end.append(min(ttr.min(), tev.min()))
        for rep in range(3):  # trial-wise folds exactly as in E4
            idx = rng(0, 9, s, rep).permutation(len(iv.trial_id))
            n = len(idx)
            ntr, nte = int(round(0.7 * n)), int(round(0.2 * n))
            tr_rows, te_rows = idx[:ntr], idx[ntr:ntr + nte]
            S1, t1, _, _ = training_windows(iv, tr_rows)
            S2, t2, _ = eval_grid(iv, te_rows)
            for sr in np.unique(S2):
                a = np.sort(t1[S1 == sr])
                b = t2[S2 == sr]
                if not a.size:
                    continue
                j = np.searchsorted(a, b)
                d = np.minimum(np.abs(b - a[np.clip(j - 1, 0, a.size - 1)]), np.abs(a[np.clip(j, 0, a.size - 1)] - b))
                overl += int((d < 0.5 - 1e-9).sum())
                tot += b.size
    print(f"earliest onset window end over all participants: {min(min_end):.2f} s after the series start "
          f"(so every 500-ms window starts >= {min(min_end) - 0.5:.2f} s into the series)")
    print(f"trial-wise onset folds: test evaluation windows sharing samples with a training window: {overl} of {tot} "
          f"({overl / tot * 100:.2f}%)")

    # ------------------------------------------------------------------------------------ O4
    hdr("O4  per-axis R2 (mean over participants)")
    ax = e1[(e1.dataset == "kin") & (e1.scheme.isin(["trialwise", "subjectwise"]))]
    tab = ax.groupby(["model", "scheme", "target"]).r2.mean().unstack("target").round(3)
    print(tab.to_string())
    ax.to_csv(OUT + "round2_o4_per_axis_per_subject.csv", index=False)

    if not optional:
        return

    # ------------------------------------------------------------------------------------ O1
    hdr("O1  purged blocked CV (0.5 s embargo; train AND validation purged), force")
    pool = Pool(data, S, "force", 50, 5)
    w = pool.w
    rows = []
    for b in (0.25, 1.0, 4.0):
        for s in S:
            sp = blocked(w, s, 0, b, 0)
            tr, va, te = split_masks(w, sp)
            keep_tr, keep_va = tr.copy(), va.copy()
            for tid in np.unique(w.trial_id[te]):
                te_e = np.sort(w.end[te & (w.trial_id == tid)])
                for msk in (keep_tr, keep_va):
                    ii = np.flatnonzero(msk & (w.trial_id == tid))
                    e = w.end[ii]
                    j = np.searchsorted(te_e, e)
                    d = np.minimum(np.abs(e - te_e[np.clip(j - 1, 0, te_e.size - 1)]), np.abs(te_e[np.clip(j, 0, te_e.size - 1)] - e))
                    msk[ii[d < 50]] = False  # |delta end| < 0.5 s -> would share samples
            # verify: no remaining train/val window shares a raw sample with a test window
            shared = 0
            for tid in np.unique(w.trial_id[te]):
                te_e = w.end[te & (w.trial_id == tid)]
                tr_e = w.end[(keep_tr | keep_va) & (w.trial_id == tid)]
                if tr_e.size:
                    shared += int((np.abs(te_e[:, None] - tr_e[None, :]) <= 49).any(1).sum())
            for model in ("ridge", "riemann", "mlp"):
                seed_everything(s)
                r = fit_eval(pool, keep_tr, keep_va, te, model, ["force"], seed=s,
                             kw={"max_epochs": 200, "patience": 10, "batch": 512} if model == "mlp" else None)
                rows.append({"block_s": b, "subject": s, "model": model, "r2": r[0], "n_train": int(keep_tr.sum()),
                             "n_val": int(keep_va.sum()), "n_test": int(te.sum()), "n_train_unpurged": int(tr.sum()),
                             "test_windows_sharing_samples": shared})
        print(f"block {b}s done", flush=True)
        pool.drop_features("cov")
    o1 = pd.DataFrame(rows)
    o1.to_csv(OUT + "round2_o1_purged_blocked.csv", index=False)
    summ = o1.groupby(["block_s", "model"]).agg(r2=("r2", "mean"), n_train=("n_train", "mean"),
                                                n_train_unpurged=("n_train_unpurged", "mean"), n_test=("n_test", "mean"),
                                                shared=("test_windows_sharing_samples", "sum")).round(3)
    print(summ.to_string())
    e6 = pd.read_csv(latest_run("e6_blocks") / "per_subject.csv")
    print("\nunpurged (E6) force R2 for comparison:")
    print(e6[(e6.dataset == "force") & e6.block_s.isin([0.25, 1.0, 4.0])].groupby(["block_s", "model"]).r2.mean().round(3).unstack())

    # ------------------------------------------------------------------------------------ O2
    hdr("O2  CAR recomputed over the 25 retained channels (ridge, Riemannian; trial-wise)")
    keep = [i for i, c in enumerate(data.sd[1].ch_names) if c not in ARTIFACT_CHANNELS]
    alt = {}
    for s in S:
        sd = data.sd[s]
        # filtering and decimation are linear, so re-referencing the stored CAR-32 signals to the mean of the
        # retained channels equals computing CAR over those 25 channels before filtering
        bb = sd.bb - sd.bb[:, keep].mean(1, keepdims=True)
        lf = sd.lf - sd.lf[:, keep].mean(1, keepdims=True)
        alt[s] = dataclasses.replace(sd, bb=bb.astype(np.float32), lf=lf.astype(np.float32))
    d3 = Data.__new__(Data)
    d3.subjects, d3.sd, d3.seg0 = S, alt, data.seg0
    e5 = pd.read_csv(latest_run("e5_artifacts") / "per_subject.csv")
    rows = []
    for target in ("force", "pos_z"):
        ds = "force" if target == "force" else "kin"
        pool = Pool(d3, S, ds, 50, 5)
        for model in ("ridge", "riemann"):
            fr = []
            for sp in make_splits(pool.w, "trialwise", S, 0, n_repeats=5):
                r, _ = run_fold(pool, sp, model, [target], 0, drop=tuple(ARTIFACT_CHANNELS))
                fr += r
            car25 = pd.DataFrame(fr).groupby("subject").r2.mean()
            allc = e5[(e5.model == model) & (e5.target == target) & (e5.channels == "all")].set_index("subject").r2
            car32 = e5[(e5.model == model) & (e5.target == target) & (e5.channels == "no_fp_occ")].set_index("subject").r2
            d = (car25 - allc).values
            rows.append({"model": model, "target": target, "all_channels": allc.mean(), "drop_CAR32": car32.mean(),
                         "drop_CAR25": car25.mean(), "change_CAR25_vs_all": d.mean(),
                         "ci_lo": bootstrap_ci(d)[0], "ci_hi": bootstrap_ci(d)[1]})
            pool.drop_features()
    o2 = pd.DataFrame(rows)
    print(o2.round(4).to_string(index=False))
    o2.to_csv(OUT + "round2_o2_car25.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-optional", action="store_true")
    main(not ap.parse_args().skip_optional)
