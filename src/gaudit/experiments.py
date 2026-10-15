"""The experiments. Each writes per-fold rows, per-subject CSVs and metrics.json to its run dir.

E1 splits     RQ1  models x split schemes x datasets (force; kinematics x/y/z)
E2 selection  RQ2  top-16 selection on all data (leaky) vs training data only
E3 chance     RQ3  trial-mismatch and time-reversal nulls on E1 test predictions
E4 onset      RQ4  causal grasp-onset detection (see onset/)
E5 artifacts  validity: drop frontal-polar and occipital channels
E6 blocks     mechanism: blocked CV, R^2 versus the block length separating train and test
"""

from __future__ import annotations

import shutil
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data.cache import load_stream
from .data.splits import make_splits
from .data.targets import TARGETS
from .eval.stats import bootstrap_ci, paired_wilcoxon
from .onset import detector as odet
from .onset.evaluate import THRESHOLDS, cue_locking, eval_grid, pick_threshold, score
from .onset.labels import intervals, training_windows
from .pipeline import ARTIFACT_CHANNELS, Data, Pool, model_spec, run_fold
from .utils.logging import get_logger
from .utils.seed import rng, seed_everything

log = get_logger()


def fold_seed(seed: int, *parts) -> int:
    return (seed * 1_000_003 + zlib.crc32("/".join(map(str, parts)).encode())) % (2 ** 31)


def subject_table(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average fold-level metrics to one value per subject (and keys)."""
    return rows.groupby(keys + ["subject"], as_index=False)[["r2", "r", "rmse"]].mean()


def summarize(sub: pd.DataFrame, keys: list[str]) -> list[dict]:
    out = []
    for k, g in sub.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        d = dict(zip(keys, k))
        for m in ("r2", "r", "rmse"):
            d[m + "_mean"] = float(g[m].mean())
            d[m + "_sd"] = float(g[m].std(ddof=1))
            d[m + "_ci"] = bootstrap_ci(g[m].values)
        d["n_subjects"] = int(g["subject"].nunique())
        out.append(d)
    return out


# ---------------------------------------------------------------------------------- E1
def rows_from_preds(path: Path, dataset: str, model: str, scheme: str, splits) -> list[dict]:
    """Rebuild fold-level metric rows from a saved prediction file (same splits, same order)."""
    from .eval.metrics import all_metrics

    z = np.load(path)
    targets = [str(t) for t in z["targets"]]
    rows = []
    for i in np.unique(z["fold"]):
        mf = z["fold"] == i
        for s in np.unique(z["subject"][mf]):
            ms = mf & (z["subject"] == s)
            for j, t in enumerate(targets):
                rows.append({"model": model, "scheme": scheme, "fold": splits[int(i)].fold, "subject": int(s),
                             "target": t, **all_metrics(z["y_true"][ms, j], z["y_pred"][ms, j]),
                             "n_train": np.nan, "n_test": int(ms.sum()), "fit_s": np.nan, "info": {},
                             "dataset": dataset})
    return rows


def e1_splits(cfg, out: Path, timer) -> dict:
    data = Data(cfg.cache_dir, list(cfg.subjects))
    rows, pred_dir = [], out / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for dataset in cfg.experiment.datasets:
        targets = TARGETS[dataset]
        pools: dict = {}
        for model in cfg.experiment.models:
            if model == "forcemap" and dataset != "force":
                continue
            spec = model_spec(model, dict(cfg.stride))
            key = (spec.win, spec.stride)
            if key not in pools:
                pools[key] = Pool(data, list(cfg.subjects), dataset, spec.win, spec.stride)
            pool = pools[key]
            with timer(f"features/{dataset}/{model}"):
                pool.feature(spec.feature)
            for scheme in cfg.experiment.schemes:
                n_rep = cfg.repeats.get(model, cfg.repeats.default)
                splits = make_splits(pool.w, scheme, list(cfg.subjects), cfg.seed, n_repeats=n_rep,
                                     n_series_folds=cfg.n_series_folds)
                fname = f"{dataset}_{model}_{scheme}.npz"
                resume = Path(cfg.resume_from) / "preds" / fname if cfg.get("resume_from") else None
                if resume is not None and resume.exists():  # finished in an interrupted run: reuse
                    rows += rows_from_preds(resume, dataset, model, scheme, splits)
                    shutil.copy(resume, pred_dir / fname)
                    log.info(f"E1 {dataset:5s} {model:8s} {scheme:11s} reused predictions from {resume.parent.parent.name}")
                    continue
                P = {"rows": [], "y_true": [], "y_pred": [], "fold": []}
                with timer(f"fit/{model}/{scheme}"):
                    for i, sp in enumerate(splits):
                        if i and i % 6 == 0:
                            log.info(f"   {dataset} {model} {scheme}: fold {i}/{len(splits)}")
                        seed_everything(fold_seed(cfg.seed, dataset, model, scheme, sp.fold))
                        r, p = run_fold(pool, sp, model, targets, fold_seed(cfg.seed, model, sp.fold),
                                        model_kw=cfg.model_kw.get(model, {}))
                        for x in r:
                            x["dataset"] = dataset
                        rows += r
                        for k in ("rows", "y_true", "y_pred"):
                            P[k].append(p[k])
                        P["fold"].append(np.full(len(p["rows"]), i))
                idx = np.concatenate(P["rows"])
                np.savez_compressed(
                    pred_dir / f"{dataset}_{model}_{scheme}.npz", targets=np.array(targets),
                    subject=pool.w.subject[idx], trial_id=pool.w.trial_id[idx], k=pool.w.k[idx],
                    t_rel=pool.w.t_rel[idx], fold=np.concatenate(P["fold"]),
                    y_true=np.concatenate(P["y_true"]), y_pred=np.concatenate(P["y_pred"]))
                df = pd.DataFrame([x for x in rows if x["model"] == model and x["scheme"] == scheme
                                   and x["dataset"] == dataset])
                log.info(f"E1 {dataset:5s} {model:8s} {scheme:11s} R2={df.r2.mean():+.3f} r={df.r.mean():.3f} "
                         f"({len(splits)} folds)")
            if model in ("eegnet", "riemann"):
                pool.drop_features(spec.feature)
    df = pd.DataFrame(rows)
    df.drop(columns=["info"]).to_csv(out / "folds.csv", index=False)
    keys = ["dataset", "model", "scheme", "target"]
    sub = subject_table(df, keys)
    sub.to_csv(out / "per_subject.csv", index=False)
    tests = []
    for (dataset, model, target), g in sub.groupby(["dataset", "model", "target"]):
        piv = g.pivot(index="subject", columns="scheme", values="r2")
        for a, b in (("shuffled", "trialwise"), ("trialwise", "subjectwise"), ("trialwise", "serieswise")):
            if a in piv and b in piv:
                tests.append({"dataset": dataset, "model": model, "target": target, "a": a, "b": b,
                              **paired_wilcoxon(piv[a].values, piv[b].values)})
    # every EEG model versus the no-EEG template under each scheme
    for (dataset, scheme, target), g in sub.groupby(["dataset", "scheme", "target"]):
        piv = g.pivot(index="subject", columns="model", values="r2")
        for model in piv.columns:
            if model != "template" and "template" in piv:
                tests.append({"dataset": dataset, "model": model, "target": target, "a": f"{model}@{scheme}",
                              "b": f"template@{scheme}", **paired_wilcoxon(piv[model].values, piv["template"].values)})
    pd.DataFrame(tests).to_csv(out / "tests.csv", index=False)
    return {"summary": summarize(sub, keys), "tests": tests}


# ---------------------------------------------------------------------------------- E2
def e2_selection(cfg, out: Path, timer) -> dict:
    data = Data(cfg.cache_dir, list(cfg.subjects))
    rows = []
    for target in cfg.experiment.targets:
        dataset = "force" if target == "force" else "kin"
        pool = Pool(data, list(cfg.subjects), dataset, 50, cfg.stride.default)
        for model in cfg.experiment.models:
            for scheme in cfg.experiment.schemes:
                splits = make_splits(pool.w, scheme, list(cfg.subjects), cfg.seed, n_repeats=cfg.repeats.default)
                for sel in cfg.experiment.select_on:
                    with timer(f"{model}/{scheme}/{sel}"):
                        for sp in splits:
                            seed_everything(fold_seed(cfg.seed, target, model, scheme, sel, sp.fold))
                            r, _ = run_fold(pool, sp, model, [target], fold_seed(cfg.seed, model, sp.fold),
                                            model_kw=cfg.model_kw.get(model, {}), select_on=sel,
                                            select_k=cfg.experiment.k)
                            for x in r:
                                x.update(dataset=dataset, select_on=sel)
                            rows += r
                    g = pd.DataFrame([x for x in rows if x["model"] == model and x["scheme"] == scheme
                                      and x["select_on"] == sel and x["target"] == target])
                    log.info(f"E2 {target:6s} {model:6s} {scheme:11s} select_on={sel:5s} R2={g.r2.mean():+.3f}")
    df = pd.DataFrame(rows)
    df.drop(columns=["info"]).to_csv(out / "folds.csv", index=False)
    keys = ["model", "scheme", "target", "select_on"]
    sub = subject_table(df, keys)
    sub.to_csv(out / "per_subject.csv", index=False)
    tests = []
    for (model, scheme, target), g in sub.groupby(["model", "scheme", "target"]):
        piv = g.pivot(index="subject", columns="select_on", values="r2")
        tests.append({"model": model, "scheme": scheme, "target": target,
                      **paired_wilcoxon(piv["all"].values, piv["train"].values)})
    pd.DataFrame(tests).to_csv(out / "tests.csv", index=False)
    return {"summary": summarize(sub, keys), "tests": tests}


# ---------------------------------------------------------------------------------- E3
def _null_targets(trial_id, k, y, g, mode):
    """Rebuild the test targets under a null: 'mismatch' (another test trial of the same
    subject, aligned at segment start, edge-padded) or 'reverse' (time-reversed)."""
    out = np.empty_like(y)
    trials = np.unique(trial_id)
    seqs = {}
    for t in trials:
        m = trial_id == t
        order = np.argsort(k[m])
        seqs[t] = (np.flatnonzero(m)[order], y[m][order])
    if mode == "reverse":
        for rows, ys in seqs.values():
            out[rows] = ys[::-1]
        return out
    perm = g.permutation(trials)
    while len(trials) > 1 and np.any(perm == trials):  # derangement: never the same trial
        perm = g.permutation(trials)
    for t, src in zip(trials, perm):
        rows, _ = seqs[t]
        ys = seqs[src][1]
        idx = np.minimum(np.arange(len(rows)), len(ys) - 1)
        out[rows] = ys[idx]
    return out


def e3_chance(cfg, out: Path, timer) -> dict:
    sub1, pred_files = load_e1(cfg.experiment.get("e1_dir"))
    honest = sub1[(sub1.scheme == "trialwise") & (sub1.model != "template")]
    best = honest.groupby("model").r2.mean().sort_values(ascending=False).index[:2].tolist()
    models = best + ["template"]
    log.info(f"E3 best two EEG models (trialwise R2): {best}")
    from .eval.metrics import pearson, r2

    rows = []
    for dataset in ("force", "kin"):
        for model in models:
            for scheme in cfg.experiment.schemes:
                f = pred_files.get((dataset, model, scheme))
                if f is None:
                    continue
                z = np.load(f)
                targets = [str(x) for x in z["targets"]]
                with timer(f"{dataset}/{model}/{scheme}"):
                    for s in np.unique(z["subject"]):
                        for fold in np.unique(z["fold"][z["subject"] == s]):
                            m = (z["subject"] == s) & (z["fold"] == fold)
                            tid, kk = z["trial_id"][m], z["k"][m]
                            for j, tname in enumerate(targets):
                                y, p = z["y_true"][m][:, j], z["y_pred"][m][:, j]
                                base = {"dataset": dataset, "model": model, "scheme": scheme, "subject": int(s),
                                        "fold": int(fold), "target": tname}
                                rows.append({**base, "null": "real", "perm": -1, "r2": r2(y, p), "r": pearson(y, p)})
                                yr = _null_targets(tid, kk, y, None, "reverse")
                                rows.append({**base, "null": "reverse", "perm": -1, "r2": r2(yr, p),
                                             "r": pearson(yr, p)})
                                g = rng(cfg.seed, 7, int(s), int(fold), j)
                                for pi in range(cfg.experiment.n_perm):
                                    ym = _null_targets(tid, kk, y, g, "mismatch")
                                    rows.append({**base, "null": "mismatch", "perm": pi, "r2": r2(ym, p),
                                                 "r": pearson(ym, p)})
                log.info(f"E3 {dataset} {model} {scheme} done")
    df = pd.DataFrame(rows)
    # per subject: average over folds (per permutation index for the mismatch null)
    agg = df.groupby(["dataset", "model", "scheme", "target", "subject", "null", "perm"], as_index=False)[
        ["r2", "r"]].mean()
    agg.to_csv(out / "null_per_subject.csv", index=False)
    res = []
    for (dataset, model, scheme, target, s), g in agg.groupby(["dataset", "model", "scheme", "target", "subject"]):
        real = g[g.null == "real"].iloc[0]
        mm = g[g.null == "mismatch"]
        rev = g[g.null == "reverse"].iloc[0]
        res.append({"dataset": dataset, "model": model, "scheme": scheme, "target": target, "subject": s,
                    "r2_real": real.r2, "r_real": real.r, "r2_null95": float(np.quantile(mm.r2, 0.95)),
                    "r_null95": float(np.quantile(mm.r, 0.95)), "r2_null_mean": float(mm.r2.mean()),
                    "r_null_mean": float(mm.r.mean()), "r2_reverse": rev.r2, "r_reverse": rev.r})
    res = pd.DataFrame(res)
    res["above_r2"] = res.r2_real > res.r2_null95
    res["above_r"] = res.r_real > res.r_null95
    res.to_csv(out / "per_subject.csv", index=False)
    summ = res.groupby(["dataset", "model", "scheme", "target"], as_index=False).agg(
        r2_real=("r2_real", "mean"), r2_null95=("r2_null95", "mean"), r2_null_mean=("r2_null_mean", "mean"),
        r_real=("r_real", "mean"), r_null95=("r_null95", "mean"), r_null_mean=("r_null_mean", "mean"),
        r2_reverse=("r2_reverse", "mean"), r_reverse=("r_reverse", "mean"),
        frac_above_r2=("above_r2", "mean"), frac_above_r=("above_r", "mean"), n=("subject", "nunique"))
    summ["r2_minus_chance"] = summ.r2_real - summ.r2_null95
    summ.to_csv(out / "summary.csv", index=False)
    return {"e1_runs": sorted({str(f.parent.parent) for f in pred_files.values()}), "models": models,
            "summary": summ.to_dict("records")}


# ---------------------------------------------------------------------------------- E4
def e4_onset(cfg, out: Path, timer) -> dict:
    subjects = list(cfg.subjects)
    streams = {s: load_stream(cfg.cache_dir, s) for s in subjects}
    ivs = {s: intervals(streams[s]) for s in subjects}
    rows, locks, curves, outcomes = [], [], [], []
    ch_names = [str(c) for c in streams[subjects[0]]["ch_names"]]
    variants = {"all": None,
                "no_occipital": [i for i, c in enumerate(ch_names) if c not in ("O1", "Oz", "O2", "PO9", "PO10")]}

    def featurize(s, rows_, kind, chans, train):
        iv = ivs[s]
        if train:
            S, t, y, _ = training_windows(iv, rows_)
            return odet.features(streams[s], S, t, kind, chans), y
        S, t, r = eval_grid(iv, rows_)
        return odet.features(streams[s], S, t, kind, chans), (t, r)

    for det_name in cfg.experiment.detectors:
        for var, chans in variants.items():
            if var != "all" and not cfg.experiment.channel_check:
                continue
            for scheme in cfg.experiment.schemes:
                folds = []
                if scheme == "trialwise":
                    for s in subjects:
                        for rep in range(cfg.experiment.n_repeats):
                            idx = np.arange(len(ivs[s].trial_id))
                            g = rng(cfg.seed, 9, s, rep)
                            idx = g.permutation(idx)
                            n = len(idx)
                            ntr, nte = int(round(0.7 * n)), int(round(0.2 * n))
                            folds.append((f"S{s}/r{rep}", {s: idx[:ntr]}, {s: idx[ntr + nte:]}, {s: idx[ntr:ntr + nte]}))
                else:
                    for s in subjects:
                        rest = [x for x in subjects if x != s]
                        va = rng(cfg.seed, 4, s).choice(rest, 2, replace=False)
                        tr = [x for x in rest if x not in va]
                        alli = {x: np.arange(len(ivs[x].trial_id)) for x in subjects}
                        folds.append((f"S{s}", {x: alli[x] for x in tr}, {x: alli[x] for x in va}, {s: alli[s]}))
                with timer(f"{det_name}/{var}/{scheme}"):
                    for fold, TR, VA, TE in folds:
                        # leak check: train/val/test trial IDs disjoint
                        ids = {k: set(np.concatenate([ivs[s].trial_id[r] for s, r in D.items()]))
                               for k, D in (("tr", TR), ("va", VA), ("te", TE))}
                        assert not (ids["tr"] & ids["te"]) and not (ids["tr"] & ids["va"]) and not (ids["va"] & ids["te"])
                        Xtr, ytr = zip(*[featurize(s, r, det_name if det_name == "lda" else "riemann", chans, True)
                                         for s, r in TR.items()])
                        clf = (odet.OnsetLDA() if det_name == "lda" else
                               odet.OnsetRiemann(seed=fold_seed(cfg.seed, fold)))
                        clf.fit(np.concatenate(Xtr), np.concatenate(ytr))
                        val_rows = []
                        for s, r in VA.items():
                            Xv, (tv, rv) = featurize(s, r, clf.kind, chans, False)
                            vr, _ = score(ivs[s], rv, tv, clf.proba(Xv))
                            val_rows.append(pd.DataFrame(vr))
                        vdf = pd.concat(val_rows).groupby("threshold", as_index=False).mean()
                        thr = pick_threshold(vdf.to_dict("records"), cfg.experiment.max_fa_per_min)
                        for s, r in TE.items():
                            # window-level AUC on the labelled windows of the test trials
                            Xl, yl = featurize(s, r, clf.kind, chans, True)
                            auc = float(roc_auc_score(yl, clf.proba(Xl)))
                            Xt, (tt, rt) = featurize(s, r, clf.kind, chans, False)
                            res, first = score(ivs[s], rt, tt, clf.proba(Xt))
                            for x in res:
                                x.update(detector=det_name, variant=var, scheme=scheme, fold=fold, subject=s)
                            curves += res
                            op = next(x for x in res if x["threshold"] == thr)
                            rows.append({**op, "operating_threshold": thr, "auc": auc})
                            lock = cue_locking(ivs[s], first["rows"], first[float(thr)])
                            locks.append({"detector": det_name, "variant": var, "scheme": scheme, "fold": fold,
                                          "subject": s, **lock})
                            # per test trial at the operating point: first in-window detection vs hand start
                            iv = ivs[s]
                            for i, fd in zip(first["rows"], first[float(thr)]):
                                cat = "never" if not np.isfinite(fd) else ("before_hs" if fd < iv.hs[i] else "after_hs_only")
                                outcomes.append({"detector": det_name, "variant": var, "scheme": scheme, "fold": fold,
                                                 "subject": s, "trial_id": int(iv.trial_id[i]), "outcome": cat,
                                                 "lead_ms": (fd - iv.hs[i]) * 1000 if np.isfinite(fd) else np.nan})
                g = pd.DataFrame([x for x in rows if x["detector"] == det_name and x["variant"] == var
                                  and x["scheme"] == scheme])
                log.info(f"E4 {det_name:7s} {var:12s} {scheme:11s} AUC={g.auc.mean():.3f} det={g.det_rate.mean():.2f} "
                         f"lead={g.median_lead_ms.median():.0f}ms FA/min={g.fa_per_min.mean():.2f}")
    # no-EEG cue-timer baseline: "detect" at LED-on + 100 ms
    cue = []
    for s in subjects:
        iv = ivs[s]
        lead = (iv.led + 0.1 - iv.hs) * 1000
        cue.append({"subject": s, "det_rate": float(np.mean(iv.led + 0.1 <= iv.hs + 0.2)),
                    "median_lead_ms": float(np.median(lead)), "fa_per_min": 0.0,
                    "median_rt_ms": float(np.median((iv.hs - iv.led) * 1000))})
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    op = pd.DataFrame(rows)
    op.to_csv(out / "operating_points.csv", index=False)
    pd.DataFrame(locks).to_csv(out / "cue_locking.csv", index=False)
    pd.DataFrame(outcomes).to_csv(out / "trial_outcomes.csv", index=False)
    pd.DataFrame(cue).to_csv(out / "cue_timer_baseline.csv", index=False)
    per_sub = op.groupby(["detector", "variant", "scheme", "subject"], as_index=False)[
        ["det_rate", "median_lead_ms", "median_latency_led_ms", "fa_per_min", "operating_threshold", "auc"]].mean()
    per_sub.to_csv(out / "per_subject.csv", index=False)
    lk = pd.DataFrame(locks).groupby(["detector", "variant", "scheme", "subject"], as_index=False)[["slope", "r"]].mean()
    summ = per_sub.groupby(["detector", "variant", "scheme"], as_index=False).agg(
        det_rate=("det_rate", "mean"), median_lead_ms=("median_lead_ms", "median"),
        median_latency_led_ms=("median_latency_led_ms", "median"), fa_per_min=("fa_per_min", "mean"),
        auc=("auc", "mean"))
    summ = summ.merge(lk.groupby(["detector", "variant", "scheme"], as_index=False).agg(
        cue_slope=("slope", "median"), cue_r=("r", "median")), on=["detector", "variant", "scheme"])
    summ.to_csv(out / "summary.csv", index=False)
    return {"summary": summ.to_dict("records"), "cue_timer": pd.DataFrame(cue).mean().to_dict(),
            "thresholds": THRESHOLDS.tolist()}


# ---------------------------------------------------------------------------------- E5
def e5_artifacts(cfg, out: Path, timer) -> dict:
    data = Data(cfg.cache_dir, list(cfg.subjects))
    rows = []
    for target in cfg.experiment.targets:
        dataset = "force" if target == "force" else "kin"
        pool = Pool(data, list(cfg.subjects), dataset, 50, cfg.stride.default)
        splits = make_splits(pool.w, "trialwise", list(cfg.subjects), cfg.seed, n_repeats=cfg.repeats.default)
        for model in cfg.experiment.models:
            for chans, drop in (("all", ()), ("no_fp_occ", tuple(ARTIFACT_CHANNELS))):
                with timer(f"{target}/{model}/{chans}"):
                    for sp in splits:
                        seed_everything(fold_seed(cfg.seed, target, model, chans, sp.fold))
                        r, _ = run_fold(pool, sp, model, [target], fold_seed(cfg.seed, model, sp.fold), drop=drop)
                        for x in r:
                            x.update(dataset=dataset, channels=chans)
                        rows += r
                g = pd.DataFrame([x for x in rows if x["model"] == model and x["target"] == target])
                log.info(f"E5 {target} {model} " + " ".join(f"{c}:{v:+.3f}" for c, v in g.groupby("channels").r2.mean().items()))
            pool.drop_features()
    df = pd.DataFrame(rows)
    df.drop(columns=["info"]).to_csv(out / "folds.csv", index=False)
    keys = ["model", "target", "channels"]
    sub = subject_table(df, keys)
    sub.to_csv(out / "per_subject.csv", index=False)
    tests = []
    for (model, target), g in sub.groupby(["model", "target"]):
        piv = g.pivot(index="subject", columns="channels", values="r2")
        tests.append({"model": model, "target": target, **paired_wilcoxon(piv["no_fp_occ"].values, piv["all"].values)})
    pd.DataFrame(tests).to_csv(out / "tests.csv", index=False)
    return {"summary": summarize(sub, keys), "tests": tests}


# ---------------------------------------------------------------------------------- E6
def e6_blocks(cfg, out: Path, timer) -> dict:
    """Blocked CV: R^2 as a function of the block length that separates train and test."""
    from .data.splits import blocked

    data = Data(cfg.cache_dir, list(cfg.subjects))
    rows = []
    for dataset in cfg.experiment.datasets:
        targets = TARGETS[dataset]
        pool = Pool(data, list(cfg.subjects), dataset, 50, cfg.stride.default)
        for model in cfg.experiment.models:
            for block_s in cfg.experiment.blocks:
                splits = [blocked(pool.w, s, cfg.seed, float(block_s), r) for s in cfg.subjects
                          for r in range(cfg.experiment.n_repeats)]
                with timer(f"{dataset}/{model}/{block_s}"):
                    for sp in splits:
                        seed_everything(fold_seed(cfg.seed, dataset, model, block_s, sp.fold))
                        r, _ = run_fold(pool, sp, model, targets, fold_seed(cfg.seed, model, sp.fold),
                                        model_kw=cfg.model_kw.get(model, {}))
                        for x in r:
                            x.update(dataset=dataset, block_s=float(block_s))
                        rows += r
                g = pd.DataFrame([x for x in rows if x["model"] == model and x["dataset"] == dataset
                                  and x["block_s"] == float(block_s)])
                log.info(f"E6 {dataset} {model} block={block_s}s R2={g.r2.mean():+.3f} n_train={g.n_train.mean():.0f}")
            pool.drop_features(model_spec(model).feature)
    df = pd.DataFrame(rows)
    df.drop(columns=["info"]).to_csv(out / "folds.csv", index=False)
    keys = ["dataset", "model", "target", "block_s"]
    sub = subject_table(df, keys)
    sub.to_csv(out / "per_subject.csv", index=False)
    return {"summary": summarize(sub, keys)}


# ---------------------------------------------------------------------------------- E7
def e7_residual(cfg, out: Path, timer) -> dict:
    """Does EEG add anything beyond the no-EEG template? Same folds for template and
    template+EEG; residual R^2 = how well EEG predicts (target - template) on test windows."""
    from .eval.metrics import pearson, r2

    data = Data(cfg.cache_dir, list(cfg.subjects))
    rows = []
    for dataset in cfg.experiment.datasets:
        targets = TARGETS[dataset]
        pool = Pool(data, list(cfg.subjects), dataset, 50, cfg.stride.default)
        for inner in cfg.experiment.models:
            name = f"template+{inner}"
            for scheme in cfg.experiment.schemes:
                splits = make_splits(pool.w, scheme, list(cfg.subjects), cfg.seed, n_repeats=cfg.repeats.default)
                with timer(f"{dataset}/{inner}/{scheme}"):
                    for sp in splits:
                        seed_everything(fold_seed(cfg.seed, dataset, name, scheme, sp.fold))
                        _, pt = run_fold(pool, sp, "template", targets, 0)
                        _, pc = run_fold(pool, sp, name, targets, fold_seed(cfg.seed, name, sp.fold),
                                         model_kw=cfg.model_kw.get(inner, {}))
                        assert np.array_equal(pt["rows"], pc["rows"])
                        subj = pool.w.subject[pc["rows"]]
                        for s in np.unique(subj):
                            m = subj == s
                            for j, t in enumerate(targets):
                                y, T, P = pc["y_true"][m, j], pt["y_pred"][m, j], pc["y_pred"][m, j]
                                rows.append({"dataset": dataset, "model": inner, "scheme": scheme, "fold": sp.fold,
                                             "subject": int(s), "target": t, "r2_template": r2(y, T),
                                             "r2_combined": r2(y, P), "r2_residual": r2(y - T, P - T),
                                             "r_residual": pearson(y - T, P - T)})
                g = pd.DataFrame([x for x in rows if x["model"] == inner and x["dataset"] == dataset
                                  and x["scheme"] == scheme])
                log.info(f"E7 {dataset} {inner} {scheme}: template {g.r2_template.mean():+.3f} -> "
                         f"template+EEG {g.r2_combined.mean():+.3f} (residual R2 {g.r2_residual.mean():+.3f})")
            pool.drop_features()
    df = pd.DataFrame(rows)
    df.to_csv(out / "folds.csv", index=False)
    cols = ["r2_template", "r2_combined", "r2_residual", "r_residual"]
    sub = df.groupby(["dataset", "model", "scheme", "target", "subject"], as_index=False)[cols].mean()
    sub.to_csv(out / "per_subject.csv", index=False)
    k = sub[sub.dataset == "kin"].groupby(["dataset", "model", "scheme", "subject"], as_index=False)[cols].mean()
    both = pd.concat([sub[sub.dataset == "force"], k.assign(target="pos")])
    tests, summ = [], []
    for (ds, m, sc, t), g in both.groupby(["dataset", "model", "scheme", "target"]):
        w = paired_wilcoxon(g.r2_combined.values, g.r2_template.values)
        tests.append({"dataset": ds, "model": m, "scheme": sc, "target": t, **w})
        summ.append({"dataset": ds, "model": m, "scheme": sc, "target": t, **{c: float(g[c].mean()) for c in cols},
                     "gain": float((g.r2_combined - g.r2_template).mean()),
                     "n_better": int((g.r2_combined > g.r2_template).sum()), "n": len(g), "p": w["p"]})
    pd.DataFrame(tests).to_csv(out / "tests.csv", index=False)
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    return {"summary": summ}


def latest_run(name: str, root: str = "results") -> Path:
    runs = sorted((Path(root) / name).glob("*/metrics.json"))
    assert runs, f"no completed {name} run under {root}/{name}"
    return runs[-1].parent


def load_e1(e1_dir: str | None = None, root: str = "results"):
    """Per-subject E1 table and prediction files, merged over completed E1 runs (E1 may be run
    in parts, e.g. EEGNet separately). For each (dataset, model) the most recent run wins."""
    runs = [Path(e1_dir)] if e1_dir else sorted(p.parent for p in (Path(root) / "e1_splits").glob("*/metrics.json"))
    assert runs, "no completed e1_splits run"
    tables, preds, seen = [], {}, set()
    for run in reversed(runs):  # newest first
        t = pd.read_csv(run / "per_subject.csv")
        t = t[[(d, m) not in seen for d, m in zip(t.dataset, t.model)]]
        new = set(zip(t.dataset, t.model))
        for f in (run / "preds").glob("*.npz"):
            dataset, model, scheme = f.stem.split("_", 2)
            if (dataset, model) in new:
                preds[(dataset, model, scheme)] = f
        seen |= new
        tables.append(t.assign(run=run.name))
    return pd.concat(tables, ignore_index=True), preds


EXPERIMENTS = {"e1_splits": e1_splits, "e2_selection": e2_selection, "e3_chance": e3_chance,
               "e4_onset": e4_onset, "e5_artifacts": e5_artifacts, "e6_blocks": e6_blocks,
               "e7_residual": e7_residual}

