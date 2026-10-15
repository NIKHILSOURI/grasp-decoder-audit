"""Round-3 analyses on saved predictions (no retraining): B4 (R2 robustness, template support), B5 (common grid),
C2 (font sizes), and A3 evidence files for the anonymous artifact.

    python scripts/round3/analysis_round3.py [--only b4,b5,a3,c2]
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from gaudit.data.splits import blocked, make_splits, split_masks
from gaudit.eval.metrics import r2
from gaudit.eval.stats import bootstrap_ci, holm
from gaudit.experiments import fold_seed, latest_run, load_e1
from gaudit.pipeline import Data, Pool

S = list(range(1, 13))
OUT = Path("results/round3")
REL = Path("../anonymous_release")
EEG = ["ridge", "mlp", "riemann", "eegnet"]
AX = {"force": "force", "pos_x": "x", "pos_y": "y", "pos_z": "z"}


def nrep(model):
    return 3 if model == "eegnet" else 5


def load_pred(preds, ds, model, scheme):
    z = np.load(preds[(ds, model, scheme)])
    d = {k: z[k] for k in ("subject", "trial_id", "k", "t_rel", "fold")}
    d["y_true"], d["y_pred"], d["targets"] = z["y_true"], z["y_pred"], [str(t) for t in z["targets"]]
    d["rep"] = d["fold"] % nrep(model) if scheme == "trialwise" else np.zeros_like(d["fold"])
    return d


def participant_mean(rows, key):
    """rows: dicts with subject, fold, target, key -> fold mean per target, then mean over x/y/z, per participant."""
    df = pd.DataFrame(rows)
    return df.groupby(["subject", "target"])[key].mean().groupby("subject").mean()


def ci_str(x):
    lo, hi = bootstrap_ci(np.asarray(x))
    return lo, hi


# ---------------------------------------------------------------------------------------------- B4
def b4(preds, e1):
    data = Data("data/cache", S)
    out, support = [], []
    for ds in ("force", "kin"):
        for scheme in ("trialwise", "subjectwise"):
            for model in ["template"] + EEG:
                z = load_pred(preds, ds, model, scheme)
                rows_tb, rows_ci, rows_pool = [], [], []
                for f in np.unique(z["fold"]):
                    mf = z["fold"] == f
                    s = int(z["subject"][mf][0])
                    for j, t in enumerate(z["targets"]):
                        y, p, tid, tr = z["y_true"][mf, j], z["y_pred"][mf, j], z["trial_id"][mf], z["t_rel"][mf]
                        per_trial = [r2(y[tid == k], p[tid == k]) for k in np.unique(tid)]
                        rows_tb.append({"subject": s, "fold": f, "target": t, "v": float(np.mean(per_trial)),
                                        "v_median": float(np.median(per_trial))})
                        ci = (tr >= 0.5 - 1e-9) & (tr <= 4.8 + 1e-9)  # LED - 0.3 s to LED + 4.0 s (LED at 0.8 s)
                        rows_ci.append({"subject": s, "fold": f, "target": t, "v": r2(y[ci], p[ci])})
                        rows_pool.append({"subject": s, "fold": f, "target": t, "v": r2(y, p)})
                tb = participant_mean(rows_tb, "v")
                tbm = participant_mean(rows_tb, "v_median")
                cm = participant_mean(rows_ci, "v")
                pm = participant_mean(rows_pool, "v")
                out.append({"dataset": ds, "scheme": scheme, "model": model, "pooled_r2_paper": pm.mean(),
                            "trial_balanced_r2": tb.mean(), "trial_balanced_ci": ci_str(tb),
                            "trial_median_r2": tbm.mean(), "common_interval_r2": cm.mean(),
                            "common_interval_ci": ci_str(cm)})
        # B4c template support (training trials reaching each 0.5-s bin)
        pool = Pool(data, S, ds, 50, 5)
        lens = {s: dict(zip(data.sd[s].trial_id, data.sd[s].length / 100)) for s in S}
        for scheme in ("trialwise", "subjectwise"):
            for sp in make_splits(pool.w, scheme, S, 0, n_repeats=5):
                tr, _, _ = split_masks(pool.w, sp)
                tids = np.unique(pool.w.trial_id[tr])
                L = np.array([lens[int(t) // 10000][int(t)] for t in tids])
                for b0 in np.arange(0, 18, 0.5):
                    support.append({"dataset": ds, "scheme": scheme, "fold": sp.fold, "bin_start_s": b0,
                                    "n_train_trials": len(L), "n_contributing": int((L >= b0 + 0.25).sum())})
    res = pd.DataFrame(out)
    res.to_csv(OUT / "b4_r2_robustness.csv", index=False)
    sup = pd.DataFrame(support)
    sup["frac"] = sup.n_contributing / sup.n_train_trials
    sup_s = sup.groupby(["dataset", "scheme", "bin_start_s"], as_index=False)[["n_contributing", "n_train_trials", "frac"]].mean()
    sup_s.to_csv(OUT / "b4_template_support.csv", index=False)
    half = sup_s[sup_s.frac < 0.5].groupby(["dataset", "scheme"]).bin_start_s.min()
    shortest = min(min(d.length) for d in data.sd.values()) / 100
    print("B4 robustness:\n", res.round(3).to_string(index=False))
    print("B4c first 0.5-s bin where < 50% of training trials contribute (s after segment start; LED at 0.8 s):\n", half)
    print(f"shortest trial segment: {shortest:.2f} s (common interval ends at 4.8 s)")
    # B4d per axis
    ax = e1[e1.dataset == "kin"]
    ax.to_csv(OUT / "b4_per_axis_per_participant.csv", index=False)
    return res, sup_s, half, shortest


# ---------------------------------------------------------------------------------------------- B5
def b5(preds, e1):
    out = []
    for ds in ("force", "kin"):
        for scheme in ("trialwise", "subjectwise"):
            zE = load_pred(preds, ds, "eegnet", scheme)
            keyE = set(zip(zE["trial_id"].tolist(), np.round(zE["t_rel"], 3).tolist(), zE["rep"].tolist()))
            scores = {}
            for model in EEG:
                z = load_pred(preds, ds, model, scheme)
                sel = (z["rep"] < 3) & (np.round((z["t_rel"] - 0.49) / 0.05).astype(int) % 2 == 0)
                if model != "eegnet":
                    keyM = set(zip(z["trial_id"][sel].tolist(), np.round(z["t_rel"][sel], 3).tolist(), z["rep"][sel].tolist()))
                    assert keyM == keyE, f"grid mismatch {ds} {scheme} {model}"
                rows = []
                for f in np.unique(z["fold"][sel]):
                    m = sel & (z["fold"] == f)
                    for j, t in enumerate(z["targets"]):
                        rows.append({"subject": int(z["subject"][m][0]), "fold": f, "target": t,
                                     "v": r2(z["y_true"][m, j], z["y_pred"][m, j])})
                scores[model] = participant_mean(rows, "v")
            for model in EEG:
                pap = e1[(e1.dataset == ds) & (e1.model == model) & (e1.scheme == scheme)]
                pap = pap.groupby(["subject", "target"]).r2.mean().groupby("subject").mean()
                d = (scores["eegnet"] - scores[model]).values
                out.append({"dataset": ds, "scheme": scheme, "model": model, "paper_r2": pap.mean(),
                            "common_grid_r2": scores[model].mean(),
                            "eegnet_minus_model": d.mean() if model != "eegnet" else np.nan,
                            "diff_ci": ci_str(d) if model != "eegnet" else None,
                            "wilcoxon_p": wilcoxon(d).pvalue if model != "eegnet" else np.nan,
                            "n_eegnet_better": int((d > 0).sum()) if model != "eegnet" else None})
    res = pd.DataFrame(out)
    res.to_csv(OUT / "b5_common_grid.csv", index=False)
    print("B5 common grid:\n", res.round(4).to_string(index=False))
    return res


# ---------------------------------------------------------------------------------------------- C2
def c2():
    import pymupdf

    out = {}
    for f in ("fig_blocks", "fig_example", "fig_onset", "fig_onset_v1"):
        p = Path("../paper/figures") / f"{f}.pdf"
        if not p.exists():
            continue
        d = pymupdf.open(p)
        w_in = d[0].rect.width / 72
        sizes = [sp["size"] for pg in d for b in pg.get_text("dict")["blocks"] for ln in b.get("lines", [])
                 for sp in ln["spans"] if sp["text"].strip()]
        out[f] = {"width_in": round(w_in, 2), "min_pt_native": round(min(sizes), 2) if sizes else None}
    print("C2 fonts:", out)
    return out


# ---------------------------------------------------------------------------------------------- A3
def a3(e1):
    rel = REL / "results"
    rel.mkdir(parents=True, exist_ok=True)
    data = Data("data/cache", S)
    # fold-level rows of E1 (both runs)
    runs = sorted(p.parent for p in Path("results/e1_splits").glob("*/metrics.json"))
    folds = []
    for run in runs:
        f = pd.read_csv(run / "folds.csv")
        folds.append(f[f.model.isin(set(e1[e1.run == run.name].model))])
    F = pd.concat(folds, ignore_index=True)
    # trial counts per fold and participant from the splits
    counts = {}
    for ds in ("force", "kin"):
        for stride, reps, models in ((5, 5, "default"), (10, 3, "eegnet")):
            pool = Pool(data, S, ds, 50, stride)
            for scheme in ("shuffled", "trialwise", "serieswise", "subjectwise"):
                for sp in make_splits(pool.w, scheme, S, 0, n_repeats=reps):
                    tr, _, te = split_masks(pool.w, sp)
                    s = sp.test_subjects[0]
                    counts[(ds, models, scheme, sp.fold)] = (len(np.unique(pool.w.trial_id[tr])),
                                                             len(np.unique(pool.w.trial_id[te & (pool.w.subject == s)])))
    tpl = F[F.model == "template"].set_index(["dataset", "scheme", "fold", "subject", "target"]).r2
    e7 = pd.read_csv(latest_run("e7_residual") / "folds.csv")
    e7["gain"] = e7.r2_combined - e7.r2_template
    e7i = e7.set_index(["dataset", "model", "scheme", "fold", "subject", "target"]).gain
    rows = []
    for _, r in F.iterrows():
        key = (r.dataset, "eegnet" if r.model == "eegnet" else "default", r.scheme, r.fold)
        ntr, nte = counts.get(key, (np.nan, np.nan))
        rep = re.search(r"/[rf](\d+)$", r.fold)
        rows.append({"subject": r.subject, "split": r.scheme, "fold": r.fold, "repeat": int(rep.group(1)) if rep else 0,
                     "target": "force" if r.target == "force" else "position", "axis": AX[r.target], "model": r.model,
                     "seed": fold_seed(0, r.model, r.fold), "n_train_trials": ntr, "n_test_trials": nte,
                     "n_test_windows": r.n_test, "R2": r.r2, "pearson_r": r.r,
                     "template_R2": tpl.get((r.dataset, r.scheme, r.fold, r.subject, r.target), np.nan),
                     "residual_gain": e7i.get((r.dataset, r.model, r.scheme, r.fold, r.subject, r.target), np.nan)})
    pd.DataFrame(rows).to_csv(rel / "results_by_subject.csv", index=False)
    print("results_by_subject.csv rows:", len(rows))

    # splits.json
    sj = {"note": "trial_id = participant*10000 + series*100 + lift; seed 0", "splits": []}
    for ds in ("force", "kin"):
        pool = Pool(data, S, ds, 50, 5)
        schemes = [(sc, make_splits(pool.w, sc, S, 0, n_repeats=5)) for sc in ("shuffled", "trialwise", "serieswise", "subjectwise")]
        schemes.append(("blocked", [blocked(pool.w, s, 0, b, 0) for b in (0.05, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0) for s in S]))
        for sc, sps in schemes:
            for sp in sps:
                tr, va, te = split_masks(pool.w, sp)
                part = {}
                for name, m in (("train", tr), ("validation", va), ("test", te)):
                    tids = np.unique(pool.w.trial_id[m])
                    part[name] = {"participants": sorted({int(t) // 10000 for t in tids}),
                                  "series": sorted({f"{int(t) // 10000}-{(int(t) // 100) % 100}" for t in tids}),
                                  "trials": [int(t) for t in tids]}
                sh = len(set(part["train"]["trials"]) & set(part["test"]["trials"]))
                shp = len(set(part["train"]["participants"]) & set(part["test"]["participants"]))
                honest = sc in ("trialwise", "serieswise", "subjectwise")
                sj["splits"].append({"dataset": ds, "scheme": sc, "fold": sp.fold, "block_s": sp.block_s or None,
                                     **part, "shared_trials_train_test": sh, "shared_participants_train_test": shp,
                                     "assertion_no_trial_in_two_partitions": (sh == 0) if honest else "not applicable (window-level, leaky by design)",
                                     "assertion_no_test_participant_in_train": (shp == 0) if sc == "subjectwise" else "not applicable"})
    (rel / "splits.json").write_text(json.dumps(sj))
    print("splits.json entries:", len(sj["splits"]))

    # stats families
    fams = []
    k = e1[e1.dataset == "kin"].groupby(["model", "scheme", "subject"], as_index=False).r2.mean().assign(dataset="kin")
    fam = pd.concat([e1[e1.dataset == "force"][["dataset", "model", "scheme", "subject", "r2"]], k])
    tmp = []
    for (ds, m), g in fam.groupby(["dataset", "model"]):
        piv = g.pivot(index="subject", columns="scheme", values="r2")
        for a, b in (("shuffled", "trialwise"), ("trialwise", "subjectwise")):
            tmp.append((f"{ds}/{m}: {a} vs {b}", wilcoxon(piv[a], piv[b]).pvalue))
    hp = holm([p for _, p in tmp])
    fams += [{"family": "split inflation (Table I text)", "family_size": len(tmp), "contrast": c, "raw_p": p, "holm_p": h,
              "n": 12, "side": "two-sided"} for (c, p), h in zip(tmp, hp)]
    for fname, path, ccol in (("artifact ablation (E5)", latest_run("e5_artifacts") / "tests.csv", ("model", "target")),
                              ("leaky selection (E2)", latest_run("e2_selection") / "tests.csv", ("model", "scheme", "target"))):
        t = pd.read_csv(path)
        h = holm(t.p.fillna(1).tolist())
        fams += [{"family": fname, "family_size": len(t), "contrast": "/".join(str(r[c]) for c in ccol) + ": variant vs reference",
                  "raw_p": r.p, "holm_p": hh, "n": int(r.n), "side": "two-sided"} for (_, r), hh in zip(t.iterrows(), h)]
    t7 = pd.read_csv(latest_run("e7_residual") / "summary.csv")
    h = holm(t7.p.tolist())
    fams += [{"family": "template + EEG (Table II)", "family_size": len(t7),
              "contrast": f"{r.dataset}/{r.model}/{r.scheme}: template+EEG vs template", "raw_p": r.p, "holm_p": hh, "n": 12,
              "side": "two-sided"} for (_, r), hh in zip(t7.iterrows(), h)]
    pd.DataFrame(fams).to_csv(rel / "stats_families.csv", index=False)
    print("stats_families rows:", len(fams))

    # environment
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    freeze = "\n".join(line for line in freeze.splitlines() if not line.startswith("-e "))
    import braindecode
    import mne
    import scipy
    import sklearn
    import torch

    env = (f"python {platform.python_version()}\nplatform {platform.system()} {platform.release()}\n"
           f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'} (6 GB)\n"
           f"torch {torch.__version__} (CUDA {torch.version.cuda})\nbraindecode {braindecode.__version__}\n"
           f"mne {mne.__version__}\nscipy {scipy.__version__}\nscikit-learn {sklearn.__version__}\n\n# pip freeze\n{freeze}\n")
    (rel / "environment.txt").write_text(env)
    (rel / "README_reproduce.md").write_text(REPRO, encoding="utf-8")


REPRO = """# Reproduce every table (one command each)

Setup (once): `make setup && make data && make preprocess && make test`
(WAY-EEG-GAL is downloaded from its original figshare record; ~10 GB.)

| Paper item | Command | Output |
|---|---|---|
| Table I (split schemes), Fig. headline | `make e1 && make tables` | `results/e1_splits/*/per_subject.csv`, `results/tables/t1_e1_r2.tex` |
| Inflation statistics (Wilcoxon, Holm) | `make tables` | `results/tables/t2_inflation.csv` |
| Table II (template + EEG) | `make e7` | `results/e7_residual/*/summary.csv` |
| Table III (onset) | `make e4` | `results/e4_onset/*/summary.csv` |
| Empirical chance | `make e3` | `results/e3_chance/*/summary.csv` |
| Leaky selection | `make e2` | `results/e2_selection/*/tests.csv` |
| Artifact channels | `make e5` | `results/e5_artifacts/*/tests.csv` |
| Blocked CV figure | `make e6` | `results/e6_blocks/*/per_subject.csv` |
| Verification numbers (round 1/2) | `python scripts/verify_paper_claims.py && python scripts/verify_round2.py` | `results/verification/` |
| Round-3 analyses | `python scripts/round3/b1_embargo.py`, `python scripts/round3/onset_round3.py`, `python scripts/round3/b2_b7_filtering_mlp.py`, `python scripts/round3/analysis_round3.py` | `results/round3/` |
| Check every printed number | `python scripts/check_paper_numbers.py` | `results/round3/paper_number_check.csv` |

All runs use seed 0; per-fold seeds are derived deterministically (see `src/gaudit/experiments.py::fold_seed`).
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="b4,b5,c2,a3")
    parts = ap.parse_args().only.split(",")
    e1, preds = load_e1()
    if "b4" in parts:
        b4(preds, e1)
    if "b5" in parts:
        b5(preds, e1)
    if "c2" in parts:
        c2()
    if "a3" in parts:
        a3(e1)
