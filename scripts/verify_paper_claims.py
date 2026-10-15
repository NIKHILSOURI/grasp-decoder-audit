"""Recompute every number questioned in the paper's verification checklist (no results are changed).

    python scripts/verify_paper_claims.py   ->  prints a report and writes results/verification/*.csv
"""

from __future__ import annotations

import json
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import wilcoxon

from gaudit.data.splits import blocked, make_splits, split_masks
from gaudit.eval.stats import bootstrap_ci, holm
from gaudit.experiments import latest_run, load_e1
from gaudit.pipeline import Data, Pool

warnings.filterwarnings("ignore")
OUT = Path("results/verification")
OUT.mkdir(parents=True, exist_ok=True)
S = list(range(1, 13))


def hdr(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


# ---------------------------------------------------------------------------------------------- A3
hdr("A3  MLP force: shuffled - trial-wise (paired, unrounded)")
sub, _ = load_e1()
f = sub[(sub.dataset == "force") & (sub.model == "mlp")].pivot(index="subject", columns="scheme", values="r2")
d = (f.shuffled - f.trialwise).values
print("per-participant shuffled:", np.round(f.shuffled.values, 4))
print("per-participant trialwise:", np.round(f.trialwise.values, 4))
print(f"mean shuffled {f.shuffled.mean():.4f}  mean trialwise {f.trialwise.mean():.4f}")
print(f"paired mean difference {d.mean():.4f}; bootstrap 95% CI over participants {np.round(bootstrap_ci(d), 4)} "
      f"(10,000 resamples of the 12 per-participant differences)")

# ---------------------------------------------------------------------------------------------- A4
hdr("A4  Wilcoxon family used for Holm correction in the inflation table (t2)")
k = sub[sub.dataset == "kin"].groupby(["model", "scheme", "subject"], as_index=False)[["r2"]].mean().assign(dataset="kin")
fam = pd.concat([sub[sub.dataset == "force"][["dataset", "model", "scheme", "subject", "r2"]], k])
rows = []
for (ds, m), g in fam.groupby(["dataset", "model"]):
    piv = g.pivot(index="subject", columns="scheme", values="r2")
    for a, b in (("shuffled", "trialwise"), ("trialwise", "subjectwise")):
        res = wilcoxon(piv[a], piv[b], zero_method="wilcox", alternative="two-sided")
        rows.append({"dataset": ds, "model": m, "contrast": f"{a}-{b}", "n_pos": int((piv[a] > piv[b]).sum()),
                     "p_raw": res.pvalue})
fam_df = pd.DataFrame(rows)
fam_df["p_holm"] = holm(fam_df.p_raw.tolist())
print(f"family size = {len(fam_df)} tests")
print(fam_df.round(5).to_string(index=False))
fam_df.to_csv(OUT / "a4_holm_family.csv", index=False)
print("\nexact two-sided Wilcoxon p for 12/12 positive:", wilcoxon(np.arange(1, 13), np.zeros(12)).pvalue)

e5 = latest_run("e5_artifacts")
t5 = pd.read_csv(e5 / "tests.csv")
t5["p_holm"] = holm(t5.p.tolist())
print("\nE5 artifact ablation (paired_wilcoxon, raw p; Holm over these 4 shown alongside):")
print(t5[["model", "target", "mean_diff", "p", "p_holm"]].round(5).to_string(index=False))

e7 = latest_run("e7_residual")
t7 = pd.read_csv(e7 / "summary.csv")
t7["p_holm8"] = holm(t7.p.tolist())
print("\nE7 Table II stars (raw p from paired Wilcoxon; Holm over the 8 rows shown alongside):")
print(t7[["dataset", "model", "scheme", "gain", "n_better", "p", "p_holm8"]].round(5).to_string(index=False))

# ---------------------------------------------------------------------------------------------- A6
hdr("A6  Blocked CV: no gap; train/test sizes; fraction of test windows overlapping a training window")
e6 = latest_run("e6_blocks")
f6 = pd.read_csv(e6 / "folds.csv")
sizes = f6.groupby(["dataset", "block_s"]).agg(n_train_mean=("n_train", "mean"), n_train_min=("n_train", "min"),
                                               n_train_max=("n_train", "max"), n_test_mean=("n_test", "mean"))
# n_train is per fold (pool of 1 participant); n_test is per participant and target row -> same per fold
print("training / test windows per participant-fold (mean, min, max over the 12 participants):")
print(sizes.round(0).to_string())

data = Data("data/cache", S)
ov_rows = []
for ds in ("force", "kin"):
    pool = Pool(data, S, ds, 50, 5)
    w = pool.w
    for b in sorted(f6.block_s.unique()):
        fr = []
        for s in S:
            sp = blocked(w, s, 0, float(b), 0)  # identical to the E6 split (seed 0, repeat 0)
            tr, _, te = split_masks(w, sp)
            # a 500-ms window covers samples [end-49, end] (100 Hz); two windows of the same trial
            # share at least one sample iff |end_i - end_j| <= 49
            hit = np.zeros(te.sum(), bool)
            te_idx = np.flatnonzero(te)
            for tid in np.unique(w.trial_id[te]):
                mt = te_idx[w.trial_id[te_idx] == tid]
                tr_e = np.sort(w.end[tr & (w.trial_id == tid)])
                if tr_e.size == 0:
                    continue
                e = w.end[mt]
                j = np.searchsorted(tr_e, e)
                dlo = np.abs(e - tr_e[np.clip(j - 1, 0, tr_e.size - 1)])
                dhi = np.abs(tr_e[np.clip(j, 0, tr_e.size - 1)] - e)
                hit[np.searchsorted(te_idx, mt)] = np.minimum(dlo, dhi) <= 49
            fr.append(hit.mean())
        ov_rows.append({"dataset": ds, "block_s": b, "frac_test_overlapping_train_mean": float(np.mean(fr)),
                        "min": float(np.min(fr)), "max": float(np.max(fr))})
ov = pd.DataFrame(ov_rows)
print("\nfraction of test windows sharing >= 1 EEG sample with a training window (mean over participants):")
print(ov.round(3).to_string(index=False))
ov.to_csv(OUT / "a6_blocked_overlap.csv", index=False)
sizes.to_csv(OUT / "a6_blocked_sizes.csv")
for ds in ("force", "kin"):  # shuffled split for comparison
    pool = Pool(data, S, ds, 50, 5)
    fr = []
    for sp in make_splits(pool.w, "shuffled", S, 0, n_repeats=1):
        tr, _, te = split_masks(pool.w, sp)
        w = pool.w
        hit = []
        for i in np.flatnonzero(te)[:2000]:
            same = tr & (w.trial_id == w.trial_id[i])
            hit.append(np.any(np.abs(w.end[same] - w.end[i]) <= 49))
        fr.append(np.mean(hit))
    print(f"shuffled split ({ds}): fraction of test windows overlapping a training window = {np.mean(fr):.3f}")

# ---------------------------------------------------------------------------------------------- A10
hdr("A10 Dataset counts")
for ds in ("force", "kin"):
    pool = Pool(data, S, ds, 50, 5)
    print(f"{ds}: {len(pool.w)} windows (stride 50 ms), {len(np.unique(pool.w.trial_id))} trials")
print("trials with targets per participant:", {s: len(data.sd[s].trial_id) for s in S})
rt = {s: round(float(np.median(data.sd[s].ev("tHandStart") - data.sd[s].ev("LEDOn"))) * 1000) for s in S}
print("median reaction time per participant (ms):", rt)
max_ov, gf_res = 0.0, []
for s in S:
    for ser in range(1, 10):
        ws = sio.loadmat(f"data/raw/P{s}/WS_P{s}_S{ser}.mat")["ws"][0, 0]
        win = ws["win"][0]
        st = np.array([w["trial_start_time"].item() for w in win])
        n = np.array([w["eeg"].shape[0] for w in win]) / 500
        start = st - 2.0
        ov_s = (start[:-1] + n[:-1]) - start[1:]
        max_ov = max(max_ov, ov_s.max())
        if ser == 1:
            K = np.concatenate([w["kin"] for w in win])
            coef = np.linalg.lstsq(np.c_[K[:, 16], K[:, 17], np.ones(len(K))], K[:, 41], rcond=None)[0]
            gf_res.append((s, *np.round(coef, 6)))
print(f"maximum overlap between consecutive stored windows: {max_ov:.3f} s")
print("GF ~ a*FZ1 + b*FZ2 + c (series 1 of every participant):", gf_res)

# ---------------------------------------------------------------------------------------------- A12
hdr("A12 Series-wise folds: series per fold and weight conditions in each test fold")
comp = []
for ds in ("force", "kin"):
    pool = Pool(data, S, ds, 50, 5)
    for sp in make_splits(pool.w, "serieswise", S, 0, n_series_folds=3):
        s = sp.test_subjects[0]
        sd = data.sd[s]
        te_ser = sorted(int(x) % 100 for x in sp.test)
        tr_ser = sorted(int(x) % 100 for x in sp.train)
        va_ser = sorted(int(x) % 100 for x in sp.val)
        m = np.isin(sd.series, te_ser) & (sd.block_type == 1 if ds == "force" else True)
        comp.append({"dataset": ds, "subject": s, "fold": sp.fold, "test_series": te_ser, "val_series": va_ser,
                     "train_series": tr_ser, "test_weights": sorted(set(sd.weight[m].tolist())),
                     "test_block_types": sorted(set(sd.block_type[m].tolist())), "n_test_trials": int(m.sum())})
comp = pd.DataFrame(comp)
comp.to_csv(OUT / "a12_series_folds.csv", index=False)
print(comp[comp.subject == 1].to_string(index=False))
print("\nall test folds contain all three weights:",
      bool(comp.test_weights.apply(lambda x: x == [1, 2, 3]).all()))
print("kinematics test folds containing all three block types (weight/friction/mixed):",
      int(comp[comp.dataset == "kin"].test_block_types.apply(lambda x: x == [1, 2, 3]).sum()), "of",
      int((comp.dataset == "kin").sum()))
w_ok = []
for _, r in comp.iterrows():
    sd = data.sd[r.subject]
    base = sd.block_type == 1 if r.dataset == "force" else np.ones(len(sd.series), bool)
    m = np.isin(sd.series, r.test_series) & base
    w_ok.append(np.round(np.bincount(sd.weight[m], minlength=4)[1:] / m.sum(), 2))
comp["weight_share_165_330_660"] = w_ok
print("\nshare of 165/330/660 g trials in each force test fold (participant 1):")
print(comp[(comp.dataset == "force") & (comp.subject == 1)][["fold", "test_series", "weight_share_165_330_660"]].to_string(index=False))

# ---------------------------------------------------------------------------------------------- A13
hdr("A13 Other numbers")
e2 = pd.read_csv(latest_run("e2_selection") / "per_subject.csv")
g = e2.pivot_table(index=["model", "scheme", "target", "subject"], columns="select_on", values="r2").reset_index()
print("max mean selection gain:", round(float((g["all"] - g["train"]).groupby([g.model, g.scheme, g.target]).mean().max()), 4))
k = sub[sub.dataset == "kin"].groupby(["model", "scheme", "subject"], as_index=False).r2.mean().assign(dataset="kin")
allr = pd.concat([sub[sub.dataset == "force"][["dataset", "model", "scheme", "subject", "r2"]], k])
mx = 0
for _key, gg in allr.groupby(["dataset", "scheme"]):
    piv = gg.pivot(index="subject", columns="model", values="r2")
    for m in piv.columns:
        if m != "template":
            mx = max(mx, int((piv[m] > piv["template"]).sum()))
print("max number of participants in which an EEG decoder beats the template (any split, any target):", mx)

# ---------------------------------------------------------------------------------------------- A14
hdr("A14 arXiv comment fields")
ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
x = ET.fromstring(urllib.request.urlopen("http://export.arxiv.org/api/query?id_list=2508.07677,2607.24081", timeout=30).read())
for e in x.findall("a:entry", ns):
    c = e.find("arxiv:comment", ns)
    print(e.find("a:id", ns).text, "| comment:", None if c is None else " ".join(c.text.split()))
print("\nwrote", sorted(p.name for p in OUT.glob("*.csv")))
json.dumps({})
