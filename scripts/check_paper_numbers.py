"""Recompute every number printed in the paper and compare it with the printed value.

    python scripts/check_paper_numbers.py
Printed values live in scripts/paper_numbers_printed.csv (edit it if the paper text changes); each row has a `key`
that selects the recomputation below. A value is OK if it equals the recomputed value at the printed precision
(|printed - recomputed| <= half a unit in the last printed digit). Output: results/round3/paper_number_check.csv
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from gaudit.eval.stats import bootstrap_ci, holm
from gaudit.experiments import latest_run, load_e1

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PRINTED = HERE / "paper_numbers_printed.csv"
OUT = Path("results/round3/paper_number_check.csv")
VER = Path("results/verification")
R3 = Path("results/round3")
MODELS = ["template", "ridge", "mlp", "riemann", "eegnet", "forcemap"]
SCHEMES = ["shuffled", "trialwise", "serieswise", "subjectwise"]

# ------------------------------------------------------------------------------------------ data
e1, _ = load_e1()
force = e1[e1.dataset == "force"].groupby(["model", "scheme", "subject"]).r2.mean()
pos = e1[e1.dataset == "kin"].groupby(["model", "scheme", "subject", "target"]).r2.mean().groupby(["model", "scheme", "subject"]).mean()
r_force = e1[e1.dataset == "force"].groupby(["model", "scheme", "subject"]).r.mean()
r_pos = e1[e1.dataset == "kin"].groupby(["model", "scheme", "subject", "target"]).r.mean().groupby(["model", "scheme", "subject"]).mean()
E = {"force": force, "pos": pos}


def ps(ds, m, sc):
    return E[ds].loc[(m, sc)]


def holm_family():
    rows = []
    for ds in ("force", "pos"):
        for m in MODELS:
            if ds == "pos" and m == "forcemap":
                continue
            for a, b in (("shuffled", "trialwise"), ("trialwise", "subjectwise")):
                x, y = ps(ds, m, a), ps(ds, m, b)
                rows.append(((ds, m, a, b), wilcoxon(x, y).pvalue))
    adj = holm([p for _, p in rows])
    return {k: (p, h) for (k, p), h in zip(rows, adj)}


HF = holm_family()
e7 = pd.read_csv(latest_run("e7_residual") / "per_subject.csv")
e7k = e7[e7.dataset == "kin"].groupby(["model", "scheme", "subject"])[["r2_template", "r2_combined", "r2_residual"]].mean()
e7f = e7[e7.dataset == "force"].set_index(["model", "scheme", "subject"])[["r2_template", "r2_combined", "r2_residual"]]
E7 = {"force": e7f, "pos": e7k}
e4 = pd.read_csv(latest_run("e4_onset") / "summary.csv")
cue = pd.read_csv(latest_run("e4_onset") / "cue_timer_baseline.csv")
e3s = pd.read_csv(latest_run("e3_chance") / "summary.csv")
e3p = pd.read_csv(latest_run("e3_chance") / "per_subject.csv")
e5p = pd.read_csv(latest_run("e5_artifacts") / "per_subject.csv")
e5t = pd.read_csv(latest_run("e5_artifacts") / "tests.csv")
e2p = pd.read_csv(latest_run("e2_selection") / "per_subject.csv")
e6 = pd.read_csv(latest_run("e6_blocks") / "per_subject.csv").groupby(["dataset", "model", "block_s", "subject"]).r2.mean()
ov = pd.read_csv(VER / "a6_blocked_overlap.csv")
nd = pd.read_csv(VER / "a6_near_duplicates.csv")
o1 = pd.read_csv(VER / "round2_o1_purged_blocked.csv")
r1 = pd.read_csv(VER / "round2_r1_series_filtering.csv")
o2 = pd.read_csv(VER / "round2_o2_car25.csv")
out4 = pd.read_csv(VER / "e4_pertrial" / "trial_outcomes.csv")
cue3 = R3 / "cue_timer_trials.csv"


def outcome_share(det, scheme, cat):
    g = out4[(out4.detector == det) & (out4.scheme == scheme)]
    sh = g.groupby(["subject", "fold"]).outcome.apply(lambda x: (x == cat).mean()).groupby("subject").mean()
    return 100 * sh.mean()


def e4row(det, scheme, col, variant="all"):
    return float(e4[(e4.detector == det) & (e4.scheme == scheme) & (e4.variant == variant)][col].iloc[0])


def gain_ci(ds, m, sc):
    g = E7[ds].loc[(m, sc)]
    return list(bootstrap_ci((g.r2_combined - g.r2_template).values))


def beat_template_max():
    mx = 0
    for ds in ("force", "pos"):
        for sc in SCHEMES:
            for m in MODELS[1:]:
                if (m, sc) in E[ds].index.droplevel(2):
                    mx = max(mx, int((ps(ds, m, sc) > ps(ds, "template", sc)).sum()))
    return mx


def mlp_delta():
    d = (ps("force", "mlp", "shuffled") - ps("force", "mlp", "trialwise")).values
    return d.mean(), bootstrap_ci(d)


def strict_cue():
    if cue3.exists():
        c = pd.read_csv(cue3)
        c = c[c.scheme == "trialwise"]
        return c.groupby(["subject", "fold"]).strict_pre.mean().groupby("subject").mean().mean()
    return np.nan


# ------------------------------------------------------------------------------------------ registry
def compute(key: str):
    p = key.split("|")
    k = p[0]
    if k == "T1":  # T1|model|ds|scheme|mean/sd
        x = ps(p[2], p[1], p[3])
        return [x.mean() if p[4] == "mean" else x.std(ddof=1)]
    if k == "T2":  # T2|ds|model|scheme|gain/res/n
        g = E7[p[1]].loc[(p[2], p[3])]
        d = g.r2_combined - g.r2_template
        return [d.mean()] if p[4] == "gain" else [g.r2_residual.mean()] if p[4] == "res" else [int((d > 0).sum())]
    if k == "T3":  # T3|det|scheme|col
        if p[3] == "pre":
            return [outcome_share(p[1], p[2], "before_hs")]
        return [e4row(p[1], p[2], p[3])]
    if k == "T3noocc":
        return [e4row(p[1], p[2], "auc", "no_occipital")]
    if k == "cue_lead":
        return [cue.median_lead_ms.median()]
    if k == "cue_strict":
        return [strict_cue()]
    if k == "cue_loose":
        return [cue.det_rate.mean()]
    if k == "delta":
        return [mlp_delta()[0]]
    if k == "delta_ci":
        return list(mlp_delta()[1])
    if k == "p":  # p|ds|model|a|b  (Holm)
        return [HF[(p[1], p[2], p[3], p[4])][1]]
    if k == "n_pos":
        return [int((ps(p[1], p[2], p[3]) > ps(p[1], p[2], p[4])).sum())]
    if k == "beat_template":
        return [beat_template_max()]
    if k == "blk":  # blk|ds|model|block
        return [e6.loc[(p[1], p[2], float(p[3]))].mean()]
    if k == "overlap_pct":
        return [100 * float(ov[(ov.dataset == p[1]) & (ov.block_s == float(p[2]))].frac_test_overlapping_train_mean.iloc[0])]
    if k == "neardup_pct":
        return [100 * float(nd[(nd.dataset == p[1]) & (nd.block_s == float(p[2]))].frac_near_duplicate.iloc[0])]
    if k == "purged":
        g = o1[(o1.model == p[1]) & (o1.block_s == float(p[2]))]
        return [g.r2.mean()]
    if k == "purged_ridge_range":
        g = o1[o1.model == "ridge"].groupby("block_s").r2.mean()
        return [g.min(), g.max()]
    if k == "purged_train_loss_pct":
        g = o1[(o1.block_s == float(p[1])) & (o1.model == "mlp")]
        return [100 * (1 - g.n_train.mean() / g.n_train_unpurged.mean())]
    if k == "filt":  # filt|ds|diff/new
        a = r1[(r1.dataset == p[1]) & (r1.filtering == "per-trial (paper)")].ridge_r2.iloc[0]
        b = r1[(r1.dataset == p[1]) & (r1.filtering == "whole-series")].ridge_r2.iloc[0]
        return [b - a] if p[2] == "diff" else [b]
    if k == "chance_n":  # chance_n|model|scheme|target
        g = e3p[(e3p.model == p[1]) & (e3p.scheme == p[2]) & (e3p.target == p[3])]
        return [int(g.above_r2.sum())]
    if k == "chance_real_null":
        g = e3s[(e3s.model == p[1]) & (e3s.scheme == p[2]) & (e3s.target == p[3])].iloc[0]
        return [g.r2_real, g.r2_null95]
    if k == "chance_subject_range":
        g = e3p[(e3p.model.isin(["eegnet", "ridge"])) & (e3p.scheme == "subjectwise") & (e3p.target.isin(["force", "pos_z"]))]
        c = g.groupby(["model", "target"]).above_r2.sum()
        return [int(c.min()), int(c.max())]
    if k == "r_subject_range":
        vals = [r_force.loc[(m, "subjectwise")].mean() for m in ["ridge", "mlp", "riemann", "eegnet", "forcemap"]]
        vals += [r_pos.loc[(m, "subjectwise")].mean() for m in ["ridge", "mlp", "riemann", "eegnet"]]
        return [min(vals), max(vals)]
    if k == "r_null_max":  # r_null_max|eeg/template
        g = e3s[e3s.model == "template"] if p[1] == "template" else e3s[e3s.model != "template"]
        return [g.r_null95.max()]
    if k == "ablation_range":
        piv = e5p.pivot_table(index=["model", "target", "subject"], columns="channels", values="r2")
        d = (piv["all"] - piv["no_fp_occ"]).groupby(["model", "target"]).mean()
        return [d.min(), d.max()]
    if k == "car25_range":
        return [-o2.change_CAR25_vs_all.max(), -o2.change_CAR25_vs_all.min()]
    if k == "ablation_pmax":
        return [e5t.p.max()]
    if k == "selection_max":
        piv = e2p.pivot_table(index=["model", "scheme", "target", "subject"], columns="select_on", values="r2")
        return [(piv["all"] - piv["train"]).groupby(["model", "scheme", "target"]).mean().max()]
    if k == "pre_range":
        v = [outcome_share(d, s, "before_hs") for d in ("lda", "riemann") for s in ("trialwise", "subjectwise")]
        return [min(v), max(v)]
    if k == "after_range":
        v = [outcome_share(d, s, "after_hs_only") for d in ("lda", "riemann") for s in ("trialwise", "subjectwise")]
        return [min(v), max(v)]
    if k == "gain_ci":
        return gain_ci(p[1], p[2], p[3])
    if k == "template_pct":
        return [100 * ps(p[1], "template", "trialwise").mean()]
    raise KeyError(key)


DEFAULT = []


def add(loc, qty, key, printed):
    DEFAULT.append({"location": loc, "quantity": qty, "key": key, "printed": printed})


T1 = {"template": ["0.53 0.09", "0.52 0.09", "0.51 0.12", "0.19 0.33", "0.79 0.10", "0.79 0.10", "0.78 0.12", "0.55 0.16"],
      "ridge": ["0.35 0.07", "0.26 0.12", "0.24 0.09", "-0.02 0.23", "0.39 0.11", "0.35 0.09", "0.31 0.08", "0.06 0.19"],
      "mlp": ["0.49 0.03", "0.05 0.03", "0.04 0.04", "-0.14 0.19", "0.40 0.06", "0.17 0.07", "0.15 0.07", "-0.01 0.09"],
      "riemann": ["0.38 0.05", "0.18 0.08", "0.14 0.07", "-0.23 0.32", "0.43 0.08", "0.32 0.10", "0.29 0.09", "-0.05 0.15"],
      "eegnet": ["0.26 0.10", "0.17 0.10", "0.08 0.06", "-0.18 0.33", "0.44 0.09", "0.39 0.09", "0.34 0.09", "0.02 0.17"],
      "forcemap": ["0.14 0.03", "0.02 0.04", "0.00 0.06", "-0.10 0.17"]}
for m, cells in T1.items():
    for i, c in enumerate(cells):
        ds, sc = ("force" if i < 4 else "pos"), SCHEMES[i % 4]
        mean, sd = c.split()
        add("Table I", f"{m} {ds} {sc} mean", f"T1|{m}|{ds}|{sc}|mean", mean)
        add("Table I", f"{m} {ds} {sc} SD", f"T1|{m}|{ds}|{sc}|sd", sd)
T2 = [("force", "ridge", "+0.001 -0.03 7", "-0.022 -0.19 5"), ("force", "mlp", "-0.005 -0.03 3", "-0.023 -0.21 3"),
      ("pos", "ridge", "+0.015 0.06 12", "+0.012 -0.11 9"), ("pos", "mlp", "+0.006 0.02 9", "-0.025 -0.22 0")]
for ds, m, tw, sw in T2:
    for sc, cells in (("trialwise", tw), ("subjectwise", sw)):
        g, r, n = cells.split()
        add("Table II", f"{ds} {m} {sc} gain", f"T2|{ds}|{m}|{sc}|gain", g)
        add("Table II", f"{ds} {m} {sc} residual R2", f"T2|{ds}|{m}|{sc}|res", r)
        add("Table II", f"{ds} {m} {sc} n improved", f"T2|{ds}|{m}|{sc}|n", n)
T3 = [("lda", "trialwise", "0.83", "13.9", "0.19", "-50", "1.2", "0.35"), ("lda", "subjectwise", "0.71", "4.7", "0.08", "-16", "1.2", "0.30"),
      ("riemann", "trialwise", "0.78", "21.5", "0.29", "-113", "3.0", "0.20"), ("riemann", "subjectwise", "0.68", "10.0", "0.13", "-186", "7.6", "0.35")]
for d, sc, auc, pre, loose, lead, fa, slope in T3:
    for col, val in (("auc", auc), ("pre", pre), ("det_rate", loose), ("median_lead_ms", lead), ("fa_per_min", fa), ("cue_slope", slope)):
        add("Table III", f"{d} {sc} {col}", f"T3|{d}|{sc}|{col}", val)
add("Table III", "cue timer lead (ms)", "cue_lead", "-315")
add("Table III", "cue timer strict Pre (\\CUEPRE)", "cue_strict", "1.00")
add("Table III", "cue timer loose", "cue_loose", "1.00")
add("Results", "LDA AUC without occipital, trial", "T3noocc|lda|trialwise", "0.82")
add("Results", "LDA AUC without occipital, subject", "T3noocc|lda|subjectwise", "0.71")
add("Results", "MLP force shuffled - trialwise", "delta", "0.43")
add("Results", "its 95% CI", "delta_ci", "0.41 0.45")
add("Results", "Holm p, 12/12 contrasts (MLP force shuffled vs trial)", "p|force|mlp|shuffled|trialwise", "0.011")
add("Results", "Holm p, EEGNet force shuffled vs trial", "p|force|eegnet|shuffled|trialwise", "0.16")
add("Results", "EEGNet force participants shuffled > trial", "n_pos|force|eegnet|shuffled|trialwise", "10")
add("Results", "Holm p, template force shuffled vs trial", "p|force|template|shuffled|trialwise", "0.68")
add("Results", "Holm p, template position shuffled vs trial", "p|pos|template|shuffled|trialwise", "0.68")
add("Results", "max participants where EEG beats template", "beat_template", "2")
add("Results", "template force variance explained (%)", "template_pct|force", "52")
add("Results", "template position variance explained (%)", "template_pct|pos", "79")
for qty, key, val in (("MLP force, 0.05-s blocks", "blk|force|mlp|0.05", "0.49"), ("MLP force, 0.25-s blocks", "blk|force|mlp|0.25", "0.08"),
                      ("Riemannian force, 0.05-s", "blk|force|riemann|0.05", "0.38"), ("Riemannian force, 8-s", "blk|force|riemann|8.0", "0.18"),
                      ("ridge force, 0.05-s", "blk|force|ridge|0.05", "0.35"), ("ridge force, 8-s", "blk|force|ridge|8.0", "0.30"),
                      ("test windows sharing samples at 0.25 s (%)", "overlap_pct|force|0.25", "98"),
                      ("near-duplicates at 0.05 s (%)", "neardup_pct|force|0.05", "91"), ("near-duplicates at 0.25 s (%)", "neardup_pct|force|0.25", "28"),
                      ("test windows sharing samples at 8 s (%)", "overlap_pct|force|8.0", "3")):
    add("Blocked CV", qty, key, val)
for qty, key, val in (("purged MLP 0.25 s", "purged|mlp|0.25", "-0.03"), ("purged MLP 1 s", "purged|mlp|1.0", "0.06"),
                      ("purged MLP 4 s", "purged|mlp|4.0", "0.06"), ("purged ridge range", "purged_ridge_range", "0.29 0.30"),
                      ("training windows removed at 0.25 s (%)", "purged_train_loss_pct|0.25", "54")):
    add("Purged blocked CV", qty, key, val)
for qty, key, val in (("ridge force change", "filt|force|diff", "+0.09"), ("ridge position change", "filt|kin|diff", "+0.12"),
                      ("ridge force whole-series", "filt|force|new", "0.35"), ("ridge position whole-series", "filt|kin|new", "0.47")):
    add("Filtering sensitivity", qty, key, val)
for qty, key, val in (("ridge trial force above chance", "chance_n|ridge|trialwise|force", "12"),
                      ("EEGNet trial force above chance", "chance_n|eegnet|trialwise|force", "11"),
                      ("ridge trial force real vs null95", "chance_real_null|ridge|trialwise|force", "0.26 0.20"),
                      ("subject-wise above chance range", "chance_subject_range", "7 11"),
                      ("subject-wise Pearson r range", "r_subject_range", "0.12 0.40"),
                      ("max null95 r, EEG decoders", "r_null_max|eeg", "0.57"), ("max null95 r, template", "r_null_max|template", "0.91")):
    add("Target mismatch", qty, key, val)
for qty, key, val in (("ablation drop range (CAR 32)", "ablation_range", "0.03 0.06"), ("ablation drop range (CAR 25)", "car25_range", "0.03 0.07"),
                      ("ablation max p", "ablation_pmax", "0.016"), ("max selection gain", "selection_max", "0.017"),
                      ("strict pre-onset range (%)", "pre_range", "4.7 21.5"), ("detected only after hand start range (%)", "after_range", "2.8 7.3"),
                      ("Table II CI position ridge trial", "gain_ci|pos|ridge|trialwise", "0.008 0.025"),
                      ("Table II CI position ridge subject", "gain_ci|pos|ridge|subjectwise", "0.001 0.021")):
    add("Other", qty, key, val)


def decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


def main():
    if not PRINTED.exists():
        pd.DataFrame(DEFAULT).to_csv(PRINTED, index=False)
    pr = pd.read_csv(PRINTED, dtype=str)
    rows = []
    for _, r in pr.iterrows():
        toks = re.findall(r"[+-]?\d+(?:\.\d+)?", r.printed)
        try:
            rec = compute(r.key)
            ok = len(rec) == len(toks) and all(
                np.isfinite(v) and abs(float(t) - v) <= 0.5 * 10 ** -decimals(t) + 1e-9 for t, v in zip(toks, rec))
            status = "OK" if ok else "DOES NOT REPRODUCE"
            rec_s = " ".join(f"{v:.4f}" for v in rec)
        except Exception as ex:  # missing input file etc.
            status, rec_s = f"NOT CHECKED ({type(ex).__name__}: {ex})", ""
        rows.append({"location": r.location, "quantity": r.quantity, "printed": r.printed, "recomputed": rec_s, "status": status})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.status.value_counts().to_string())
    bad = df[df.status != "OK"]
    if len(bad):
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
