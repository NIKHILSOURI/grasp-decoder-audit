"""Summaries for A1 (cue timer), B3 (onset controls), B6 (variability, FA budgets) and the E4 reproduction check.

    python scripts/round3/onset_summary.py   (after onset_round3.py)
Participant = unit of inference: rates are computed per fold, averaged over folds within a participant, then
summarized over the 12 participants (bootstrap 95% CI, 10,000 resamples, seed 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gaudit.eval.stats import bootstrap_ci
from gaudit.experiments import latest_run

R3 = "results/round3/"
tr = pd.read_csv(R3 + "onset_trials.csv.gz")
folds = pd.read_csv(R3 + "onset_folds.csv")
cue = pd.read_csv(R3 + "cue_timer_trials.csv")


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, hi = bootstrap_ci(x)
    return f"{x.mean():.3f} [{lo:.3f}, {hi:.3f}]"


def per_participant(g):
    """g: per-trial rows at one threshold -> per participant strict, loose, FA/min, detected count, median lead."""
    f = g.groupby(["subject", "fold"]).agg(strict=("strict_pre", "mean"), loose=("loose_hit", "mean"),
                                           fa=("n_false_alarms", "sum"), rest=("rest_seconds", "sum"),
                                           n_det=("loose_hit", "sum"))
    f["fa_min"] = f.fa / (f.rest / 60)
    p = f.groupby("subject").agg(strict=("strict", "mean"), loose=("loose", "mean"), fa_min=("fa_min", "mean"),
                                 n_det=("n_det", "sum"))
    det = g[g.first_detection_time.notna()]
    p["lead_med"] = ((det.first_detection_time - det.hand_start) * 1000).groupby(det.subject).median()
    return p


def at_op(det, scheme):
    g = tr[(tr.detector == det) & (tr.scheme == scheme)]
    return g[g.threshold == g.operating_threshold]


def main():
    lines = []
    # ------------------------------------------------------------------ A1
    a1 = []
    for scheme in ("trialwise", "subjectwise"):
        c = cue[cue.scheme == scheme]
        pp = c.groupby(["subject", "fold"])[["strict_pre", "loose_hit"]].mean().groupby("subject").mean()
        q = np.percentile(c.lead_ms, [25, 50, 75])
        a1.append({"scheme": scheme, "strict_pre_rate": pp.strict_pre.mean(), "loose_rate": pp.loose_hit.mean(),
                   "n_test_trial_evaluations": len(c), "n_unique_trials": c.trial_id.nunique(),
                   "n_failing_strict": int((c.strict_pre == 0).sum()), "n_failing_loose": int((c.loose_hit == 0).sum()),
                   "min_rt_s": c.rt_s.min(), "lead_median_ms": q[1], "lead_q25_ms": q[0], "lead_q75_ms": q[2]})
    a1 = pd.DataFrame(a1)
    a1.to_csv(R3 + "a1_cue_timer.csv", index=False)
    allrt = cue.drop_duplicates("trial_id").rt_s
    lines.append(f"A1 min RT over all 3,528 trials: {allrt.min():.3f} s; trials with RT <= 0.1 s: {(allrt <= 0.1).sum()}; "
                 f"RT < -0.1 s: {(allrt < -0.1).sum()}")
    # ------------------------------------------------------------------ reproduction of Table III
    e4 = pd.read_csv(latest_run("e4_onset") / "summary.csv")
    rep = []
    for det in ("lda", "riemann"):
        for scheme in ("trialwise", "subjectwise"):
            f = folds[(folds.detector == det) & (folds.scheme == scheme)].groupby("subject")[
                ["auc", "det_rate_loose", "fa_per_min", "median_lead_ms"]].mean()
            ref = e4[(e4.detector == det) & (e4.scheme == scheme) & (e4.variant == "all")].iloc[0]
            rep.append({"detector": det, "scheme": scheme, "auc": f.auc.mean(), "auc_paper": ref.auc,
                        "loose": f.det_rate_loose.mean(), "loose_paper": ref.det_rate, "fa": f.fa_per_min.mean(),
                        "fa_paper": ref.fa_per_min, "lead": f.median_lead_ms.median(), "lead_paper": ref.median_lead_ms})
    rep = pd.DataFrame(rep)
    rep.to_csv(R3 + "onset_reproduction_check.csv", index=False)
    # ------------------------------------------------------------------ B3
    b3 = []
    for det in ("lda", "riemann", "lda_timematched", "riemann_timematched", "cue_only", "lda_plus_cue"):
        for scheme in ("trialwise", "subjectwise"):
            p = per_participant(at_op(det, scheme))
            auc = folds[(folds.detector == det) & (folds.scheme == scheme)].groupby("subject").auc.mean()
            rules = folds[(folds.detector == det) & (folds.scheme == scheme)].neg_rule.value_counts().to_dict()
            b3.append({"detector": det, "scheme": scheme, "window_auc": ci(auc), "strict_pre": ci(p.strict),
                       "loose": ci(p.loose), "test_fa_per_min": ci(p.fa_min), "median_lead_ms_detected": ci(p.lead_med),
                       "negative_rule": "; ".join(f"{k} ({v} folds)" for k, v in rules.items())})
    b3 = pd.DataFrame(b3)
    b3.to_csv(R3 + "b3_onset_controls.csv", index=False)
    # ------------------------------------------------------------------ B6
    b6 = []
    for det in ("lda", "riemann"):
        for scheme in ("trialwise", "subjectwise"):
            g = at_op(det, scheme)
            p = per_participant(g)
            d = g[g.first_detection_time.notna()]
            lead = (d.first_detection_time - d.hand_start) * 1000

            def mq(x):
                q = np.nanpercentile(x, [25, 50, 75])
                return f"{q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]"

            b6.append({"detector": det, "scheme": scheme, "strict_median_IQR": mq(p.strict), "loose_median_IQR": mq(p.loose),
                       "fa_median_IQR": mq(p.fa_min), "n_detected_min_med_max": f"{int(p.n_det.min())}/{int(p.n_det.median())}/{int(p.n_det.max())}",
                       "n_detected_total": int(len(d)), "lead_all_detected_median_IQR_ms":
                       f"{np.median(lead):.0f} [{np.percentile(lead, 25):.0f}, {np.percentile(lead, 75):.0f}]"})
    b6 = pd.DataFrame(b6)
    b6.to_csv(R3 + "b6_onset_variability.csv", index=False)
    budget = []
    for det in ("lda", "riemann"):
        for scheme in ("trialwise", "subjectwise"):
            g = tr[(tr.detector == det) & (tr.scheme == scheme)]
            f = g.groupby(["subject", "fold", "threshold"]).agg(strict=("strict_pre", "mean"), fa=("n_false_alarms", "sum"),
                                                                 rest=("rest_seconds", "sum"))
            f["fa_min"] = f.fa / (f.rest / 60)
            pc = f.groupby(["subject", "threshold"])[["strict", "fa_min"]].mean()
            for bud in (0.5, 1, 2, 5, 10):
                vals = []
                for _s, h in pc.groupby(level="subject"):
                    ok = h[h.fa_min <= bud]
                    vals.append(ok.strict.max() if len(ok) else 0.0)
                budget.append({"detector": det, "scheme": scheme, "fa_budget_per_min": bud, "strict_pre": ci(vals)})
    budget = pd.DataFrame(budget)
    budget.to_csv(R3 + "b6_fa_budget.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("A1\n", a1.round(4).to_string(index=False))
        print("\n".join(lines))
        print("\nreproduction\n", rep.round(3).to_string(index=False))
        print("\nB3\n", b3.to_string(index=False))
        print("\nB6\n", b6.to_string(index=False))
        print("\nFA budget\n", budget.pivot_table(index=["detector", "scheme"], columns="fa_budget_per_min", values="strict_pre", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
