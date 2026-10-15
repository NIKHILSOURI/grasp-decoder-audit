"""LaTeX booktabs tables (results/tables/*.tex) and a Markdown digest (results/RESULTS.md)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gaudit.eval.stats import bootstrap_ci, holm
from gaudit.experiments import latest_run, load_e1

OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)
SCHEMES = ["shuffled", "trialwise", "serieswise", "subjectwise"]
SCH = {"shuffled": "Shuffled$^\\dagger$", "trialwise": "Trial", "serieswise": "Series", "subjectwise": "Subject"}
MODELS = ["template", "ridge", "mlp", "riemann", "eegnet", "forcemap"]
MLAB = {"template": "No-EEG template", "ridge": "Ridge", "mlp": "MLP", "riemann": "Riemannian",
        "eegnet": "EEGNet", "forcemap": "ForceMap-style"}
md: list[str] = []


def fmt(v, sd=None, d=2):
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:.{d}f}".replace("-", "$-$")
    return s if sd is None else f"{s} $\\pm$ {sd:.{d}f}"


def write(name, body, caption, label, cols):
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
           "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{" + cols + "}\n\\toprule\n" + body
           + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT / f"{name}.tex").write_text(tex)
    print("wrote", name)


def kin_collapse(sub):
    k = sub[sub.dataset == "kin"].groupby(["model", "scheme", "subject"], as_index=False)[["r2", "r", "rmse"]].mean()
    return {"force": sub[sub.dataset == "force"], "kin": k}


def t1_main(sub):
    parts = kin_collapse(sub)
    body = "& \\multicolumn{4}{c}{Grip force $R^2$} & \\multicolumn{4}{c}{Hand position $R^2$ (mean x,y,z)} \\\\\n"
    body += "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nDecoder & " + " & ".join(SCH[s] for s in SCHEMES) * 1
    body += " & " + " & ".join(SCH[s] for s in SCHEMES) + " \\\\\n\\midrule\n"
    md.append("\n## E1: R^2 by split scheme (mean +/- SD over 12 participants)\n")
    md.append("| decoder | " + " | ".join(f"force {s}" for s in SCHEMES) + " | " + " | ".join(f"pos {s}" for s in SCHEMES) + " |")
    md.append("|" + "---|" * 9)
    for m in MODELS:
        cells, mcells = [], []
        for ds in ("force", "kin"):
            d = parts[ds]
            for s in SCHEMES:
                v = d[(d.model == m) & (d.scheme == s)].r2
                cells.append(fmt(v.mean(), v.std(ddof=1)) if len(v) else "--")
                mcells.append(f"{v.mean():+.2f} +/- {v.std(ddof=1):.2f}" if len(v) else "-")
        body += MLAB[m] + " & " + " & ".join(cells) + " \\\\\n"
        if m == "template":
            body += "\\midrule\n"
        md.append(f"| {MLAB[m]} | " + " | ".join(mcells) + " |")
    write("t1_e1_r2", body, "Test $R^2$ (mean $\\pm$ SD over 12 participants) by split scheme. "
          "$^\\dagger$Leaky on purpose: overlapping windows of the same trial in train and test. "
          "The no-EEG template predicts the mean cue-locked target curve of the training trials.",
          "tab:e1", "l" + "c" * 8)
    # Pearson r version
    body = "Decoder & " + " & ".join(SCH[s] for s in SCHEMES) + " & " + " & ".join(SCH[s] for s in SCHEMES) + " \\\\\n\\midrule\n"
    for m in MODELS:
        cells = []
        for ds in ("force", "kin"):
            d = parts[ds]
            for s in SCHEMES:
                v = d[(d.model == m) & (d.scheme == s)].r
                cells.append(fmt(v.mean()) if len(v) else "--")
        body += MLAB[m] + " & " + " & ".join(cells) + " \\\\\n"
    write("t1b_e1_r", body, "Pearson $r$ for the same runs (force: columns 2--5; hand position: 6--9). "
          "Compare with $R^2$ in Table~\\ref{tab:e1}.", "tab:e1r", "l" + "c" * 8)


def t2_inflation(sub):
    parts = kin_collapse(sub)
    from scipy.stats import wilcoxon

    rows = []
    for ds in ("force", "kin"):
        d = parts[ds]
        for m in MODELS:
            piv = d[d.model == m].pivot(index="subject", columns="scheme", values="r2")
            for a, b in (("shuffled", "trialwise"), ("trialwise", "subjectwise")):
                if a in piv and b in piv:
                    diff = (piv[a] - piv[b]).dropna()
                    p = wilcoxon(piv[a], piv[b]).pvalue if len(diff) > 2 and not np.allclose(diff, 0) else np.nan
                    lo, hi = bootstrap_ci(diff.values)
                    rows.append({"dataset": ds, "model": m, "contrast": f"{a} - {b}", "mean": diff.mean(), "lo": lo,
                                 "hi": hi, "p": p, "n_pos": int((diff > 0).sum()), "n": len(diff)})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p.fillna(1).tolist())
    df.to_csv(OUT / "t2_inflation.csv", index=False)
    body = "Target & Decoder & Contrast & $\\Delta R^2$ [95\\% CI] & $n_{>0}$ & $p_\\text{Holm}$ \\\\\n\\midrule\n"
    md.append("\n## Inflation (paired Wilcoxon, Holm-corrected)\n")
    md.append("| target | decoder | contrast | mean dR2 [95% CI] | subjects > 0 | p (Holm) |\n|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        tgt = "Force" if r.dataset == "force" else "Position"
        body += (f"{tgt} & {MLAB[r.model]} & {r.contrast.replace('shuffled', 'shuf.').replace('trialwise', 'trial').replace('subjectwise', 'subj.')} & "
                 f"{fmt(r['mean'])} [{fmt(r.lo)}, {fmt(r.hi)}] & {r.n_pos}/{r.n} & {r.p_holm:.3g} \\\\\n")
        md.append(f"| {tgt} | {MLAB[r.model]} | {r.contrast} | {r['mean']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {r.n_pos}/{r.n} | {r.p_holm:.3g} |")
    write("t2_inflation", body, "Score inflation between split schemes: paired differences in $R^2$ across "
          "participants, bootstrap 95\\% CI (10,000 resamples), Wilcoxon signed-rank $p$ (Holm-corrected).",
          "tab:inflation", "llllcc")


def t3_template(sub):
    parts = kin_collapse(sub)
    rows = []
    for ds in ("force", "kin"):
        d = parts[ds]
        for s in SCHEMES:
            piv = d[d.scheme == s].pivot(index="subject", columns="model", values="r2")
            for m in MODELS[1:]:
                if m in piv and "template" in piv:
                    diff = (piv[m] - piv["template"]).dropna()
                    rows.append({"dataset": ds, "scheme": s, "model": m, "mean": diff.mean(),
                                 "n_pos": int((diff > 0).sum()), "n": len(diff)})
    df = pd.DataFrame(rows)
    body = "Decoder & " + " & ".join(SCH[s] for s in SCHEMES) + " & " + " & ".join(SCH[s] for s in SCHEMES) + " \\\\\n\\midrule\n"
    md.append("\n## EEG decoder minus no-EEG template (R^2; subjects where EEG beats template)\n")
    md.append("| decoder | " + " | ".join(f"force {s}" for s in SCHEMES) + " | " + " | ".join(f"pos {s}" for s in SCHEMES) + " |")
    md.append("|" + "---|" * 9)
    for m in MODELS[1:]:
        cells, mc = [], []
        for ds in ("force", "kin"):
            for s in SCHEMES:
                r = df[(df.dataset == ds) & (df.scheme == s) & (df.model == m)]
                if len(r):
                    r = r.iloc[0]
                    cells.append(f"{fmt(r['mean'])} ({r.n_pos}/{r.n})")
                    mc.append(f"{r['mean']:+.2f} ({r.n_pos}/{r.n})")
                else:
                    cells.append("--")
                    mc.append("-")
        body += MLAB[m] + " & " + " & ".join(cells) + " \\\\\n"
        md.append(f"| {MLAB[m]} | " + " | ".join(mc) + " |")
    write("t3_vs_template", body, "EEG decoder minus the no-EEG cue-locked template ($R^2$), with the number "
          "of participants where the EEG decoder is better.", "tab:template", "l" + "c" * 8)


def t4_chance(e3):
    s = pd.read_csv(e3 / "summary.csv")
    body = "Decoder & Split & Target & $R^2$ & null$_{95}$ & $r$ & $r$ null$_{95}$ & $>$chance ($R^2$) & $>$chance ($r$) \\\\\n\\midrule\n"
    md.append("\n## E3: empirical chance (trial-mismatch null, 100 permutations)\n")
    md.append("| decoder | split | target | R2 | R2 null95 | r | r null95 | frac > chance (R2) | frac > chance (r) | R2 time-reversed |\n|---|---|---|---|---|---|---|---|---|---|")
    for _, r in s.sort_values(["model", "scheme", "target"]).iterrows():
        body += (f"{MLAB[r.model]} & {SCH[r.scheme]} & {r.target.replace('_', ' ')} & {fmt(r.r2_real)} & {fmt(r.r2_null95)} & "
                 f"{fmt(r.r_real)} & {fmt(r.r_null95)} & {r.frac_above_r2 * 100:.0f}\\% & {r.frac_above_r * 100:.0f}\\% \\\\\n")
        md.append(f"| {MLAB[r.model]} | {r.scheme} | {r.target} | {r.r2_real:+.3f} | {r.r2_null95:+.3f} | {r.r_real:.3f} | "
                  f"{r.r_null95:.3f} | {r.frac_above_r2 * 100:.0f}% | {r.frac_above_r * 100:.0f}% | {r.r2_reverse:+.3f} |")
    write("t4_chance", body, "Empirical chance levels: test EEG paired with the target of another test trial of "
          "the same participant (100 permutations). null$_{95}$: 95th percentile of the null, averaged over "
          "participants; last columns: share of participants whose real score exceeds their own null$_{95}$.",
          "tab:chance", "lllcccccc")


def t5_onset(e4):
    s = pd.read_csv(e4 / "summary.csv")
    cue = pd.read_csv(e4 / "cue_timer_baseline.csv")
    body = ("Detector & Channels & Split & AUC & Det. rate & Lead (ms) & Lat. LED (ms) & FA/min & Cue slope \\\\\n\\midrule\n")
    md.append("\n## E4: causal grasp-onset detection (operating point: FA <= 1/min on validation)\n")
    md.append("| detector | channels | split | window AUC | detection rate | median lead vs hand start (ms) | median latency from LED (ms) | FA/min (test) | cue-locking slope |\n|---|---|---|---|---|---|---|---|---|")
    for _, r in s.iterrows():
        body += (f"{'LDA' if r.detector == 'lda' else 'Riemannian'} & {'all' if r.variant == 'all' else 'no occ.'} & "
                 f"{SCH[r.scheme]} & {r.auc:.2f} & {r.det_rate:.2f} & {r.median_lead_ms:.0f} & {r.median_latency_led_ms:.0f} & "
                 f"{r.fa_per_min:.2f} & {r.cue_slope:.2f} \\\\\n")
        md.append(f"| {r.detector} | {r.variant} | {r.scheme} | {r.auc:.3f} | {r.det_rate:.2f} | {r.median_lead_ms:.0f} | "
                  f"{r.median_latency_led_ms:.0f} | {r.fa_per_min:.2f} | {r.cue_slope:.2f} |")
    body += "\\midrule\n"
    body += (f"Cue timer (no EEG) & -- & -- & -- & {cue.det_rate.mean():.2f} & {cue.median_lead_ms.median():.0f} & 100 & 0 & 0 \\\\\n")
    md.append(f"| cue timer (no EEG) | - | - | - | {cue.det_rate.mean():.2f} | {cue.median_lead_ms.median():.0f} | 100 | 0 | 0 |")
    write("t5_onset", body, "Causal grasp-onset detection, threshold chosen on validation data for "
          "$\\leq$1 false alarm per minute. Lead: first detection relative to hand start (negative = before "
          "movement). Cue slope: regression of detection latency on reaction time (1 = tracks movement, "
          "0 = tracks the LED cue). The cue timer fires 100 ms after LED-on without any EEG.", "tab:onset",
          "lllcccccc")


def t6_selection(e2):
    d = pd.read_csv(e2 / "per_subject.csv")
    body = "Decoder & Split & Target & select on train & select on all & $\\Delta R^2$ & $n_{>0}$ \\\\\n\\midrule\n"
    md.append("\n## E2: leaky feature selection (top 16 by |r|)\n")
    md.append("| decoder | split | target | R2 select-on-train | R2 select-on-all | gain | subjects > 0 |\n|---|---|---|---|---|---|---|")
    for (m, sc, t), g in d.groupby(["model", "scheme", "target"]):
        piv = g.pivot(index="subject", columns="select_on", values="r2")
        diff = piv["all"] - piv["train"]
        body += (f"{MLAB[m]} & {SCH[sc]} & {t.replace('_', ' ')} & {fmt(piv['train'].mean())} & {fmt(piv['all'].mean())} & "
                 f"{fmt(diff.mean(), d=3)} & {(diff > 0).sum()}/{len(diff)} \\\\\n")
        md.append(f"| {MLAB[m]} | {sc} | {t} | {piv['train'].mean():+.3f} | {piv['all'].mean():+.3f} | {diff.mean():+.3f} | {(diff > 0).sum()}/{len(diff)} |")
    write("t6_selection", body, "Leakage factor 2: selecting the 16 features most correlated with the target "
          "on all windows (test included) versus on training windows only.", "tab:selection", "lllcccc")


def t7_artifacts(e5):
    d = pd.read_csv(e5 / "per_subject.csv")
    body = "Decoder & Target & all channels & no Fp/occipital & $\\Delta R^2$ \\\\\n\\midrule\n"
    md.append("\n## E5: artifact-prone channels removed (trial-wise)\n")
    md.append("| decoder | target | R2 all | R2 without Fp1/2, O1/z/2, PO9/10 | change |\n|---|---|---|---|---|")
    for (m, t), g in d.groupby(["model", "target"]):
        piv = g.pivot(index="subject", columns="channels", values="r2")
        diff = piv["no_fp_occ"] - piv["all"]
        body += f"{MLAB[m]} & {t.replace('_', ' ')} & {fmt(piv['all'].mean())} & {fmt(piv['no_fp_occ'].mean())} & {fmt(diff.mean(), d=3)} \\\\\n"
        md.append(f"| {MLAB[m]} | {t} | {piv['all'].mean():+.3f} | {piv['no_fp_occ'].mean():+.3f} | {diff.mean():+.3f} |")
    write("t7_artifacts", body, "Validity check: removing frontal-polar and occipital channels (trial-wise split).",
          "tab:artifacts", "llccc")


def t8_blocks(e6):
    d = pd.read_csv(e6 / "per_subject.csv").groupby(["dataset", "model", "block_s", "subject"], as_index=False).r2.mean()
    blocks = sorted(d.block_s.unique())
    body = "Target & Decoder & " + " & ".join(f"{b:g}\\,s" for b in blocks) + " \\\\\n\\midrule\n"
    md.append("\n## E6: blocked CV, R^2 versus block length (0.05 s = shuffled windows)\n")
    md.append("| target | decoder | " + " | ".join(f"{b:g}s" for b in blocks) + " |\n|" + "---|" * (len(blocks) + 2))
    for (ds, m), g in d.groupby(["dataset", "model"]):
        v = g.groupby("block_s").r2.mean().reindex(blocks)
        body += ("Force" if ds == "force" else "Position") + f" & {MLAB[m]} & " + " & ".join(fmt(x) for x in v) + " \\\\\n"
        md.append(f"| {ds} | {MLAB[m]} | " + " | ".join(f"{x:+.2f}" for x in v) + " |")
    write("t8_blocks", body, "Blocked cross-validation within trials: contiguous blocks of the given length go to "
          "train, validation or test (7:2:1). 0.05\\,s blocks reproduce the shuffled-window split.", "tab:blocks",
          "ll" + "c" * len(blocks))


def _try(name):
    try:
        return latest_run(name)
    except AssertionError:
        return None


def main():
    md.append("# Results digest (auto-generated by scripts/make_tables.py)\n")
    try:
        sub, _ = load_e1()
    except AssertionError:
        print("no E1 run yet")
        return
    sub = sub[sub.model.isin(MODELS)]
    md.append(f"E1 runs: {', '.join(sorted(sub.run.unique()))}")
    t1_main(sub)
    t2_inflation(sub)
    t3_template(sub)
    for name, fn in (("e3_chance", t4_chance), ("e4_onset", t5_onset), ("e2_selection", t6_selection),
                     ("e5_artifacts", t7_artifacts), ("e6_blocks", t8_blocks)):
        run = _try(name)
        if run is not None:
            md.append(f"\n({name}: run {run.name})")
            fn(run)
    timing = {}
    for p in Path("results").glob("e*/*/timing.json"):
        timing[f"{p.parent.parent.name}/{p.parent.name}"] = round(json.loads(p.read_text())["wall_s"] / 60, 1)
    md.append("\n## Compute (wall-clock minutes per run)\n")
    md.extend(f"- {k}: {v} min" for k, v in sorted(timing.items()))
    Path("results/RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote results/RESULTS.md")


if __name__ == "__main__":
    main()
