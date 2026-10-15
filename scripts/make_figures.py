"""Paper figures (PDF + PNG) from the latest completed runs -> results/figures/.

 fig0_split_schemes      how each split scheme assigns windows of a series (schematic, real timing)
 fig1_headline           R2 by split scheme for every decoder, with the no-EEG template (headline)
 fig2_paired_subjects    per-subject paired lines across split schemes
 fig3_null               chance-level null distributions with the real score marked
 fig4_example_trial      true vs predicted force: shuffled vs honest vs template
 fig5_onset              onset detection rate vs false alarms per minute, lead-time distribution
 fig6_traj3d             3D hand trajectories: true vs decoded (honest) vs template
 fig7_blocks3d           3D leakage landscape: R2 vs CV block length and decoder (E6)
 fig8_topomaps           scalp maps of EEG-target correlation (force, height)
 fig9_metric_illusion    Pearson r vs R2 under the null: why r flatters slow targets
 fig10_selection_artifacts  E2 leaky selection gain and E5 artifact-channel check
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from gaudit.eval.stats import bootstrap_ci  # noqa: E402
from gaudit.experiments import latest_run, load_e1  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

SCHEMES = ["shuffled", "trialwise", "serieswise", "subjectwise"]
SCHEME_LABEL = {"shuffled": "Shuffled windows\n(leaky)", "trialwise": "Trial-wise", "serieswise": "Series-wise",
                "subjectwise": "Subject-wise"}
SCHEME_COLOR = {"shuffled": "#c0392b", "trialwise": "#2e86c1", "serieswise": "#17a589", "subjectwise": "#7d3c98"}
MODELS = ["ridge", "mlp", "riemann", "eegnet", "forcemap"]
MODEL_LABEL = {"template": "No-EEG template", "ridge": "Ridge (lagged LF)", "mlp": "MLP (band power)",
               "riemann": "Riemannian", "eegnet": "EEGNet", "forcemap": "ForceMap-style"}
SHORT = {"template": "Template", "ridge": "Ridge", "mlp": "MLP", "riemann": "Riemann.", "eegnet": "EEGNet",
         "forcemap": "ForceMap\n-style"}
MODEL_COLOR = {"ridge": "#1f77b4", "mlp": "#ff7f0e", "riemann": "#2ca02c", "eegnet": "#d62728",
               "forcemap": "#9467bd", "template": "#555555"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def kin_mean(sub: pd.DataFrame) -> pd.DataFrame:
    """Collapse x/y/z into one 'kinematics' score per subject (mean over axes)."""
    k = sub[sub.dataset == "kin"].groupby(["dataset", "model", "scheme", "subject"], as_index=False)[
        ["r2", "r", "rmse"]].mean()
    k["target"] = "pos (x,y,z)"
    return pd.concat([sub[sub.dataset == "force"], k], ignore_index=True)


# ------------------------------------------------------------------------------- fig 0
def fig0_split_schemes():
    from gaudit.data.cache import load_stream

    st = load_stream("data/cache", 1)
    m = st["series"] == 1
    lo, ln = st["seg_start_abs"][m][:6], st["seg_len_s"][m][:6]
    t0 = lo[0]
    fig, axes = plt.subplots(4, 1, figsize=(7.16, 2.9), sharex=True)
    g = np.random.default_rng(3)
    colors = {"train": "#2e86c1", "val": "#f5b041", "test": "#c0392b"}
    for ax, scheme in zip(axes, SCHEMES):
        for i, (a, n) in enumerate(zip(lo, ln)):
            a -= t0
            ends = np.arange(a + 0.49, a + n, 0.05)
            if scheme == "shuffled":
                lab = g.choice(["train", "test", "val"], size=len(ends), p=[0.7, 0.2, 0.1])
            elif scheme == "trialwise":
                lab = np.full(len(ends), ["train", "test", "train", "val", "train", "train"][i])
            elif scheme == "serieswise":
                lab = np.full(len(ends), "test")
            else:
                lab = np.full(len(ends), "test")
            for e, lb in zip(ends, lab):
                ax.add_patch(Rectangle((e - 0.05, 0.15), 0.05, 0.7, color=colors[lb], lw=0))
            ax.add_patch(Rectangle((a, 0.05), n, 0.9, fill=False, ec="k", lw=0.6))
            if scheme == "shuffled" and i == 0:
                ax.annotate("one 500-ms window\noverlaps its 9 neighbours", xy=(a + 2.0, 0.85), xytext=(a + 0.3, 1.25),
                            fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5), annotation_clip=False)
        note = {"shuffled": "windows of one trial land in train AND test",
                "trialwise": "whole trials (disjoint segments) per partition",
                "serieswise": "the whole recording series is held out",
                "subjectwise": "the whole participant is held out"}[scheme]
        ax.text(1.005, 0.5, note, transform=ax.transAxes, fontsize=6, va="center")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(SCHEME_LABEL[scheme].replace("\n", " "), rotation=0, ha="right", va="center", fontsize=7)
        ax.grid(False)
        for sp in ("left", "top", "right"):
            ax.spines[sp].set_visible(False)
        if ax is not axes[-1]:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)
    axes[-1].set_xlim(0, lo[-1] + ln[-1] - t0)
    axes[-1].set_xlabel("time in series (s) - six consecutive grasp-and-lift trials of participant 1")
    fig.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()], loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.45, 1.04))
    save(fig, "fig0_split_schemes")


# ------------------------------------------------------------------------------- fig 1
def fig1_headline(sub):
    s = kin_mean(sub)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), sharey=True)
    for ax, (target, title) in zip(axes, [("force", "Grip force"), ("pos (x,y,z)", "Hand position (mean of x, y, z)")]):
        d = s[s.target == target]
        models = [m for m in MODELS if m in set(d.model)]
        w = 0.8 / len(SCHEMES)
        for j, scheme in enumerate(SCHEMES):
            for i, m in enumerate(models):
                v = d[(d.model == m) & (d.scheme == scheme)].r2.values
                if not len(v):
                    continue
                lo, hi = bootstrap_ci(v)
                x = i + (j - 1.5) * w
                ax.bar(x, v.mean(), w * 0.92, color=SCHEME_COLOR[scheme], label=SCHEME_LABEL[scheme] if i == 0 else None)
                ax.errorbar(x, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], color="k", lw=0.6, capsize=1.2)
        # no-EEG template, per scheme, as short dashed lines spanning the plot
        for scheme in SCHEMES:
            v = d[(d.model == "template") & (d.scheme == scheme)].r2
            if len(v):
                ax.plot([-0.5, len(models) - 0.5], [v.mean()] * 2, ls=(0, (3, 2)), color=SCHEME_COLOR[scheme], lw=0.9)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([SHORT[m] for m in models])
        ax.set_title(title)
        ax.set_xlim(-0.5, len(models) - 0.5)
    axes[0].set_ylabel("Test $R^2$ (mean over 12 participants, 95% CI)")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], ls=(0, (3, 2)), color="k", lw=0.9))
    lab.append("No-EEG template\n(same split, dashed)")
    fig.legend(h, [x.replace("\n", " ") for x in lab], loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.07))
    ymin = min(-0.5, s[s.model.isin(models)].groupby(["model", "scheme", "target"]).r2.mean().min() - 0.05)
    axes[0].set_ylim(max(ymin, -1.2), 1.0)
    save(fig, "fig1_headline")


# ------------------------------------------------------------------------------- fig 2
def fig2_paired(sub):
    s = kin_mean(sub)
    models = ["template"] + [m for m in MODELS if m in set(s.model)]
    fig, axes = plt.subplots(2, len(models), figsize=(7.16, 3.6), sharey="row")
    for r, target in enumerate(["force", "pos (x,y,z)"]):
        for c, m in enumerate(models):
            ax = axes[r, c]
            d = s[(s.target == target) & (s.model == m)]
            if d.empty:
                ax.axis("off")
                continue
            piv = d.pivot(index="subject", columns="scheme", values="r2").reindex(columns=SCHEMES)
            for _, row in piv.iterrows():
                ax.plot(range(4), row.values, color="0.6", lw=0.5, marker="o", ms=1.8, alpha=0.8)
            ax.plot(range(4), piv.mean().values, color=MODEL_COLOR[m], lw=1.8, marker="o", ms=3)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["shuf", "trial", "series", "subj"], rotation=0)
            if r == 0:
                ax.set_title(MODEL_LABEL[m].split(" (")[0], fontsize=8)
            if c == 0:
                ax.set_ylabel(("Grip force" if r == 0 else "Hand position") + "\ntest $R^2$")
            ax.set_ylim(-1.0, 1.0)
    fig.tight_layout(w_pad=0.3, h_pad=0.6)
    save(fig, "fig2_paired_subjects")


# ------------------------------------------------------------------------------- fig 3 / 9
def fig3_null(e3):
    ps = pd.read_csv(e3 / "per_subject.csv")
    nulls = pd.read_csv(e3 / "null_per_subject.csv")
    """Per participant: grey bar = 5th-95th percentile of the trial-mismatch null, dot = real score
    (filled when above the participant's own 95th percentile)."""
    models = [m for m in ps.model.unique() if m != "template"] + ["template"]
    panels = [(m, sc) for m in models for sc in ("trialwise", "subjectwise")]
    fig, axes = plt.subplots(2, len(panels), figsize=(7.16, 3.6), sharey="row", sharex=True)
    for c, (m, sc) in enumerate(panels):
        for r, (target, ttl) in enumerate([("force", "Grip force"), ("pos_z", "Hand height (z)")]):
            ax = axes[r, c]
            d = nulls[(nulls.model == m) & (nulls.scheme == sc) & (nulls.target == target) & (nulls.null == "mismatch")]
            p = ps[(ps.model == m) & (ps.scheme == sc) & (ps.target == target)].sort_values("subject")
            if d.empty:
                ax.axis("off")
                continue
            q = d.groupby("subject").r2.quantile([0.05, 0.95]).unstack().reindex(p.subject)
            x = np.arange(len(p))
            ax.vlines(x, q[0.05], q[0.95], color="0.6", lw=2.2, alpha=0.8)
            col = MODEL_COLOR.get(m, "k")
            above = p.above_r2.values
            ax.scatter(x[above], p.r2_real.values[above], s=9, color=col, zorder=3, edgecolor="k", linewidth=0.3)
            ax.scatter(x[~above], p.r2_real.values[~above], s=9, facecolor="white", edgecolor=col, zorder=3, linewidth=0.8)
            ax.axhline(0, color="k", lw=0.4)
            ax.text(0.03, 0.04, f"{above.sum()}/{len(p)} > chance", transform=ax.transAxes, fontsize=5.5)
            ax.set_xticks([])
            if r == 0:
                ax.set_title(f"{SHORT[m].replace(chr(10), '')}\n{sc[:-4]}-wise", fontsize=6.5)
            if c == 0:
                ax.set_ylabel(f"{ttl}\ntest $R^2$")
            if r == 1:
                ax.set_xlabel("participants", fontsize=6)
    for ax in axes[1]:
        ax.set_ylim(-1.0, 1.0)
    for ax in axes[0]:
        ax.set_ylim(-1.0, 1.0)
    handles = [Line2D([], [], color="0.6", lw=2.2, label="chance range: EEG paired with another trial's target (5th-95th pct, 100 perms)"),
               Line2D([], [], marker="o", ls="", color="#d62728", markeredgecolor="k", markersize=3.5, label="real score, above chance"),
               Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#d62728", markersize=3.5,
                      label="real score, not above chance")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=6)
    fig.tight_layout(w_pad=0.3, h_pad=0.4)
    save(fig, "fig3_null")


def fig9_metric_illusion(e3):
    nulls = pd.read_csv(e3 / "null_per_subject.csv")
    d = nulls[(nulls.null == "mismatch")]
    real = nulls[nulls.null == "real"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    for ax, ds_targets, ttl in [(axes[0], ["force"], "Grip force"), (axes[1], ["pos_x", "pos_y", "pos_z"], "Hand position")]:
        dd = d[d.target.isin(ds_targets)].groupby(["model", "scheme", "target", "subject"], as_index=False)[["r2", "r"]].mean()
        rr = real[real.target.isin(ds_targets)]
        ax.scatter(dd.r, dd.r2, s=5, color="0.55", label="null (mismatched trials), per subject", alpha=0.7)
        for m in rr.model.unique():
            x = rr[rr.model == m]
            ax.scatter(x.r, x.r2, s=7, color=MODEL_COLOR.get(m, "k"), label=f"real: {MODEL_LABEL[m].split(' (')[0]}", alpha=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("$R^2$")
        ax.set_title(ttl)
        ax.set_xlim(-0.2, 1)
        ax.set_ylim(-1.5, 1)
    axes[1].legend(loc="lower right", fontsize=6, frameon=False)
    fig.tight_layout()
    save(fig, "fig9_metric_illusion")


# ------------------------------------------------------------------------------- fig 4
def _trial_curve(z, tid, true=False):
    m = z["trial_id"] == tid
    m &= z["fold"] == z["fold"][m][0]
    o = np.argsort(z["t_rel"][m])
    return z["t_rel"][m][o] - 0.8, (z["y_true"] if true else z["y_pred"])[m][o]


def fig4_example(sub, preds):
    """One test trial: true force vs predictions of the leakiest decoder under each split."""
    f = sub[(sub.dataset == "force") & sub.model.isin(MODELS)]
    piv = f.groupby(["model", "scheme"]).r2.mean().unstack()
    model = (piv["shuffled"] - piv["trialwise"]).idxmax()  # largest shuffled-vs-honest gap
    files = {sc: preds.get(("force", model, sc)) for sc in SCHEMES}
    tfile = preds.get(("force", "template", "trialwise"))
    if not all(files.values()) or tfile is None:
        return
    Z = {sc: np.load(fl) for sc, fl in files.items()}
    T = np.load(tfile)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), sharey=True)
    for ax, subj in zip(axes, (3, 9)):
        cand = set(Z["trialwise"]["trial_id"][Z["trialwise"]["subject"] == subj])
        cand &= set(Z["subjectwise"]["trial_id"][Z["subjectwise"]["subject"] == subj])
        cand &= set(T["trial_id"][T["subject"] == subj])
        cand &= set(Z["shuffled"]["trial_id"][Z["shuffled"]["subject"] == subj])
        tid = sorted(cand)[len(cand) // 2]
        t, y = _trial_curve(Z["subjectwise"], tid, true=True)
        ax.plot(t, y[:, 0], color="k", lw=1.8, label="measured grip force")
        zs = Z["shuffled"]
        m = zs["trial_id"] == tid
        ax.scatter(zs["t_rel"][m] - 0.8, zs["y_pred"][m, 0], s=5, color=SCHEME_COLOR["shuffled"], zorder=3,
                   label="shuffled windows (leaky): test windows of this trial")
        for sc in ("trialwise", "subjectwise"):
            t, p = _trial_curve(Z[sc], tid)
            ax.plot(t, p[:, 0], color=SCHEME_COLOR[sc], lw=1.0, label=f"{sc[:-4]}-wise split")
        t, p = _trial_curve(T, tid)
        ax.plot(t, p[:, 0], color="0.45", lw=1.0, ls="--", label="no-EEG template")
        ax.axvline(0, color="0.5", lw=0.5, ls=":")
        ax.set_xlabel("time from LED-on (s)")
        ax.set_title(f"participant {subj}, trial {tid % 10000}", fontsize=8)
    axes[0].set_ylabel("grip force (N)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12), fontsize=6.5)
    fig.suptitle(f"{MODEL_LABEL[model]}: the decoder with the largest shuffled-vs-trial-wise gap", y=1.16, fontsize=8)
    save(fig, "fig4_example_trial")


# ------------------------------------------------------------------------------- fig 5
def fig5_onset(e4):
    cur = pd.read_csv(e4 / "curves.csv")
    op = pd.read_csv(e4 / "operating_points.csv")
    cue = pd.read_csv(e4 / "cue_timer_baseline.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    styles = {("lda", "trialwise"): ("#1f77b4", "-"), ("lda", "subjectwise"): ("#1f77b4", "--"),
              ("riemann", "trialwise"): ("#2ca02c", "-"), ("riemann", "subjectwise"): ("#2ca02c", "--")}
    for ax, var in zip(axes[:2], ["all", "no_occipital"]):
        for (det, sc), (c, ls) in styles.items():
            d = cur[(cur.detector == det) & (cur.scheme == sc) & (cur.variant == var) & (cur.threshold >= 0.5)]
            if d.empty:
                continue
            g = d.groupby("threshold", as_index=False)[["det_rate", "fa_per_min"]].mean()
            ax.plot(g.fa_per_min, g.det_rate, color=c, ls=ls, lw=1.1,
                    label=f"{'LDA' if det == 'lda' else 'Riemannian'}, {sc[:-4]}-wise")
            o = op[(op.detector == det) & (op.scheme == sc) & (op.variant == var)]
            ax.scatter(o.fa_per_min.mean(), o.det_rate.mean(), color=c, s=14, marker="o", zorder=4,
                       edgecolor="k", linewidth=0.4)
        ax.scatter([0.0], [cue.det_rate.mean()], marker="*", s=60, color="k", zorder=5, clip_on=False,
                   label="cue timer, no EEG (0 FA/min)")
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xticks([0, 0.1, 1, 10, 50])
        ax.set_xticklabels(["0", "0.1", "1", "10", "50"])
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("false alarms per minute of rest")
        ax.set_title("all channels" if var == "all" else "without occipital channels")
    axes[0].set_ylabel("detection rate\n(LED-on to hand start + 200 ms)")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", color="0.5", ls="", markeredgecolor="k", markersize=4))
    lab.append("operating point (FA <= 1/min on validation)")
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.45, 1.1), fontsize=6)
    ax = axes[2]
    d = op[(op.variant == "all")]
    data, labels = [], []
    for (det, sc) in styles:
        v = d[(d.detector == det) & (d.scheme == sc)].median_lead_ms.dropna()
        data.append(v.values)
        labels.append(("LDA" if det == "lda" else "Riem.") + "\n" + ("trial" if sc == "trialwise" else "subj."))
    data.append(cue.median_lead_ms.values)
    labels.append("cue\ntimer")
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], ["#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c", "0.4"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_xticklabels(labels, fontsize=6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("median lead vs hand start (ms)\n(negative = before movement)")
    ax.set_title("lead time at operating point")
    fig.tight_layout()
    save(fig, "fig5_onset")


# ------------------------------------------------------------------------------- fig 6
def fig6_traj3d(sub, preds):
    """3D wrist paths of two held-out trials: measured vs decoded, best EEG decoder vs template."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cand = sub[(sub.dataset == "kin") & (sub.scheme == "trialwise") & (sub.model.isin(MODELS))]
    best = cand.groupby("model").r2.mean().idxmax()
    keys = [("trialwise", best), ("subjectwise", best), ("trialwise", "template")]
    files = {k: preds.get(("kin", k[1], k[0])) for k in keys}
    if any(f is None for f in files.values()):
        return
    Z = {k: np.load(f) for k, f in files.items()}
    subj = 2
    tw = Z[keys[0]]
    common = set(tw["trial_id"][tw["subject"] == subj]) & set(Z[keys[1]]["trial_id"][Z[keys[1]]["subject"] == subj])
    common &= set(Z[keys[2]]["trial_id"][Z[keys[2]]["subject"] == subj])
    tids = sorted(common)
    tids = [tids[len(tids) // 3], tids[2 * len(tids) // 3]]
    cols = ["#1f77b4", "#d62728"]
    titles = [f"{MODEL_LABEL[best]}\ntrial-wise split", f"{MODEL_LABEL[best]}\nsubject-wise split",
              "No-EEG template\ntrial-wise split"]
    true_all = np.concatenate([_trial_curve(Z[keys[0]], t, true=True)[1] for t in tids])
    lo, hi = true_all.min(0), true_all.max(0)
    pad = 0.25 * (hi - lo)
    fig = plt.figure(figsize=(7.16, 3.0))
    for i, k in enumerate(keys):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        for c, tid in zip(cols, tids):
            _, yt = _trial_curve(Z[k], tid, true=True)
            _, yp = _trial_curve(Z[k], tid)
            ax.plot(yt[:, 0], yt[:, 1], yt[:, 2], color=c, lw=1.8, alpha=0.9)
            ax.plot(yp[:, 0], yp[:, 1], yp[:, 2], color=c, lw=1.3, ls=(0, (2, 1)), alpha=1.0)
            ax.scatter(*yt[0], color=c, s=10, marker="o")
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
        ax.set_box_aspect((0.6, 1.0, 0.8), zoom=1.05)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_title(titles[i], fontsize=7, pad=-2)
        ax.set_xlabel("x (cm)", fontsize=6, labelpad=-8)
        ax.set_ylabel("y (cm)", fontsize=6, labelpad=-8)
        ax.set_zlabel("z (cm)", fontsize=6, labelpad=-9)
        ax.tick_params(labelsize=5, pad=-3)
        ax.view_init(elev=14, azim=-18)
    fig.legend(handles=[Line2D([], [], color="k", lw=1.8, label="measured wrist path"),
                        Line2D([], [], color="k", lw=0.9, ls=(0, (2, 1)), label="decoded"),
                        Line2D([], [], color="k", marker="o", ls="", ms=3, label="trial start")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, 0.0, f"Participant {subj}: two held-out reach-grasp-lift-replace trials (colours).", ha="center", fontsize=6)
    fig.subplots_adjust(wspace=0.12, left=0.0, right=0.97, top=0.86, bottom=0.08)
    save(fig, "fig6_traj3d")


# ------------------------------------------------------------------------------- fig 7
def fig7_blocks3d(e6, sub):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    d = pd.read_csv(e6 / "per_subject.csv")
    d = d.groupby(["dataset", "model", "block_s", "subject"], as_index=False).r2.mean()
    fig = plt.figure(figsize=(7.16, 3.1))
    for i, (ds, ttl) in enumerate([("force", "Grip force"), ("kin", "Hand position")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        # rows ordered short-to-tall from front to back (MLP is low except at 0.05 s)
        models = [m for m in ["mlp", "riemann", "ridge"] if m in set(d.model)]
        blocks = sorted(d.block_s.unique())
        gx = np.arange(len(blocks))
        row = 1.6  # spacing between decoder rows
        for j, m in enumerate(models):
            g = d[(d.dataset == ds) & (d.model == m)].groupby("block_s").r2.mean().reindex(blocks).values
            ax.bar3d(gx - 0.3, np.full(len(gx), j * row - 0.3), np.zeros(len(gx)), 0.6, 0.6, np.maximum(g, 0),
                     color=MODEL_COLOR[m], alpha=0.9, edgecolor="k", linewidth=0.15, shade=True)
            tw = sub[(sub.dataset == ds) & (sub.model == m) & (sub.scheme == "trialwise")].groupby("subject").r2.mean().mean()
            ax.plot([-0.5, len(gx) - 0.5], [j * row] * 2, [tw, tw], color="k", ls="--", lw=1.0, zorder=10)
        ax.set_xticks(gx)
        ax.set_xticklabels([f"{b:g}" for b in blocks], fontsize=5)
        ax.set_yticks([j * row for j in range(len(models))])
        ax.set_yticklabels([MODEL_LABEL[m].split(" (")[0] for m in models], fontsize=5.5)
        ax.set_xlabel("CV block length (s)", fontsize=6, labelpad=-4)
        ax.set_zlabel("test $R^2$", fontsize=6, labelpad=-4)
        ax.tick_params(axis="both", labelsize=5, pad=-2)
        ax.set_title(ttl, fontsize=8)
        ax.view_init(elev=26, azim=-58)
    fig.text(0.5, 0.02, "Blocked CV within trials: 0.05 s = shuffled windows; longer blocks leave fewer overlapping "
             "edge windows. Dashed: honest trial-wise score.", ha="center", fontsize=6)
    save(fig, "fig7_blocks3d")


# ------------------------------------------------------------------------------- fig 8
def fig8_topomaps():
    import mne

    from gaudit.data.cache import SUBJECTS, load_cached

    maps = {"force": [], "pos_z": []}
    names = None
    for s in SUBJECTS:
        d = load_cached("data/cache", s)
        names = d.ch_names
        w = d.block_type == 1
        idx = np.concatenate([np.arange(o, o + n) for o, n, k in zip(d.offset, d.length, w) if k])
        for t in maps:
            y = d.target(t)[idx]
            X = d.lf[idx]
            Xc, yc = X - X.mean(0), y - y.mean()
            maps[t].append((Xc * yc[:, None]).sum(0) / np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()))
    info = mne.create_info(names, 100, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4), gridspec_kw={"width_ratios": [1, 1, 0.06]})
    vmax = max(np.abs(np.mean(v, 0)).max() for v in maps.values())
    for ax, (t, ttl) in zip(axes[:2], [("force", "grip force"), ("pos_z", "hand height (z)")]):
        im, _ = mne.viz.plot_topomap(np.mean(maps[t], 0), info, axes=ax, show=False, cmap="RdBu_r",
                                     vlim=(-vmax, vmax), contours=4, sensors=True)
        ax.set_title(f"r(0.1-3 Hz EEG, {ttl})\nmean of 12 participants", fontsize=7)
    fig.colorbar(im, cax=axes[2], label="Pearson r")
    save(fig, "fig8_topomaps")


# ------------------------------------------------------------------------------- fig 10
def fig10_selection_artifacts(e2, e5):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    if e2 is not None:
        d = pd.read_csv(e2 / "per_subject.csv")
        piv = d.pivot_table(index=["model", "scheme", "target", "subject"], columns="select_on", values="r2").reset_index()
        piv["gain"] = piv["all"] - piv["train"]
        groups = piv.groupby(["model", "scheme", "target"])
        labels, data = [], []
        for (m, sc, t), g in groups:
            labels.append(f"{m}\n{'trial' if sc == 'trialwise' else 'subj.'}\n{t}")
            data.append(g.gain.values)
        ax = axes[0]
        ax.boxplot(data, widths=0.55, showfliers=False)
        for i, v in enumerate(data):
            ax.scatter(np.full(len(v), i + 1) + np.random.default_rng(i).normal(0, 0.05, len(v)), v, s=4, color="#c0392b")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticklabels(labels, fontsize=5.5)
        ax.set_ylabel("$R^2$ gain from selecting on all data")
        ax.set_title("E2: top-16 selection fit on test data too")
    if e5 is not None:
        d = pd.read_csv(e5 / "per_subject.csv")
        ax = axes[1]
        k = 0
        ticks = []
        for (m, t), g in d.groupby(["model", "target"]):
            piv = g.pivot(index="subject", columns="channels", values="r2")
            for _, row in piv.iterrows():
                ax.plot([k, k + 0.6], [row["all"], row["no_fp_occ"]], color="0.6", lw=0.5, marker="o", ms=1.8)
            ax.plot([k, k + 0.6], [piv["all"].mean(), piv["no_fp_occ"].mean()], color=MODEL_COLOR[m], lw=2, marker="o", ms=3)
            ticks.append((k + 0.3, f"{m}\n{t}"))
            k += 1.3
        ax.set_xticks([x for x, _ in ticks])
        ax.set_xticklabels([lab for _, lab in ticks], fontsize=6)
        ax.set_ylabel("trial-wise test $R^2$")
        ax.set_title("E5: all channels (left) vs no Fp/occipital (right)")
    fig.tight_layout()
    save(fig, "fig10_selection_artifacts")


def _try_run(name):
    try:
        return latest_run(name)
    except AssertionError:
        print(f"skip: no {name} run")
        return None


def main():
    fig0_split_schemes()
    try:
        sub, preds = load_e1()
    except AssertionError:
        print("no E1 results yet")
        return
    sub = sub[sub.model.isin(MODELS + ["template"])]
    fig1_headline(sub)
    fig2_paired(sub)
    fig4_example(sub, preds)
    fig6_traj3d(sub, preds)
    fig8_topomaps()
    e3, e4, e2, e5, e6 = (_try_run(n) for n in ("e3_chance", "e4_onset", "e2_selection", "e5_artifacts", "e6_blocks"))
    if e3 is not None:
        fig3_null(e3)
        fig9_metric_illusion(e3)
    if e4 is not None:
        fig5_onset(e4)
    if e6 is not None:
        fig7_blocks3d(e6, sub)
    if e2 is not None or e5 is not None:
        fig10_selection_artifacts(e2, e5)


if __name__ == "__main__":
    main()
