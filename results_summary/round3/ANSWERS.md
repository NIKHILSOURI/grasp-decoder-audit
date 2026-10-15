# Round-3 answers

Rules followed: no existing result file or paper number was changed (new results only in `results/round3/`; the two
regenerated figures keep their previous versions as `*_v1.pdf`). Unit of inference = participant (folds averaged within
participant first); bootstrap 10,000 resamples over participants, seed 0; Wilcoxon two-sided, exact. Same splits, seeds and
preprocessing as the paper unless stated. Every task ran in full (no reduced versions were needed).

## A1 cue timer: strict_pre = 0.9967 ; loose = 0.9981 ; trials = 2,124 ; failing strict = 7 ; min RT = -0.366 s ; lead median -314 ms (IQR -478 to -234)

Trial-wise onset folds (same 3 x 12 folds as E4; 2,124 test-trial evaluations of 1,728 distinct trials). Subject-wise folds
(all 3,528 trials once): strict 0.9977, loose 0.9991, failing strict 8, failing loose 3, lead median -314 ms (IQR -477 to -234).
Over all 3,528 trials, 8 have RT <= 0.1 s and 3 have RT < -0.1 s (the hand moved before the cue; min -0.366 s).
**`\CUEPRE` = 1.00 is correct at two decimals; more precisely 0.997 (99.7%).** File: `a1_cue_timer.csv`, `cue_timer_trials.csv`.

## A2 number check: 185 checked, 185 OK, 0 DOES NOT REPRODUCE

`python scripts/check_paper_numbers.py` -> `results/round3/paper_number_check.csv` (columns location, quantity, printed,
recomputed, status). Covered: all Table I cells (48 means + 48 SDs), all Table II cells (24), all Table III cells (24 + cue
timer lead, strict and loose), Δ = 0.43 and CI [0.41, 0.45], p = 0.011 / 0.16 / 0.68 (two template contrasts), EEGNet 10/12,
blocked CV (0.49, 0.08, 0.38, 0.18, 0.35, 0.30, 98%, 91%, 28%, 3%), purged (-0.03, 0.06, 0.06, 0.29 to 0.30, 54%), filtering
(+0.09, +0.12, 0.35, 0.47), mismatch (12/12, 11/12, 0.26 vs 0.20, 7 to 11, 0.12 to 0.40, 0.57, 0.91), ablation (0.03 to 0.06,
0.03 to 0.07, p <= 0.016), selection (+0.017), onset (AUCs incl. no-occipital 0.82/0.71, Pre, Loose, leads, FA/min, slopes,
4.7 to 21.5%, 2.8 to 7.3%), Table II CIs [0.008, 0.025] and [0.001, 0.021], "at most 2 of 12", 52% and 79%.
**Caveat:** I do not have your current Overleaf `main.tex`; the printed values are taken from my copy plus the numbers I
gave you in rounds 1-2. They live in `scripts/paper_numbers_printed.csv` (columns location, quantity, key, printed): edit the
`printed` column if your text differs and re-run. **Purged MLP at 0.25 s (-0.03) reproduces from the round-2 file but is
seed-dependent (see B1 and the last section).**

## A3 artifact files: list + anonymity scan result

Written to `anonymous_release/results/` (nothing uploaded):
- `results_by_subject.csv` (3,336 rows): subject, split, fold, repeat, target, axis, model, seed, n_train_trials,
  n_test_trials, n_test_windows, R2, pearson_r, template_R2, residual_gain. (template_R2 for EEGNet rows comes from the
  template on the same fold and participant at the 50-ms grid; residual_gain exists for ridge/MLP trial-/subject-wise.)
- `onset_events.csv` (16,956 rows): subject, split, fold, trial_id, LED_on, hand_start, detector (lda, riemann, cue_timer),
  threshold (validation-selected), first_detection_time, strict_pre, loose_hit, rest_seconds, n_false_alarms.
- `splits.json` (504 split entries: shuffled, trial-wise, series-wise, subject-wise, blocked; force and kinematics): participant,
  series and trial IDs for train/validation/test plus recorded assertions. All honest splits pass: trial-wise 120/120,
  series-wise 72/72 (no trial in two partitions), subject-wise 24/24 (no trial and no test participant in training). Shuffled
  and blocked splits are flagged "not applicable (window-level, leaky by design)".
- `stats_families.csv` (42 tests in 5 families: split inflation 22, artifact ablation 4, leaky selection 8, Table II 8):
  family, family_size, contrast, raw_p, holm_p, n, side (two-sided).
- `environment.txt` (Python 3.11.9, RTX 3060 Laptop 6 GB, torch 2.6.0+cu124, braindecode 1.8.1, MNE 1.13.2, SciPy 1.17.1,
  scikit-learn 1.9.1, full `pip freeze`).
- `README_reproduce.md` (one command per table).
- Also refreshed: code (`src/`, `scripts/` incl. round 3, `configs/`, `tests/`), docs, and `results/round3/` (all round-3 CSVs).

**Anonymity scan of the whole folder (149 files, incl. decompressed .gz): no hits** for names, e-mail, user paths, local
drive paths or the spec handle.

## B1 embargo table (markdown table) + overlap check + train-size-matched control

Force, blocked CV (E6 splits). Embargo e removes train AND validation windows whose end lies < e s from a test-window end in
the same trial. R² mean [95% CI] over 12 participants; train windows = mean per participant.

| block | variant | MLP | Ridge | Riemannian | train windows (unpurged 17.2-17.3k) | test windows | test windows sharing samples |
|---|---|---|---|---|---|---|---|
| 0.25 s | embargo 0.49 s | -0.055 [-0.16, 0.02] | 0.287 [0.25, 0.32] | 0.072 [0.01, 0.13] | 8,004 | 4,930 | 0 |
| 0.25 s | embargo 0.5 s | -0.076 [-0.18, 0.01] | 0.287 [0.25, 0.32] | 0.072 [0.01, 0.13] | 8,004 | 4,930 | 0 |
| 0.25 s | embargo 1.0 s | -0.106 [-0.20, -0.02] | 0.224 [0.18, 0.27] | -0.108 [-0.21, -0.01] | 3,692 | 4,930 | 0 |
| 0.25 s | embargo 2.2 s | -0.264 [-0.32, -0.20] | -0.081 [-0.14, -0.02] | -0.373 [-0.48, -0.27] | 900 | 4,930 | 0 |
| 0.25 s | embargo 3.3 s | -0.282 [-0.48, -0.08] | -0.427 [-0.70, -0.15] | -0.232 [-0.40, -0.07] | 338 | 4,930 | 0 |
| 0.25 s | **size-matched (8,004, unpurged)** | **0.088 [0.07, 0.11]** | **0.323 [0.29, 0.36]** | **0.288 [0.25, 0.32]** | 8,004 | 4,930 | 58,098 |
| 1 s | embargo 0.49 s | 0.069 [0.05, 0.09] | 0.297 [0.25, 0.34] | 0.201 [0.17, 0.24] | 14,183 | 4,937 | 0 |
| 1 s | embargo 0.5 s | 0.066 [0.05, 0.09] | 0.297 [0.25, 0.34] | 0.201 [0.17, 0.24] | 14,183 | 4,937 | 0 |
| 1 s | embargo 1.0 s | 0.049 [0.03, 0.07] | 0.286 [0.24, 0.33] | 0.185 [0.15, 0.22] | 11,652 | 4,937 | 0 |
| 1 s | embargo 2.2 s | 0.035 [0.02, 0.06] | 0.270 [0.22, 0.32] | 0.145 [0.11, 0.18] | 7,747 | 4,937 | 0 |
| 1 s | embargo 3.3 s | 0.022 [-0.00, 0.05] | 0.225 [0.16, 0.28] | 0.111 [0.08, 0.14] | 5,765 | 4,937 | 0 |
| 1 s | **size-matched (14,183)** | 0.076 [0.06, 0.10] | 0.317 [0.28, 0.36] | 0.230 [0.20, 0.27] | 14,183 | 4,937 | 40,564 |
| 4 s | embargo 0.49 s | 0.057 [0.04, 0.07] | 0.301 [0.24, 0.36] | 0.175 [0.13, 0.22] | 16,769 | 4,926 | 0 |
| 4 s | embargo 0.5 s | 0.063 [0.05, 0.08] | 0.301 [0.24, 0.36] | 0.175 [0.13, 0.22] | 16,769 | 4,926 | 0 |
| 4 s | embargo 1.0 s | 0.053 [0.04, 0.07] | 0.297 [0.24, 0.36] | 0.172 [0.13, 0.21] | 16,193 | 4,926 | 0 |
| 4 s | embargo 2.2 s | 0.055 [0.04, 0.07] | 0.296 [0.24, 0.36] | 0.165 [0.12, 0.21] | 14,862 | 4,926 | 0 |
| 4 s | embargo 3.3 s | 0.050 [0.03, 0.07] | 0.291 [0.23, 0.35] | 0.166 [0.12, 0.21] | 13,811 | 4,926 | 0 |
| 4 s | **size-matched (16,769)** | 0.065 [0.05, 0.08] | 0.304 [0.24, 0.36] | 0.184 [0.14, 0.23] | 16,769 | 4,926 | 7,541 |

Overlap check: **0 test windows share a raw sample with any training or validation window for every embargo >= 0.49 s**
(window ends lie on a 50-ms grid, so the 0.49-s and 0.5-s embargoes remove exactly the same windows; their MLP scores differ
only by seed). Reading: at 0.25-s blocks the size-matched control keeps Riemannian at 0.29 while the 0.5-s embargo gives 0.07,
so the Riemannian drop is caused by removing nearby (sample-sharing) windows, not by less data; for long embargoes at 0.25 s
the training set collapses (338 windows at 3.3 s) and those cells mostly measure data loss. At 4-s blocks even a 3.3-s embargo
(full FIR length) changes R² by <= 0.014: filter-support leakage is negligible. Files: `b1_embargo.csv`, `b1_embargo_summary.csv`.

## B2 filtering: (a) per-participant table, (b) series-wise whole-series table, (c) 0.3 Hz result

**(a) Ridge, trial-wise, per-trial (paper) vs whole-series filtering** (per participant; position = mean of x, y, z):

| P | force per-trial | force whole-series | diff | position per-trial | position whole-series | diff |
|---|---|---|---|---|---|---|
| 1 | -0.068 | 0.343 | +0.411 | 0.478 | 0.665 | +0.187 |
| 2 | 0.290 | 0.320 | +0.030 | 0.181 | 0.280 | +0.098 |
| 3 | 0.272 | 0.338 | +0.066 | 0.364 | 0.474 | +0.110 |
| 4 | 0.373 | 0.463 | +0.090 | 0.431 | 0.578 | +0.146 |
| 5 | 0.252 | 0.311 | +0.059 | 0.373 | 0.531 | +0.158 |
| 6 | 0.199 | 0.268 | +0.068 | 0.306 | 0.469 | +0.163 |
| 7 | 0.228 | 0.324 | +0.096 | 0.390 | 0.519 | +0.129 |
| 8 | 0.329 | 0.377 | +0.048 | 0.279 | 0.323 | +0.044 |
| 9 | 0.345 | 0.416 | +0.071 | 0.343 | 0.459 | +0.116 |
| 10 | 0.266 | 0.382 | +0.115 | 0.450 | 0.650 | +0.200 |
| 11 | 0.374 | 0.373 | -0.001 | 0.281 | 0.303 | +0.022 |
| 12 | 0.288 | 0.315 | +0.027 | 0.277 | 0.391 | +0.114 |
| mean | 0.262 | 0.352 | **+0.090 [0.047, 0.156]**, p = 0.0010, 11/12 | 0.346 | 0.470 | **+0.124 [0.094, 0.152]**, p = 0.0005, 12/12 |

**(b) Series-wise splits with whole-series filtering (leak-free: each held-out series is filtered on its own):**

| Target | Model | paper series-wise | whole-series | diff [95% CI] | Wilcoxon p | improved |
|---|---|---|---|---|---|---|
| force | ridge | 0.237 | 0.305 [0.27, 0.34] | +0.068 [0.041, 0.098] | 0.0010 | 11/12 |
| force | MLP | 0.037 | 0.037 [0.01, 0.06] | -0.000 [-0.001, 0.000] | 0.57 | 8/12 |
| force | Riemannian | 0.144 | 0.145 [0.11, 0.18] | +0.001 [-0.000, 0.001] | 0.15 | 8/12 |
| force | template | 0.514 | 0.514 (unchanged, no EEG) | 0 | - | - |
| position | ridge | 0.308 | 0.431 [0.37, 0.50] | +0.123 [0.092, 0.154] | 0.0005 | 12/12 |
| position | MLP | 0.150 | 0.150 [0.11, 0.19] | -0.000 [-0.000, 0.000] | 0.13 | 3/12 |
| position | Riemannian | 0.291 | 0.293 [0.24, 0.34] | +0.002 [0.001, 0.004] | 0.0024 | 11/12 |
| position | template | 0.781 | 0.781 (unchanged) | 0 | - | - |

**(c) Per-trial filtering with 0.3-3 Hz (ridge, trial-wise):** force 0.023 [-0.011, 0.048] vs 0.262 (diff -0.239
[-0.276, -0.198], p = 0.0005, 0/12 improved); position 0.073 [0.058, 0.090] vs 0.346 (diff -0.273 [-0.307, -0.237],
p = 0.0005, 0/12). Files: `b2_per_participant.csv`, `b2_b7_summary.csv`.

## B3 onset controls: table for (a) (b) (c) + which negative rule was used

Mean over participants [95% CI]; test data at the validation-selected threshold (<= 1 FA/min on validation).
Window AUC for (a) uses the time-matched labels; for (b), (c) and the E4 rows the E4 labels.

| Detector | Split | window AUC | strict Pre | Loose | test FA/min | median lead of detected (ms) | negative rule |
|---|---|---|---|---|---|---|---|
| E4 LDA | trial | 0.830 [0.812, 0.846] | 0.139 [0.101, 0.177] | 0.190 [0.143, 0.240] | 1.19 [0.90, 1.48] | -66 [-105, -33] | rest + hold |
| E4 LDA | subject | 0.709 [0.681, 0.737] | 0.047 [0.023, 0.075] | 0.080 [0.047, 0.118] | 1.20 [0.58, 2.02] | -20 [-68, 16] | rest + hold |
| E4 Riemannian | trial | 0.779 [0.746, 0.813] | 0.215 [0.159, 0.269] | 0.288 [0.225, 0.349] | 3.02 [2.50, 3.51] | -126 [-174, -79] | rest + hold |
| E4 Riemannian | subject | 0.685 [0.653, 0.715] | 0.100 [0.025, 0.201] | 0.128 [0.031, 0.252] | 7.64 [0.79, 16.84] | -203 [-306, -108] | rest + hold |
| (a) LDA, time-matched negatives | trial | 0.702 [0.683, 0.719] | 0.055 [0.031, 0.082] | 0.087 [0.053, 0.124] | 1.21 [0.93, 1.47] | -26 [-63, 6] | time-matched (36/36 folds) |
| (a) LDA, time-matched | subject | 0.618 [0.597, 0.637] | 0.032 [0.012, 0.056] | 0.052 [0.022, 0.090] | 1.24 [0.37, 2.75] | -49 [-102, -1] | time-matched (12/12) |
| (a) Riemannian, time-matched | trial | 0.603 [0.576, 0.626] | 0.271 [0.224, 0.325] | 0.395 [0.340, 0.457] | **16.97 [12.20, 22.33]** | -92 [-109, -73] | time-matched (36/36) |
| (a) Riemannian, time-matched | subject | 0.553 [0.534, 0.575] | 0.028 [0.001, 0.073] | 0.045 [0.004, 0.115] | 2.27 [0.13, 6.33] | -65 [-238, 82] | time-matched (12/12) |
| (b) cue-age only (no EEG) | trial | 0.984 [0.974, 0.994] | **0.969 [0.960, 0.977]** | 0.969 [0.960, 0.977] | 0.03 [0.00, 0.08] | -402 [-484, -333] | rest + hold |
| (b) cue-age only | subject | 0.986 [0.976, 0.993] | 0.968 [0.967, 0.969] | 0.969 [0.967, 0.969] | 0.00 [0.00, 0.00] | -400 [-482, -332] | rest + hold |
| (c) LDA + cue age | trial | 0.983 [0.973, 0.991] | 0.933 [0.903, 0.958] | 0.947 [0.929, 0.964] | 1.30 [0.94, 1.65] | -391 [-475, -318] | rest + hold |
| (c) LDA + cue age | subject | 0.980 [0.971, 0.988] | 0.962 [0.956, 0.969] | 0.965 [0.958, 0.971] | 1.59 [0.51, 2.96] | -398 [-480, -330] | rest + hold |

**Negative rule for (a):** the primary rule was used in every fold (no fallback). Negatives = windows ending at the same times
after the cue as the training positives (range of positive cue-ages in the training fold, 50-ms grid) in trials whose hand start
is more than 300 ms after the window end. Cue age = time since the most recent LED-on in the same series (causal).
Files: `b3_onset_controls.csv`, `onset_folds.csv`, `onset_trials.csv.gz`.

## B4 template/R2 robustness: (a) (b) tables, (c) support numbers, (d) per-axis

Mean over participants (pooled = paper's metric).

| Target | Split | Model | pooled (paper) | (a) trial-balanced mean | (a) median over trials | (b) common interval [95% CI] |
|---|---|---|---|---|---|---|
| force | trial | template | 0.520 | -0.113 | 0.478 | 0.581 [0.540, 0.619] |
| force | trial | ridge | 0.262 | -0.299 | 0.163 | 0.264 [0.160, 0.337] |
| force | trial | MLP | 0.053 | -0.474 | -0.141 | 0.038 [0.019, 0.056] |
| force | trial | Riemannian | 0.183 | -0.364 | 0.022 | 0.171 [0.130, 0.214] |
| force | trial | EEGNet | 0.173 | -0.378 | 0.000 | 0.164 [0.109, 0.224] |
| force | subject | template | 0.192 | -1.049 | -0.150 | 0.333 [0.186, 0.460] |
| force | subject | ridge | -0.023 | -1.010 | -0.336 | 0.033 [-0.085, 0.146] |
| force | subject | MLP | -0.141 | -1.023 | -0.465 | -0.133 [-0.219, -0.063] |
| force | subject | Riemannian | -0.227 | -1.470 | -0.704 | -0.180 [-0.328, -0.058] |
| force | subject | EEGNet | -0.176 | -1.283 | -0.606 | -0.130 [-0.302, -0.022] |
| position | trial | template | 0.789 | 0.787 | 0.847 | 0.895 [0.855, 0.927] |
| position | trial | ridge | 0.346 | 0.329 | 0.396 | 0.332 [0.277, 0.385] |
| position | trial | MLP | 0.168 | 0.149 | 0.161 | 0.162 [0.108, 0.211] |
| position | trial | Riemannian | 0.322 | 0.305 | 0.330 | 0.327 [0.269, 0.380] |
| position | trial | EEGNet | 0.391 | 0.377 | 0.405 | 0.383 [0.326, 0.432] |
| position | subject | template | 0.549 | 0.543 | 0.611 | 0.736 [0.640, 0.818] |
| position | subject | ridge | 0.055 | 0.029 | 0.065 | 0.022 [-0.128, 0.129] |
| position | subject | MLP | -0.014 | -0.042 | -0.024 | -0.050 [-0.128, 0.022] |
| position | subject | Riemannian | -0.050 | -0.080 | -0.042 | -0.024 [-0.140, 0.082] |
| position | subject | EEGNet | 0.016 | -0.007 | 0.027 | 0.061 [-0.022, 0.133] |

(a) Single-trial force R² has a heavy negative tail (trials with little within-trial force variance), so the trial-balanced mean
is negative for every model, including the template; the median and the ordering (template best in every row) are stable.
(b) Common interval = windows ending LED - 0.3 s to LED + 4.0 s (t = 0.5-4.8 s after segment start); the shortest trial segment
is 5.46 s, so every trial covers it. The template's lead over the EEG decoders grows in this interval.
(c) Template support (training trials reaching each time after segment start; LED at 0.8 s; trial-wise training sets have 119
force / 206 position trials per fold, subject-wise 1,530 / 2,646): 100% up to 5.0 s, 99% at 5.5 s, 94% at 6.0 s, 84%
(force) / 91% (position) at 6.5 s, 67-68% / 80-81% at 7.0 s, **< 50% from 7.5 s (force) and 8.0 s (position)**, 25% / 35% at
8.0 s, 10% / 15% at 8.5 s. Only 6.6% (force) and 9.0% (position) of test windows end after 7.5 s.
(d) Per-axis R² (mean over participants), trial-wise / subject-wise: template x 0.806/0.587, y 0.811/0.588, z 0.750/0.472;
ridge 0.358/0.059, 0.356/0.071, 0.324/0.035; MLP 0.176/-0.028, 0.183/0.021, 0.145/-0.037; Riemannian 0.341/-0.041,
0.342/-0.047, 0.282/-0.061; EEGNet 0.415/0.035, 0.422/0.020, 0.336/-0.006. Files: `b4_r2_robustness.csv`,
`b4_template_support.csv`, `b4_per_axis_per_participant.csv`.

## B5 common grid: table

Ridge, MLP and Riemannian rescored on exactly EEGNet's test timestamps (100-ms grid, trial-wise repeats r0-r2; equality of the
timestamp sets asserted). Paired difference = EEGNet minus model, same timestamps.

| Target | Split | Model | paper R² | common-grid R² | EEGNet - model [95% CI] | Wilcoxon p | EEGNet better |
|---|---|---|---|---|---|---|---|
| force | trial | ridge | 0.262 | 0.236 | -0.063 [-0.170, 0.071] | 0.15 | 3/12 |
| force | trial | MLP | 0.053 | 0.051 | +0.123 [0.080, 0.173] | 0.0005 | 12/12 |
| force | trial | Riemannian | 0.183 | 0.182 | -0.009 [-0.046, 0.027] | 0.68 | 5/12 |
| force | trial | EEGNet | 0.173 | 0.173 | - | - | - |
| force | subject | ridge | -0.023 | -0.025 | -0.151 [-0.298, -0.019] | 0.064 | 3/12 |
| force | subject | MLP | -0.141 | -0.141 | -0.035 [-0.230, 0.128] | 0.38 | 9/12 |
| force | subject | Riemannian | -0.227 | -0.227 | +0.052 [-0.071, 0.181] | 0.38 | 7/12 |
| force | subject | EEGNet | -0.176 | -0.176 | - | - | - |
| position | trial | ridge | 0.346 | 0.340 | +0.052 [-0.012, 0.118] | 0.18 | 8/12 |
| position | trial | MLP | 0.168 | 0.167 | +0.224 [0.197, 0.252] | 0.0005 | 12/12 |
| position | trial | Riemannian | 0.322 | 0.325 | +0.066 [0.053, 0.082] | 0.0005 | 12/12 |
| position | trial | EEGNet | 0.391 | 0.391 | - | - | - |
| position | subject | ridge | 0.055 | 0.055 | -0.038 [-0.180, 0.114] | 0.42 | 5/12 |
| position | subject | MLP | -0.015 | -0.013 | +0.030 [-0.079, 0.115] | 0.23 | 9/12 |
| position | subject | Riemannian | -0.050 | -0.049 | +0.065 [-0.018, 0.141] | 0.18 | 8/12 |
| position | subject | EEGNet | 0.016 | 0.016 | - | - | - |

The common grid changes no conclusion (largest shift: ridge force trial-wise 0.262 -> 0.236, mostly from using 3 instead of
5 repeats). File: `b5_common_grid.csv`.

## B6 onset variability: table + FA-budget table

Test data at the validated threshold. Per-participant values: median [IQR] over the 12 participants. Detected test trials per
participant are summed over that participant's folds (trial-wise: 3 repeats). Lead over all detected trials pooled.

| Detector | Split | strict Pre | Loose | test FA/min | detected per participant min/median/max (total) | lead, all detected, median [IQR] ms |
|---|---|---|---|---|---|---|
| LDA | trial | 0.150 [0.066, 0.172] | 0.186 [0.109, 0.223] | 1.14 [0.76, 1.75] | 14/33/63 (404) | -57 [-160, 6] |
| LDA | subject | 0.029 [0.009, 0.088] | 0.070 [0.027, 0.113] | 0.57 [0.45, 1.56] | 2/20/61 (284) | -17 [-144, 50] |
| Riemannian | trial | 0.246 [0.131, 0.301] | 0.316 [0.185, 0.376] | 3.13 [2.60, 3.48] | 21/56/81 (611) | -128 [-270, 1] |
| Riemannian | subject | 0.017 [0.003, 0.108] | 0.017 [0.003, 0.137] | 0.35 [0.04, 4.00] | **0/5/197** (450) | -159 [-304, -30] |

Strict Pre at fixed TEST FA budgets (per participant: the best strict rate among thresholds whose fold-averaged test FA/min is
within budget; descriptive, read off the test curves). Mean [95% CI]:

| Detector | Split | 0.5/min | 1/min | 2/min | 5/min | 10/min |
|---|---|---|---|---|---|---|
| LDA | trial | 0.063 [0.039, 0.089] | 0.118 [0.084, 0.152] | 0.196 [0.145, 0.248] | 0.331 [0.268, 0.395] | 0.433 [0.370, 0.498] |
| LDA | subject | 0.014 [0.008, 0.022] | 0.033 [0.021, 0.046] | 0.062 [0.042, 0.086] | 0.112 [0.086, 0.139] | 0.221 [0.175, 0.272] |
| Riemannian | trial | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.025 [0.000, 0.076] | 0.260 [0.183, 0.337] | 0.381 [0.306, 0.455] |
| Riemannian | subject | 0.027 [0.014, 0.045] | 0.039 [0.020, 0.062] | 0.060 [0.035, 0.089] | 0.128 [0.084, 0.181] | 0.185 [0.134, 0.240] |

(The trial-wise Riemannian detector never reaches <= 1 test FA/min at any threshold for most participants.) Files:
`b6_onset_variability.csv`, `b6_fa_budget.csv`.

## B7 (if run)

Run in full. MLP with 1.0-s windows (100 samples, same bands, Welch nperseg 100):

| Target | Split | 0.5-s (paper) | 1.0-s | diff [95% CI] | p |
|---|---|---|---|---|---|
| force | trial-wise | 0.053 | 0.054 [0.037, 0.070] | +0.000 [-0.009, 0.010] | 1.00 |
| force | shuffled | 0.487 | **0.927** [0.923, 0.930] | +0.440 [0.426, 0.453] | 0.0005 |
| position | trial-wise | 0.168 | 0.180 [0.140, 0.228] | +0.012 [-0.009, 0.035] | 0.20 |
| position | shuffled | 0.401 | **0.810** [0.793, 0.825] | +0.408 [0.389, 0.428] | 0.0005 |

Longer windows do not help honest decoding but double the shuffled-split inflation (force: 0.43 -> 0.87).

## C figures: which files changed, smallest font per figure

| File | Status | Size | Smallest text at print width |
|---|---|---|---|
| `paper/figures/fig_onset.pdf` | **NEW (C1)**: strict / loose vs test FA (symlog, linthresh 0.1, 95% CI bands), validation operating points, cue-timer star at 0.997, lead boxes with n detected (404 / 284 / 611 / 450 / 2,124 cue); line styles + markers + hatching for grayscale | 7.0 x 2.3 in | 7.0 pt (7.16 pt at \textwidth) |
| `paper/figures/fig_onset_v1.pdf` | previous onset figure (kept) | 7.0 x 2.0 in | 5.8 pt |
| `paper/figures/fig_blocks.pdf` | **REGENERATED (C2)**: only change = y-label uses a full-size "R²" glyph (the mathtext superscript was 5.6 pt) | 3.5 x 1.9 in | 7.0 pt |
| `paper/figures/fig_blocks_v1.pdf` | previous version (kept) | 3.5 x 1.9 in | 5.6 pt (superscript "2") |
| `paper/figures/fig_blocks_v2.pdf` | **NEW (C3)**: + hollow markers for 0.5-s (circle) and 3.3-s (square) embargoes on the force panel; values < 0 as hollow down-triangles on the axis | 3.5 x 1.9 in | 7.0 pt |
| `paper/figures/fig_example.pdf` | unchanged since round 1 (already >= 7 pt) | 3.11 x 2.27 in | 7.0 pt native, 7.9 pt at \columnwidth |

**Upload to Overleaf:** `fig_onset.pdf` (new), `fig_blocks.pdf` (font fix), optionally `fig_blocks_v2.pdf`.
The new onset figure is 0.3 in taller than v1 (2.3 vs 2.0 in).

## D compile check: all items

Compiled `paper/main.tex` (my local copy: it still has the illustrated Fig. 1, the 3D blocked-CV figure and the
headline/chance/trajectory figures; your NCC version differs, see below) with the new `fig_onset.pdf`
-> `paper/compiled_check.pdf`:
- **Page count: 7** (6 before the new onset figure; page 7 holds only the last ~3 lines of the reference list).
- References start on page 5. No figure or table appears after the references (page 7 has no captions).
- Text bounds: left 48.5 pt, right 563.0 pt on a 612-pt page (right margin 49.0 pt).
- Body font 10 pt (most common size). Smallest text in the PDF: 4.3 pt inside `fig_traj3d.pdf` (not in your NCC version);
  table captions use IEEE small caps measured at 6.4 pt (template default); all figures in your version are >= 7 pt.
- PDF metadata: Creator "TeX", Producer "MiKTeX pdfTeX-1.40.28"; **no Author, no Title**.
- All 28 fonts embedded (Type 1 and TrueType subsets).
I cannot compile your Overleaf `main.tex` (0.75-in margins, Fig. 1 removed). With 0.75-in margins the left text edge should be
54 pt; please run the same checks on the Overleaf PDF, or send me the `.tex` and I will run them.

## Anything surprising or that contradicts the paper (most important section)

1. **The purged MLP value -0.03 (0.25-s blocks) is seed-dependent.** With identical training data (0.49-s vs 0.5-s embargo
   remove the same windows) the MLP gives -0.055 and -0.076; round 2 gave -0.03. Ridge and Riemannian are deterministic and
   reproduce exactly (0.29, 0.07). Suggest: "MLP force R² was about -0.03 to -0.08 (seed-dependent; CI includes 0)".
2. **Onset detection is largely cue/time-locked (B3).** With negatives taken at the same time after the cue, LDA AUC falls
   0.83 -> 0.70 (trial) and 0.71 -> 0.62 (subject), and strict pre-onset detection 13.9% -> 5.5%. A detector that sees only the
   time since the cue reaches AUC 0.98 and 97% strict pre-onset detection at ~0 FA/min (lead -400 ms); adding EEG to the cue time
   does not beat it (0.93-0.96). This strengthens the paper's "no advantage demonstrated" wording; I would state it explicitly.
3. **Ridge's honest score lives in the 0.1-0.3 Hz band (B2c).** Raising the high-pass to 0.3 Hz drops ridge from 0.26 to 0.02
   (force) and 0.35 to 0.07 (position), 0/12 participants improved. Very slow potentials (and possibly drifts/artifacts) carry
   the ridge signal; worth one limitation sentence.
4. **Whole-series filtering helps ridge even when it is leak-free (B2b):** series-wise +0.07 (force) and +0.12 (position);
   MLP/Riemannian unchanged. So per-trial filtering is conservative for ridge (not a leakage artifact); the main conclusion is
   unchanged (ridge 0.31/0.43 vs template 0.51/0.78 series-wise).
5. **The Riemannian leak at short blocks is about partial overlap, not training size (B1):** size-matched 0.29 vs embargoed
   0.07 at 0.25-s blocks. MLP needs near-duplicates (earlier finding); Riemannian also exploits partially overlapping windows.
6. **Longer windows double the inflation (B7):** shuffled MLP force 0.93 with 1-s windows vs 0.05 trial-wise.
7. **Trial-balanced force R² is negative for all models (B4a),** template included, because single-trial R² explodes when a
   trial's force barely varies; use pooled or median-trial R², not the trial-balanced mean, if you mention it.
8. **Subject-wise Riemannian onset detections are dominated by one participant** (detected trials per participant 0/5/197);
   its mean Det. 0.13 hides a median of 0.017.
9. **Cue timer strict rate is 0.997, not exactly 1** (8 of 3,528 trials start within 0.1 s of the cue or before it).
10. **My local layout went to 7 pages with the taller onset figure;** check your NCC layout after swapping in `fig_onset.pdf`.

## New files created this round

Code: `scripts/check_paper_numbers.py`, `scripts/paper_numbers_printed.csv`, `scripts/round3/b1_embargo.py`,
`scripts/round3/onset_round3.py`, `scripts/round3/onset_summary.py`, `scripts/round3/b2_b7_filtering_mlp.py`,
`scripts/round3/analysis_round3.py`, `scripts/round3/release_round3.py`; `paper/make_fig_onset_v2.py`,
`paper/make_fig_blocks_v2.py` (and a one-line label change in `paper/make_fig_blocks.py`).
Results (`results/round3/`): `ANSWERS.md`, `paper_number_check.csv`, `a1_cue_timer.csv`, `cue_timer_trials.csv`,
`onset_trials.csv.gz`, `onset_folds.csv`, `onset_reproduction_check.csv`, `b1_embargo.csv`, `b1_embargo_summary.csv`,
`b2_per_participant.csv`, `b2_b7_summary.csv`, `b3_onset_controls.csv`, `b4_r2_robustness.csv`, `b4_template_support.csv`,
`b4_per_axis_per_participant.csv`, `b5_common_grid.csv`, `b6_onset_variability.csv`, `b6_fa_budget.csv`, and logs
(`b1_embargo.log`, `onset_round3.log`, `b2_b7.log`, `b4.log`).
Data (derived, not results): `data/cache_round3/P1-P12.npz` (whole-series and 0.3-Hz filtered EEG).
Figures: `paper/figures/fig_onset.pdf` (+ `.png`), `fig_onset_v1.pdf`, `fig_blocks.pdf` (regenerated), `fig_blocks_v1.pdf`,
`fig_blocks_v2.pdf` (+ `.png`); `paper/compiled_check.pdf`.
Artifact (`anonymous_release/results/`): `results_by_subject.csv`, `onset_events.csv`, `splits.json`, `stats_families.csv`,
`environment.txt`, `README_reproduce.md`, `round3/` (copies of all round-3 CSVs); code and docs refreshed.
