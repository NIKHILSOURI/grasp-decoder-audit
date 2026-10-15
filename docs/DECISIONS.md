# Decisions and deviations from the build spec

Each entry: what changed, why, and where it lives in the code. Entries 1-6 come from facts found
in the data files, so they are corrections rather than choices.

## Data facts that changed the design

1. **WS windows overlap; trials are cut into disjoint segments.** Consecutive WS windows of a series
   overlap (58% of consecutive pairs, up to 4.7 s; elsewhere they leave gaps). A trial-wise split on WS windows would put the
   same raw samples in train and test. Every series is cut into disjoint segments
   `[LEDon_k - 0.8 s, min(LEDon_(k+1) - 0.8 s, window end))`. `data/load_wayeeggal.py`,
   `tests/test_windows.py::test_trial_segments_are_disjoint_on_series_clock`.
2. **Clock.** `trial_start_time` is the LED-on time on the HS clock; window k starts at HS sample
   `round(trial_start_time*500) - 1000`. Asserted sample-exact for all 3,528 trials.
3. **Grip force is the mean, not the sum,** of the two plates' normal forces: `GF = (FZ1+FZ2)/2`,
   stored negative. We use `-GF` (N). The scale does not affect R^2 or r.
4. **3,528 trials have targets, not 3,936.** One weight series per participant has EEG only (`ST`).
   Force decoding uses the constant-surface (weight) series: 2,040 trials (170 per participant).
5. **Hand position = sensor 4 (wrist).** Sensor 1 is the object.
6. **The pre-movement window overlaps the cue response.** Median reaction time is 0.3-0.8 s, so the
   "hand start - 300 ms" window often begins at or right after LED-on. This motivated the cue-timer
   baseline, the cue-locking slope and the no-occipital variant in E4.

## Additions to the spec (to make the audit conclusive)

7. **No-EEG template baseline** (`models/template.py`): predicts the mean cue-locked target curve of the
   training trials, using only the time since the segment start. It runs in every split scheme.
8. **Series-wise split** (`data/splits.py`): within subject, whole recording series held out (3 folds).
   It tests robustness to drift within a session, which trial-wise splits cannot show.
9. **EEGForceMap-style decoder** (`features/forcemap.py`): 9 channels as published (C3, CP1, CP2, Cz,
   FC2, FC6, Fp1, P3, C4), 100 ms windows with a 50 ms step, ERP statistics + theta/mu power + ERD/ERS,
   PCA (95 %, fit on training data), MLP. Described as "in the style of"; details the paper leaves
   open (filters, envelope, baseline) are our choices.
10. **E6 blocked cross-validation** (`splits.blocked`): within each trial, contiguous blocks of B seconds
    (0.05, 0.25, 0.5, 1, 2, 4, 8 s) are assigned 7:2:1 to train/val/test. B = 0.05 s reproduces the
    shuffled split; longer blocks leave fewer test windows that overlap a training window. It shows
    *why* shuffled windows inflate scores while the training-set size stays constant.
    A first design (purge every training window within a gap of any test window) was abandoned: in a
    shuffled split test windows are so dense (one every ~0.25 s) that a 1 s gap removed every training
    window, confounding leakage with training-set size.
11. **Holm correction** across the Wilcoxon tests in the inflation table.
12. **Window-level AUC** for the onset classifiers, in addition to the online metrics.

## Implementation choices

13. **Offline filters per trial segment** (zero-phase, reflect-padded) so a test trial's filtered EEG
    never contains samples from a neighbouring training trial. The 0.1-3 Hz band uses a zero-phase
    Butterworth (order 2) after removing the pre-cue offset, because a linear-phase FIR with a 0.1 Hz
    edge would be longer than a trial. The 1-40 Hz band uses a 1,651-tap zero-phase FIR as specified.
14. **Online (causal) filters run over the continuous HS series**, as a real-time system would.
15. **Ridge alpha** is chosen per output on the validation set from logspace(-2, 4, 13), solved in
    closed form (one eigendecomposition). Equivalent to RidgeCV with a fixed validation set.
16. **MLP in PyTorch** (GPU), not sklearn's MLPRegressor, so early stopping uses *our* validation split
    (sklearn would carve its own validation set out of the training windows). Hidden (128, 64), ReLU,
    AdamW, weight decay 1e-4, batch 512, max 200 epochs, patience 10.
17. **Multi-output heads.** For kinematics, MLP, EEGNet and the ForceMap-style MLP predict x, y and z
    jointly (3 outputs); ridge, Riemannian and template are fit per output (closed form, identical to
    separate models). This cuts EEGNet cost by 3x.
18. **Riemannian reference** = Karcher mean of 2,000 random training covariances (GPU implementation,
    matches pyriemann to 5e-11 relative error; `tests/test_covariance.py`). Tangent mapping in
    float32 on the GPU (relative error ~3e-5 versus float64). OAS covariance vectorised (matches
    sklearn). No per-subject re-centering in the subject-wise split (that would use test-subject data).
19. **EEGNet**: braindecode 1.8 `EEGNet` (the renamed EEGNetv4), regression head, per-channel input
    scaling from training windows, AdamW (lr 1e-3, weight decay 1e-2), batch 256, max 40 epochs,
    patience 8, AMP.
20. **Strict subject-wise normalisation**: targets are z-scored with pooled training-subject statistics;
    no per-subject statistics of the test participant are used anywhere.
21. **Splits** return ID lists (window, trial, series or subject IDs); `split_masks` asserts disjointness
    and, for honest schemes, that no trial straddles partitions.

## Budget reductions (spec order: stride, epochs, permutations, subjects)

22. **EEGNet repeats**: 3 repeats (instead of 5) of the shuffled and trial-wise splits. Stride
    (100 ms), max epochs (40), subjects (12) and permutations (100) are as specified.
23. **E1 run in two processes** (EEGNet separately) sharing one GPU; analysis merges completed E1 runs
    (`experiments.load_e1`).
24. **Total compute exceeded the 2-hour cap**, mainly EEGNet (40 epochs rarely stop early on the small
    within-subject training sets) plus the added experiments. Measured wall-clock times per run are
    listed in `results/RESULTS.md` and each run's `timing.json`. No result depends on the cap.
    Approximate totals: E1 main 140 min; E1 EEGNet ~140 min of GPU time (spread over interrupted and
    resumed runs, see 25); E2 52; E3 3; E4 30; E5 16; E6 37.
25. **GPU memory overcommit and resume.** Running three or four GPU jobs at once on the 6 GB card filled
    its memory; Windows then pages GPU memory to system RAM and CUDA slows by orders of magnitude.
    Stalled runs were stopped and rerun with at most two GPU jobs at a time. E1 gained a
    `+resume_from=<run dir>` option: schemes whose prediction files exist are scored from those files
    instead of retrained (verified to reproduce stored metrics to 1e-8). The final EEGNet E1 run
    (`*_eegnet`) reused the 7 completed schemes and trained only kinematics subject-wise; its folds.csv
    has no `n_train`/`fit_s` for the reused schemes.
26. **E6 repeats**: one repeat of the blocked split per participant and block length.

## Onset detection (E4)

27. **Rest negatives**: windows ending in `[HandStop_(k-1) + 0.5 s, LEDon_k - 0.5 s]` instead of
    "more than 1,000 ms before LED-on", because inter-trial rests are often shorter than that. The same
    intervals are used to count false alarms. The first trial of each series starts its rest 2 s into
    the stream (causal-filter warm-up).
28. **Operating point** chosen on validation trials (trial-wise) or validation participants
    (subject-wise): the lowest threshold with <= 1 false alarm per minute. Full threshold sweeps are
    reported as curves.
29. **Trial-wise onset folds**: 3 repeats of a 7:2:1 split of all 294 trials per participant.
30. **Cue-timer baseline**: a "detector" that fires 100 ms after LED-on, using no EEG.
31. **Cue-locking slope**: regression of detection latency (from LED-on) on reaction time. A detector
    that tracks movement onset has slope ~1; one that tracks the visual cue has slope ~0.

## Chance controls (E3)

32. The two best EEG decoders are chosen by mean trial-wise R^2 in E1; the template is added for
    reference. Mismatched targets are aligned at the segment start (0.8 s before the cue) and
    edge-padded; a trial is never paired with itself (derangement, `tests/test_chance.py`). Per
    participant, null scores are averaged over folds for each permutation index.

## Environment

33. No conda on the build machine: a pip virtual environment (`.venv`) with PyTorch 2.6.0+cu124.
    `environment.yml` pins the same versions for conda users. torchaudio is pinned to 2.6.0 because
    braindecode imports it and a mismatched build fails to load on Windows.

## Residual analysis (E7, added for the paper)

34. **Template plus EEG** (`models/residual.py`, `experiments.e7_residual`): the no-EEG template is fit on
    the training trials and a ridge or MLP decoder is trained on its residual; the prediction is template +
    predicted residual. Run trial-wise (5 repeats) and subject-wise, force and kinematics. The template and
    the combined model use the same folds, so the gain and the residual R^2 are paired per fold.
