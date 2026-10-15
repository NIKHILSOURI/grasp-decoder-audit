# Evaluation checklist for EEG movement decoding (one page)

Use it before you trust, report or review a continuous EEG decoding score.

## 1. Splits

- [ ] **No shuffled overlapping windows.** Windows of one trial must never be split across train and test.
      If windows overlap or targets are smooth, a random window split measures memorisation.
- [ ] **Trials are really disjoint.** Check that stored trial epochs do not overlap in time (in
      WAY-EEG-GAL they do, by up to ~2 s); cut a continuous recording into disjoint segments.
- [ ] **Report at least trial-wise and subject-wise results**, and ideally series/session-wise
      (drift between runs can turn a positive R^2 negative).
- [ ] **Subject-independent means leave-one-subject-out over all subjects**, not one held-out subject.

## 2. Fitting

- [ ] Every fitted step sees training data only: scalers, channel/feature selection, PCA,
      covariance reference, target normalisation, hyperparameters, early stopping.
- [ ] Test data are scored once; no model or threshold is selected on test scores.
- [ ] Filtering is stateless and does not mix test and training samples (filter per trial, or causally).

## 3. Baselines and chance

- [ ] **A no-EEG baseline**: predict the target from time since the cue (the mean cue-locked curve).
      In cued, stereotyped tasks this alone can reach R^2 = 0.6-0.9. EEG must beat it, not zero.
- [ ] **An empirical chance level**: pair test EEG with targets of *other* trials (and time-reversed
      targets). Report the share of participants above their own 95th percentile.
- [ ] Report R^2 (or RMSE in physical units) as the main metric. Pearson r ignores offset and scale and
      stays high for slow targets even under chance pairings.

## 4. Validity

- [ ] Check artifact-prone channels (frontal-polar for eyes, occipital for visual cues, temporal for
      muscle): does the score survive without them?
- [ ] Inspect predictions over whole trials, not just a score.
- [ ] For "intention" or onset detection: report lead time *and* false alarms per minute of rest,
      compare against a cue-timer baseline, and test whether detections track the movement or the cue.

## 5. Reporting

- [ ] Per-participant results, paired tests across participants, confidence intervals.
- [ ] Exact channel count, sampling rate, window length, stride, and which trials/series were used.
- [ ] Release code and the split IDs.
