# Is your grasp decoder cheating?

A leakage-controlled re-evaluation of EEG grip-force and hand-kinematics decoding on the open
WAY-EEG-GAL dataset (12 participants, 32-channel EEG at 500 Hz, grasp-and-lift task).

It answers four questions:

| | Question | Experiment |
|---|---|---|
| RQ1 | How much do shuffled overlapping windows inflate scores versus trial-, series- and subject-wise splits? | E1, E6 |
| RQ2 | How much does channel/feature selection on all data inflate scores? | E2 |
| RQ3 | How much honest decoding is above an empirical chance level, and above a no-EEG baseline? | E1, E3 |
| RQ4 | How early, and with how many false alarms, can causal EEG detect grasp onset? | E4 |
| validity | Do the scores survive removing artifact-prone channels? | E5 |

## Key results (12 participants; full tables in `results/RESULTS.md` and `results/tables/`)

![headline](results/figures/fig1_headline.png)

**Test R² (mean over participants)**

| Decoder | Force: shuffled (leaky) | trial-wise | series-wise | subject-wise | Position: shuffled | trial-wise | series-wise | subject-wise |
|---|---|---|---|---|---|---|---|---|
| **No-EEG template** | **0.53** | **0.52** | **0.51** | **0.19** | **0.79** | **0.79** | **0.78** | **0.55** |
| Ridge (lagged 0.1-3 Hz) | 0.35 | 0.26 | 0.24 | -0.02 | 0.39 | 0.35 | 0.31 | 0.06 |
| MLP (band power) | 0.49 | 0.05 | 0.04 | -0.14 | 0.40 | 0.17 | 0.15 | -0.01 |
| Riemannian | 0.38 | 0.18 | 0.14 | -0.23 | 0.43 | 0.32 | 0.29 | -0.05 |
| EEGNet | 0.26 | 0.17 | 0.08 | -0.18 | 0.44 | 0.39 | 0.34 | 0.02 |
| EEGForceMap-style | 0.14 | 0.02 | 0.00 | -0.10 | - | - | - | - |

1. **Shuffled windows inflate scores** for every EEG decoder: shuffled > trial-wise in 12/12
   participants (Wilcoxon, Holm-corrected p = 0.011) for all decoder-target pairs except EEGNet on
   force (10/12, p = 0.16). The no-EEG template is not inflated (p = 0.68), as expected. The inflation
   depends on model capacity: +0.43 R² for the MLP on force (0.49 -> 0.05), +0.20 Riemannian, +0.09
   ridge. Blocked CV (E6) shows the mechanism: the MLP's score collapses as soon as train and test
   windows stop overlapping (0.25 s blocks).
2. **A predictor that never sees EEG beats every EEG decoder** in every split scheme. It outputs the
   mean cue-locked force or position curve of the training trials. No EEG decoder beats it in more
   than 2 of 12 participants under any split.
3. **Across participants, honest EEG decoding is at or below zero R²** (force -0.23 to -0.02,
   position -0.05 to 0.06), while Pearson r stays 0.12-0.40. Correlation flatters slow targets.
4. **There is real, small trial-specific information.** Against an empirical chance level (EEG paired
   with another trial's target), ridge and EEGNet beat chance in 92-100 % of participants trial-wise,
   but only by about 0.05-0.10 R², and chance itself is high (R² 0.12-0.32), because every trial shares
   the same cue-locked shape.
5. **Leaky feature selection barely matters here** (top-16 selection on all data: +0.00 to +0.02 R²,
   mostly n.s.). With tens of thousands of windows the dominant leak is window overlap, not selection.
6. **Removing frontal-polar and occipital channels lowers honest scores by 0.03-0.06 R²**, so part of
   the signal is eye/visual activity.
7. **Causal grasp-onset detection is weak at a usable false-alarm rate.** Window AUC 0.68-0.83, but at
   <= 1 false alarm per minute (threshold chosen on validation data) only 8-29 % of grasps are detected,
   15-186 ms before the hand moves. A cue timer that fires 100 ms after the LED, using no EEG, detects
   100 % of grasps a median 315 ms early. Detection latencies only partly track reaction time
   (slope 0.2-0.6), i.e. the detectors partly respond to the visual cue.
8. **The EEGForceMap-style pipeline does not reproduce the published R² of 0.82**: we obtain 0.14 even
   with the leaky shuffled split and 0.02 trial-wise (details that the paper leaves open may differ).

Figures (`results/figures/`, PDF + PNG): `fig0` split schematic, `fig1` headline, `fig2` per-participant
lines, `fig3` chance levels, `fig4` example trials, `fig5` onset detection, `fig6` 3D wrist paths,
`fig7` 3D blocked-CV leakage landscape, `fig8` scalp maps, `fig9` r versus R², `fig10` selection and
artifact checks.

## Reproduce

Requires Python 3.11, an NVIDIA GPU (6 GB is enough), about 16 GB RAM and 15 GB disk.
The frozen results of the paper are in [`results_summary/`](results_summary/README.md) if you only want to inspect them.

**Linux / macOS / Git Bash**

```bash
make setup        # .venv with PyTorch 2.6 (CUDA 12.4) + this package
make data         # ~10 GB from figshare (doi:10.6084/m9.figshare.988376), MD5-verified, resumable
make preprocess   # ~1 min: disjoint trial segments, filters, 100 Hz cache in data/cache/
make test         # 40 tests: splits, leakage spies, windows, metrics, chance, causality
make pilot        # optional timing pilot
make e1 e2 e3 e4 e5 e6 e7
make figures tables
```

**Windows (CMD)**

```bat
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -I scripts\download_data.py --out data\raw
.venv\Scripts\python scripts\preprocess_all.py --jobs 4
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\run_experiment.py experiment=e1_splits
.venv\Scripts\python scripts\make_figures.py
.venv\Scripts\python scripts\make_tables.py
```
(Run the other experiments with `experiment=e2_selection`, `e3_chance`, `e4_onset`, `e5_artifacts`, `e6_blocks`, `e7_residual`.)

**Audit checks** (after the experiments): `python scripts/verify_paper_claims.py`, `python scripts/verify_round2.py`,
the scripts in `scripts/round3/`, and `python scripts/check_paper_numbers.py`, which recomputes every number printed in the
paper and compares it with `scripts/paper_numbers_printed.csv`.

Single experiments: `python scripts/run_experiment.py experiment=e1_splits [seed=0] [subjects=[1,2,3]]`.
Every run writes `results/<experiment>/<timestamp>/` with the resolved config, git hash, seed,
library versions and GPU name (`metrics.json`), per-fold and per-subject CSVs, and `timing.json`.
Figures go to `results/figures/` (PDF + PNG), LaTeX tables to `results/tables/`, and a Markdown
digest of all numbers to `results/RESULTS.md`.

## What is in here

```
configs/                Hydra configs: config.yaml, model/, split/, target/, experiment/
scripts/                download, preprocess, run_experiment, time_pilot, make_figures, make_tables, make_data_card,
                        verification and number-check scripts
scripts/round3/         round-3 audit analyses (embargo, filtering, onset controls, robustness, artifact files)
src/gaudit/data/        loader (WS/HS/AllLifts), preprocessing, cache, windows, splits, targets
src/gaudit/features/    lagged low-frequency EEG, band power, OAS covariance + tangent space, selection, ForceMap-style
src/gaudit/models/      ridge, MLP, Riemannian, EEGNet, no-EEG template, template + EEG residual, torch trainer
src/gaudit/onset/       labels, causal detectors (LDA, Riemannian), online evaluation
src/gaudit/eval/        metrics (R^2, r, RMSE), statistics (Wilcoxon, bootstrap, Holm)
src/gaudit/experiments.py  E1-E7 (chance controls in e3_chance)
tests/                  40 tests, must pass before experiments
results_summary/        frozen per-participant results, tables, split manifests and audit outputs
docs/DATA_CARD.md       files, fields, channel names, trial counts and verified facts about the release
docs/DECISIONS.md       every deviation from the build spec, with reasons
docs/CHECKLIST.md       one-page evaluation checklist for EEG movement decoding
```

## Split schemes

| Scheme | Rule | Leaky? |
|---|---|---|
| `shuffled` | within subject, shuffle all 500 ms windows (50 ms stride), 7:2:1 | **yes, on purpose** |
| `trialwise` | within subject, whole trials 7:2:1, 5 seeds | no |
| `serieswise` | within subject, whole recording series held out, 3 folds | no |
| `subjectwise` | leave one participant out, 2 validation participants | no |

Trials are first cut into **disjoint** segments, because the dataset's stored trial windows overlap
(58% of consecutive pairs, by up to 4.7 s; see `docs/DATA_CARD.md`). Every fitted step (target scaler, feature scaler, selection,
PCA, Riemannian reference, ridge alpha, early stopping) sees training data only; `tests/test_no_leakage.py`
plants a sentinel in every test row and spies on each fit.

## Data and licence

WAY-EEG-GAL: Luciw, Jarocka & Edin (2014), *Scientific Data* 1:140047, doi:10.1038/sdata.2014.47
(CC BY 4.0). The download script fetches the files from the original figshare source; this
repository does not redistribute them.

## Related work this re-evaluates

The leaky protocol is implemented "in the style of" EEGForceMap (Dangi & Meena, IEEE SMC 2025,
arXiv:2508.07677), which reports a random 7:2:1 split of each participant's windows and a
subject-independent test on one held-out participant. Differences from that pipeline are listed in
`docs/DECISIONS.md`. Nothing here assumes intent; it measures what each protocol can and cannot show.
