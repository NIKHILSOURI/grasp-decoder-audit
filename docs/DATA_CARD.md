# Data card: WAY-EEG-GAL as used in this project

Source: Luciw, Jarocka & Edin (2014), *Multi-channel EEG recordings during 3,936 grasp and lift
trials with varying weight and friction*, Scientific Data 1:140047, doi:10.1038/sdata.2014.47.
Files: figshare collection 988376 (doi:10.6084/m9.figshare.988376), one archive per participant,
fetched by `scripts/download_data.py` and verified against figshare's MD5. Not redistributed here.

**License.** The paper states CC BY 4.0; the figshare records list CC0. We follow the stricter
reading (CC BY 4.0): cite the dataset paper.

## Archives

| Participant | Archive | Size | MD5 |
|---|---|---|---|
| P1 | P1.zip | 839 MB | `db738ebf022abd6826879d9afee3202c` |
| P2 | P2.zip | 950 MB | `1bd75993ae538c35f8c403e4e5ff40e5` |
| P3 | P3.zip | 744 MB | `891bfca8516056201e5a214b2fde4558` |
| P4 | P4.zip | 802 MB | `b921d5e1d82151bf00fc041419ef22e9` |
| P5 | P5.zip | 842 MB | `2ef0bc0c9bc2840fd28190a6fc5e9602` |
| P6 | P6.zip | 862 MB | `5cf49ca627cbc3e8ef8e3c03a180c1d3` |
| P7 | P7.zip | 1037 MB | `125e2958872b3c06aeb3b729f282482a` |
| P8 | P8.zip | 782 MB | `56e39cfa3e7ea8361d51e8810fd65cb2` |
| P9 | P9.zip | 832 MB | `6e3eda3b36de5c1bbf09c5432258af79` |
| P10 | P10.zip | 852 MB | `4037a88eedc22bc6a3cbb786ab993ac2` |
| P11 | P11.zip | 888 MB | `45b7e844786b4707f5cd0dda42ecb63a` |
| P12 | P12.zip | 909 MB | `d8e1d2bfd8cf58ceff8d9d21a7114e85` |

## Files per participant (verified)

* `HS_P<n>_S<s>.mat` (s = 1..9): continuous series. Struct `hs` with fields `name`, `participant`,
  `series` and substructures `eeg`, `kin`, `emg`, `env`, `misc`, each with fields ['names', 'sig', 'samplingrate'].
  EEG 500 Hz (32 ch), kin 500 Hz (36 ch), EMG 4 kHz (5 ch). `misc` channels: ['Button', 'Magnet', 'SurfaceLED', 'ParticipantLED', 'Temperature1', 'Temperature2'].
* `HS_P<n>_ST.mat`: EEG-only series (the weight series whose non-EEG data were withheld). Not used.
* `WS_P<n>_S<s>.mat`: windowed lifts. Struct `ws.win(k)` with fields: `eeg`, `kin`, `emg`, `eeg_t`, `emg_t`, `trial_start_time`, `LEDon`, `LEDoff`, `trial_end_time`, `weight`, `weight_id`, `surf`, `surf_id`, `weight_prev`, `weight_prev_id`, `surf_prev`, `surf_prev_id`.
  EEG `(n, 32)` float32, kin `(n, 45)` (36 raw + 9 derived), EMG `(n_emg, 5)`.
* `P<n>_AllLifts.mat`: `P.AllLifts` (lifts x 43) with column names `P.ColNames`.

**EEG channels (32):** Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, TP9, CP5, CP1, CP2, CP6, TP10, P7, P3, Pz, P4, P8, PO9, O1, Oz, O2, PO10

**EMG channels (5):** Anterior Deltoid, Brachoradial, Flexor Digitorum, Common Extensor Digitorum, First Dorsal Interosseus

**Kinematic/force channels in WS (45):** Ae1 - angle e sensor 1, Ae2 - angle e sensor 2, Ae3 - angle e sensor 3, Ae4 - angle e sensor 4, Ar1 - angle r sensor 1, Ar2 - angle r sensor 2, Ar3 - angle r sensor 3, Ar4 - angle r sensor 4, Az1 - angle z sensor 1, Az2 - angle z sensor 2, Az3 - angle z sensor 3, Az4 - angle z sensor 4, FX1 - force x plate 1, FX2 - force x plate 2, FY1 - force y plate 1, FY2 - force y plate 2, FZ1 - force z plate 1, FZ2 - force z plate 2, Px1 - position x sensor 1, Px2 - position x sensor 2, Px3 - position x sensor 3, Px4 - position x sensor 4, Py1 - position y sensor 1, Py2 - position y sensor 2, Py3 - position y sensor 3, Py4 - position y sensor 4, Pz1 - position z sensor 1, Pz2 - position z sensor 2, Pz3 - position z sensor 3, Pz4 - position z sensor 4, TX1 - torque x plate 1, TX2 - torque x plate 2, TY1 - torque y plate 1, TY2 - torque y plate 2, TZ1 - torque z plate 1, TZ2 - torque z plate 2, IndLF, ThuLF, LF, IndGF, ThuGF, GF, IndRatio, ThuRatio, GFLFRatio

**AllLifts columns (43):** Part, Run, Lift, CurW, CurS, PrevW, PrevS, StartTime, LEDOn, LEDOff, BlockType, tIndTouch, tThumbTouch, tFirstDigitTouch, tBothDigitTouch, tIndStartLoadPhase, tThuStartLoadPhase, tBothStartLoadPhase, tLiftOff, tReplace, tIndRelease, tThuRelease, tBothReleased, GF_Max, LF_Max, dGF_Max, dLF_Max, tGF_Max, tLF_Max, tdGF_Max, tdLF_Max, GF_Hold, LF_Hold, tHandStart, tHandStop, tPeakVelHandReach, tPeakVelHandRetract, GripAparture_Max, tGripAparture_Max, Dur_Reach, Dur_Preload, Dur_LoadPhase, Dur_Release

## Facts verified on the files (they change how the data must be used)

1. **Clock.** `trial_start_time` (WS) and `StartTime` (AllLifts) are the LED-on time on the HS clock,
   not the window start. WS window k starts at HS sample `round(trial_start_time*500) - 1000` and is a
   sample-exact copy of the HS EEG and kinematics there (asserted for every one of the 3,528 trials).
2. **Overlap.** WS windows run from 2 s before LED-on to 3 s after LED-off, so consecutive windows of
   a series overlap (58% of consecutive pairs; median 1.5 s, up to 4.7 s); elsewhere they leave gaps of up to about 12 s.
   A trial-wise split on the raw WS windows would therefore put identical samples in train and test.
   We cut every series into **disjoint segments** `[LEDon_k - 0.8 s, min(LEDon_(k+1) - 0.8 s, window end))`.
3. **Grip force.** `GF` = (FZ1 + FZ2) / 2, the *mean* normal force of the two plates (exact to 1e-15
   on P1), stored with a negative sign; we use `-GF` in newtons. (The spec assumed a sum.)
4. **Hand position.** Position sensor 4 (`Px4, Py4, Pz4`, cm) is the wrist sensor; sensor 1 is the object.
5. **Trials with targets.** One weight series per participant has no kinematics/force (`ST` files), so
   **3528 trials** have targets (294 per participant), not 3,936. Force decoding uses the
   constant-surface (weight) series only: **2040 trials**.
6. **Events.** AllLifts event times are seconds relative to the WS window start (`LEDOn == 2`).
   Median reaction time (HandStart - LEDOn) is only 0.3-0.8 s, so "pre-movement" windows overlap the
   visual response to the LED cue.

## Per participant

S = series, W/F/M = weight/friction/mixed block. RT = median HandStart - LEDOn. Duration = median
disjoint-segment length used for decoding.

| Participant | .mat files | Trials with targets | Force trials (weight series) | Trials per series | RT (ms) | Duration (s) | Max grip force (N) |
|---|---|---|---|---|---|---|---|
| P1 | 20 | 294 | 170 | S1:34W S2:34F S3:28M S4:34W S5:28M S6:34F S7:34W S8:34W S9:34W | 318 | 7.28 | 15.4 |
| P2 | 20 | 294 | 170 | S1:28M S2:34F S3:34W S4:34F S5:34W S6:28M S7:34W S8:34W S9:34W | 648 | 8.74 | 11.0 |
| P3 | 20 | 294 | 170 | S1:34W S2:28M S3:34F S4:28M S5:34F S6:34W S7:34W S8:34W S9:34W | 318 | 7.22 | 13.2 |
| P4 | 20 | 294 | 170 | S1:34F S2:28M S3:34W S4:28M S5:34W S6:34F S7:34W S8:34W S9:34W | 312 | 7.39 | 20.5 |
| P5 | 20 | 294 | 170 | S1:28M S2:34W S3:34F S4:34W S5:34F S6:28M S7:34W S8:34W S9:34W | 535 | 8.09 | 9.3 |
| P6 | 20 | 294 | 170 | S1:34W S2:34F S3:28M S4:34W S5:28M S6:34F S7:34W S8:34W S9:34W | 426 | 8.09 | 15.0 |
| P7 | 20 | 294 | 170 | S1:34F S2:34W S3:28M S4:34F S5:28M S6:34W S7:34W S8:34W S9:34W | 772 | 10.37 | 24.3 |
| P8 | 20 | 294 | 170 | S1:28M S2:34F S3:34W S4:34F S5:34W S6:28M S7:34W S8:34W S9:34W | 376 | 7.59 | 16.0 |
| P9 | 20 | 294 | 170 | S1:34W S2:28M S3:34F S4:28M S5:34F S6:34W S7:34W S8:34W S9:34W | 392 | 7.92 | 20.7 |
| P10 | 20 | 294 | 170 | S1:34F S2:28M S3:34W S4:28M S5:34W S6:34F S7:34W S8:34W S9:34W | 408 | 8.20 | 16.7 |
| P11 | 20 | 294 | 170 | S1:28M S2:34W S3:34F S4:34W S5:34F S6:28M S7:34W S8:34W S9:34W | 421 | 8.32 | 28.3 |
| P12 | 20 | 294 | 170 | S1:34W S2:34F S3:28M S4:34W S5:28M S6:34F S7:34W S8:34W S9:34W | 490 | 8.50 | 18.2 |

## Preprocessing (cache: `data/cache/P<n>.npz`, `P<n>_stream.npz`)

* Offline (decoding): per trial segment, common average reference, zero-phase FIR 1-40 Hz (1651 taps)
  and zero-phase Butterworth 0.1-3 Hz (order 2); decimated to 100 Hz. Targets low-passed (20 Hz,
  zero-phase) and decimated to 100 Hz.
* Online (onset): causal Butterworth 1-40 Hz (order 4) and 0.1-3 Hz (order 2) over the continuous HS
  series; decimated to 100 Hz.
