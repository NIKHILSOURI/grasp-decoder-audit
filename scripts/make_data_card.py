"""Write docs/DATA_CARD.md from the raw release and the cache (files, fields, trial counts)."""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from gaudit.data.cache import SUBJECTS, load_cached
from gaudit.data.load_wayeeggal import BLOCK_TYPES

RAW, CACHE, OUT = Path("data/raw"), Path("data/cache"), Path("docs/DATA_CARD.md")


def main():
    man = json.loads((RAW / "manifest.json").read_text()) if (RAW / "manifest.json").exists() else {}
    ws = sio.loadmat(RAW / "P1" / "WS_P1_S1.mat")["ws"][0, 0]
    hs = sio.loadmat(RAW / "P1" / "HS_P1_S1.mat")["hs"][0, 0]
    al = sio.loadmat(RAW / "P1" / "P1_AllLifts.mat")["P"][0, 0]
    eeg = [str(x[0]) for x in np.ravel(ws["names"][0, 0]["eeg"])]
    kin = [str(x[0]) for x in np.ravel(ws["names"][0, 0]["kin"])]
    emg = [str(x[0]) for x in np.ravel(ws["names"][0, 0]["emg"])]
    cols = [str(c[0][0]) for c in al["ColNames"]]
    win_fields = list(ws["win"].dtype.names)
    hs_fields = {k: list(hs[k][0, 0].dtype.names) for k in ("eeg", "kin", "emg", "env", "misc")}
    misc = [str(x[0]) for x in np.ravel(hs["misc"][0, 0]["names"])]

    rows, tot, tot_force = [], 0, 0
    for s in SUBJECTS:
        d = load_cached(CACHE, s)
        files = sorted(p.name for p in (RAW / f"P{s}").glob("*.mat"))
        per_series = {int(k): int((d.series == k).sum()) for k in np.unique(d.series)}
        bt = {int(k): BLOCK_TYPES[int(d.block_type[d.series == k][0])][0].upper() for k in np.unique(d.series)}
        rt = np.median(d.ev("tHandStart") - d.ev("LEDOn")) * 1000
        dur = np.median(d.length) / 100
        nf = int((d.block_type == 1).sum())
        tot += len(d.trial_id)
        tot_force += nf
        rows.append(f"| P{s} | {len(files)} | {len(d.trial_id)} | {nf} | "
                    + " ".join(f"S{k}:{per_series[k]}{bt[k]}" for k in per_series)
                    + f" | {rt:.0f} | {dur:.2f} | {d.force.max():.1f} |")

    zips = "\n".join(f"| P{k} | {v['file']} | {v['size'] / 1e6:.0f} MB | `{v['md5']}` |" for k, v in sorted(
        man.items(), key=lambda kv: int(kv[0])))
    text = f"""# Data card: WAY-EEG-GAL as used in this project

Source: Luciw, Jarocka & Edin (2014), *Multi-channel EEG recordings during 3,936 grasp and lift
trials with varying weight and friction*, Scientific Data 1:140047, doi:10.1038/sdata.2014.47.
Files: figshare collection 988376 (doi:10.6084/m9.figshare.988376), one archive per participant,
fetched by `scripts/download_data.py` and verified against figshare's MD5. Not redistributed here.

**License.** The paper states CC BY 4.0; the figshare records list CC0. We follow the stricter
reading (CC BY 4.0): cite the dataset paper.

## Archives

| Participant | Archive | Size | MD5 |
|---|---|---|---|
{zips}

## Files per participant (verified)

* `HS_P<n>_S<s>.mat` (s = 1..9): continuous series. Struct `hs` with fields `name`, `participant`,
  `series` and substructures `eeg`, `kin`, `emg`, `env`, `misc`, each with fields {hs_fields['eeg']}.
  EEG 500 Hz (32 ch), kin 500 Hz (36 ch), EMG 4 kHz (5 ch). `misc` channels: {misc}.
* `HS_P<n>_ST.mat`: EEG-only series (the weight series whose non-EEG data were withheld). Not used.
* `WS_P<n>_S<s>.mat`: windowed lifts. Struct `ws.win(k)` with fields: {', '.join(f'`{f}`' for f in win_fields)}.
  EEG `(n, 32)` float32, kin `(n, 45)` (36 raw + 9 derived), EMG `(n_emg, 5)`.
* `P<n>_AllLifts.mat`: `P.AllLifts` (lifts x 43) with column names `P.ColNames`.

**EEG channels (32):** {', '.join(eeg)}

**EMG channels (5):** {', '.join(emg)}

**Kinematic/force channels in WS (45):** {', '.join(kin)}

**AllLifts columns (43):** {', '.join(cols)}

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
   **{tot} trials** have targets ({tot // 12} per participant), not 3,936. Force decoding uses the
   constant-surface (weight) series only: **{tot_force} trials**.
6. **Events.** AllLifts event times are seconds relative to the WS window start (`LEDOn == 2`).
   Median reaction time (HandStart - LEDOn) is only 0.3-0.8 s, so "pre-movement" windows overlap the
   visual response to the LED cue.

## Per participant

S = series, W/F/M = weight/friction/mixed block. RT = median HandStart - LEDOn. Duration = median
disjoint-segment length used for decoding.

| Participant | .mat files | Trials with targets | Force trials (weight series) | Trials per series | RT (ms) | Duration (s) | Max grip force (N) |
|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + """

## Preprocessing (cache: `data/cache/P<n>.npz`, `P<n>_stream.npz`)

* Offline (decoding): per trial segment, common average reference, zero-phase FIR 1-40 Hz (1651 taps)
  and zero-phase Butterworth 0.1-3 Hz (order 2); decimated to 100 Hz. Targets low-passed (20 Hz,
  zero-phase) and decimated to 100 Hz.
* Online (onset): causal Butterworth 1-40 Hz (order 4) and 0.1-3 Hz (order 2) over the continuous HS
  series; decimated to 100 Hz.
"""
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
