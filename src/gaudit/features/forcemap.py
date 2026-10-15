"""Features "in the style of" EEGForceMap (Dangi & Meena, SMC 2025; arXiv 2508.07677).

As described in that paper: 9 premotor-parietal (+Fp1) channels; 100 ms windows with a
50 ms step; per-channel ERP statistics (mean, mean |x|, variance, skewness, kurtosis, area),
theta (4-8 Hz) and mu (9-11 Hz) power, and ERD/ERS of both bands against a baseline.
Our re-implementation differs in details the paper does not fix (filter design, envelope
estimator, baseline = the trial's own pre-cue period). Everything here is stateless per
trial; the PCA (95 % variance) the paper applies is fit inside the decoder on training data.
"""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

from ..data.cache import FS, SubjectData

FM_CHANNELS = ["C3", "CP1", "CP2", "Cz", "FC2", "FC6", "Fp1", "P3", "C4"]
FM_WIN, FM_STRIDE = 10, 5  # 100 ms, 50 ms
BANDS = {"theta": (4.0, 8.0), "mu": (9.0, 11.0)}
BASE_S = 0.5  # pre-cue baseline: first 0.5 s of every segment (segments start 0.8 s pre-cue)


def band_power_signals(sd: SubjectData, chans: list[int]) -> dict[str, np.ndarray]:
    """Per-trial zero-phase band-pass + squared Hilbert envelope, and per-trial baseline."""
    out = {}
    for name, (lo, hi) in BANDS.items():
        sos = signal.butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
        P = np.empty((sd.bb.shape[0], len(chans)), np.float32)
        B = np.empty_like(P)
        for off, n in zip(sd.offset, sd.length):
            x = sd.bb[off:off + n][:, chans].astype(np.float64)
            env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x, axis=0), axis=0)) ** 2
            P[off:off + n] = env
            B[off:off + n] = env[: int(BASE_S * FS)].mean(0)
        out[name] = P
        out[name + "_base"] = B
    return out


def forcemap_features(sd: SubjectData, end: np.ndarray, bp: dict | None = None) -> np.ndarray:
    chans = [sd.ch_names.index(c) for c in FM_CHANNELS]
    bp = bp or band_power_signals(sd, chans)
    idx = end[:, None] - np.arange(FM_WIN - 1, -1, -1)[None, :]
    x = sd.bb[:, chans][idx].astype(np.float64)  # (n, T, C)
    erp = [x.mean(1), np.abs(x).mean(1), x.var(1), stats.skew(x, axis=1), stats.kurtosis(x, axis=1),
           np.trapezoid(x, dx=1.0 / FS, axis=1)]
    pw = []
    for name in BANDS:
        p = bp[name][idx].mean(1)
        base = bp[name + "_base"][end]
        pw += [np.log10(p + 1e-12), (p - base) / (base + 1e-12)]  # PSD and ERD/ERS
    F = np.concatenate(erp + pw, axis=1)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
