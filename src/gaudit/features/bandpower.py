"""Log band power per channel from Welch over each 500 ms window."""

import numpy as np
from scipy import signal

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "lowgamma": (30, 40)}
# premotor and parietal channels (FC, C, CP, P families) of the 32-channel cap
PREMOTOR_PARIETAL = ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6",
                     "P7", "P3", "Pz", "P4", "P8"]


def bandpower(win: np.ndarray, fs: int = 100, chunk: int = 50000) -> np.ndarray:
    """win: (n, C, T) -> (n, C*len(BANDS)) log10 band power, band-major."""
    out = []
    for i in range(0, len(win), chunk):
        f, p = signal.welch(win[i:i + chunk], fs=fs, nperseg=win.shape[-1], nfft=2 * win.shape[-1], axis=-1)
        bp = [p[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS.values()]
        out.append(np.log10(np.stack(bp, axis=1) + 1e-12).reshape(len(p), -1))
    return np.concatenate(out).astype(np.float32)
