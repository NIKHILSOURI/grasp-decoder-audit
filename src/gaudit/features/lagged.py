"""Lagged low-frequency EEG (classic linear kinematics decoding)."""

import numpy as np

LAGS = (0, 5, 10, 15, 20, 25)  # samples at 100 Hz: 0..250 ms back in 50 ms steps


def lagged(x: np.ndarray, end: np.ndarray, lags=LAGS, channels=None) -> np.ndarray:
    """x: (N, C) signal; end: window last-sample indices -> (n, C*len(lags)), lag-major."""
    if channels is not None:
        x = x[:, channels]
    return np.concatenate([x[end - lag] for lag in lags], axis=1).astype(np.float32)


def raw_windows(x: np.ndarray, end: np.ndarray, win: int = 50, channels=None) -> np.ndarray:
    """(n, C, win) windows ending at `end` (inclusive)."""
    if channels is not None:
        x = x[:, channels]
    idx = end[:, None] - np.arange(win - 1, -1, -1)[None, :]
    return np.ascontiguousarray(x[idx].transpose(0, 2, 1)).astype(np.float32)
