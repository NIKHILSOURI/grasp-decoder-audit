"""Top-k channel/feature selection by absolute Pearson correlation with the target.

`select_on='train'` fits inside the decoder on training windows only (honest);
`select_on='all'` is computed on every window of the pool, test included (leaky on purpose).
"""

import numpy as np


def corr_rank(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    y = y.reshape(len(y), -1).mean(1)  # multi-output: rank by the mean target
    Xc = X - X.mean(0)
    yc = y - y.mean()
    r = (Xc * yc[:, None]).sum(0) / (np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12)
    return np.sort(np.argsort(-np.abs(r), kind="stable")[:k])
