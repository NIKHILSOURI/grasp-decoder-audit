"""Causal grasp-onset classifiers and the online detection rule.

Features at a window ending at time t use only stream samples at or before t, from the
causally filtered streams (see data/preprocess.py: online_filters):
  * `lda`:     lagged 0.1-3 Hz EEG (6 lags, 0-250 ms back) -> shrinkage LDA
  * `riemann`: OAS covariance of the last 500 ms of 1-40 Hz EEG -> tangent space
               (reference mean fit on training windows) -> logistic regression
Detection: probability above threshold for 3 consecutive 20 ms steps.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..features.covariance import TangentSpace, oas_cov
from ..features.lagged import LAGS

FS = 100
WIN = 50


def stream_index(t: np.ndarray) -> np.ndarray:
    """Window-end time (s, series clock) -> 100 Hz stream index (the sample at or before t)."""
    return np.floor(np.round(t * FS, 6)).astype(np.int64)


def features(stream: dict, series: np.ndarray, t: np.ndarray, kind: str, channels=None) -> np.ndarray:
    X = None
    for s in np.unique(series):
        m = series == s
        j = stream_index(t[m])
        key = ("lf_S" if kind == "lda" else "bb_S") + str(int(s))
        x = stream[key] if channels is None else stream[key][:, channels]
        assert j.min() >= WIN and j.max() < len(x), "window outside the stream"
        if kind == "lda":
            f = np.concatenate([x[j - lag] for lag in LAGS], axis=1)
        else:
            idx = j[:, None] - np.arange(WIN - 1, -1, -1)[None, :]  # past samples only, ends at j
            f = oas_cov(np.ascontiguousarray(x[idx].transpose(0, 2, 1))).astype(np.float32)
        if X is None:
            X = np.empty((len(t),) + f.shape[1:], np.float32)
        X[m] = f
    return X


class OnsetLDA:
    kind = "lda"

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])
        self.m.fit(self.sc.transform(X), y)
        return self

    def proba(self, X):
        return self.m.predict_proba(self.sc.transform(X))[:, 1]


class OnsetRiemann:
    kind = "riemann"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        self.ts = TangentSpace(seed=self.seed).fit(X)
        T = self.ts.transform(X)
        self.sc = StandardScaler().fit(T)
        self.m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        self.m.fit(self.sc.transform(T), y)
        return self

    def proba(self, X):
        return self.m.predict_proba(self.sc.transform(self.ts.transform(X)))[:, 1]


def detection_events(p: np.ndarray, thr: float, n_consec: int = 3) -> np.ndarray:
    """Indices where the run of consecutive above-threshold steps first reaches n_consec."""
    above = p > thr
    run = np.zeros(len(p), np.int32)
    c = 0
    for i, a in enumerate(above):
        c = c + 1 if a else 0
        run[i] = c
    return np.flatnonzero(run == n_consec)
