"""Shared decoder interface: fit(X_train, y_train, X_val, y_val) and predict(X).

Targets arrive z-scored with training statistics; predictions are returned on that scale.
Every fitted step (scaler, selection, PCA, Riemannian reference, alpha, early stopping)
sees training data only, plus validation data for hyperparameter choice. Never test data.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..features.selection import corr_rank


class Decoder:
    feature: str = ""  # feature kind the runner must supply
    info: dict

    def fit(self, Xtr, ytr, Xva, yva):
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError


class Selector:
    """Optional top-k selection; `fixed` = indices chosen outside (leaky select_on=all)."""

    def __init__(self, k: int | None = None, fixed: np.ndarray | None = None):
        self.k, self.fixed = k, fixed

    def fit(self, X, y):
        if self.fixed is not None:
            self.idx_ = np.asarray(self.fixed)
        elif self.k:
            self.idx_ = corr_rank(X, y, self.k)
        else:
            self.idx_ = None
        return self

    def transform(self, X):
        return X if self.idx_ is None else X[:, self.idx_]


def make_scaler() -> StandardScaler:
    return StandardScaler()
