"""Ridge regression with the penalty chosen per output on the validation set."""

from __future__ import annotations

import numpy as np

from .base import Decoder, Selector, make_scaler

ALPHAS = np.logspace(-2, 4, 13)


class RidgeVal:
    """Closed-form ridge for a grid of alphas via one eigendecomposition of the Gram matrix."""

    def __init__(self, alphas=ALPHAS):
        self.alphas = np.asarray(alphas)

    def fit(self, Z, y, Zva, yva):
        y = y.reshape(len(y), -1)
        yva = yva.reshape(len(yva), -1)
        self.zmu_, self.ymu_ = Z.mean(0), y.mean(0)
        Zc = (Z - self.zmu_).astype(np.float64)
        G = Zc.T @ Zc
        b = Zc.T @ (y - self.ymu_)
        w, V = np.linalg.eigh(G)
        Vb = V.T @ b
        Zva_c = (Zva - self.zmu_).astype(np.float64)
        best = np.full(y.shape[1], np.inf)
        self.coef_ = np.zeros((Z.shape[1], y.shape[1]))
        self.alpha_ = np.zeros(y.shape[1])
        for a in self.alphas:
            coef = V @ (Vb / (np.maximum(w, 0) + a)[:, None])
            mse = (((Zva_c @ coef + self.ymu_) - yva) ** 2).mean(0)
            better = mse < best
            best[better] = mse[better]
            self.coef_[:, better] = coef[:, better]
            self.alpha_[better] = a
        return self

    def predict(self, Z):
        return (Z - self.zmu_) @ self.coef_ + self.ymu_


class RidgeDecoder(Decoder):
    feature = "lagged"

    def __init__(self, select_k=None, select_fixed=None, feature="lagged"):
        self.feature = feature
        self.sel = Selector(select_k, select_fixed)

    def fit(self, Xtr, ytr, Xva, yva):
        self.sel.fit(Xtr, ytr)
        Xtr, Xva = self.sel.transform(Xtr), self.sel.transform(Xva)
        self.scaler = make_scaler().fit(Xtr)
        self.reg = RidgeVal().fit(self.scaler.transform(Xtr), ytr, self.scaler.transform(Xva), yva)
        self.info = {"alpha": self.reg.alpha_.tolist(), "n_features": int(Xtr.shape[1])}
        return self

    def predict(self, X):
        return self.reg.predict(self.scaler.transform(self.sel.transform(X)))
