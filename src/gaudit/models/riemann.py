"""Tangent-space ridge on OAS window covariances; reference mean fit on training windows only."""

from __future__ import annotations

import numpy as np

from ..features.covariance import TangentSpace
from .base import Decoder, make_scaler
from .ridge import RidgeVal


class RiemannDecoder(Decoder):
    feature = "cov"

    def __init__(self, seed=0, channels=None):
        self.seed, self.channels = seed, channels

    def _sub(self, C):
        if self.channels is None:
            return C
        ch = np.asarray(self.channels)
        return C[:, ch][:, :, ch]

    def fit(self, Xtr, ytr, Xva, yva):
        Ctr = self._sub(Xtr)
        self.ts = TangentSpace(seed=self.seed).fit(Ctr)
        Ttr = self.ts.transform(Ctr)
        Tva = self.ts.transform(self._sub(Xva))
        self.scaler = make_scaler().fit(Ttr)
        self.reg = RidgeVal().fit(self.scaler.transform(Ttr), ytr, self.scaler.transform(Tva), yva)
        self.info = {"alpha": self.reg.alpha_.tolist(), "n_features": int(Ttr.shape[1])}
        return self

    def predict(self, X):
        return self.reg.predict(self.scaler.transform(self.ts.transform(self._sub(X))))
