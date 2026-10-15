"""Small MLP (128, 64) on band-power features, early stopping on the validation set."""

from __future__ import annotations

import numpy as np
from torch import nn

from .base import Decoder, Selector, make_scaler
from .train_torch import fit_regressor, predict


def mlp_net(n_in: int, n_out: int, hidden=(128, 64)) -> nn.Module:
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class MLPDecoder(Decoder):
    feature = "bandpower"

    def __init__(self, select_k=None, select_fixed=None, max_epochs=200, patience=10, batch=512, seed=0,
                 feature="bandpower", pca=None):
        self.feature = feature
        self.sel = Selector(select_k, select_fixed)
        self.kw = dict(max_epochs=max_epochs, patience=patience, batch=batch, seed=seed, weight_decay=1e-4)
        self.pca = pca

    def _prep(self, X):
        Z = self.scaler.transform(self.sel.transform(X))
        return self.pca_.transform(Z) if self.pca else Z

    def fit(self, Xtr, ytr, Xva, yva):
        self.sel.fit(Xtr, ytr)
        self.scaler = make_scaler().fit(self.sel.transform(Xtr))
        if self.pca:
            from sklearn.decomposition import PCA

            self.pca_ = PCA(n_components=self.pca, svd_solver="full", random_state=0)
            self.pca_.fit(self.scaler.transform(self.sel.transform(Xtr)))
        Ztr, Zva = self._prep(Xtr), self._prep(Xva)
        y2 = ytr.reshape(len(ytr), -1)
        self.net, self.info = fit_regressor(mlp_net(Ztr.shape[1], y2.shape[1]), Ztr, y2, Zva, yva, **self.kw)
        self.info.pop("val_curve", None)
        self.info["n_features"] = int(Ztr.shape[1])
        return self

    def predict(self, X):
        return predict(self.net, self._prep(X).astype(np.float32))
